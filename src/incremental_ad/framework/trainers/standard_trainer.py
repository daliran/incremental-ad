import copy
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Self, Sized, cast

import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from incremental_ad.framework.contracts.dataset import DataLoaderConfig, Segment
from incremental_ad.framework.contracts.model import Model
from incremental_ad.framework.contracts.trainer import TrainSummary, Trainer
from incremental_ad.framework.core.device import move_to_device

log = logging.getLogger(__name__)

OptimizerType = Literal["adamw"]
SchedulerType = Literal["cosine", "constant"]


class StandardTrainer(Trainer):

    def __init__(
        self,
        n_epochs: int,
        patience: int,
        optimizer: OptimizerType,
        weight_decay: float,
        learning_rate: float,
        grad_clip: float,
        scheduler: SchedulerType,
        warmup_ratio: float,
        loader_config: DataLoaderConfig,
        device: str = "auto",
        checkpoint_interval: int = 0,
        reg_lambda: float = 0.0,
        reg_exclude: list[str] | None = None,
    ) -> None:
        self.n_epochs = n_epochs
        self.patience = patience
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.scheduler_type = scheduler
        self.warmup_ratio = warmup_ratio
        self.loader_config = loader_config
        self.checkpoint_interval = checkpoint_interval
        # reference_state regularization (L2-SP): penalize drift from a fixed anchor passed
        # into fit() (e.g. IncrementalTaskArithmeticPipeline's baseline_state). reg_lambda=0
        # (default) makes this a no-op, so plain baseline/StandardPipeline training is unaffected.
        self.reg_lambda = reg_lambda
        self.reg_exclude = reg_exclude if reg_exclude is not None else ["norm", "bias"]
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_n_epochs", type=int, required=True)
        parser.add_argument(f"--{p}_patience", type=int, required=True)
        parser.add_argument(f"--{p}_optimizer", choices=["adamw"], default="adamw")
        parser.add_argument(f"--{p}_weight_decay", type=float, required=True)
        parser.add_argument(f"--{p}_learning_rate", type=float, required=True)
        parser.add_argument(f"--{p}_grad_clip", type=float, required=True)
        parser.add_argument(f"--{p}_scheduler", choices=["cosine", "constant"], required=True)
        parser.add_argument(f"--{p}_warmup_ratio", type=float, default=0.1)
        parser.add_argument(f"--{p}_device", type=str, default="auto")
        parser.add_argument(f"--{p}_checkpoint_interval", type=int, default=0)
        parser.add_argument(
            f"--{p}_reg_lambda", type=float, default=0.0,
            help="L2-SP weight: penalizes drift from the reference_state passed into fit() "
                 "(e.g. baseline weights during incremental fine-tuning). 0 = disabled. "
                 "Has no effect unless the caller actually passes a reference_state.",
        )
        parser.add_argument(
            f"--{p}_reg_exclude", nargs="*", default=["norm", "bias"],
            help="Parameter-name substrings excluded from the reg_lambda penalty "
                 "(default excludes LayerNorm and bias params, per common L2-SP practice). "
                 "Pass no values to disable exclusion.",
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            n_epochs=getattr(cfg, f"{p}_n_epochs"),
            patience=getattr(cfg, f"{p}_patience"),
            optimizer=getattr(cfg, f"{p}_optimizer"),
            weight_decay=getattr(cfg, f"{p}_weight_decay"),
            learning_rate=getattr(cfg, f"{p}_learning_rate"),
            grad_clip=getattr(cfg, f"{p}_grad_clip"),
            scheduler=getattr(cfg, f"{p}_scheduler"),
            warmup_ratio=getattr(cfg, f"{p}_warmup_ratio"),
            loader_config=DataLoaderConfig.from_config(cfg),
            device=getattr(cfg, f"{p}_device"),
            checkpoint_interval=getattr(cfg, f"{p}_checkpoint_interval"),
            reg_lambda=getattr(cfg, f"{p}_reg_lambda"),
            reg_exclude=getattr(cfg, f"{p}_reg_exclude"),
        )

    def fit(
        self,
        model: Model,
        segment: Segment,
        checkpoint_dir: Path | None = None,
        secondary_loaders: dict[str, DataLoader] | None = None,
        step_name: str = "",
        step_offset: int = 0,
        reference_state: dict[str, Tensor] | None = None,
    ) -> TrainSummary:

        if len(cast(Sized, segment.train)) == 0:
            raise ValueError(
                f"Training segment has 0 windows (step_name={step_name!r}). "
                "Check dataset split config: window_len or stride may be too large for the segment size."
            )

        # A non-None but empty val set silently makes the val loss inf every epoch, which
        # corrupts early-stopping/checkpoint selection (it keeps epoch-1 weights). Fail loudly
        # instead — this happens when the val slice is shorter than window_len.
        if segment.val is not None and len(cast(Sized, segment.val)) == 0:
            raise ValueError(
                f"Validation segment has 0 windows (step_name={step_name!r}). "
                "The val slice is shorter than window_len, so early stopping would be meaningless. "
                "Increase the val fraction, reduce n_finetune_segments, or reduce window_len so each "
                "segment's val length (val_fraction × segment_size) exceeds the window length."
            )

        train_loader = self.loader_config.make_loader(segment.train, shuffle=True)

        val_loader = (
            self.loader_config.make_loader(segment.val, shuffle=False)
            if segment.val is not None
            else None
        )

        model.to(self.device)

        # Moved once per fit() call (not per-batch): one transient extra copy of the
        # reference weights on-device, released when this call returns.
        reference_state_dev = (
            {k: v.to(self.device) for k, v in reference_state.items()}
            if reference_state is not None
            else None
        )

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss: float | None = None
        best_epoch: int | None = None
        best_epoch_train_loss: float = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0
        final_train_loss = float("inf")
        best_train_loss = float("inf")
        final_val_loss: float | None = None
        final_secondary_losses: dict[str, float] = {}
        epochs_trained = 0

        epoch_w = len(str(self.n_epochs))
        prefix = f"{step_name}/" if step_name else ""
        label = f"[{step_name}] " if step_name else ""

        with logging_redirect_tqdm():
            for epoch in range(1, self.n_epochs + 1):
                epochs_trained = epoch

                train_loss, grad_norm = self._train_epoch(
                    model, train_loader, optimizer, epoch, reference_state_dev
                )

                final_train_loss = train_loss

                if train_loss < best_train_loss:
                    best_train_loss = train_loss

                if scheduler is not None:
                    scheduler.step()

                val_loss: float | None = None

                if val_loader is not None:
                    val_loss = self._compute_loader_loss(model, val_loader, reference_state_dev)
                    final_val_loss = val_loss

                if secondary_loaders:
                    final_secondary_losses = {
                        name: self._compute_loader_loss(model, loader, reference_state_dev)
                        for name, loader in secondary_loaders.items()
                    }

                if checkpoint_dir is not None:
                    self._save_checkpoint(
                        checkpoint_dir / "last.pt",
                        model.state_dict(),
                        epoch,
                        train_loss,
                        val_loss,
                    )

                    if (
                        self.checkpoint_interval > 0
                        and epoch % self.checkpoint_interval == 0
                    ):
                        self._save_checkpoint(
                            checkpoint_dir / f"epoch_{epoch:04d}.pt",
                            model.state_dict(),
                            epoch,
                            train_loss,
                            val_loss,
                        )

                lr = optimizer.param_groups[0]["lr"]

                is_best = False
                if val_loss is not None:
                    if best_val_loss is None or val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_epoch_train_loss = train_loss
                        best_state = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        is_best = True
                        if wandb.run is not None:
                            wandb.run.summary[f"{prefix}best_val_loss"] = val_loss
                            wandb.run.summary[f"{prefix}best_epoch"] = epoch
                    else:
                        epochs_no_improve += 1

                val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
                secondary_str = ""
                if final_secondary_losses:
                    items = "  ".join(
                        f"{k}={v:.4f}" for k, v in final_secondary_losses.items()
                    )
                    secondary_str = f"  [{items}]"
                pat_str = (
                    f"  pat={'best' if is_best else f'{epochs_no_improve}/{self.patience}'}"
                    if val_loss is not None
                    else ""
                )

                log.info(
                    f"{label}Epoch {epoch:{epoch_w}}/{self.n_epochs} "
                    f"— train={train_loss:.4f}{val_str}{secondary_str}  "
                    f"lr={lr:.2e}  gnorm={grad_norm:.3f}{pat_str}"
                )

                wandb_metrics: dict[str, float] = {
                    f"{prefix}loss/train": train_loss,
                    f"{prefix}train/lr": lr,
                    f"{prefix}train/grad_norm": grad_norm,
                }
                if val_loss is not None:
                    wandb_metrics[f"{prefix}loss/val"] = val_loss
                for name, loss in final_secondary_losses.items():
                    wandb_metrics[f"{prefix}loss/secondary/{name}"] = loss
                if wandb.run is not None:
                    wandb.log(wandb_metrics, step=step_offset + epoch)

                if val_loss is None:
                    continue

                if epochs_no_improve >= self.patience:
                    log.info(
                        f"{label}Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        checkpoint_path: Path | None = None

        if best_state is not None:
            model.load_state_dict(best_state)
            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "best.pt"
                self._save_checkpoint(
                    checkpoint_path,
                    best_state,
                    best_epoch,
                    best_epoch_train_loss,
                    best_val_loss,
                )

        return TrainSummary(
            final_train_loss=final_train_loss,
            best_train_loss=best_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            checkpoint_path=checkpoint_path,
            secondary_val_losses=final_secondary_losses,
        )

    @staticmethod
    def _save_checkpoint(
        path: Path,
        state_dict: dict,
        epoch: int | None,
        train_loss: float,
        val_loss: float | None,
    ) -> None:
        torch.save(
            {
                "model_state_dict": state_dict,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            path,
        )

    def _train_epoch(
        self,
        model: Model,
        loader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        reference_state: dict[str, Tensor] | None = None,
    ) -> tuple[float, float]:
        model.train()

        total_loss, total_gnorm, count = 0.0, 0.0, 0

        with tqdm(loader, desc=f"Epoch {epoch}", leave=False, unit="batch") as pbar:
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                optimizer.zero_grad()
                loss = model.compute_loss(batch) + self._reg_penalty(model, reference_state)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.grad_clip
                )
                optimizer.step()
                total_loss += loss.item()
                total_gnorm += gnorm.item()
                count += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = count if count > 0 else 1
        return total_loss / n, total_gnorm / n

    def _compute_loader_loss(
        self, model: Model, loader, reference_state: dict[str, Tensor] | None = None
    ) -> float:
        model.eval()

        total, count = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, self.device)
                total += model.compute_loss(batch).item()
                total += self._reg_penalty(model, reference_state).item()
                count += 1

        return total / count if count > 0 else float("inf")

    def _reg_penalty(
        self, model: Model, reference_state: dict[str, Tensor] | None
    ) -> Tensor:
        """L2-SP penalty: lambda * sum ||p - reference_state[name]||^2 over included params.
        No-op (returns 0) when there's no reference_state or reg_lambda is 0."""
        if reference_state is None or self.reg_lambda == 0.0:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for name, p in model.named_parameters():
            if any(pat in name for pat in self.reg_exclude):
                continue
            penalty = penalty + (p - reference_state[name]).pow(2).sum()
        return self.reg_lambda * penalty

    def _build_optimizer(self, model: Model) -> torch.optim.Optimizer:
        if self.optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer '{self.optimizer_type}'")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.scheduler_type == "cosine":
            warmup_epochs = max(1, int(self.warmup_ratio * self.n_epochs))
            cosine_epochs = self.n_epochs - warmup_epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            if cosine_epochs <= 0:
                return warmup
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        if self.scheduler_type == "constant":
            return None
        raise ValueError(f"Unknown scheduler '{self.scheduler_type}'")
