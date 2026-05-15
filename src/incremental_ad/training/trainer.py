import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from incremental_ad.core.checkpoint import Metrics, save_checkpoint as _save_ckpt
from incremental_ad.core.cli import pluck
from incremental_ad.models.base_model import BaseModel

log = logging.getLogger(__name__)

OptimizerType = Literal["adamw"]
SchedulerType = Literal["cosine", "constant"]


@dataclass
class TrainingConfig:
    seed: int
    epochs: int
    patience: int
    batch_size: int
    optimizer: OptimizerType
    weight_decay: float
    learning_rate: float
    grad_clip: float
    scheduler: SchedulerType
    warmup_ratio: float


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--train-epochs", type=int, required=True)
    parser.add_argument("--train-patience", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--train-optimizer", choices=["adamw"], required=True)
    parser.add_argument("--train-weight-decay", type=float, required=True)
    parser.add_argument("--train-learning-rate", type=float, required=True)
    parser.add_argument("--train-grad-clip", type=float, required=True)
    parser.add_argument(
        "--train-scheduler", choices=["cosine", "constant"], required=True
    )
    parser.add_argument("--train-warmup-ratio", type=float, required=True)


def make_config(args: Namespace) -> TrainingConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "train")
    return TrainingConfig(**fields)


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_id: str,
        run_dir: Path,
        wandb_group: str,
        dataset_name: str,
        dataset_cfg: Any,
        model_name: str,
        model_cfg: Any,
        config: TrainingConfig,
    ) -> None:

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_id = run_id
        self.run_dir = run_dir
        self.wandb_group = wandb_group
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.config = config

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.start_epoch = 1
        self.early_stop_counter = 0
        self.best_val_loss = float("inf")

    def train(self) -> None:

        # Used when tqdm and logger are used at the same time to avoid conflicts.
        with logging_redirect_tqdm():
            for epoch in range(self.start_epoch, self.config.epochs + 1):

                # run an epoch on all batches of the training set
                train_loss, grad_norm = self._train_epoch(epoch)

                # run an epoch on all batches of the validation set
                val_loss = self._val_epoch(epoch)

                current_lr = self.optimizer.param_groups[0]["lr"]

                log.info(
                    f"Epoch {epoch}/{self.config.epochs} — "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f} lr={current_lr:.2e}"
                )

                wandb.log(
                    {
                        "loss/train": train_loss,
                        "loss/val": val_loss,
                        "train/lr": current_lr,
                        "train/grad_norm": grad_norm,
                    },
                    step=epoch,
                )

                self._save_checkpoint("last.pt", epoch, train_loss, val_loss)

                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0

                    self._save_checkpoint("best.pt", epoch, train_loss, val_loss)

                    if wandb.run is not None:
                        wandb.run.summary["best_val_loss"] = val_loss
                        wandb.run.summary["best_epoch"] = epoch

                    log.info(
                        f"New best checkpoint at epoch {epoch} "
                        f"(val_loss={val_loss:.6f})"
                    )

                else:
                    self.early_stop_counter += 1

                    log.info(
                        f"No improvement for {self.early_stop_counter}/{self.config.patience} epochs"
                    )

                if self.early_stop_counter >= self.config.patience:
                    log.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no val loss improvement for {self.config.patience} consecutive epochs)"
                    )
                    break

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        total_grad_norm = 0.0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)

        for batch in progress:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # computes loss (forward)
            loss = self.model.training_step(batch)

            # accumulate gradient (backward)
            loss.backward()

            # clip (this must happen after the backward and before the optimizer step)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip
            )

            # update the weights
            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if self.scheduler:
            self.scheduler.step()

        progress.close()

        n = len(self.train_loader)
        avg_loss = total_loss / n
        avg_grad_norm = total_grad_norm / n

        return avg_loss, avg_grad_norm

    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()

        total_loss = 0.0

        progress = tqdm(self.val_loader, desc=f"Epoch {epoch} [val]", leave=False)

        with torch.no_grad():
            for batch in progress:
                batch = batch.to(self.device)
                loss = self.model.training_step(batch)
                total_loss += loss.item()

                progress.set_postfix(loss=f"{loss.item():.4f}")

        progress.close()

        n = len(self.val_loader)
        avg_loss = total_loss / n

        return avg_loss

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer

        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer '{name}'")

    def _build_scheduler(self):
        name = self.config.scheduler

        if name == "cosine":
            warmup_epochs = max(
                1,
                int(self.config.warmup_ratio * self.config.epochs),
            )

            cosine_epochs = self.config.epochs - warmup_epochs

            # warmup phase
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            # normal scheduling phase
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
            )

            # combined scheduler putting together the two
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        elif name == "constant":
            return None
        else:
            raise ValueError(f"Unknown scheduler '{name}'")

    def _save_checkpoint(
        self, filename: str, epoch: int, train_loss: float, val_loss: float
    ) -> None:
        _save_ckpt(
            self.run_dir / "checkpoints" / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            run_id=self.run_id,
            wandb_group=self.wandb_group,
            dataset_name=self.dataset_name,
            dataset_cfg=self.dataset_cfg,
            model_name=self.model_name,
            model_cfg=self.model_cfg,
            train_cfg=self.config,
            metrics=Metrics(
                best_val_loss=self.best_val_loss,
                val_loss=val_loss,
                train_loss=train_loss,
            ),
        )

    def load_checkpoint(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model_state"])

        if ckpt["optimizer_state"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if ckpt["scheduler_state"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["metrics"]["best_val_loss"]

        self.early_stop_counter = 0

        log.info(
            f"Resumed from epoch {ckpt['epoch']} "
            f"(best_val_loss={self.best_val_loss:.6f})"
        )
