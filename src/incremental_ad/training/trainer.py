import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from incremental_ad.core.cli import pluck

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
        model,
        run_dir: Path,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:

        self.model = model
        self.run_dir = run_dir
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.start_epoch = 1
        self.early_stop_counter = 0
        self.best_val_loss = float("inf")

        # todo checkpoint?

    def train(self) -> None:
        pass

    def _train_epoch(self):
        pass

    def _val_epoch(self):
        pass

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

            # combined scheuduler putting together the two
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler '{name}'")
