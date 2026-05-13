from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

from incremental_ad.core.cli import pluck

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
