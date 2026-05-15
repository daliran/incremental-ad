from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset

from incremental_ad.core.cli import pluck

Split = Literal["val", "test"]

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATASET_NAME = "SWaT"
N_FEATURES = 51

Normalization = Literal["standard", "none"]


@dataclass
class SWaTConfig:
    window_len: int
    stride: int
    normalization: Normalization


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--swat-window-len", type=int, required=True)
    parser.add_argument("--swat-stride", type=int, required=True)
    parser.add_argument(
        "--swat-normalization", choices=["standard", "none"], required=True
    )


def make_config(args: Namespace) -> SWaTConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "swat")
    return SWaTConfig(**fields)


def load_train(config: SWaTConfig) -> tuple[Dataset, Dataset]:
    """Returns (train_dataset, val_dataset)."""
    raise NotImplementedError


def load_eval(config: SWaTConfig, split: Split) -> Dataset:
    """Returns the dataset for the requested evaluation split."""
    raise NotImplementedError
