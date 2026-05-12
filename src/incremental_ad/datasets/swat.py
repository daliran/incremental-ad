from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

from incremental_ad.core.cli import pluck

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATASET_NAME = "SWaT"

Normalization = Literal["standard", "none"]


@dataclass
class SWaTConfig:
    window_len: int
    stride: int
    normalization: Normalization


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument("--swat-window-len", type=int, required=True)
    parser.add_argument("--swat-stride", type=int, required=True)
    parser.add_argument(
        "--swat-normalization", choices=["standard", "none"], required=True
    )


def make_config(args: Namespace) -> SWaTConfig:
    fields = pluck(args, "swat")
    return SWaTConfig(**fields)
