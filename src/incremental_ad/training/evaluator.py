from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

from incremental_ad.core.cli import pluck

Split = Literal["val", "test"]


@dataclass
class EvalConfig:
    seed: int
    split: Split


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], required=True)


def make_config(args: Namespace) -> EvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "eval")
    return EvalConfig(**fields)
