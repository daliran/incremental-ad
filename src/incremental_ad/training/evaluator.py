from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from incremental_ad.core.cli import pluck


@dataclass
class EvalConfig:
    seed: int


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--eval-seed", type=int, required=True)


def make_config(args: Namespace) -> EvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "eval")
    return EvalConfig(**fields)
