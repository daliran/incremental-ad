import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader

from incremental_ad.core.cli import pluck
from incremental_ad.models.base_model import BaseModel

log = logging.getLogger(__name__)

Split = Literal["val", "test"]


@dataclass
class EvalConfig:
    seed: int
    split: Split
    batch_size: int


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)


def make_config(args: Namespace) -> EvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "eval")
    return EvalConfig(**fields)


class Evaluator:
    def __init__(
        self,
        model: BaseModel,
        loader: DataLoader,
        device: torch.device,
        config: EvalConfig,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device
        self.config = config

    def load_checkpoint(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model_state"])
        log.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(best_val_loss={ckpt['metrics']['best_val_loss']:.6f})"
        )

    def evaluate(self) -> None:
        if self.config.split == "val":
            self._evaluate_val()
        else:
            self._evaluate_test()

    def _evaluate_val(self) -> None:
        raise NotImplementedError

    def _evaluate_test(self) -> None:
        raise NotImplementedError
