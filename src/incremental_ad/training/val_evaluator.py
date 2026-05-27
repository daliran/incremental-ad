import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from incremental_ad.core.cli import pluck
from incremental_ad.models.base_model import BaseModel
from incremental_ad.training import scoring

log = logging.getLogger(__name__)


@dataclass
class ValEvalConfig:
    seed: int
    batch_size: int


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--val-eval-seed", type=int, required=True)
    parser.add_argument("--val-eval-batch-size", type=int, required=True)


def make_config(args: Namespace) -> ValEvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "val_eval")
    return ValEvalConfig(**fields)


class ValEvaluator:
    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        val_loader: DataLoader,
        run_dir: Path,
        run_id: str,
    ) -> None:
        self.model = model
        self.device = device
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.run_id = run_id

    def load_checkpoint(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model_state"])
        log.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(best_val_loss={ckpt['metrics']['best_val_loss']:.6f})"
        )

    def evaluate(self) -> None:
        log.info("Running val reconstruction evaluation...")

        # scores per window — same stochastic eval as used for the test set.
        scores = scoring.collect_scores(
            self.model, self.val_loader, self.device, desc="Scoring (val reconstruction)"
        )

        stats = {
            "val_reconstruction/score_mean": float(np.mean(scores)),
            "val_reconstruction/score_std": float(np.std(scores)),
            "val_reconstruction/score_p50": float(np.percentile(scores, 50)),
            "val_reconstruction/score_p95": float(np.percentile(scores, 95)),
            "val_reconstruction/score_p99": float(np.percentile(scores, 99)),
        }

        for k, v in stats.items():
            log.info(f"  {k}: {v:.6f}")

        if wandb.run is not None:
            wandb.run.summary.update(stats)

        path = self.run_dir / "val_reconstruction.json"
        path.write_text(json.dumps(stats, indent=2))
        log.info(f"Val reconstruction results saved to {path}")
