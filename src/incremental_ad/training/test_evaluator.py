import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from incremental_ad.core.cli import pluck
from incremental_ad.datasets.base_dataset import BaseDataset
from incremental_ad.models.base_model import BaseModel
from incremental_ad.training import metrics, scoring
from incremental_ad.training.run_debug import RunDebugger

log = logging.getLogger(__name__)

ThresholdStrategy = Literal["oracle", "train_percentile"]


@dataclass
class TestEvalConfig:
    seed: int
    batch_size: int
    threshold_strategy: ThresholdStrategy
    threshold_percentile: float


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--test-eval-seed", type=int, required=True)
    parser.add_argument("--test-eval-batch-size", type=int, required=True)
    parser.add_argument(
        "--test-eval-threshold-strategy",
        choices=["oracle", "train_percentile"],
        required=True,
    )
    parser.add_argument("--test-eval-threshold-percentile", type=float, required=True)


def make_config(args: Namespace) -> TestEvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "test_eval")
    return TestEvalConfig(**fields)


class TestEvaluator:
    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        eval_loader: DataLoader,
        train_loader: DataLoader,
        run_dir: Path,
        run_id: str,
        config: TestEvalConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.eval_loader = eval_loader
        self.train_loader = train_loader
        self.run_dir = run_dir
        self.run_id = run_id
        self.config = config

    def load_checkpoint(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["model_state"])
        log.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(best_val_loss={ckpt['metrics']['best_val_loss']:.6f})"
        )

    def evaluate(self) -> None:
        self._evaluate_test()

    def _resolve_threshold(self) -> float | None:
        if self.config.threshold_strategy != "train_percentile":
            return None

        # collecting the distribution from the training set.
        # this uses the evaluator stride, not the training stride.
        train_scores = scoring.collect_scores(
            self.model,
            self.train_loader,
            self.device,
            desc="Computing threshold (train)",
        )

        # calculating the percentile from the training set distribution.
        threshold = float(np.percentile(train_scores, self.config.threshold_percentile))

        log.info(
            f"Threshold (p{self.config.threshold_percentile:.0f} of train scores): "
            f"{threshold:.6f}"
        )

        return threshold

    def _evaluate_test(self) -> None:
        # IMPORTANT: most of the metrics here rely on the fact that stride = 1.

        log.info("Running test evaluation...")

        threshold = self._resolve_threshold()

        # scores per window.
        # with stride = 1 the number of windows ~ the number of timesteps.
        scores = scoring.collect_scores(
            self.model, self.eval_loader, self.device, desc="Scoring (test)"
        )

        dataset = cast(BaseDataset, self.eval_loader.dataset)

        if dataset.labels is None:
            raise ValueError("Test dataset must have labels")

        # extract timestep labels and windows size from the dataset loaded into the dataloader.
        labels = dataset.labels.numpy()
        window_size = dataset.window_size

        # the label assigned to each window/timestep is the label of the last timestep in the window.
        timestep_labels = labels[window_size - 1 :]

        # a window is anomalous if at least one timestep inside it is anomalous.
        window_labels = (
            np.lib.stride_tricks.sliding_window_view(labels, window_size)
            .any(axis=1)
            .astype(int)
        )

        results: dict = {
            "window": metrics.eval_classification(scores, window_labels, threshold),
            "point": metrics.eval_classification(scores, timestep_labels, threshold),
            "point_adjusted": metrics.eval_point_adjusted(
                scores, timestep_labels, threshold
            ),
            "event": metrics.eval_event(scores, timestep_labels, threshold),
        }

        # Resolve a concrete threshold value for debug output.
        # Oracle mode picks the best window-level F1 threshold; train_percentile already has one.
        debug_threshold = (
            threshold
            if threshold is not None
            else metrics.find_best_f1_threshold(scores, window_labels)
        )

        RunDebugger(
            run_dir=self.run_dir,
            model=self.model,
            device=self.device,
            eval_dataset=dataset,
        ).run(scores, window_labels, debug_threshold)

        self._report(results, scores, window_labels, timestep_labels)

    def _report(
        self,
        results: dict,
        scores: np.ndarray,
        window_labels: np.ndarray,
        timestep_labels: np.ndarray,
    ) -> None:

        # save results on file
        path = self.run_dir / "test_results.json"
        path.write_text(json.dumps(results, indent=2))
        log.info(f"Test results saved to {path}")

        # flatten the nested dictionary into a one level dictionary putting a prefix in each key.
        flat = {
            f"{level}/{k}": v
            for level, metrics_dict in results.items()
            for k, v in metrics_dict.items()
        }

        # log each flattened key.
        for k, v in flat.items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.4f}")
            else:
                log.info(f"  {k}: {v}")

        # send each flattened key to wandb.
        if wandb.run is not None:
            wandb.run.summary.update(
                {k: v for k, v in flat.items() if isinstance(v, (float, int))}
            )

        # log curves on wandb
        self._log_curves(scores, window_labels, prefix="window")
        self._log_curves(scores, timestep_labels, prefix="point")

    def _log_curves(self, scores: np.ndarray, labels: np.ndarray, prefix: str) -> None:
        if wandb.run is None:
            return

        # wandb's built-in curve plots expect class probabilities in [0, 1].
        # min-max normalise the scores so the relative ordering is preserved.
        lo, hi = scores.min(), scores.max()
        scores_norm = (scores - lo) / (hi - lo + 1e-8)

        # shape: (n_samples, 2) — column 0: P(normal), column 1: P(anomaly).
        y_prob = np.column_stack([1.0 - scores_norm, scores_norm])

        wandb.log(
            {
                f"{prefix}/roc_curve": wandb.plot.roc_curve(
                    labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
                f"{prefix}/pr_curve": wandb.plot.pr_curve(
                    labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
            }
        )
