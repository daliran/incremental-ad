import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from incremental_ad.core.cli import pluck
from incremental_ad.datasets.base_dataset import BaseDataset, Split
from incremental_ad.models.base_model import BaseModel

log = logging.getLogger(__name__)

ThresholdStrategy = Literal["oracle", "train_percentile"]


@dataclass
class EvalConfig:
    seed: int
    split: Split
    batch_size: int
    threshold_strategy: ThresholdStrategy
    threshold_percentile: float


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)
    parser.add_argument(
        "--eval-threshold-strategy",
        choices=["oracle", "train_percentile"],
        required=True,
    )
    parser.add_argument(
        "--eval-threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile of train scores used as threshold (only for train_percentile strategy).",
    )


def make_config(args: Namespace) -> EvalConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "eval")
    return EvalConfig(**fields)


class Evaluator:
    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        eval_loader: DataLoader,
        train_loader: DataLoader,
        run_dir: Path,
        run_id: str,
        config: EvalConfig,
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
        if self.config.split == "val":
            self._evaluate_val()
        else:
            self._evaluate_test()

    def _collect_scores(self, loader: DataLoader, desc: str) -> np.ndarray:

        self.model.eval()

        # (batches/n_windows)
        # List lists.
        # Since batch size > 1, each sub list item is the score/reconstruction error of an individual batch/window.
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = batch.to(self.device)
                scores = self.model.eval_step(batch)

                # Add the scores to the list of lists
                all_scores.append(scores.cpu().numpy())

        # Flatten the list of lists to create a unique list of scores, removing the grouping from batch size.
        return np.concatenate(all_scores, axis=0)

    def _evaluate_val(self) -> None:
        log.info("Running val evaluation...")

        # scores per window.
        scores = self._collect_scores(self.eval_loader, desc="Scoring (val)")

        stats = {
            "val/score_mean": float(np.mean(scores)),
            "val/score_std": float(np.std(scores)),
            "val/score_p50": float(np.percentile(scores, 50)),
            "val/score_p95": float(np.percentile(scores, 95)),
            "val/score_p99": float(np.percentile(scores, 99)),
        }

        # Log the stats.
        for k, v in stats.items():
            log.info(f"  {k}: {v:.6f}")

        # Send the stats to wandb.
        if wandb.run is not None:
            wandb.run.summary.update(stats)

        # Save the scores, to allow plotting the data and select a threshold externally.
        scores_path = self.run_dir / "val_scores.npy"
        np.save(scores_path, scores)

        log.info(f"Val scores saved to {scores_path}")

    def _evaluate_test(self) -> None:
        # IMPORTANT: point-wise and event-wise metrics require dataset stride = 1.
        # with stride > 1, gaps appear in the per-timestep score sequence and those metrics become invalid.

        log.info("Running test evaluation...")

        # Resolve threshold before scoring the test set.
        fixed_threshold: float | None = None

        if self.config.threshold_strategy == "train_percentile":
            train_scores = self._collect_scores(
                self.train_loader, desc="Computing threshold (train)"
            )

            fixed_threshold = float(
                np.percentile(train_scores, self.config.threshold_percentile)
            )

            log.info(
                f"Threshold (p{self.config.threshold_percentile:.0f} of train scores): "
                f"{fixed_threshold:.6f}"
            )

        # scores per window.
        # with stride = 1 the number of windows ~ the number of timesteps.
        scores = self._collect_scores(self.eval_loader, desc="Scoring (test)")

        dataset = cast(BaseDataset, self.eval_loader.dataset)

        if dataset.labels is None:
            raise ValueError("Test dataset must have labels")

        # extract timestep labels and windows size from the dataset loaded into the dataloader.
        labels = dataset.labels.numpy()
        window_size = dataset.window_size

        # since the number of windows is ~ the number timesteps, this assigns to the window/timestep a label.
        # this is used to simulate score and label at timestep level (point wise).
        timestep_scores, timestep_labels = self._point_wise(scores, labels, window_size)

        # a window is anomalous if at least one timestep inside it is anomalous.
        n_windows = len(scores)

        window_labels = np.array(
            [int(labels[i : i + window_size].any()) for i in range(n_windows)]
        )

        results: dict = {"threshold_strategy": self.config.threshold_strategy}
        if fixed_threshold is not None:
            results["threshold"] = {
                "value": fixed_threshold,
                "percentile": self.config.threshold_percentile,
            }

        results["window"] = self._eval_window(scores, window_labels, fixed_threshold)
        results["point"] = self._eval_point(timestep_scores, timestep_labels, fixed_threshold)
        results["point_adjusted"] = self._eval_point_adjusted(
            timestep_scores, timestep_labels, fixed_threshold
        )
        results["event"] = self._eval_event(timestep_scores, timestep_labels, fixed_threshold)

        self._log_test_results(results)
        self._save_test_results(results)

        self._log_curves(scores, window_labels, prefix="window")
        self._log_curves(timestep_scores, timestep_labels, prefix="point")

    def _point_wise(
        self, scores: np.ndarray, labels: np.ndarray, window_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # the label assigned to each window/timestep is the label of the last timestep in the window.
        # this is a convention.
        return scores, labels[window_size - 1 :]

    def _eval_window(
        self, scores: np.ndarray, window_labels: np.ndarray, threshold: float | None
    ) -> dict:
        if threshold is None:
            threshold = _find_best_f1_threshold(scores, window_labels)
        preds = (scores >= threshold).astype(int)

        return {
            "auroc": float(roc_auc_score(window_labels, scores)),
            "auprc": float(average_precision_score(window_labels, scores)),
            "f1": float(f1_score(window_labels, preds, zero_division=0)),
            "precision": float(precision_score(window_labels, preds, zero_division=0)),
            "recall": float(recall_score(window_labels, preds, zero_division=0)),
        }

    def _eval_point(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float | None
    ) -> dict:
        if threshold is None:
            threshold = _find_best_f1_threshold(scores, labels)
        preds = (scores >= threshold).astype(int)

        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        }

    def _eval_point_adjusted(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float | None
    ) -> dict:
        if threshold is not None:
            # Fixed threshold: apply PA directly without sweeping.
            preds = (scores >= threshold).astype(int)
            preds_pa = _apply_point_adjustment(preds, labels)
            return {
                "f1": float(f1_score(labels, preds_pa, zero_division=0)),
                "precision": float(precision_score(labels, preds_pa, zero_division=0)),
                "recall": float(recall_score(labels, preds_pa, zero_division=0)),
            }

        # Oracle: sweep all exact threshold candidates, applying PA at each step.
        _, _, all_thresholds = precision_recall_curve(labels, scores)

        # Downsample to at most 500 candidates: the PA-F1 curve is smooth so
        # evenly-spaced sampling loses negligible accuracy while avoiding O(N²).
        n_candidates = 500
        if len(all_thresholds) > n_candidates:
            idx = np.linspace(0, len(all_thresholds) - 1, n_candidates, dtype=int)
            thresholds = all_thresholds[idx]
        else:
            thresholds = all_thresholds

        best: dict = {"f1": -1.0, "precision": 0.0, "recall": 0.0}

        for t in thresholds:
            preds = (scores >= t).astype(int)

            preds_pa = _apply_point_adjustment(preds, labels)

            f = float(f1_score(labels, preds_pa, zero_division=0))

            if f > best["f1"]:
                best = {
                    "f1": f,
                    "precision": float(
                        precision_score(labels, preds_pa, zero_division=0)
                    ),
                    "recall": float(recall_score(labels, preds_pa, zero_division=0)),
                }

        return best

    def _eval_event(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float | None
    ) -> dict:
        if threshold is None:
            threshold = _find_best_f1_threshold(scores, labels)
        segments = _find_segments(labels)

        if len(segments) == 0:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        detected = scores >= threshold

        tp = sum(1 for s, e in segments if detected[s:e].any())
        fn = len(segments) - tp

        # FP: contiguous detected regions that don't overlap any anomaly segment
        pred_segments = _find_segments(detected.astype(int))
        fp = sum(
            1
            for ps, pe in pred_segments
            if not any(s < pe and ps < e for s, e in segments)
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "n_segments": len(segments),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def _log_test_results(self, results: dict) -> None:

        # flatten the nested dictionary into a one level dictionary putting a prefix in each key.
        # skip non-dict entries (e.g. threshold_strategy string).
        flat = {
            f"{level}/{k}": v
            for level, metrics in results.items()
            if isinstance(metrics, dict)
            for k, v in metrics.items()
        }

        # log each flattened key.
        for k, v in flat.items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.4f}")
            else:
                log.info(f"  {k}: {v}")

        # send each flattend key to wandb.
        if wandb.run is not None:
            wandb.run.summary.update(
                {k: v for k, v in flat.items() if isinstance(v, float)}
            )

    def _save_test_results(self, results: dict) -> None:
        path = self.run_dir / "test_results.json"
        path.write_text(json.dumps(results, indent=2))
        log.info(f"Test results saved to {path}")

    def _log_curves(self, scores: np.ndarray, labels: np.ndarray, prefix: str) -> None:
        if wandb.run is None:
            return

        # wandb's built-in curve plots expect class probabilities in [0, 1].
        # min-max normalise the scores so the relative ordering is preserved.
        lo, hi = scores.min(), scores.max()
        scores_norm = (scores - lo) / (hi - lo + 1e-8)

        # shape: (n_samples, 2) — column 0: P(normal), column 1: P(anomaly).
        y_probas = np.column_stack([1.0 - scores_norm, scores_norm])

        wandb.log(
            {
                f"{prefix}/roc_curve": wandb.plot.roc_curve(
                    labels.tolist(), y_probas.tolist(), labels=["normal", "anomaly"]
                ),
                f"{prefix}/pr_curve": wandb.plot.pr_curve(
                    labels.tolist(), y_probas.tolist(), labels=["normal", "anomaly"]
                ),
            }
        )


def _find_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact best-F1 threshold derived from the precision-recall curve (O(N log N))."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1s = np.zeros_like(denom)
    np.divide(2 * precision[:-1] * recall[:-1], denom, out=f1s, where=denom > 0)
    return float(thresholds[np.argmax(f1s)])


def _apply_point_adjustment(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """If any timestep in an anomalous segment is detected, mark the entire segment as detected."""
    adjusted = preds.copy()
    for start, end in _find_segments(labels):
        if preds[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def _find_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Returns (start, end) index pairs for each contiguous run of 1s. End is exclusive."""
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(labels):
        if val == 1 and not in_segment:
            in_segment = True
            start = i
        elif val == 0 and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(labels)))

    return segments
