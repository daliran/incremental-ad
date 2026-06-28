import numpy as np
import torch
import wandb
from torch import Tensor

from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.evaluators._metrics import (
    best_f1_threshold,
    eval_classification,
    eval_event,
    eval_point,
    eval_point_adjusted,
)


class AdTestEvaluator(Evaluator[tuple[Tensor, Tensor]]):
    """
    Accumulates per-window anomaly scores and binary labels, then computes:
      - Window-level: AUROC, AUPRC, F1 / precision / recall
      - Point-adjusted (PA): F1 / precision / recall
      - Event-level: F1 / precision / recall + TP / FP / FN

    threshold_strategy="oracle"     → best-F1 sweep over PR-curve candidates
    threshold_strategy="percentile" → fixed threshold from set_reference_scores()
    """

    def __init__(
        self,
        threshold_strategy: str = "oracle",
        threshold_percentile: float = 95.0,
    ) -> None:
        self._threshold_strategy = threshold_strategy
        self._threshold_percentile = threshold_percentile
        self._threshold: float | None = None
        self._scores: list[Tensor] = []
        self._labels: list[Tensor] = []
        self._last_scores: np.ndarray | None = None
        self._last_labels: np.ndarray | None = None
        self._last_point_labels: np.ndarray | None = None
        self._last_threshold: float | None = None

    def needs_reference_scores(self) -> bool:
        return self._threshold_strategy == "percentile"

    def set_reference_scores(self, scores: np.ndarray) -> None:
        """Compute and store the percentile threshold from reference (baseline) scores."""
        self._threshold = float(np.percentile(scores, self._threshold_percentile))

    def update(self, outputs: tuple[Tensor, Tensor]) -> None:
        """outputs: (scores [B], labels [B, W]) — per-window anomaly score and full label slice."""
        scores, labels = outputs
        self._scores.append(scores.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        scores = torch.cat(self._scores).numpy().astype(np.float64)
        labels_2d = torch.cat(self._labels).numpy().astype(np.int32)  # [N, W]
        window_labels = labels_2d.max(axis=1)                          # [N]
        point_labels = labels_2d[:, -1]                                # [N] last timestep per window

        # Oracle mode: pass None so each function sweeps its own optimal threshold.
        # Percentile mode: pass the fixed threshold to all.
        per_metric_threshold = self._threshold if self._threshold_strategy == "percentile" else None

        # Window-oracle threshold is used for debug/wandb visualisation only.
        viz_threshold = (
            self._threshold
            if self._threshold_strategy == "percentile"
            else best_f1_threshold(scores, window_labels)
        )

        self._last_scores = scores
        self._last_labels = window_labels
        self._last_point_labels = point_labels
        self._last_threshold = viz_threshold

        window = eval_classification(scores, window_labels, per_metric_threshold)
        point = eval_point(scores, point_labels, per_metric_threshold)
        pa = eval_point_adjusted(scores, point_labels, per_metric_threshold)
        event = eval_event(scores, point_labels, per_metric_threshold)

        return {
            "window_auroc": window["auroc"],
            "window_auprc": window["auprc"],
            "window_f1": window["f1"],
            "window_precision": window["precision"],
            "window_recall": window["recall"],
            "point_auroc": point["auroc"],
            "point_auprc": point["auprc"],
            "point_f1": point["f1"],
            "point_precision": point["precision"],
            "point_recall": point["recall"],
            "pa_f1": pa["f1"],
            "pa_precision": pa["precision"],
            "pa_recall": pa["recall"],
            "event_f1": event["f1"],
            "event_precision": event["precision"],
            "event_recall": event["recall"],
            "event_tp": float(event["tp"]),
            "event_fp": float(event["fp"]),
            "event_fn": float(event["fn"]),
        }

    def debug_data(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Return (scores, labels, threshold) from the last compute() call, or None if not yet computed."""
        if self._last_scores is None:
            return None
        assert self._last_labels is not None
        assert self._last_threshold is not None
        return self._last_scores, self._last_labels, self._last_threshold

    def log_wandb_charts(self, step_name: str) -> None:
        if wandb.run is None or self._last_scores is None:
            return
        assert self._last_labels is not None
        assert self._last_point_labels is not None
        scores = self._last_scores
        lo, hi = float(scores.min()), float(scores.max())
        scores_norm = (scores - lo) / (hi - lo + 1e-8)
        y_prob = np.column_stack([1.0 - scores_norm, scores_norm])
        prefix = f"{step_name}/" if step_name else ""
        wandb.log(
            {
                f"{prefix}window/roc_curve": wandb.plot.roc_curve(
                    self._last_labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
                f"{prefix}window/pr_curve": wandb.plot.pr_curve(
                    self._last_labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
                f"{prefix}point/roc_curve": wandb.plot.roc_curve(
                    self._last_point_labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
                f"{prefix}point/pr_curve": wandb.plot.pr_curve(
                    self._last_point_labels.tolist(), y_prob.tolist(), labels=["normal", "anomaly"]
                ),
            }
        )

    def reset(self) -> None:
        self._scores = []
        self._labels = []
        self._last_scores = None
        self._last_labels = None
        self._last_point_labels = None
        self._last_threshold = None
