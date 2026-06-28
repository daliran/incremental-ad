from typing import Any

import numpy as np
from torch import Tensor

from incremental_ad.framework.contracts.evaluator import Evaluator


class AdValEvaluator(Evaluator[tuple[Tensor, Any]]):
    """Accumulates per-window reconstruction scores and returns summary statistics. Used for val monitoring."""

    def __init__(self) -> None:
        self._scores: list[float] = []

    def update(self, outputs: tuple[Tensor, Any]) -> None:
        """outputs: (scores [B], target) — target is ignored (reconstruction target or None)."""
        scores, _ = outputs
        self._scores.extend(scores.tolist())

    def compute(self) -> dict[str, float]:
        scores = np.array(self._scores)
        return {
            "reconstruction/score_mean": float(np.mean(scores)),
            "reconstruction/score_std": float(np.std(scores)),
            "reconstruction/score_p50": float(np.percentile(scores, 50)),
            "reconstruction/score_p95": float(np.percentile(scores, 95)),
            "reconstruction/score_p99": float(np.percentile(scores, 99)),
        }

    def reset(self) -> None:
        self._scores = []
