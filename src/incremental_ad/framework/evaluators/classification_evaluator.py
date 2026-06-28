import torch
from torch import Tensor

from incremental_ad.framework.contracts.evaluator import Evaluator


class ClassificationEvaluator(Evaluator[tuple[Tensor, Tensor]]):
    """Accumulates (logits, labels) and computes accuracy and per-class macro-F1."""

    def __init__(self) -> None:
        self._logits: list[Tensor] = []
        self._labels: list[Tensor] = []

    def update(self, outputs: tuple[Tensor, Tensor]) -> None:
        """outputs: (logits [B, n_classes], labels [B])."""
        logits, labels = outputs
        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        logits = torch.cat(self._logits)          # [N, n_classes]
        labels = torch.cat(self._labels).long()   # [N]
        preds = logits.argmax(dim=1)              # [N]
        n_classes = logits.size(1)

        accuracy = (preds == labels).float().mean().item()

        # Macro-F1: average F1 over all classes
        f1_sum = 0.0
        for c in range(n_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + recall
            f1_sum += (2 * precision * recall / denom) if denom > 0 else 0.0

        return {
            "classification/accuracy": accuracy,
            "classification/macro_f1": f1_sum / n_classes,
        }

    def reset(self) -> None:
        self._logits = []
        self._labels = []
