import torch
import torch.nn.functional as F
from torch import Tensor

from incremental_ad.framework.contracts.evaluator import Evaluator


class ForecastingEvaluator(Evaluator[tuple[Tensor, Tensor]]):
    """Accumulates (predictions, targets) and computes MSE / MAE / RMSE."""

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, outputs: tuple[Tensor, Tensor]) -> None:
        """outputs: (predictions [B, H, F], targets [B, H, F])."""
        preds, targets = outputs
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, float]:
        preds = torch.cat(self._preds)
        targets = torch.cat(self._targets)
        mse = F.mse_loss(preds, targets).item()
        mae = F.l1_loss(preds, targets).item()
        return {
            "forecast/mse": mse,
            "forecast/rmse": mse ** 0.5,
            "forecast/mae": mae,
        }

    def reset(self) -> None:
        self._preds = []
        self._targets = []
