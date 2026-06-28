import torch
import torch.nn.functional as F
from torch import Tensor

from incremental_ad.framework.contracts.evaluator import Evaluator


class ImputationEvaluator(Evaluator[tuple[Tensor, Tensor]]):
    """Accumulates (reconstructions, originals) and computes MSE / MAE / RMSE."""

    def __init__(self) -> None:
        self._recons: list[Tensor] = []
        self._originals: list[Tensor] = []

    def update(self, outputs: tuple[Tensor, Tensor]) -> None:
        """outputs: (predictions [B, n_masked*patch_len, F], originals [B, n_masked*patch_len, F])
        — values at the masked positions only, so MSE reflects true imputation quality."""
        recon, original = outputs
        self._recons.append(recon.detach().cpu())
        self._originals.append(original.detach().cpu())

    def compute(self) -> dict[str, float]:
        recon = torch.cat(self._recons)
        original = torch.cat(self._originals)
        mse = F.mse_loss(recon, original).item()
        mae = F.l1_loss(recon, original).item()
        return {
            "imputation/mse": mse,
            "imputation/rmse": mse ** 0.5,
            "imputation/mae": mae,
        }

    def reset(self) -> None:
        self._recons = []
        self._originals = []
