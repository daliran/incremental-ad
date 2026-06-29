"""Debug visualization for MaeTx forecasting.

Picks a handful of evaluation windows and plots, per feature, the visible context
followed by the actual vs. predicted future. Saves PNGs under step_dir/debug/forecast_samples/.

Uses the test set when available, otherwise falls back to the val set — so it works
even when the dataset has no held-out test split. (Note: the pipelines only invoke the
debugger after the test-eval stage, so in practice a test set must exist for it to run;
the val fallback covers callers that invoke it directly.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from incremental_ad.framework.contracts.dataset import Dataset, DatasetCapability
from incremental_ad.framework.contracts.debugger import Debugger
from incremental_ad.framework.contracts.evaluator import Evaluator

log = logging.getLogger(__name__)


class MaeTxForecastDebugger(Debugger):
    def __init__(self, n_samples: int = 6, max_features: int = 6) -> None:
        self._n_samples = n_samples
        self._max_features = max_features

    def run(
        self,
        model: Any,
        dataset: Any,
        evaluator: Evaluator[Any],
        step_dir: Path,
    ) -> None:
        if not isinstance(dataset, Dataset):
            return
        caps = dataset.capabilities
        if DatasetCapability.TEST in caps:
            eval_ds, split = dataset.get_test_dataset(), "test"
        elif DatasetCapability.VAL in caps:
            eval_ds, split = dataset.get_val_eval_dataset(), "val"
        else:
            return

        n = len(eval_ds)
        if n == 0:
            return

        device = next(model.parameters()).device
        out = step_dir / "debug" / "forecast_samples"
        out.mkdir(parents=True, exist_ok=True)

        idxs = np.unique(np.linspace(0, n - 1, min(self._n_samples, n)).astype(int))
        model.eval()
        with torch.no_grad():
            for rank, idx in enumerate(idxs):
                full_window, future = eval_ds[int(idx)]
                pred = model.score(full_window.unsqueeze(0).to(device))[0]  # [H, F]
                self._plot(
                    full_window.cpu().numpy(),
                    future.cpu().numpy(),
                    pred.cpu().numpy(),
                    out / f"{split}_{rank:02d}_win{int(idx)}.png",
                )

        self._log_to_wandb(out)
        log.info("  forecast debug: %d sample plots (%s split)", len(idxs), split)

    def _plot(
        self, full_window: np.ndarray, future: np.ndarray, pred: np.ndarray, path: Path
    ) -> None:
        # full_window [W, F]; future / pred [H, F]
        window_len, n_features = full_window.shape
        horizon = future.shape[0]
        context_len = window_len - horizon

        n_show = min(self._max_features, n_features)
        t_ctx = np.arange(context_len)
        t_fut = np.arange(context_len, window_len)

        fig, axes = plt.subplots(
            n_show, 1, figsize=(12, 2.2 * n_show), sharex=True, squeeze=False
        )
        for fi in range(n_show):
            ax = axes[fi, 0]
            ax.plot(t_ctx, full_window[:context_len, fi], color="#444444", lw=0.8, label="context")
            ax.plot(t_fut, future[:, fi], color="#0055cc", lw=1.3, label="actual")
            ax.plot(t_fut, pred[:, fi], color="#cc2222", lw=1.3, ls="--", label="predicted")
            ax.axvline(context_len - 0.5, color="#999999", lw=0.6, ls=":")
            ax.set_ylabel(f"feature {fi}", fontsize=7)
            ax.tick_params(labelsize=6)
            if fi == 0:
                ax.legend(fontsize=7, loc="upper left")
        axes[-1, 0].set_xlabel("timestep")

        mse = float(np.mean((pred - future) ** 2))
        mae = float(np.mean(np.abs(pred - future)))
        fig.suptitle(f"forecast sample — MSE={mse:.4f}  MAE={mae:.4f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    def _log_to_wandb(self, out: Path) -> None:
        if wandb.run is None:
            return
        imgs = {
            f"debug/forecast/{p.stem}": wandb.Image(str(p))
            for p in sorted(out.glob("*.png"))
        }
        if imgs:
            wandb.log(imgs)
