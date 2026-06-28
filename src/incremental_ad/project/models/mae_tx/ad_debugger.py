"""Debug visualization for the MaeTx anomaly detection pipeline.

Produces under step_dir/debug/:
  score_timeline.png       — anomaly score + threshold + GT labels over time
  score_distributions.png  — score histograms for normal vs anomaly windows
  event_analysis.csv       — per-GT-segment detection stats
  samples/<cat>_<rank>_<idx>.png — reconstruction heatmaps for FP / FN / TP windows
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from incremental_ad.framework.contracts.debugger import Debugger
from incremental_ad.framework.contracts.evaluator import DebugDataEvaluator, Evaluator
from incremental_ad.framework.evaluators._metrics import find_segments
from incremental_ad.project.models.mae_tx.mae import (
    MaeTx,
    TrainingMode,
    _create_random_mask,
    _get_by_mask,
)


class MaeTxAdDebugger(Debugger):
    def __init__(self, n_samples: int = 5) -> None:
        self._n_samples = n_samples

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(
        self,
        model: object,
        dataset: object,
        evaluator: Evaluator[Any],
        step_dir: Path,
    ) -> None:
        assert isinstance(
            evaluator, DebugDataEvaluator
        ), f"MaeTxAdDebugger requires a DebugDataEvaluator, got {type(evaluator).__name__}"
        data = evaluator.debug_data()
        if data is None:
            return
        scores, labels, threshold = data

        out = step_dir / "debug"
        out.mkdir(parents=True, exist_ok=True)

        self._plot_score_timeline(scores, labels, threshold, out)
        self._plot_score_distributions(scores, labels, threshold, out)
        self._write_event_analysis(scores, labels, threshold, out)
        self._plot_reconstructions(scores, labels, threshold, model, dataset, out)
        self._log_to_wandb(out)

    # ── Wandb image logging ───────────────────────────────────────────────────

    def _log_to_wandb(self, out: Path) -> None:
        if wandb.run is None:
            return
        to_log = {}
        for name, path in [
            ("debug/score_timeline", out / "score_timeline.png"),
            ("debug/score_distributions", out / "score_distributions.png"),
        ]:
            if path.exists():
                to_log[name] = wandb.Image(str(path))
        samples_dir = out / "samples"
        if samples_dir.exists():
            for img_path in sorted(samples_dir.glob("*.png")):
                to_log[f"debug/samples/{img_path.stem}"] = wandb.Image(str(img_path))
        if to_log:
            wandb.log(to_log)

    # ── Score timeline ────────────────────────────────────────────────────────

    def _plot_score_timeline(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float, out: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 3))
        t = np.arange(len(scores))
        ax.plot(t, scores, lw=0.6, color="#4C72B0", label="score")
        ax.axhline(
            threshold,
            color="crimson",
            lw=1.0,
            ls="--",
            label=f"threshold={threshold:.4f}",
        )

        segments = find_segments(labels)
        for i, (s, e) in enumerate(segments):
            ax.axvspan(
                s, e, alpha=0.18, color="orange", label="anomaly" if i == 0 else ""
            )

        ax.set_xlabel("window index")
        ax.set_ylabel("anomaly score")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Score timeline")
        fig.tight_layout()
        fig.savefig(out / "score_timeline.png", dpi=120)
        plt.close(fig)

    # ── Score distributions ───────────────────────────────────────────────────

    def _plot_score_distributions(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float, out: Path
    ) -> None:
        normal = scores[labels == 0]
        anomaly = scores[labels == 1]
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(scores.min(), scores.max(), 60)
        if len(normal):
            ax.hist(
                normal,
                bins=bins,
                alpha=0.6,
                color="#4C72B0",
                label="normal",
                density=True,
            )
        if len(anomaly):
            ax.hist(
                anomaly,
                bins=bins,
                alpha=0.6,
                color="orange",
                label="anomaly",
                density=True,
            )
        ax.axvline(
            threshold,
            color="crimson",
            lw=1.2,
            ls="--",
            label=f"threshold={threshold:.4f}",
        )
        ax.set_xlabel("anomaly score")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.set_title("Score distributions")
        fig.tight_layout()
        fig.savefig(out / "score_distributions.png", dpi=120)
        plt.close(fig)

    # ── Event analysis CSV ────────────────────────────────────────────────────

    def _write_event_analysis(
        self, scores: np.ndarray, labels: np.ndarray, threshold: float, out: Path
    ) -> None:
        preds = (scores >= threshold).astype(int)
        rows = []
        for seg_id, (s, e) in enumerate(find_segments(labels)):
            seg_scores = scores[s:e]
            seg_preds = preds[s:e]
            rows.append(
                {
                    "seg_id": seg_id,
                    "start": s,
                    "end": e,
                    "length": e - s,
                    "detected": bool(seg_preds.any()),
                    "max_score": float(seg_scores.max()),
                    "mean_score": float(seg_scores.mean()),
                    "detected_fraction": float(seg_preds.mean()),
                }
            )
        if rows:
            with (out / "event_analysis.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # ── Sample reconstructions ────────────────────────────────────────────────

    def _plot_reconstructions(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        model: object,
        dataset: object,
        out: Path,
    ) -> None:
        mae_model = cast(MaeTx, model)
        device = next(mae_model.parameters()).device

        # Use the test loader's underlying dataset for indexed access
        from incremental_ad.framework.contracts.dataset import (
            Dataset,
            DatasetCapability,
        )

        if (
            not isinstance(dataset, Dataset)
            or DatasetCapability.TEST not in dataset.capabilities
        ):
            return
        test_ds = dataset.get_test_dataset()

        preds = (scores >= threshold).astype(int)
        categories = {
            "tp": self._pick_indices(
                np.where((preds == 1) & (labels == 1))[0], scores, "high"
            ),
            "fp": self._pick_indices(
                np.where((preds == 1) & (labels == 0))[0], scores, "high"
            ),
            "fn": self._pick_indices(
                np.where((preds == 0) & (labels == 1))[0], scores, "low"
            ),
        }

        samples_dir = out / "samples"
        samples_dir.mkdir(exist_ok=True)

        for cat, indices in categories.items():
            for rank, idx in enumerate(indices):
                window, _ = test_ds[idx]
                window_t = window.unsqueeze(0).to(device)
                recon = self._reconstruct(mae_model, window_t)
                self._plot_reconstruction(
                    recon,
                    scores[idx],
                    threshold,
                    samples_dir / f"{cat}_{rank:02d}_win{idx}.png",
                )

    def _pick_indices(
        self, candidates: np.ndarray, scores: np.ndarray, order: str
    ) -> list[int]:
        if len(candidates) == 0:
            return []
        sorted_idx = np.argsort(scores[candidates])
        if order == "high":
            sorted_idx = sorted_idx[::-1]
        return candidates[sorted_idx[: self._n_samples]].tolist()

    def _plot_reconstruction(
        self,
        recon: dict[str, Tensor],
        score: float,
        threshold: float,
        path: Path,
    ) -> None:
        original = recon["original"].squeeze(0).cpu().numpy()  # (T', F)
        reconstruction = recon["reconstruction"].squeeze(0).cpu().numpy()
        token_errors = recon["token_errors"].squeeze(0).cpu().numpy()  # (n_patches,)

        n_patches = len(token_errors)
        vmin = min(original.min(), reconstruction.min())
        vmax = max(original.max(), reconstruction.max())

        fig = plt.figure(figsize=(max(10, original.shape[1] * 0.15 + 4), 10))
        gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

        for row, (data, title, cmap) in enumerate(
            [
                (original, "Original", "RdBu_r"),
                (reconstruction, "Reconstruction", "RdBu_r"),
                (np.abs(original - reconstruction), "|Error|", "hot"),
            ]
        ):
            ax = fig.add_subplot(gs[row])
            im = ax.imshow(
                data.T,
                aspect="auto",
                cmap=cmap,
                vmin=vmin if row < 2 else None,
                vmax=vmax if row < 2 else None,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("feature", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

        ax4 = fig.add_subplot(gs[3])
        ax4.bar(np.arange(n_patches), token_errors, color="#4C72B0")
        ax4.set_xlabel("patch index", fontsize=7)
        ax4.set_ylabel("MSE", fontsize=7)
        ax4.set_title("Per-patch error", fontsize=9)
        ax4.tick_params(labelsize=6)

        status = "ANOMALY" if score >= threshold else "NORMAL"
        fig.suptitle(
            f"score={score:.4f}  threshold={threshold:.4f}  [{status}]", fontsize=10
        )
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Reconstruction (moved here from MaeTx) ────────────────────────────────

    def _reconstruct(self, model: MaeTx, batch: Tensor) -> dict[str, Tensor]:
        model.eval()
        with torch.no_grad():
            if model.config.training_mode == TrainingMode.RANDOM_MASK:
                return self._reconstruct_random_mask(model, batch)
            return self._reconstruct_deterministic(model, batch)

    def _reconstruct_random_mask(
        self, model: MaeTx, batch: Tensor
    ) -> dict[str, Tensor]:
        assert model.n_features is not None, "model not built"
        B = batch.size(0)
        D = model.config.patch_len * model.n_features
        error_sum = torch.zeros(B, model.n_patches, device=batch.device)
        error_counts = torch.zeros(B, model.n_patches, device=batch.device)
        recon_sum = torch.zeros(B, model.n_patches, D, device=batch.device)
        recon_counts = torch.zeros(B, model.n_patches, device=batch.device)
        tokens_ref: Tensor | None = None

        for _ in range(model.config.n_eval_passes):
            mask_indices, unmask_indices = _create_random_mask(
                B, model.n_patches, model.config.mask_ratio, batch.device
            )
            decoder_output, tokens = model._forward(batch, unmask_indices)
            if tokens_ref is None:
                tokens_ref = tokens
            model._accumulate_errors(
                decoder_output, tokens, mask_indices, error_sum, error_counts
            )
            batch_idx = torch.arange(B, device=batch.device).unsqueeze(1)
            recon_sum[batch_idx, mask_indices] += _get_by_mask(
                decoder_output, mask_indices
            )
            recon_counts[batch_idx, mask_indices] += 1

        assert tokens_ref is not None, "n_eval_passes must be > 0"
        return {
            "original": _unpatchify(model, tokens_ref),
            "reconstruction": _unpatchify(
                model, recon_sum / recon_counts.clamp(min=1).unsqueeze(-1)
            ),
            "token_errors": error_sum / error_counts.clamp(min=1),
        }

    def _reconstruct_deterministic(
        self, model: MaeTx, batch: Tensor
    ) -> dict[str, Tensor]:
        B = batch.size(0)
        mask_indices, unmask_indices = model._create_mask(B, batch.device)
        decoder_output, tokens = model._forward(batch, unmask_indices)
        gt = _get_by_mask(tokens, mask_indices)
        pred = _get_by_mask(decoder_output, mask_indices)
        if model.config.patch_norm:
            gt = model._normalize_patches(gt)
        patch_errors = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
        token_errors = torch.zeros(B, model.n_patches, device=batch.device)
        token_errors[
            torch.arange(B, device=batch.device).unsqueeze(1), mask_indices
        ] = patch_errors
        return {
            "original": _unpatchify(model, tokens),
            "reconstruction": _unpatchify(model, decoder_output),
            "token_errors": token_errors,
        }


def _unpatchify(model: MaeTx, patches: Tensor) -> Tensor:
    """(B, n_patches, patch_len * n_features) -> (B, n_patches * patch_len, n_features)"""
    assert model.n_features is not None, "model not built"
    B = patches.size(0)
    return patches.reshape(
        B, model.n_patches * model.config.patch_len, model.n_features
    )
