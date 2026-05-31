"""Run-level debug outputs written alongside test_results.json.

Generated inside <run_dir>/debug/:
  score_timeline.png      Anomaly scores over time with GT shading and threshold line
  score_distributions.png Score histograms for normal vs anomaly windows
  event_analysis.csv      Per-anomaly-event scores and detection status (hardest first)
  samples/                Reconstruction plots for FN / FP / TP examples
"""
from __future__ import annotations

import logging
from pathlib import Path

import wandb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch

from incremental_ad.datasets.base_dataset import BaseDataset
from incremental_ad.models.base_model import BaseModel
from incremental_ad.training.metrics import find_segments

log = logging.getLogger(__name__)

N_SAMPLES = 5  # examples per category (FN, FP, TP)


class RunDebugger:
    def __init__(
        self,
        run_dir: Path,
        model: BaseModel,
        device: torch.device,
        eval_dataset: BaseDataset,
    ) -> None:
        self.debug_dir = run_dir / "debug"
        self.model = model
        self.device = device
        self.dataset = eval_dataset

    def run(
        self,
        scores: np.ndarray,         # (n_windows,)
        window_labels: np.ndarray,  # (n_windows,) 0/1
        threshold: float,
    ) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Writing debug outputs to {self.debug_dir}")
        
        try:
            self._score_timeline(scores, window_labels, threshold)
            self._score_distributions(scores, window_labels, threshold)
            self._event_analysis(scores, window_labels, threshold)
            self._sample_reconstructions(scores, window_labels, threshold)
            self._log_to_wandb()
        except Exception:
            log.exception("RunDebugger encountered an error; outputs may be partial")

    # ------------------------------------------------------------------
    # wandb upload
    # ------------------------------------------------------------------

    def _log_to_wandb(self) -> None:
        if wandb.run is None:
            return

        to_log: dict[str, wandb.Image] = {}

        for name, path in [
            ("debug/score_timeline", self.debug_dir / "score_timeline.png"),
            ("debug/score_distributions", self.debug_dir / "score_distributions.png"),
        ]:
            if path.exists():
                to_log[name] = wandb.Image(str(path))

        samples_dir = self.debug_dir / "samples"
        if samples_dir.exists():
            for img_path in sorted(samples_dir.glob("*.png")):
                to_log[f"debug/samples/{img_path.stem}"] = wandb.Image(str(img_path))

        if to_log:
            wandb.log(to_log)
            log.info(f"  Logged {len(to_log)} debug images to wandb")

    # ------------------------------------------------------------------
    # Score timeline
    # ------------------------------------------------------------------

    def _score_timeline(
        self, scores: np.ndarray, window_labels: np.ndarray, threshold: float
    ) -> None:
        t = np.arange(len(scores))
        normal_mask = window_labels == 0
        anomaly_mask = ~normal_mask

        fig, ax = plt.subplots(figsize=(18, 4))

        for s, e in find_segments(window_labels):
            ax.axvspan(s, e, color="#ffcccc", alpha=0.45, linewidth=0)

        if normal_mask.any():
            ax.scatter(
                t[normal_mask], scores[normal_mask],
                s=0.5, c="#888888", alpha=0.3, label="Normal", rasterized=True,
            )
        if anomaly_mask.any():
            ax.scatter(
                t[anomaly_mask], scores[anomaly_mask],
                s=1.5, c="#cc2222", alpha=0.7, label="Anomaly", rasterized=True,
            )

        ax.axhline(
            threshold, color="#0055cc", linewidth=1.2, linestyle="--",
            label=f"Threshold ({threshold:.4f})",
        )

        ax.set_xlabel("Window index (test set)")
        ax.set_ylabel("Anomaly score")
        ax.set_title("Score timeline")
        ax.set_xlim(0, len(scores))
        ax.legend(loc="upper right", fontsize=8, markerscale=6)
        plt.tight_layout()
        fig.savefig(self.debug_dir / "score_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  debug/score_timeline.png")

    # ------------------------------------------------------------------
    # Score distributions
    # ------------------------------------------------------------------

    def _score_distributions(
        self, scores: np.ndarray, window_labels: np.ndarray, threshold: float
    ) -> None:
        normal_scores = scores[window_labels == 0]
        anomaly_scores = scores[window_labels == 1]

        bins = np.linspace(scores.min(), scores.max(), 80)
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(normal_scores, bins=bins, density=True, alpha=0.55, color="#444488", label=f"Normal ({len(normal_scores):,})")
        ax.hist(anomaly_scores, bins=bins, density=True, alpha=0.55, color="#cc2222", label=f"Anomaly ({len(anomaly_scores):,})")
        ax.axvline(
            threshold, color="#0055cc", linewidth=1.5, linestyle="--",
            label=f"Threshold ({threshold:.4f})",
        )

        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Density")
        ax.set_title("Score distributions — normal vs anomaly windows")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.debug_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  debug/score_distributions.png")

    # ------------------------------------------------------------------
    # Event analysis CSV
    # ------------------------------------------------------------------

    def _event_analysis(
        self, scores: np.ndarray, window_labels: np.ndarray, threshold: float
    ) -> None:
        segments = find_segments(window_labels)
        if not segments:
            log.info("  debug/event_analysis.csv  (no anomaly events found)")
            return

        records = []
        for s, e in segments:
            seg = scores[s:e]
            max_score = float(seg.max())
            records.append(
                {
                    "start_win": s,
                    "end_win": e,
                    "duration_wins": e - s,
                    "max_score": round(max_score, 6),
                    "mean_score": round(float(seg.mean()), 6),
                    "min_score": round(float(seg.min()), 6),
                    "threshold": round(threshold, 6),
                    # positive → detected, negative → how far below threshold the best score is
                    "margin": round(max_score - threshold, 6),
                    "detected": max_score >= threshold,
                    "detection_ratio": round(float((seg >= threshold).mean()), 4),
                }
            )

        df = (
            pd.DataFrame(records)
            .sort_values("max_score", ascending=True)  # hardest first
            .reset_index(drop=True)
        )
        df.to_csv(self.debug_dir / "event_analysis.csv", index=False)
        n_missed = int((~df["detected"]).sum())
        log.info(f"  debug/event_analysis.csv  ({len(df)} events, {n_missed} missed)")

    # ------------------------------------------------------------------
    # Sample reconstructions
    # ------------------------------------------------------------------

    def _sample_reconstructions(
        self, scores: np.ndarray, window_labels: np.ndarray, threshold: float
    ) -> None:
        preds = (scores >= threshold).astype(int)
        is_anomaly = window_labels == 1
        is_normal = ~is_anomaly

        def pick(mask: np.ndarray, ascending: bool) -> list[int]:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return []
            order = np.argsort(scores[idx])
            if not ascending:
                order = order[::-1]
            return idx[order[:N_SAMPLES]].tolist()

        categories = {
            "fn": pick(is_anomaly & (preds == 0), ascending=True),   # missed anomalies, lowest score first
            "fp": pick(is_normal  & (preds == 1), ascending=False),  # false alarms, highest score first
            "tp": pick(is_anomaly & (preds == 1), ascending=False),  # caught anomalies, highest score first
        }

        samples_dir = self.debug_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        for cat, indices in categories.items():
            for rank, win_idx in enumerate(indices):
                debug_info = self._fetch_debug(win_idx)
                self._plot_window(
                    debug_info=debug_info,
                    win_idx=win_idx,
                    score=float(scores[win_idx]),
                    threshold=threshold,
                    label=int(window_labels[win_idx]),
                    category=cat,
                    rank=rank,
                    out_dir=samples_dir,
                )

        counts = {k: len(v) for k, v in categories.items()}
        log.info(f"  debug/samples/  fn:{counts['fn']} fp:{counts['fp']} tp:{counts['tp']}")

    def _fetch_debug(self, win_idx: int) -> dict | None:
        """Call model.debug_step for one window and convert tensors to numpy."""
        try:
            window = self.dataset[win_idx].unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                raw = self.model.debug_step(window)
            if raw is None:
                return None
            return {
                k: v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in raw.items()
            }
        except Exception:
            log.exception(f"debug_step failed for window {win_idx}")
            return None

    def _plot_window(
        self,
        debug_info: dict | None,
        win_idx: int,
        score: float,
        threshold: float,
        label: int,
        category: str,
        rank: int,
        out_dir: Path,
    ) -> None:
        if debug_info is None:
            return
        if "original" not in debug_info or "reconstruction" not in debug_info:
            return

        orig = debug_info["original"]           # (T', F)
        recon = debug_info["reconstruction"]    # (T', F)

        abs_err = np.abs(orig - recon)

        # Shared symmetric z-score colormap: data is already z-score normalised so
        # clipping at ±3 captures >99 % of normal variation without per-window scaling.
        vmax = float(max(3.0, np.percentile(np.abs(orig), 99)))
        kw_zscore = dict(aspect="auto", interpolation="nearest",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        has_patch_errors = "token_errors" in debug_info
        n_rows = 4 if has_patch_errors else 3
        height_ratios = [3, 3, 3, 1.5] if has_patch_errors else [3, 3, 3]

        fig, axes = plt.subplots(
            n_rows, 1, figsize=(14, 3 * n_rows + 2),
            gridspec_kw={"height_ratios": height_ratios},
        )

        kw = dict(aspect="auto", interpolation="nearest")

        axes[0].imshow(orig.T, **kw_zscore)
        axes[0].set_title(f"Original  (z-score, symmetric ±{vmax:.1f} colormap)")
        axes[0].set_ylabel("Feature")

        axes[1].imshow(recon.T, **kw_zscore)
        axes[1].set_title("Reconstruction  (same colormap as original)")
        axes[1].set_ylabel("Feature")

        im = axes[2].imshow(abs_err.T, cmap="hot", **kw)
        axes[2].set_title("|Original − Reconstruction|  per (timestep, feature)")
        axes[2].set_ylabel("Feature")
        plt.colorbar(im, ax=axes[2], shrink=0.8)

        if has_patch_errors:
            token_errors: np.ndarray = debug_info["token_errors"]
            n_patches = len(token_errors)
            cmap_bar = plt.cm.RdYlBu_r  # type: ignore[attr-defined]
            norm_bar = mcolors.Normalize(vmin=0, vmax=float(token_errors.max()) + 1e-8)
            bar_colors = [cmap_bar(norm_bar(e)) for e in token_errors]
            axes[3].bar(np.arange(n_patches), token_errors, color=bar_colors, edgecolor="none")
            axes[3].axhline(
                score, color="#333", linewidth=1.2, linestyle="--",
                label=f"score = {score:.5f}",
            )
            axes[3].set_xlabel("Patch index")
            axes[3].set_ylabel("MSE")
            axes[3].set_title("Per-patch error")
            axes[3].legend(fontsize=8)

        cat_label = {"fn": "False Negative", "fp": "False Positive", "tp": "True Positive"}[category]
        gt_str = "Anomaly" if label == 1 else "Normal"
        fig.suptitle(
            f"[{cat_label}]  win={win_idx}  GT={gt_str}  "
            f"score={score:.5f}  threshold={threshold:.5f}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / f"{category}_{rank:02d}_win{win_idx}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

