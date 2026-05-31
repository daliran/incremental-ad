"""
Dataset-level analysis for anomaly detection datasets.

Outputs (written to --out-dir, default debug/dataset/<name>):
  stats.csv            Per-feature stats: train / test-normal / test-anomaly / delta-z
  anomaly_events.csv   Segment catalog sorted by hardness ascending (hardest first)
  timeseries.pdf       Multi-page time series with anomaly shading and ±2σ reference
  anomaly_heatmap.png  Features × time z-score heatmap with GT label strip

Adding a new dataset:
  1. Implement  load_<name>() -> DatasetBundle  (raw/unscaled arrays)
  2. Register it in LOADERS at the bottom of this file

Usage:
  python analysis/analyze_dataset.py --dataset swat
  python analysis/analyze_dataset.py --dataset swat --out-dir path/to/output
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    name: str
    feature_names: list[str]
    train_data: np.ndarray  # (T_train, F) raw/unscaled
    test_data: np.ndarray   # (T_test,  F) raw/unscaled
    test_labels: np.ndarray  # (T_test,)   int, 0=normal 1=anomaly


# ---------------------------------------------------------------------------
# Dataset loaders  — add new datasets here
# ---------------------------------------------------------------------------


def load_swat() -> DatasetBundle:
    from incremental_ad.datasets.swat import load_for_analysis

    train_data, test_data, test_labels, feature_names = load_for_analysis()
    log.info(
        f"SWaT — train: {len(train_data):,}  test: {len(test_data):,} "
        f"({int(test_labels.sum()):,} anomaly pts, {test_labels.mean():.1%})"
    )
    return DatasetBundle("swat", feature_names, train_data, test_data, test_labels)


LOADERS: dict[str, Callable[[], DatasetBundle]] = {
    "swat": load_swat,
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _anomaly_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_inclusive) for each contiguous anomaly run."""
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_event:
            in_event, start = True, i
        elif lbl == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def _shade_anomaly_regions(ax: matplotlib.axes.Axes, labels: np.ndarray) -> None:
    for s, e in _anomaly_events(labels):
        ax.axvspan(s, e + 1, color="#ffcccc", alpha=0.45, linewidth=0)


# ---------------------------------------------------------------------------
# Analysis: statistics
# ---------------------------------------------------------------------------


def compute_stats(bundle: DatasetBundle) -> pd.DataFrame:
    """Per-feature descriptive stats for train, test-normal, and test-anomaly splits."""
    train = bundle.train_data
    test_normal = bundle.test_data[bundle.test_labels == 0]
    test_anomaly = bundle.test_data[bundle.test_labels == 1]

    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0) + 1e-8

    def _block(arr: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_mean": arr.mean(axis=0),
            f"{prefix}_std": arr.std(axis=0),
            f"{prefix}_min": arr.min(axis=0),
            f"{prefix}_max": arr.max(axis=0),
            f"{prefix}_p5": np.percentile(arr, 5, axis=0),
            f"{prefix}_p25": np.percentile(arr, 25, axis=0),
            f"{prefix}_p75": np.percentile(arr, 75, axis=0),
            f"{prefix}_p95": np.percentile(arr, 95, axis=0),
        }

    cols: dict[str, np.ndarray] = {}
    cols.update(_block(train, "train"))
    cols.update(_block(test_normal, "test_normal"))
    if len(test_anomaly) > 0:
        cols.update(_block(test_anomaly, "test_anomaly"))
        cols["anomaly_delta_z"] = (test_anomaly.mean(axis=0) - train_mean) / train_std

    df = pd.DataFrame(cols, index=bundle.feature_names)
    df.index.name = "feature"
    return df.round(4)


# ---------------------------------------------------------------------------
# Analysis: anomaly event catalog
# ---------------------------------------------------------------------------


def compute_anomaly_catalog(bundle: DatasetBundle) -> pd.DataFrame:
    """Segment catalog sorted by mean |z-score| ascending (hardest events first)."""
    train_mean = bundle.train_data.mean(axis=0)
    train_std = bundle.train_data.std(axis=0) + 1e-8

    events = _anomaly_events(bundle.test_labels)
    if not events:
        return pd.DataFrame()

    records = []
    for start, end in events:
        seg = bundle.test_data[start : end + 1]
        z = np.abs((seg.mean(axis=0) - train_mean) / train_std)
        top = np.argsort(z)[::-1]
        records.append(
            {
                "start": start,
                "end": end,
                "duration": end - start + 1,
                "mean_abs_z": round(float(z.mean()), 4),
                "max_abs_z": round(float(z.max()), 4),
                "top1_feature": bundle.feature_names[top[0]],
                "top1_z": round(float(z[top[0]]), 4),
                "top2_feature": bundle.feature_names[top[1]],
                "top2_z": round(float(z[top[1]]), 4),
                "top3_feature": bundle.feature_names[top[2]],
                "top3_z": round(float(z[top[2]]), 4),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("mean_abs_z", ascending=True)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plot: multi-page time series PDF
# ---------------------------------------------------------------------------

_FEATURES_PER_PAGE = 6


def plot_timeseries(bundle: DatasetBundle, out_path: Path) -> None:
    """
    One PDF page per group of features. Each subplot shows the test-set signal
    with anomaly regions shaded red and ±2σ dashed lines from the training mean.
    """
    train_mean = bundle.train_data.mean(axis=0)
    train_std = bundle.train_data.std(axis=0) + 1e-8
    T = len(bundle.test_data)
    t = np.arange(T)
    n_feat = len(bundle.feature_names)
    n_pages = (n_feat + _FEATURES_PER_PAGE - 1) // _FEATURES_PER_PAGE

    with PdfPages(out_path) as pdf:
        for page, page_start in enumerate(range(0, n_feat, _FEATURES_PER_PAGE)):
            feat_idx = list(range(page_start, min(page_start + _FEATURES_PER_PAGE, n_feat)))
            fig, axes_2d = plt.subplots(
                len(feat_idx), 1,
                figsize=(18, 3 * len(feat_idx)),
                sharex=True,
                squeeze=False,
            )
            axes = axes_2d[:, 0]

            for ax, fi in zip(axes, feat_idx):
                _shade_anomaly_regions(ax, bundle.test_labels)
                ax.plot(t, bundle.test_data[:, fi], color="#222222", linewidth=0.6)
                mu, sigma = float(train_mean[fi]), float(train_std[fi])
                ax.axhline(mu, color="#0055cc", linewidth=0.5, linestyle=":")
                ax.axhline(mu + 2 * sigma, color="#0055cc", linewidth=0.9, linestyle="--", alpha=0.7)
                ax.axhline(mu - 2 * sigma, color="#0055cc", linewidth=0.9, linestyle="--", alpha=0.7)
                ax.set_ylabel(bundle.feature_names[fi], fontsize=8, rotation=0, ha="right", va="center")
                ax.tick_params(labelsize=7)
                ax.set_xlim(0, T)

            axes[-1].set_xlabel("Timestep (test set)")
            axes[0].legend(
                handles=[
                    mpatches.Patch(facecolor="#ffcccc", alpha=0.7, label="Anomaly region"),
                    Line2D([0], [0], color="#0055cc", linestyle="--", label="±2σ (train)"),
                ],
                loc="upper right",
                fontsize=8,
            )
            fig.suptitle(
                f"{bundle.name.upper()} — Features {page_start + 1}–{feat_idx[-1] + 1}"
                f" of {n_feat}  (page {page + 1}/{n_pages})",
                fontsize=10,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    log.info(f"Saved time series: {out_path}")


# ---------------------------------------------------------------------------
# Plot: z-score heatmap
# ---------------------------------------------------------------------------


def plot_anomaly_heatmap(bundle: DatasetBundle, out_path: Path, max_cols: int = 5000) -> None:
    """
    Features × time z-score heatmap with a GT label strip underneath.
    Features are sorted top-to-bottom by mean |z| during anomaly periods
    (most detectable first), making hard anomalies visible as faint rows.
    """
    train_mean = bundle.train_data.mean(axis=0)
    train_std = bundle.train_data.std(axis=0) + 1e-8
    T = len(bundle.test_data)
    stride = max(1, T // max_cols)

    data_ds = bundle.test_data[::stride]
    labels_ds = bundle.test_labels[::stride]
    z = (data_ds - train_mean) / train_std  # (T', F)

    anomaly_mask = labels_ds == 1
    mean_z_anomaly = (
        np.abs(z[anomaly_mask]).mean(axis=0) if anomaly_mask.any() else np.zeros(z.shape[1])
    )
    sort_idx = np.argsort(mean_z_anomaly)[::-1]
    z_sorted = z[:, sort_idx].T  # (F, T')
    names_sorted = [bundle.feature_names[i] for i in sort_idx]

    vmax = float(max(3.0, np.percentile(np.abs(z_sorted), 98)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, (ax_heat, ax_lbl) = plt.subplots(
        2, 1,
        figsize=(18, 12),
        gridspec_kw={"height_ratios": [14, 1]},
        sharex=True,
    )

    im = ax_heat.imshow(z_sorted, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, label="Z-score (relative to train mean)", shrink=0.8, pad=0.01)

    n_feat = len(names_sorted)
    ytick_step = max(1, n_feat // 30)
    ax_heat.set_yticks(range(0, n_feat, ytick_step))
    ax_heat.set_yticklabels(names_sorted[::ytick_step], fontsize=7)
    ax_heat.set_ylabel("Feature  (top = most detectable during anomalies)")

    ax_lbl.imshow(
        labels_ds[None, :],
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_lbl.set_yticks([0])
    ax_lbl.set_yticklabels(["GT label"], fontsize=8)
    ax_lbl.set_xlabel(f"Timestep (stride={stride})")

    fig.suptitle(
        f"{bundle.name.upper()} — Z-score heatmap  |  features sorted by anomaly detectability",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved heatmap: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(LOADERS), help="Dataset to analyse"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: debug/dataset/<name>)",
    )
    args = parser.parse_args()

    bundle = LOADERS[args.dataset]()
    out_dir: Path = args.out_dir or Path("debug") / "dataset" / bundle.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing to: {out_dir.resolve()}")

    log.info("Computing sensor statistics…")
    stats = compute_stats(bundle)
    stats.to_csv(out_dir / "stats.csv")
    log.info(f"  stats.csv  ({len(stats)} features)")

    log.info("Computing anomaly event catalog…")
    catalog = compute_anomaly_catalog(bundle)
    catalog.to_csv(out_dir / "anomaly_events.csv", index=False)
    log.info(f"  anomaly_events.csv  ({len(catalog)} events)")

    log.info("Plotting time series…")
    plot_timeseries(bundle, out_dir / "timeseries.pdf")

    log.info("Plotting anomaly heatmap…")
    plot_anomaly_heatmap(bundle, out_dir / "anomaly_heatmap.png")

    log.info("Done.")


if __name__ == "__main__":
    main()