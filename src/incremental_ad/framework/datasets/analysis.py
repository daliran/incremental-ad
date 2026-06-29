"""Reusable exploratory analysis for sliding-window time-series datasets.

Operates on raw (unscaled) numpy arrays + feature names, with optional per-timestep
anomaly labels. Datasets call `run_timeseries_analysis()` from their `analyze()` method.
Writes, under the given output dir:

  feature_stats.csv        — per-feature mean/std/min/max/percentiles for train and test
                             (split into normal/anomaly + delta-z when labels are given)
  anomaly_catalog.csv      — per anomaly segment: duration, mean/max |z|, top driving
                             features (labelled datasets only)
  test_timeseries.pdf      — every feature's test signal, ±2σ train bands, anomaly shading
  train_timeseries.pdf     — every feature's train signal, ±2σ bands
  test_heatmap.png         — feature×time z-score heatmap, sorted by anomaly detectability
                             (labelled) or temporal drift (unlabelled), + GT strip if labelled
  train_heatmap.png        — feature×time z-score heatmap, sorted by temporal drift

Pure plotting/stats — no torch, no model, no project knowledge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

log = logging.getLogger(__name__)

_FEATURES_PER_PAGE = 6
_HEATMAP_MAX_COLS = 5000
_DRIFT_WINDOW_FRAC = 0.05


def run_timeseries_analysis(
    out_dir: Path,
    *,
    name: str,
    train: np.ndarray,
    test: np.ndarray | None = None,
    feature_names: list[str],
    test_labels: np.ndarray | None = None,
) -> None:
    """Write the full analysis bundle for a dataset to out_dir.

    train: raw (unscaled) [T, F] array. test: optional held-out [T_test, F] array — when
    absent (None/empty) only train-side artifacts are produced. test_labels: optional
    [T_test] 0/1 labels — when present, enables the anomaly-split stats, catalog, shading,
    and detectability sort.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.asarray(train, dtype=float)
    has_test = test is not None and len(test) > 0
    test = np.asarray(test, dtype=float) if has_test else None
    ref_mean = train.mean(axis=0)
    ref_std = train.std(axis=0) + 1e-8

    labels = None if (test_labels is None or not has_test) else np.asarray(test_labels).astype(int)
    has_anomalies = labels is not None and int(labels.sum()) > 0

    _write_feature_stats(out_dir, train, test, feature_names, labels, ref_mean, ref_std)
    if has_test:
        if has_anomalies:
            _write_anomaly_catalog(out_dir, test, labels, feature_names, ref_mean, ref_std)
        _plot_feature_timeseries(
            out_dir / "test_timeseries.pdf", f"{name} — test", test, feature_names,
            ref_mean, ref_std, labels if has_anomalies else None,
        )
        _plot_zscore_heatmap(
            out_dir / "test_heatmap.png", f"{name} — test z-score heatmap", test,
            ref_mean, ref_std, feature_names,
            sort="detectability" if has_anomalies else "drift",
            labels=labels if has_anomalies else None,
        )

    _plot_feature_timeseries(
        out_dir / "train_timeseries.pdf", f"{name} — train", train, feature_names,
        ref_mean, ref_std, None,
    )
    _plot_zscore_heatmap(
        out_dir / "train_heatmap.png", f"{name} — train z-score heatmap (drift-sorted)",
        train, ref_mean, ref_std, feature_names, sort="drift", labels=None,
    )
    log.info("Dataset analysis written to %s", out_dir)


# ── Segment helpers ─────────────────────────────────────────────────────────────


def _anomaly_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_inclusive) for each contiguous run of 1s."""
    events: list[tuple[int, int]] = []
    in_event, start = False, 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_event:
            in_event, start = True, i
        elif lbl == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events


# ── Stats CSVs ──────────────────────────────────────────────────────────────────


def _stat_block(arr: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
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


def _write_feature_stats(
    out_dir: Path,
    train: np.ndarray,
    test: np.ndarray | None,
    feature_names: list[str],
    labels: np.ndarray | None,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
) -> None:
    cols: dict[str, np.ndarray] = {}
    cols.update(_stat_block(train, "train"))
    if test is None:
        pass
    elif labels is not None:
        normal, anomaly = test[labels == 0], test[labels == 1]
        if len(normal):
            cols.update(_stat_block(normal, "test_normal"))
        if len(anomaly):
            cols.update(_stat_block(anomaly, "test_anomaly"))
            cols["anomaly_delta_z"] = (anomaly.mean(axis=0) - ref_mean) / ref_std
    else:
        cols.update(_stat_block(test, "test"))
    df = pd.DataFrame(cols, index=feature_names)
    df.index.name = "feature"
    df.round(4).to_csv(out_dir / "feature_stats.csv")


def _write_anomaly_catalog(
    out_dir: Path,
    test: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
) -> None:
    events = _anomaly_events(labels)
    if not events:
        return
    n_top = min(3, len(feature_names))
    records = []
    for start, end in events:
        seg = test[start : end + 1]
        z = np.abs((seg.mean(axis=0) - ref_mean) / ref_std)
        top = np.argsort(z)[::-1]
        rec: dict[str, object] = {
            "start": start,
            "end": end,
            "duration": end - start + 1,
            "mean_abs_z": round(float(z.mean()), 4),
            "max_abs_z": round(float(z.max()), 4),
        }
        for k in range(n_top):
            rec[f"top{k + 1}_feature"] = feature_names[top[k]]
            rec[f"top{k + 1}_z"] = round(float(z[top[k]]), 4)
        records.append(rec)
    (
        pd.DataFrame(records)
        .sort_values("mean_abs_z", ascending=True)  # hardest events first
        .reset_index(drop=True)
        .to_csv(out_dir / "anomaly_catalog.csv", index=False)
    )


# ── Plots ───────────────────────────────────────────────────────────────────────


def _plot_feature_timeseries(
    out_path: Path,
    title_prefix: str,
    data: np.ndarray,
    feature_names: list[str],
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    labels: np.ndarray | None,
) -> None:
    T = len(data)
    t = np.arange(T)
    n_feat = len(feature_names)
    n_pages = (n_feat + _FEATURES_PER_PAGE - 1) // _FEATURES_PER_PAGE
    events = _anomaly_events(labels) if labels is not None else []

    with PdfPages(out_path) as pdf:
        for page, page_start in enumerate(range(0, n_feat, _FEATURES_PER_PAGE)):
            feat_idx = list(range(page_start, min(page_start + _FEATURES_PER_PAGE, n_feat)))
            fig, axes_2d = plt.subplots(
                len(feat_idx), 1, figsize=(18, 3 * len(feat_idx)), sharex=True, squeeze=False
            )
            axes = axes_2d[:, 0]
            for ax, fi in zip(axes, feat_idx):
                for s, e in events:
                    ax.axvspan(s, e + 1, color="#ffcccc", alpha=0.45, linewidth=0)
                ax.plot(t, data[:, fi], color="#222222", linewidth=0.6)
                mu, sigma = float(ref_mean[fi]), float(ref_std[fi])
                ax.axhline(mu, color="#0055cc", linewidth=0.5, linestyle=":")
                ax.axhline(mu + 2 * sigma, color="#0055cc", linewidth=0.9, linestyle="--", alpha=0.7)
                ax.axhline(mu - 2 * sigma, color="#0055cc", linewidth=0.9, linestyle="--", alpha=0.7)
                ax.set_ylabel(feature_names[fi], fontsize=8, rotation=0, ha="right", va="center")
                ax.tick_params(labelsize=7)
                ax.set_xlim(0, T)
            axes[-1].set_xlabel("timestep")
            handles = [Line2D([0], [0], color="#0055cc", linestyle="--", label="±2σ (train)")]
            if events:
                handles.insert(0, mpatches.Patch(facecolor="#ffcccc", alpha=0.7, label="anomaly"))
            axes[0].legend(handles=handles, loc="upper right", fontsize=8)
            fig.suptitle(
                f"{title_prefix} — features {page_start + 1}–{feat_idx[-1] + 1} of {n_feat}"
                f"  (page {page + 1}/{n_pages})",
                fontsize=10,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
    log.info("  %s", out_path.name)


def _plot_zscore_heatmap(
    out_path: Path,
    title: str,
    data: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
    feature_names: list[str],
    sort: str,
    labels: np.ndarray | None,
) -> None:
    T = len(data)
    stride = max(1, T // _HEATMAP_MAX_COLS)
    z = (data[::stride] - ref_mean) / ref_std  # [T', F]
    labels_ds = labels[::stride] if labels is not None else None

    if sort == "detectability" and labels_ds is not None and (labels_ds == 1).any():
        key = np.abs(z[labels_ds == 1]).mean(axis=0)
        ylabel = "feature (top = most detectable during anomalies)"
    else:  # drift: how much each feature's rolling mean wanders
        window = max(1, int(len(z) * _DRIFT_WINDOW_FRAC))
        key = pd.DataFrame(z).rolling(window, center=True, min_periods=1).mean().std(axis=0).values
        ylabel = "feature (top = highest temporal drift)"

    order = np.argsort(key)[::-1]
    z_sorted = z[:, order].T  # [F, T']
    names_sorted = [feature_names[i] for i in order]

    vmax = float(max(3.0, np.percentile(np.abs(z_sorted), 98)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    if labels_ds is not None:
        fig, (ax_heat, ax_lbl) = plt.subplots(
            2, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [14, 1]}, sharex=True
        )
    else:
        fig, ax_heat = plt.subplots(figsize=(18, 12))
        ax_lbl = None

    im = ax_heat.imshow(z_sorted, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, label="z-score (vs train mean)", shrink=0.8, pad=0.01)
    n_feat = len(names_sorted)
    ytick_step = max(1, n_feat // 30)
    ax_heat.set_yticks(range(0, n_feat, ytick_step))
    ax_heat.set_yticklabels(names_sorted[::ytick_step], fontsize=7)
    ax_heat.set_ylabel(ylabel)

    if ax_lbl is not None:
        ax_lbl.imshow(labels_ds[None, :], aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
        ax_lbl.set_yticks([0])
        ax_lbl.set_yticklabels(["GT"], fontsize=8)
        ax_lbl.set_xlabel(f"timestep (stride={stride})")
    else:
        ax_heat.set_xlabel(f"timestep (stride={stride})")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s", out_path.name)
