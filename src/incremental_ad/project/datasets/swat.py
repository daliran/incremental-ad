import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from incremental_ad.framework.contracts.dataset import (
    DatasetCapability,
    PartitionedDataset,
    Segment,
    SplitConfig,
    TimeSeriesDataset,
)
from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset

log = logging.getLogger(__name__)

_HF_DATASET_PATH = "thuml/Time-Series-Library"
_HF_DATASET_NAME = "SWaT"

Normalization = Literal["standard", "none"]


class Swat(TimeSeriesDataset, PartitionedDataset):

    def __init__(
        self,
        window_len: int,
        stride: int,
        normalization: Normalization,
        split_config: SplitConfig,
    ) -> None:
        self._window_len = window_len
        self.stride = stride
        self._eval_stride = 1
        self.split_config = split_config
        self._train_data, self._test_data, self._test_labels = _prepare_data(
            normalization
        )

    @property
    def n_features(self) -> int:
        return self._train_data.shape[1]

    @property
    def window_len(self) -> int:
        return self._window_len

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)  # registers SplitConfig args
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(f"--{p}_stride", type=int, required=True)
        parser.add_argument(
            f"--{p}_normalization", choices=["standard", "none"], required=True
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            window_len=getattr(cfg, f"{p}_window_len"),
            stride=getattr(cfg, f"{p}_stride"),
            normalization=getattr(cfg, f"{p}_normalization"),
            split_config=cls._split_config_from_cfg(cfg, prefix),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    @property
    def capabilities(self) -> set[DatasetCapability]:
        caps = {DatasetCapability.TEST, DatasetCapability.TEST_LABELS}
        if self.split_config.val_fraction > 0:
            caps.add(DatasetCapability.VAL)
        return caps

    def get_train_segment(self) -> Segment:
        return self._make_segment(0, len(self._train_data))

    def get_baseline(self) -> Segment:
        n = len(self._train_data)
        baseline_end = int(n * self.split_config.baseline_fraction)
        use_end = int(baseline_end * self.split_config.baseline_use_fraction)
        return self._make_segment(0, use_end)

    def get_incremental_segments(self) -> list[Segment]:
        if self.split_config.n_finetune_segments == 0:
            return []
        n = len(self._train_data)
        remainder_start = int(n * self.split_config.baseline_fraction)
        segment_size = (n - remainder_start) // self.split_config.n_finetune_segments
        return [
            self._make_segment(
                remainder_start + i * segment_size,
                remainder_start + (i + 1) * segment_size,
            )
            for i in range(self.split_config.n_finetune_segments)
        ]

    def get_train_eval_dataset(self) -> SlidingWindowDataset:
        return SlidingWindowDataset(self._train_data, self.window_len, self._eval_stride)

    def get_val_eval_dataset(self) -> SlidingWindowDataset:
        n = len(self._train_data)
        val_start = n - int(n * self.split_config.val_fraction)
        return SlidingWindowDataset(
            self._train_data[val_start:], self.window_len, self._eval_stride,
            labels=torch.zeros(n - val_start, dtype=torch.uint8),
        )

    def get_test_dataset(self) -> SlidingWindowDataset:
        return SlidingWindowDataset(
            self._test_data, self.window_len, self._eval_stride, self._test_labels
        )

    def analyze(self, analysis_dir: Path) -> None:
        done_flag = analysis_dir / "done.flag"
        if done_flag.exists():
            log.debug("Skipping dataset analysis — already written to %s", analysis_dir)
            return

        analysis_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing dataset analysis to %s ...", analysis_dir)

        import matplotlib

        matplotlib.use("Agg")

        train_df, test_df = _load_raw()
        train_samples, _ = _extract_samples_and_labels(train_df)
        test_samples, test_labels_raw = _extract_samples_and_labels(test_df)
        features = list(train_samples.columns)

        _write_feature_stats(
            train_samples, test_samples, test_labels_raw, features, analysis_dir
        )
        _plot_anomaly_timeline(test_labels_raw.to_numpy(), analysis_dir)
        _plot_feature_overview(train_samples, test_samples, features, analysis_dir)

        done_flag.touch()
        log.info("Dataset analysis complete.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_segment(self, start: int, end: int) -> Segment:
        if self.split_config.val_fraction > 0:
            val_start = end - int((end - start) * self.split_config.val_fraction)
            return Segment(
                train=SlidingWindowDataset(self._train_data[start:val_start], self.window_len, self.stride),
                val=SlidingWindowDataset(self._train_data[val_start:end], self.window_len, self.stride),
            )
        return Segment(
            train=SlidingWindowDataset(self._train_data[start:end], self.window_len, self.stride),
            val=None,
        )


# ── Data loading and preprocessing ────────────────────────────────────────────


def _prepare_data(normalization: Normalization) -> tuple[Tensor, Tensor, Tensor]:
    """Load, scale, and return (train_data, test_data, test_labels) as tensors.

    Scaler is fit on the full training series so all segments share the same
    normalization — call once at Swat construction, not per segment.
    """
    train_df, test_df = _load_raw()
    train_samples, _ = _extract_samples_and_labels(train_df)
    test_samples, test_labels = _extract_samples_and_labels(test_df)

    log.info(f"Normalization: {normalization}")

    if normalization == "standard":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_samples.values).astype(np.float32)
        test_scaled = scaler.transform(test_samples.values).astype(np.float32)
    else:
        train_scaled = train_samples.values.astype(np.float32)
        test_scaled = test_samples.values.astype(np.float32)

    return (
        torch.tensor(train_scaled),
        torch.tensor(test_scaled),
        torch.tensor(test_labels.values, dtype=torch.long),
    )


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading SWaT dataset from HuggingFace...")
    raw = load_dataset(_HF_DATASET_PATH, _HF_DATASET_NAME)
    train_df = cast(pd.DataFrame, raw["train"].to_pandas())
    test_df = cast(pd.DataFrame, raw["test"].to_pandas())
    log.info(f"Loaded — train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df


def _extract_samples_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = pd.to_numeric(df["Normal/Attack"], errors="raise").astype(int)
    samples = df.drop(columns=["Normal/Attack"])
    return samples, labels


# ── Analysis helpers ───────────────────────────────────────────────────────────


def _write_feature_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_labels: pd.Series,
    features: list[str],
    out: Path,
) -> None:
    test_normal = test[test_labels == 0]
    test_anomaly = test[test_labels == 1]
    rows = []
    for f in features:
        rows.append(
            {
                "feature": f,
                "train_mean": float(train[f].mean()),
                "train_std": float(train[f].std()),
                "test_normal_mean": (
                    float(test_normal[f].mean()) if len(test_normal) else float("nan")
                ),
                "test_normal_std": (
                    float(test_normal[f].std()) if len(test_normal) else float("nan")
                ),
                "test_anomaly_mean": (
                    float(test_anomaly[f].mean()) if len(test_anomaly) else float("nan")
                ),
                "test_anomaly_std": (
                    float(test_anomaly[f].std()) if len(test_anomaly) else float("nan")
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out / "feature_stats.csv", index=False)


def _plot_anomaly_timeline(labels: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 2))
    ax.fill_between(np.arange(len(labels)), labels, alpha=0.7, color="orange")
    ax.set_xlabel("timestep")
    ax.set_ylabel("anomaly")
    ax.set_title("SWaT test set — anomaly label timeline")
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    fig.savefig(out / "anomaly_timeline.png", dpi=120)
    plt.close(fig)


def _plot_feature_overview(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], out: Path
) -> None:
    import matplotlib.pyplot as plt

    # Pick top 16 features by mean shift relative to train std
    train_mean = train.mean()
    train_std = train.std().replace(0, 1)
    test_mean = test.mean()
    shift = ((test_mean - train_mean) / train_std).abs()
    top_features = shift.nlargest(16).index.tolist()

    n = len(top_features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        tr = train[feat].values
        te = test[feat].values
        bins = np.linspace(min(tr.min(), te.min()), max(tr.max(), te.max()), 40)
        ax.hist(tr, bins=bins, alpha=0.6, color="#4C72B0", density=True, label="train")
        ax.hist(te, bins=bins, alpha=0.6, color="orange", density=True, label="test")
        ax.set_title(feat, fontsize=7)
        ax.tick_params(labelsize=5)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Top 16 features by train→test mean shift (train=blue, test=orange)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "feature_overview.png", dpi=120)
    plt.close(fig)
