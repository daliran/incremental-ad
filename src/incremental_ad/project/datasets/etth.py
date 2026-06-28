"""ETTh1 — hourly electricity transformer temperature dataset.

Source: thuml/Time-Series-Library on HuggingFace (config name: ETTh1).
No anomaly labels — suitable for forecasting and imputation only.

Two concrete classes:
  EtthForecastDataset   — forecasting + IncrementalTaskArithmetic (PartitionedDataset)
  EtthImputationDataset — masked-patch imputation
"""

import logging
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

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
_HF_DATASET_NAME = "ETTh1"
_DATE_COL = "date"

Normalization = Literal["standard", "none"]


# ── Eval window dataset helpers ────────────────────────────────────────────────


class _ForecastWindowDataset(TorchDataset):
    """Yields (full_window [W, F], future [forecast_len, F]) 2-tuples."""

    def __init__(self, data: Tensor, window_len: int, forecast_len: int, stride: int) -> None:
        self.data = data
        self.window_len = window_len
        self.context_len = window_len - forecast_len
        self.forecast_len = forecast_len
        self.stride = stride

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.stride
        full_window = self.data[start : start + self.window_len]
        return full_window, full_window[self.context_len :]


class _ImputationWindowDataset(TorchDataset):
    """Yields (masked_window [W, F], (original [W, F], visible_idx [n_visible])) tuples.

    Mask is deterministic per window index for reproducible evaluation.
    """

    def __init__(
        self, data: Tensor, window_len: int, patch_len: int, mask_ratio: float, stride: int
    ) -> None:
        self.data = data
        self.window_len = window_len
        self.patch_len = patch_len
        self.n_patches = window_len // patch_len
        self.n_masked = int(self.n_patches * mask_ratio)
        self.stride = stride

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        start = idx * self.stride
        window = self.data[start : start + self.window_len]
        gen = torch.Generator().manual_seed(idx)
        perm = torch.randperm(self.n_patches, generator=gen)
        visible_idx = perm[self.n_masked :].sort().values
        mask_idx = perm[: self.n_masked].sort().values
        masked_window = window.clone()
        for p in mask_idx.tolist():
            masked_window[p * self.patch_len : (p + 1) * self.patch_len] = 0.0
        return masked_window, (window, visible_idx)


# ── Shared base ────────────────────────────────────────────────────────────────


class _EtthBase(TimeSeriesDataset, ABC):
    """Shared data loading, properties, and analysis for ETTh1 datasets.

    Subclasses implement get_train_segment(), get_test_dataset(), capabilities,
    and the task-specific protocol property (forecast_len or mask_patch_len).
    """

    def __init__(
        self,
        window_len: int,
        stride: int,
        normalization: Normalization,
    ) -> None:
        self._window_len = window_len
        self.stride = stride
        self._eval_stride = 1
        self._train_data, self._test_data = _prepare_data(normalization)

    @property
    def n_features(self) -> int:
        return self._train_data.shape[1]

    @property
    def window_len(self) -> int:
        return self._window_len

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
        features = _feature_cols(train_df)
        _write_feature_stats(train_df[features], test_df[features], features, analysis_dir)
        _plot_feature_overview(train_df[features], test_df[features], features, analysis_dir)
        done_flag.touch()
        log.info("Dataset analysis complete.")


# ── Forecasting ────────────────────────────────────────────────────────────────


class EtthForecastDataset(_EtthBase, PartitionedDataset):
    """ETTh1 for time-series forecasting.

    Implements the ForecastDataset protocol (forecast_len) and PartitionedDataset
    so it works with all three pipelines: Standard, IncrementalTaskArithmetic, Eval.

    window_len = context_len + forecast_len (total window fed to the model).
    MaeTxForecastingConfigurator forces CAUSAL_MASK and derives forecast_patches
    automatically from forecast_len // patch_len.
    """

    ARG_PREFIX = "dataset"

    def __init__(
        self,
        window_len: int,
        forecast_len: int,
        stride: int,
        normalization: Normalization,
        split_config: SplitConfig,
    ) -> None:
        _EtthBase.__init__(self, window_len, stride, normalization)
        self._forecast_len = forecast_len
        self.split_config = split_config

    @property
    def forecast_len(self) -> int:
        return self._forecast_len

    # ── Configurable ───────────────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)  # PartitionedDataset → SplitConfig args
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(f"--{p}_forecast_len", type=int, required=True)
        parser.add_argument(f"--{p}_stride", type=int, required=True)
        parser.add_argument(
            f"--{p}_normalization", choices=["standard", "none"], required=True
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            window_len=getattr(cfg, f"{p}_window_len"),
            forecast_len=getattr(cfg, f"{p}_forecast_len"),
            stride=getattr(cfg, f"{p}_stride"),
            normalization=getattr(cfg, f"{p}_normalization"),
            split_config=cls._split_config_from_cfg(cfg, prefix),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    @property
    def capabilities(self) -> set[DatasetCapability]:
        caps = {DatasetCapability.TEST}
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

    def get_val_eval_dataset(self) -> _ForecastWindowDataset:
        n = len(self._train_data)
        val_start = n - int(n * self.split_config.val_fraction)
        return _ForecastWindowDataset(
            self._train_data[val_start:], self._window_len, self._forecast_len, self._eval_stride
        )

    def get_test_dataset(self) -> _ForecastWindowDataset:
        return _ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_segment(self, start: int, end: int) -> Segment:
        if self.split_config.val_fraction > 0:
            val_start = end - int((end - start) * self.split_config.val_fraction)
            return Segment(
                train=SlidingWindowDataset(
                    self._train_data[start:val_start], self._window_len, self.stride
                ),
                val=_ForecastWindowDataset(
                    self._train_data[val_start:end],
                    self._window_len,
                    self._forecast_len,
                    self.stride,
                ),
            )
        return Segment(
            train=SlidingWindowDataset(
                self._train_data[start:end], self._window_len, self.stride
            ),
            val=None,
        )


# ── Imputation ─────────────────────────────────────────────────────────────────


class EtthImputationDataset(_EtthBase):
    """ETTh1 for masked-patch imputation.

    Implements the ImputationDataset protocol (mask_patch_len).
    patch_len must equal --mae_tx_patch_len so masked token positions align
    with the model's tokenisation. MaeTxImputationConfigurator enforces this.
    """

    ARG_PREFIX = "dataset"

    def __init__(
        self,
        window_len: int,
        patch_len: int,
        mask_ratio: float,
        stride: int,
        normalization: Normalization,
        val_fraction: float,
    ) -> None:
        _EtthBase.__init__(self, window_len, stride, normalization)
        self._patch_len = patch_len
        self._mask_ratio = mask_ratio
        self.val_fraction = val_fraction

    @property
    def mask_patch_len(self) -> int:
        return self._patch_len

    # ── Configurable ───────────────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(
            f"--{p}_patch_len",
            type=int,
            required=True,
            help="Must match --mae_tx_patch_len so masking aligns with tokenisation.",
        )
        parser.add_argument(f"--{p}_mask_ratio", type=float, default=0.5)
        parser.add_argument(f"--{p}_stride", type=int, required=True)
        parser.add_argument(
            f"--{p}_normalization", choices=["standard", "none"], required=True
        )
        parser.add_argument(f"--{p}_val_fraction", type=float, default=0.1)

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            window_len=getattr(cfg, f"{p}_window_len"),
            patch_len=getattr(cfg, f"{p}_patch_len"),
            mask_ratio=getattr(cfg, f"{p}_mask_ratio"),
            stride=getattr(cfg, f"{p}_stride"),
            normalization=getattr(cfg, f"{p}_normalization"),
            val_fraction=getattr(cfg, f"{p}_val_fraction"),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    @property
    def capabilities(self) -> set[DatasetCapability]:
        caps = {DatasetCapability.TEST}
        if self.val_fraction > 0:
            caps.add(DatasetCapability.VAL)
        return caps

    def get_train_segment(self) -> Segment:
        n = len(self._train_data)
        if self.val_fraction > 0:
            val_start = n - int(n * self.val_fraction)
            return Segment(
                train=SlidingWindowDataset(
                    self._train_data[:val_start], self._window_len, self.stride
                ),
                val=_ImputationWindowDataset(
                    self._train_data[val_start:],
                    self._window_len,
                    self._patch_len,
                    self._mask_ratio,
                    self.stride,
                ),
            )
        return Segment(
            train=SlidingWindowDataset(self._train_data, self._window_len, self.stride),
            val=None,
        )

    def get_val_eval_dataset(self) -> _ImputationWindowDataset:
        n = len(self._train_data)
        val_start = n - int(n * self.val_fraction)
        return _ImputationWindowDataset(
            self._train_data[val_start:],
            self._window_len,
            self._patch_len,
            self._mask_ratio,
            self._eval_stride,
        )

    def get_test_dataset(self) -> _ImputationWindowDataset:
        return _ImputationWindowDataset(
            self._test_data,
            self._window_len,
            self._patch_len,
            self._mask_ratio,
            self._eval_stride,
        )


# ── Data loading and preprocessing ────────────────────────────────────────────


def _prepare_data(normalization: Normalization) -> tuple[Tensor, Tensor]:
    """Load, scale, and return (train_data, test_data) as float32 tensors.

    Scaler is fit on the training split so test normalization uses training statistics.
    """
    train_df, test_df = _load_raw()
    features = _feature_cols(train_df)
    train_raw = train_df[features].values.astype(np.float32)
    test_raw = test_df[features].values.astype(np.float32)

    log.info(
        f"ETTh1 — train: {len(train_raw)} rows, test: {len(test_raw)} rows, "
        f"features: {len(features)}"
    )
    log.info(f"Normalization: {normalization}")

    if normalization == "standard":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)
    else:
        train_scaled, test_scaled = train_raw, test_raw

    return torch.tensor(train_scaled), torch.tensor(test_scaled)


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading ETTh1 from HuggingFace...")
    raw = load_dataset(_HF_DATASET_PATH, _HF_DATASET_NAME)
    train_df = cast(pd.DataFrame, raw["train"].to_pandas())
    test_df = cast(pd.DataFrame, raw["test"].to_pandas())
    log.info(f"Loaded — train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != _DATE_COL]


# ── Analysis helpers ───────────────────────────────────────────────────────────


def _write_feature_stats(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], out: Path
) -> None:
    rows = [
        {
            "feature": f,
            "train_mean": float(train[f].mean()),
            "train_std": float(train[f].std()),
            "test_mean": float(test[f].mean()),
            "test_std": float(test[f].std()),
        }
        for f in features
    ]
    pd.DataFrame(rows).to_csv(out / "feature_stats.csv", index=False)


def _plot_feature_overview(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], out: Path
) -> None:
    import matplotlib.pyplot as plt

    train_mean = train.mean()
    train_std = train.std().replace(0, 1)
    shift = ((test.mean() - train_mean) / train_std).abs()
    top_features = shift.nlargest(min(16, len(features))).index.tolist()

    n = len(top_features)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2.5), squeeze=False)
    axes_flat = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes_flat[i]
        tr, te = train[feat].values, test[feat].values
        bins = np.linspace(min(tr.min(), te.min()), max(tr.max(), te.max()), 40)
        ax.hist(tr, bins=bins, alpha=0.6, color="#4C72B0", density=True, label="train")
        ax.hist(te, bins=bins, alpha=0.6, color="orange", density=True, label="test")
        ax.set_title(feat, fontsize=7)
        ax.tick_params(labelsize=5)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Features by train→test mean shift (train=blue, test=orange)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "feature_overview.png", dpi=120)
    plt.close(fig)
