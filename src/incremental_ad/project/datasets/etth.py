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
from torch.utils.data import ConcatDataset, Dataset as TorchDataset

from incremental_ad.framework.contracts.dataset import (
    DatasetCapability,
    PartitionedDataset,
    Segment,
    SplitConfig,
    TimeSeriesDataset,
)
from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset
from incremental_ad.framework.datasets.splitting import (
    all_segment_ranges,
    baseline_range,
    finetune_ranges,
    val_tail_split,
)

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

        from incremental_ad.framework.datasets import analysis

        train_df, test_df = _load_raw()
        features = _feature_cols(train_df)
        analysis.run_timeseries_analysis(
            analysis_dir,
            name="ETTh1",
            train=train_df[features].to_numpy(dtype=float),
            test=test_df[features].to_numpy(dtype=float),
            feature_names=features,
            test_labels=None,  # ETTh1 has no anomaly labels
        )
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
        self.split_config.validate(len(self._train_data))

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
        caps = {DatasetCapability.TEST, DatasetCapability.TEST_LABELS}
        if self.split_config.val_fraction > 0:
            caps.add(DatasetCapability.VAL)
        return caps

    def get_train_segment(self) -> Segment:
        return self._make_segment(0, len(self._train_data))

    def get_baseline(self) -> Segment:
        start, end = baseline_range(len(self._train_data), self.split_config)
        return self._make_segment(start, end)

    def get_incremental_segments(self) -> list[Segment]:
        return [
            self._make_segment(start, end)
            for start, end in finetune_ranges(len(self._train_data), self.split_config)
        ]

    def _val_eval_for_range(self, start: int, end: int) -> _ForecastWindowDataset:
        """Eval-stride windowing over the val tail _make_segment() holds out from
        [start, end) — same val range, but eval_stride=1 covers every timestep."""
        _, (val_start, val_end) = val_tail_split(start, end, self.split_config.val_fraction)
        return _ForecastWindowDataset(
            self._train_data[val_start:val_end], self._window_len, self._forecast_len, self._eval_stride
        )

    def get_val_eval_dataset(self) -> _ForecastWindowDataset:
        """StandardPipeline val-eval: the global val tail (mirrors get_train_segment())."""
        return self._val_eval_for_range(0, len(self._train_data))

    def get_baseline_val_eval_dataset(self) -> _ForecastWindowDataset:
        """Incremental baseline/val: the baseline segment's own held-out slice."""
        start, end = baseline_range(len(self._train_data), self.split_config)
        return self._val_eval_for_range(start, end)

    def get_incremental_val_eval_dataset(self) -> ConcatDataset:
        """Incremental merged/val: union of every segment's held-out val slice
        (baseline + each finetune), so the merge is checked across all regimes."""
        return ConcatDataset(
            [
                self._val_eval_for_range(start, end)
                for start, end in all_segment_ranges(len(self._train_data), self.split_config)
            ]
        )

    def get_test_dataset(self) -> _ForecastWindowDataset:
        return _ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_segment(self, start: int, end: int) -> Segment:
        if self.split_config.val_fraction > 0:
            (tr_start, tr_end), (val_start, val_end) = val_tail_split(
                start, end, self.split_config.val_fraction
            )
            return Segment(
                train=SlidingWindowDataset(
                    self._train_data[tr_start:tr_end], self._window_len, self.stride
                ),
                val=_ForecastWindowDataset(
                    self._train_data[val_start:val_end],
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
        caps = {DatasetCapability.TEST, DatasetCapability.TEST_LABELS}
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
                # Clean windows: the trainer's val loss uses model.compute_loss
                # (random-mask MAE), so it must see uncorrupted inputs like training.
                # The fixed-mask imputation metric is evaluated separately via
                # get_val_eval_dataset() (the _ImputationWindowDataset path).
                val=SlidingWindowDataset(
                    self._train_data[val_start:], self._window_len, self.stride
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
