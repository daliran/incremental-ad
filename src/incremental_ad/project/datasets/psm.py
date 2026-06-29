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
from torch.utils.data import ConcatDataset

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
_HF_DATA_NAME = "PSM-data"
_HF_LABEL_NAME = "PSM-label"
_TIMESTAMP_COL = "timestamp_(min)"

Normalization = Literal["standard", "none"]


class Psm(TimeSeriesDataset, PartitionedDataset):

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
        self.split_config.validate(len(self._train_data))

    @property
    def n_features(self) -> int:
        return self._train_data.shape[1]

    @property
    def window_len(self) -> int:
        return self._window_len

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
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
        start, end = baseline_range(len(self._train_data), self.split_config)
        return self._make_segment(start, end)

    def get_incremental_segments(self) -> list[Segment]:
        return [
            self._make_segment(start, end)
            for start, end in finetune_ranges(len(self._train_data), self.split_config)
        ]

    def get_train_eval_dataset(self) -> SlidingWindowDataset:
        return SlidingWindowDataset(self._train_data, self.window_len, self._eval_stride)

    def _val_eval_for_range(self, start: int, end: int) -> SlidingWindowDataset:
        """Eval-stride windowing over the val tail _make_segment() holds out from
        [start, end) — same val range, but eval_stride=1 covers every timestep."""
        _, (val_start, val_end) = val_tail_split(start, end, self.split_config.val_fraction)
        return SlidingWindowDataset(
            self._train_data[val_start:val_end], self.window_len, self._eval_stride,
            labels=torch.zeros(val_end - val_start, dtype=torch.uint8),
        )

    def get_val_eval_dataset(self) -> SlidingWindowDataset:
        """StandardPipeline val-eval: the global val tail (mirrors get_train_segment())."""
        return self._val_eval_for_range(0, len(self._train_data))

    def get_baseline_val_eval_dataset(self) -> SlidingWindowDataset:
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

        from incremental_ad.framework.datasets import analysis

        train_df, test_df, label_df = _load_raw()
        train_samples = _extract_features(train_df).ffill().bfill()
        test_samples = _extract_features(test_df)
        test_labels_raw = label_df["label"].astype(int)
        features = list(train_samples.columns)

        analysis.run_timeseries_analysis(
            analysis_dir,
            name="PSM",
            train=train_samples.to_numpy(dtype=float),
            test=test_samples.to_numpy(dtype=float),
            feature_names=features,
            test_labels=test_labels_raw.to_numpy(),
        )

        done_flag.touch()
        log.info("Dataset analysis complete.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_segment(self, start: int, end: int) -> Segment:
        if self.split_config.val_fraction > 0:
            (tr_start, tr_end), (val_start, val_end) = val_tail_split(
                start, end, self.split_config.val_fraction
            )
            return Segment(
                train=SlidingWindowDataset(self._train_data[tr_start:tr_end], self.window_len, self.stride),
                val=SlidingWindowDataset(self._train_data[val_start:val_end], self.window_len, self.stride),
            )
        return Segment(
            train=SlidingWindowDataset(self._train_data[start:end], self.window_len, self.stride),
            val=None,
        )


# ── Data loading and preprocessing ────────────────────────────────────────────


def _prepare_data(normalization: Normalization) -> tuple[Tensor, Tensor, Tensor]:
    """Load, scale, and return (train_data, test_data, test_labels) as tensors."""
    train_df, test_df, label_df = _load_raw()

    train_samples = _extract_features(train_df).ffill().bfill()
    test_samples = _extract_features(test_df)
    test_labels = label_df["label"].astype(int)

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


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading PSM dataset from HuggingFace...")
    raw = load_dataset(_HF_DATASET_PATH, _HF_DATA_NAME)
    raw_label = load_dataset(_HF_DATASET_PATH, _HF_LABEL_NAME)
    train_df = cast(pd.DataFrame, raw["train"].to_pandas())
    test_df = cast(pd.DataFrame, raw["test"].to_pandas())
    label_df = cast(pd.DataFrame, raw_label["test_label"].to_pandas())
    log.info(f"Loaded — train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df, label_df


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[_TIMESTAMP_COL])
