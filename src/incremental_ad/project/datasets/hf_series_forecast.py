"""Generic single-continuous-series forecasting dataset, backed by
thuml/Time-Series-Library on HuggingFace.

Factored out of etth.py once ETTh1's forecasting logic turned out to be 100%
dataset-agnostic (it never referenced ETTh1 specifics beyond the HF config name and
a display name for analysis plots) — every method here is identical to what
EtthForecastDataset had. etth.py itself is left as its own, separately-loaded
implementation (not retrofitted onto this base) to avoid touching already-verified
code; new single-series HF forecast datasets (weather, traffic, exchange_rate, ...)
should subclass this instead of copy-pasting etth.py.

Subclassing contract: set _HF_DATASET_NAME and _DISPLAY_NAME as class attributes.
Nothing else needs overriding — see weather.py/traffic.py/exchange_rate.py.
"""

import logging
from abc import abstractmethod
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
from incremental_ad.framework.datasets.forecast_window import ForecastWindowDataset
from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset
from incremental_ad.framework.datasets.splitting import (
    all_segment_ranges,
    baseline_range,
    finetune_ranges,
    val_tail_split,
)

log = logging.getLogger(__name__)

Normalization = Literal["standard", "none"]


class HfSeriesForecastDataset(TimeSeriesDataset, PartitionedDataset):
    """Single continuous series (only a 'train' split exists on HuggingFace) for
    time-series forecasting. Implements the ForecastDataset protocol (forecast_len)
    and PartitionedDataset so it works with StandardPipeline, IncrementalTaskArithmetic,
    and EvalPipeline alike.

    window_len = context_len + forecast_len (total window fed to the model).
    MaeTxForecastingConfigurator forces CAUSAL_MASK and derives forecast_patches
    automatically from forecast_len // patch_len.
    """

    ARG_PREFIX = "dataset"
    _HF_DATASET_PATH = "thuml/Time-Series-Library"
    _DATE_COL = "date"

    # Subclasses override these two as plain class attributes (e.g. _HF_DATASET_NAME =
    # "weather") — declared as abstract properties here purely so that
    # inspect.isabstract() stays True for this base class, keeping it out of the
    # --dataset registry (Dataset.__init_subclass__ auto-registers concrete subclasses
    # only). A plain string class attribute in a subclass correctly satisfies an
    # abstract property in Python's ABCMeta.
    @property
    @abstractmethod
    def _HF_DATASET_NAME(self) -> str: ...

    @property
    @abstractmethod
    def _DISPLAY_NAME(self) -> str: ...

    def __init__(
        self,
        window_len: int,
        forecast_len: int,
        stride: int,
        normalization: Normalization,
        split_config: SplitConfig,
        test_fraction: float,
        series_fraction: float = 1.0,
    ) -> None:
        self._window_len = window_len
        self.stride = stride
        self._eval_stride = 1
        self._test_fraction = test_fraction
        self._series_fraction = series_fraction
        self._train_data, self._test_data = self._prepare_data(
            normalization, test_fraction, series_fraction
        )
        self._forecast_len = forecast_len
        self.split_config = split_config
        self.split_config.validate(len(self._train_data))

    @property
    def n_features(self) -> int:
        return self._train_data.shape[1]

    @property
    def window_len(self) -> int:
        return self._window_len

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
        parser.add_argument(
            f"--{p}_test_fraction", type=float, default=0.2,
            help="Fraction of the series held out (chronologically) as the test set. "
                 "0 = no test set (train + val only).",
        )
        parser.add_argument(
            f"--{p}_series_fraction", type=float, default=1.0,
            help="Truncate the series to this fraction BEFORE the train/test split. "
                 "Moves the rolling-origin cut without changing the test block's "
                 "relative size: 1.0 tests on [0.8,1.0] of the series, 0.875 on "
                 "[0.70,0.875], 0.75 on [0.60,0.75]. Data past the cut is discarded, "
                 "so no origin ever trains on its own future.",
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
            test_fraction=getattr(cfg, f"{p}_test_fraction"),
            series_fraction=getattr(cfg, f"{p}_series_fraction", 1.0),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    @property
    def capabilities(self) -> set[DatasetCapability]:
        caps: set[DatasetCapability] = set()
        if len(self._test_data) >= self._window_len:
            caps |= {DatasetCapability.TEST, DatasetCapability.TEST_LABELS}
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

    def _val_eval_for_range(self, start: int, end: int) -> ForecastWindowDataset:
        """Eval-stride windowing over the val tail _make_segment() holds out from
        [start, end) — same val range, but eval_stride=1 covers every timestep."""
        _, (val_start, val_end) = val_tail_split(
            start, end, self.split_config.val_fraction
        )
        return ForecastWindowDataset(
            self._train_data[val_start:val_end],
            self._window_len,
            self._forecast_len,
            self._eval_stride,
        )

    def get_val_eval_dataset(self) -> ForecastWindowDataset:
        """StandardPipeline val-eval: the global val tail (mirrors get_train_segment())."""
        return self._val_eval_for_range(0, len(self._train_data))

    def get_baseline_val_eval_dataset(self) -> ForecastWindowDataset:
        """Incremental baseline/val: the baseline segment's own held-out slice."""
        start, end = baseline_range(len(self._train_data), self.split_config)
        return self._val_eval_for_range(start, end)

    def get_merged_val_eval_dataset(self) -> ConcatDataset:
        """Incremental merged/val: union of every segment's held-out val slice
        (baseline + each finetune), so the merge is checked across all regimes."""
        return ConcatDataset(
            [
                self._val_eval_for_range(start, end)
                for start, end in all_segment_ranges(
                    len(self._train_data), self.split_config
                )
            ]
        )

    def get_finetune_val_eval_dataset(self, index: int) -> ForecastWindowDataset:
        """Incremental finetune_i/val: that finetune segment's own held-out slice."""
        start, end = finetune_ranges(len(self._train_data), self.split_config)[index]
        return self._val_eval_for_range(start, end)

    def get_test_dataset(self) -> ForecastWindowDataset:
        return ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )

    def analyze(self, analysis_dir: Path) -> None:
        done_flag = analysis_dir / "done.flag"
        if done_flag.exists():
            log.debug("Skipping dataset analysis — already written to %s", analysis_dir)
            return
        analysis_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing dataset analysis to %s ...", analysis_dir)

        from incremental_ad.framework.datasets import analysis

        train_df = self._load_raw()
        features = self._feature_cols(train_df)
        series = train_df[features].to_numpy(dtype=float)
        n = len(series)
        test_start = n - int(n * self._test_fraction) if self._test_fraction > 0 else n
        analysis.run_timeseries_analysis(
            analysis_dir,
            name=self._DISPLAY_NAME,
            train=series[:test_start],
            test=series[test_start:] if test_start < n else None,
            feature_names=features,
            test_labels=None,
        )
        done_flag.touch()
        log.info("Dataset analysis complete.")

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
                # Same shape as train (bare windows, not (window, future) pairs) — see
                # etth.py's note on why Segment.val must match train's batch format.
                val=SlidingWindowDataset(
                    self._train_data[val_start:val_end], self._window_len, self.stride
                ),
            )
        return Segment(
            train=SlidingWindowDataset(
                self._train_data[start:end], self._window_len, self.stride
            ),
            val=None,
        )

    def _prepare_data(
        self, normalization: Normalization, test_fraction: float,
        series_fraction: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Load the single series and carve a chronological train/test split.

        Only a 'train' split exists on HuggingFace, so the last test_fraction of it is
        held out as the test set (test_fraction=0 → no test set). The scaler is fit on
        the train portion only, then applied to both, so test normalization uses
        training statistics with no leakage.

        ``series_fraction`` < 1.0 truncates the series **before** the split, which moves the
        train/test origin without changing the test block's relative size. It is the knob for a
        rolling-origin evaluation: f=1.0 tests on [0.8, 1.0] of the series, f=0.875 on
        [0.70, 0.875], f=0.75 on [0.60, 0.75]. Every result in this project rests on a single
        origin (f=1.0), and one cut gives no error bar for the choice of cut — see EXPERIMENTS.md.
        Data after the truncation point is discarded, never used for training, so causality holds
        at every origin.
        """
        train_df = self._load_raw()
        features = self._feature_cols(train_df)
        series = train_df[features].values.astype(np.float32)

        if series_fraction < 1.0:
            series = series[: int(len(series) * series_fraction)]

        n = len(series)
        test_start = n - int(n * test_fraction) if test_fraction > 0 else n
        train_raw, test_raw = series[:test_start], series[test_start:]

        log.info(
            f"{self._DISPLAY_NAME} — {n} rows ({len(train_raw)} train / {len(test_raw)} test), "
            f"features: {len(features)}"
        )
        log.info(f"Normalization: {normalization}")

        if normalization == "standard":
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw) if len(test_raw) else test_raw
        else:
            train_scaled, test_scaled = train_raw, test_raw

        return torch.tensor(train_scaled), torch.tensor(test_scaled)

    def _load_raw(self) -> pd.DataFrame:
        log.info(f"Loading {self._DISPLAY_NAME} from HuggingFace...")
        raw = load_dataset(self._HF_DATASET_PATH, self._HF_DATASET_NAME)
        train_df = cast(pd.DataFrame, raw["train"].to_pandas())
        log.info(f"Loaded — train: {len(train_df)} rows")
        return train_df

    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c != self._DATE_COL]
