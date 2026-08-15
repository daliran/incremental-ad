"""PSM as a *forecasting* dataset, with a genuinely separate test capture.

Every forecasting dataset in this project is one recording cut at 80% — so "train" and "test"
are two stretches of the same series, and whether they share a regime is an accident of where
the cut landed (EXPERIMENTS.md). PSM is different: it ships **21 weeks of eBay application-server
telemetry split as 13 weeks train / 8 weeks test**, two separate captures of the same system.
That makes it the one dataset here where a merging technique can be studied without the
train/test similarity being a property of an arbitrary cut.

**Anomalies are excluded from scoring, not from the data.** PSM's test split contains labelled
anomalies; forecasting error during an anomaly measures "how badly does a normal-behaviour model
predict an incident", which is a different question and would swamp the metric. A test window is
kept only when **every** point it spans is labelled normal — inputs *and* the forecast horizon,
since an anomaly in the horizon corrupts the target just as badly as one in the input. The
labels are used only to decide which windows to score.

Why PSM rather than SWaT: SWaT's within-series drift is 0.112, the lowest of any dataset
measured here — a tightly controlled testbed running a fixed process, so its periods barely
differ and merging would have almost nothing to work with. PSM's is 0.465, the highest of the
anomaly-detection pair, over real server telemetry that genuinely moves.

Neither dataset was designed for forecasting, so there is no published baseline to compare
against. That is acceptable for a *relative* study — technique X against standard merging — and
should not be presented as a benchmark result.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from incremental_ad.framework.datasets.forecast_window import ForecastWindowDataset
from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset

log = logging.getLogger(__name__)

_HF_DATASET_PATH = "thuml/Time-Series-Library"
_HF_DATA_NAME = "PSM-data"
_HF_LABEL_NAME = "PSM-label"


class PsmForecastDataset(HfSeriesForecastDataset):
    """PSM train split for training; PSM test split, anomaly-masked, for evaluation."""

    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = _HF_DATA_NAME
    _DISPLAY_NAME = "PSM-forecast"
    _DATE_COL = "timestamp_(min)"

    def _load_raw(self) -> pd.DataFrame:
        from datasets import load_dataset

        log.info("Loading PSM (forecasting view) from HuggingFace...")
        raw = load_dataset(_HF_DATASET_PATH, _HF_DATA_NAME)
        train_df = cast(pd.DataFrame, raw["train"].to_pandas())
        log.info("Loaded — train: %d rows", len(train_df))
        return train_df

    def _load_test_raw(self) -> tuple[pd.DataFrame, np.ndarray]:
        """The separate test capture and its point labels (1 = anomalous)."""
        from datasets import load_dataset

        raw = load_dataset(_HF_DATASET_PATH, _HF_DATA_NAME)
        labels = load_dataset(_HF_DATASET_PATH, _HF_LABEL_NAME)
        test_df = cast(pd.DataFrame, raw["test"].to_pandas())
        label_df = cast(pd.DataFrame, labels["test_label"].to_pandas())
        return test_df, label_df["label"].astype(int).to_numpy()

    def _prepare_data(
        self, normalization, test_fraction: float, series_fraction: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        """Train from the train capture; test from the **test capture**, not a tail cut.

        `test_fraction` is ignored on purpose: the split is given by the dataset rather than
        chosen by us, which is the entire reason for using PSM here. `series_fraction` still
        truncates the *training* capture, so a rolling origin over the training weeks remains
        available; the test capture is never truncated.
        """
        from sklearn.preprocessing import StandardScaler

        train_df = self._load_raw()
        test_df, labels = self._load_test_raw()
        features = self._feature_cols(train_df)
        train_series = train_df[features].to_numpy(dtype=np.float32)
        test_series = test_df[features].to_numpy(dtype=np.float32)
        train_series = np.nan_to_num(train_series)
        test_series = np.nan_to_num(test_series)

        if series_fraction < 1.0:
            train_series = train_series[: int(len(train_series) * series_fraction)]

        self._test_labels = labels[: len(test_series)]
        log.info("PSM-forecast — train %d rows (separate capture), test %d rows, "
                 "%d anomalous points (%.1f%%)", len(train_series), len(test_series),
                 int(self._test_labels.sum()), 100.0 * self._test_labels.mean())

        if normalization == "standard":
            scaler = StandardScaler().fit(train_series)
            train_scaled = scaler.transform(train_series).astype(np.float32)
            test_scaled = scaler.transform(test_series).astype(np.float32)
        else:
            train_scaled, test_scaled = train_series, test_series
        return torch.tensor(train_scaled), torch.tensor(test_scaled)

    def get_test_dataset(self) -> ForecastWindowDataset:
        """Test windows that are entirely normal — input span **and** forecast horizon."""
        full = ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )
        span = self._window_len + self._forecast_len
        labels = self._test_labels
        keep = [i for i in range(len(full))
                if labels[i * self._eval_stride: i * self._eval_stride + span].sum() == 0]
        if not keep:
            raise ValueError("no anomaly-free test window — check the label alignment")
        log.info("PSM-forecast — scoring %d of %d test windows (%.1f%% anomaly-free)",
                 len(keep), len(full), 100.0 * len(keep) / len(full))
        return cast(ForecastWindowDataset, torch.utils.data.Subset(full, keep))
