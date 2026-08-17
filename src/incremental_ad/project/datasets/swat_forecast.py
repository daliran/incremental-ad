"""SWaT as a *forecasting* dataset, with a genuinely separate test capture.

SWaT is 11 days of continuous operation on the Secure Water Treatment testbed: **7 days of
normal operation** (the train split) followed by **4 days containing 41 attacks** (the test
split), 51 sensors and actuators. Unlike every forecasting dataset in this project, which is one
recording cut at 80%, train and test here are two separate stretches of plant operation.

Same scoring rule as `PsmForecastDataset`: a test window is kept only when **every** point it
spans is labelled normal — input span *and* forecast horizon — because forecasting error during
an attack measures how badly a normal-behaviour model predicts an attack, which is a different
question and would dominate the metric. The labels decide which windows to score, nothing else.

⚠️ **Expect this to separate very little.** SWaT's within-series drift is 0.112, the lowest of
any dataset measured here (EXPERIMENTS.md 0.1b): a tightly controlled testbed running a fixed
process, so its periods barely differ and there is little for a merging technique to exploit.
`PsmForecastDataset` is the better home for that question — PSM drifts at 0.465 over real server
telemetry. SWaT is worth running as the *saturated control*: a dataset where the honest answer
is "nothing separates" is useful precisely because it shows what that looks like.

Neither dataset was designed for forecasting, so there is no published baseline. That is fine
for a relative study — technique X against standard merging — and must not be presented as a
benchmark result.
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
_HF_DATASET_NAME = "SWaT"
_LABEL_COL = "Normal/Attack"


class SwatForecastDataset(HfSeriesForecastDataset):
    """SWaT normal-operation capture for training; the attack capture, masked, for evaluation."""

    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = _HF_DATASET_NAME
    _DISPLAY_NAME = "SWaT-forecast"

    def _load_raw(self) -> pd.DataFrame:
        from datasets import load_dataset

        log.info("Loading SWaT (forecasting view) from HuggingFace...")
        raw = load_dataset(_HF_DATASET_PATH, _HF_DATASET_NAME)
        train_df = cast(pd.DataFrame, raw["train"].to_pandas())
        if _LABEL_COL in train_df.columns:
            train_df = train_df.drop(columns=[_LABEL_COL])
        log.info("Loaded — train: %d rows", len(train_df))
        return train_df

    def _load_test_raw(self) -> tuple[pd.DataFrame, np.ndarray]:
        """The attack capture and its point labels (1 = attack)."""
        from datasets import load_dataset

        raw = load_dataset(_HF_DATASET_PATH, _HF_DATASET_NAME)
        test_df = cast(pd.DataFrame, raw["test"].to_pandas())
        labels = pd.to_numeric(test_df[_LABEL_COL], errors="raise").astype(int).to_numpy()
        return test_df.drop(columns=[_LABEL_COL]), labels

    def _prepare_data(
        self, normalization, test_fraction: float, series_fraction: float = 1.0
    ) -> tuple[Tensor, Tensor]:
        """Train from the normal capture; test from the **attack capture**, not a tail cut.

        `test_fraction` is ignored: the split is given by the dataset, which is the reason for
        using SWaT here. `series_fraction` still truncates the *training* capture so a rolling
        origin over the normal days remains available; the test capture is never truncated.
        """
        from sklearn.preprocessing import StandardScaler

        train_df = self._load_raw()
        test_df, labels = self._load_test_raw()
        features = [c for c in train_df.columns if c in test_df.columns]
        train_series = np.nan_to_num(train_df[features].to_numpy(dtype=np.float32))
        test_series = np.nan_to_num(test_df[features].to_numpy(dtype=np.float32))

        if series_fraction < 1.0:
            train_series = train_series[: int(len(train_series) * series_fraction)]

        self._test_labels = labels[: len(test_series)]
        log.info("SWaT-forecast — train %d rows (normal capture), test %d rows, "
                 "%d attack points (%.1f%%), %d features", len(train_series), len(test_series),
                 int(self._test_labels.sum()), 100.0 * self._test_labels.mean(), len(features))

        if normalization == "standard":
            scaler = StandardScaler().fit(train_series)
            train_scaled = scaler.transform(train_series).astype(np.float32)
            test_scaled = scaler.transform(test_series).astype(np.float32)
        else:
            train_scaled, test_scaled = train_series, test_series
        return torch.tensor(train_scaled), torch.tensor(test_scaled)

    def get_test_dataset(self) -> ForecastWindowDataset:
        """Test windows that are entirely attack-free — input span **and** forecast horizon."""
        full = ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )
        span = self._window_len + self._forecast_len
        labels = self._test_labels
        keep = [i for i in range(len(full))
                if labels[i * self._eval_stride: i * self._eval_stride + span].sum() == 0]
        if not keep:
            raise ValueError("no attack-free test window — check the label alignment")
        log.info("SWaT-forecast — scoring %d of %d test windows (%.1f%% attack-free)",
                 len(keep), len(full), 100.0 * len(keep) / len(full))
        return cast(ForecastWindowDataset, torch.utils.data.Subset(full, keep))
