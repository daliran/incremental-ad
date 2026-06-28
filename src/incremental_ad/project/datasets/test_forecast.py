"""Synthetic sine-wave forecasting dataset for testing the forecasting pipeline.

NOT for production use. Generates deterministic multi-variate sine waves so the
pipeline can be exercised end-to-end without downloading real data.

Training split  → SlidingWindowDataset (single tensors, self-supervised MAE training)
Val / test split → _ForecastWindowDataset (inputs, future) 2-tuples for eval
"""
import math
from argparse import ArgumentParser, Namespace
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from incremental_ad.framework.contracts.dataset import (
    Dataset,
    DatasetCapability,
    Segment,
    TimeSeriesDataset,
)
from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset


class _ForecastWindowDataset(TorchDataset):
    """Yields (full_window [W, F], future [forecast_len, F]) 2-tuples.

    full_window spans context_len + forecast_len timesteps.
    future is a view into the last forecast_len steps of full_window — zero extra copy.
    """

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
        full_window = self.data[start : start + self.window_len]  # [W, F]
        future = full_window[self.context_len :]                  # [forecast_len, F] — view
        return full_window, future


class TestForecastDataset(TimeSeriesDataset):
    """Synthetic multi-variate sine-wave dataset for forecasting pipeline tests.

    Each feature is a sine wave with a distinct frequency. Small Gaussian noise
    is added so the task is non-trivial. The full window passed to the model is
    context_len + forecast_len timesteps; the model predicts the last forecast_len
    steps from the first context_len steps (causal masking).

    window_len  = context_len + forecast_len  (total window fed to the model)
    forecast_len = number of future timesteps to predict
    """

    ARG_PREFIX = "dataset"

    def __init__(
        self,
        window_len: int,
        forecast_len: int,
        n_features: int = 8,
        n_train_steps: int = 8000,
        n_test_steps: int = 2000,
        val_fraction: float = 0.1,
        stride: int = 1,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        if forecast_len >= window_len:
            raise ValueError("forecast_len must be < window_len (context must have at least 1 step)")
        self._window_len = window_len
        self._forecast_len = forecast_len
        self._n_features = n_features
        self.val_fraction = val_fraction
        self.stride = stride
        self._eval_stride = 1

        train_data, test_data = _generate_sine(
            n_train_steps, n_test_steps, n_features, noise, seed
        )
        self._train_data = train_data
        self._test_data = test_data

    # ── TimeSeriesDataset interface ────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def window_len(self) -> int:
        return self._window_len

    @property
    def forecast_len(self) -> int:
        return self._forecast_len

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
                val=_ForecastWindowDataset(
                    self._train_data[val_start:], self._window_len, self._forecast_len, self.stride
                ),
            )
        return Segment(
            train=SlidingWindowDataset(self._train_data, self._window_len, self.stride),
            val=None,
        )

    def get_val_eval_dataset(self) -> _ForecastWindowDataset:
        n = len(self._train_data)
        val_start = n - int(n * self.val_fraction)
        return _ForecastWindowDataset(
            self._train_data[val_start:], self._window_len, self._forecast_len, self._eval_stride
        )

    def get_test_dataset(self) -> _ForecastWindowDataset:
        return _ForecastWindowDataset(
            self._test_data, self._window_len, self._forecast_len, self._eval_stride
        )

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(f"--{p}_forecast_len", type=int, required=True)
        parser.add_argument(f"--{p}_n_features", type=int, default=8)
        parser.add_argument(f"--{p}_n_train_steps", type=int, default=8000)
        parser.add_argument(f"--{p}_n_test_steps", type=int, default=2000)
        parser.add_argument(f"--{p}_val_fraction", type=float, default=0.1)
        parser.add_argument(f"--{p}_stride", type=int, default=1)
        parser.add_argument(f"--{p}_noise", type=float, default=0.05)
        parser.add_argument(f"--{p}_seed", type=int, default=0)

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            window_len=getattr(cfg, f"{p}_window_len"),
            forecast_len=getattr(cfg, f"{p}_forecast_len"),
            n_features=getattr(cfg, f"{p}_n_features"),
            n_train_steps=getattr(cfg, f"{p}_n_train_steps"),
            n_test_steps=getattr(cfg, f"{p}_n_test_steps"),
            val_fraction=getattr(cfg, f"{p}_val_fraction"),
            stride=getattr(cfg, f"{p}_stride"),
            noise=getattr(cfg, f"{p}_noise"),
            seed=getattr(cfg, f"{p}_seed"),
        )


# ── Synthetic data generation ──────────────────────────────────────────────────


def _generate_sine(
    n_train: int, n_test: int, n_features: int, noise: float, seed: int
) -> tuple[Tensor, Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    total = n_train + n_test
    t = torch.arange(total, dtype=torch.float32)
    freqs = torch.linspace(0.01, 0.07, n_features)
    phases = torch.linspace(0.0, math.pi, n_features)
    data = torch.stack(
        [torch.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, phases)], dim=1
    )
    data = data + noise * torch.randn(total, n_features, generator=rng)
    return data[:n_train], data[n_train:]
