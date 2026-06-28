"""Synthetic multi-class time-series dataset for testing the classification pipeline.

NOT for production use. Generates deterministic sine waves where each class has a
distinct dominant frequency, so the task is learnable without real data.

Training split  → _ClassWindowDataset (window, label) 2-tuples — supervised
Val / test split → _ClassWindowDataset (window, label) 2-tuples — same format
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


class _ClassWindowDataset(TorchDataset):
    """Yields (window [W, F], label scalar) 2-tuples.

    Label is taken from the class index of the window's centre timestep, so windows
    that straddle a boundary get the label of whichever class is at the midpoint.
    """

    def __init__(self, data: Tensor, class_ids: Tensor, window_len: int, stride: int) -> None:
        self.data = data
        self.class_ids = class_ids   # [T] integer class index per timestep
        self.window_len = window_len
        self.stride = stride

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.stride
        window = self.data[start : start + self.window_len]         # [W, F]
        centre = start + self.window_len // 2
        label = self.class_ids[centre]                               # scalar int64
        return window, label


class TestClassificationDataset(TimeSeriesDataset):
    """Synthetic multi-class time-series classification dataset.

    Generates a time series that cycles through `n_classes` regimes, each with a
    distinct dominant frequency. Classifier must learn to identify the regime from
    a fixed-length window.

    Each class occupies `segment_len` consecutive timesteps, and the pattern repeats
    for the full duration of the split.
    """

    ARG_PREFIX = "dataset"

    def __init__(
        self,
        window_len: int,
        n_classes: int = 4,
        segment_len: int = 200,
        n_features: int = 8,
        n_train_steps: int = 8000,
        n_test_steps: int = 2000,
        val_fraction: float = 0.1,
        stride: int = 1,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._window_len = window_len
        self._n_classes = n_classes
        self._n_features = n_features
        self.val_fraction = val_fraction
        self.stride = stride
        self._eval_stride = 1

        train_data, train_ids, test_data, test_ids = _generate_multiclass(
            n_train_steps, n_test_steps, n_classes, segment_len, n_features, noise, seed
        )
        self._train_data = train_data
        self._train_ids = train_ids
        self._test_data = test_data
        self._test_ids = test_ids

    # ── TimeSeriesDataset interface ────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def window_len(self) -> int:
        return self._window_len

    @property
    def n_classes(self) -> int:
        return self._n_classes

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
                train=_ClassWindowDataset(
                    self._train_data[:val_start],
                    self._train_ids[:val_start],
                    self._window_len,
                    self.stride,
                ),
                val=_ClassWindowDataset(
                    self._train_data[val_start:],
                    self._train_ids[val_start:],
                    self._window_len,
                    self.stride,
                ),
            )
        return Segment(
            train=_ClassWindowDataset(
                self._train_data, self._train_ids, self._window_len, self.stride
            ),
            val=None,
        )

    def get_val_eval_dataset(self) -> _ClassWindowDataset:
        n = len(self._train_data)
        val_start = n - int(n * self.val_fraction)
        return _ClassWindowDataset(
            self._train_data[val_start:], self._train_ids[val_start:], self._window_len, self._eval_stride
        )

    def get_test_dataset(self) -> _ClassWindowDataset:
        return _ClassWindowDataset(
            self._test_data, self._test_ids, self._window_len, self._eval_stride
        )

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(f"--{p}_n_classes", type=int, default=4)
        parser.add_argument(f"--{p}_segment_len", type=int, default=200)
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
            n_classes=getattr(cfg, f"{p}_n_classes"),
            segment_len=getattr(cfg, f"{p}_segment_len"),
            n_features=getattr(cfg, f"{p}_n_features"),
            n_train_steps=getattr(cfg, f"{p}_n_train_steps"),
            n_test_steps=getattr(cfg, f"{p}_n_test_steps"),
            val_fraction=getattr(cfg, f"{p}_val_fraction"),
            stride=getattr(cfg, f"{p}_stride"),
            noise=getattr(cfg, f"{p}_noise"),
            seed=getattr(cfg, f"{p}_seed"),
        )


# ── Synthetic data generation ──────────────────────────────────────────────────


def _generate_multiclass(
    n_train: int,
    n_test: int,
    n_classes: int,
    segment_len: int,
    n_features: int,
    noise: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate a time series with `n_classes` repeating frequency regimes.

    Class k uses base_freq * (k + 1) as its dominant frequency so each class
    is spectrally distinct. Returns (data, class_ids) for train and test splits.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    total = n_train + n_test

    t = torch.arange(total, dtype=torch.float32)
    class_ids = (t.long() // segment_len) % n_classes  # [T] — cycles 0,1,...,k-1,0,1,...

    base_freq = 0.02
    # Build the data: each timestep's features are dominated by its class frequency
    data = torch.zeros(total, n_features)
    for k in range(n_classes):
        mask = (class_ids == k)
        freq = base_freq * (k + 1)
        phases = torch.linspace(0.0, math.pi, n_features)
        segment_signal = torch.stack(
            [torch.sin(2 * math.pi * freq * t + p) for p in phases], dim=1
        )  # [T, F]
        data[mask] = segment_signal[mask]

    data = data + noise * torch.randn(total, n_features, generator=rng)

    return data[:n_train], class_ids[:n_train], data[n_train:], class_ids[n_train:]
