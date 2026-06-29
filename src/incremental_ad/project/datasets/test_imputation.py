"""Synthetic sine-wave imputation dataset for testing the imputation pipeline.

NOT for production use. Generates deterministic multi-variate sine waves so the
pipeline can be exercised end-to-end without downloading real data.

Training split  → SlidingWindowDataset (single tensors, self-supervised MAE training)
Val / test split → _ImputationWindowDataset — yields
                   (masked_window [W, F], (original [W, F], visible_idx [n_visible]))
                   with a deterministic per-window mask so the evaluator can compare
                   predictions at masked positions against the held-out ground truth.
"""
import math
from argparse import ArgumentParser, Namespace
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from incremental_ad.framework.contracts.dataset import (
    DatasetCapability,
    Segment,
    TimeSeriesDataset,
)
from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset


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


class _ImputationWindowDataset(TorchDataset):
    """Yields (masked_window [W, F], (original [W, F], visible_idx [n_visible])) tuples.

    The mask is deterministic per window index — the same patches are always hidden
    for the same window, so validation numbers are reproducible across evaluations.
    Zeroing out masked patches signals the encoder which tokens are absent.
    """

    def __init__(
        self,
        data: Tensor,
        window_len: int,
        patch_len: int,
        mask_ratio: float,
        stride: int,
    ) -> None:
        self.data = data
        self.window_len = window_len
        self.patch_len = patch_len
        self.n_patches = window_len // patch_len
        self.n_masked = int(self.n_patches * mask_ratio)
        self.n_visible = self.n_patches - self.n_masked
        self.stride = stride

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        start = idx * self.stride
        window = self.data[start : start + self.window_len]  # [W, F]

        # Deterministic mask seeded by window index
        gen = torch.Generator().manual_seed(idx)
        perm = torch.randperm(self.n_patches, generator=gen)
        visible_idx = perm[self.n_masked :].sort().values   # [n_visible]
        mask_idx = perm[: self.n_masked].sort().values      # [n_masked]

        # Zero out masked patches so the encoder cannot see them
        masked_window = window.clone()
        for p in mask_idx.tolist():
            masked_window[p * self.patch_len : (p + 1) * self.patch_len] = 0.0

        return masked_window, (window, visible_idx)


class TestImputationDataset(TimeSeriesDataset):
    """Synthetic multi-variate sine-wave dataset for imputation pipeline tests.

    Each feature is a sine wave with a distinct frequency. Small Gaussian noise
    is added so the task is non-trivial.

    Val / test datasets return (masked_window, (original, visible_idx)) tuples.
    The model encodes only the visible patches and predicts the masked ones;
    the evaluator compares predictions vs. ground truth at masked positions only.

    patch_len must match the model's patch_len so the masking aligns with tokenisation.
    mask_ratio controls how many patches are hidden at eval time (independent of the
    model's training mask_ratio).
    """

    ARG_PREFIX = "dataset"

    def __init__(
        self,
        window_len: int,
        patch_len: int,
        mask_ratio: float = 0.5,
        n_features: int = 8,
        n_train_steps: int = 8000,
        n_test_steps: int = 2000,
        val_fraction: float = 0.1,
        stride: int = 1,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._window_len = window_len
        self._patch_len = patch_len
        self._mask_ratio = mask_ratio
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
    def mask_patch_len(self) -> int:
        return self._patch_len

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
            self._train_data[val_start:], self._window_len, self._patch_len, self._mask_ratio, self._eval_stride
        )

    def get_test_dataset(self) -> _ImputationWindowDataset:
        return _ImputationWindowDataset(
            self._test_data,
            self._window_len,
            self._patch_len,
            self._mask_ratio,
            self._eval_stride,
        )

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_window_len", type=int, required=True)
        parser.add_argument(f"--{p}_patch_len", type=int, required=True,
                            help="Must match mae_tx_patch_len so masking aligns with tokenisation.")
        parser.add_argument(f"--{p}_mask_ratio", type=float, default=0.5,
                            help="Fraction of patches hidden at eval time.")
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
            patch_len=getattr(cfg, f"{p}_patch_len"),
            mask_ratio=getattr(cfg, f"{p}_mask_ratio"),
            n_features=getattr(cfg, f"{p}_n_features"),
            n_train_steps=getattr(cfg, f"{p}_n_train_steps"),
            n_test_steps=getattr(cfg, f"{p}_n_test_steps"),
            val_fraction=getattr(cfg, f"{p}_val_fraction"),
            stride=getattr(cfg, f"{p}_stride"),
            noise=getattr(cfg, f"{p}_noise"),
            seed=getattr(cfg, f"{p}_seed"),
        )
