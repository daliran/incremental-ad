import inspect
import platform
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from torch.utils.data import DataLoader, Dataset as TorchDataset

from incremental_ad.framework.contracts.configurable import Configurable


class DatasetCapability(Enum):
    TEST = "test"  # a test split exists
    VAL = "val"  # val is carved out inside each segment
    TEST_LABELS = "test_labels"  # test set carries supervision signal (targets, labels, or ground truth)


@dataclass(frozen=True)
class SplitConfig:
    baseline_fraction: float = (
        1.0  # fraction of the whole dataset allocated to the baseline
    )
    baseline_use_fraction: float = (
        1.0  # fraction of the baseline portion actually used for training
    )
    n_finetune_segments: int = (
        0  # number of equal fine-tuning segments from the remainder
    )
    val_fraction: float = (
        0.1  # fraction carved out as val within each segment (0 = no val)
    )

    def validate(self, n: int) -> None:
        """Fail fast on split params that would produce empty/degenerate segments for a
        series of length n. Restores the guards the old splitting.equal_chunks() enforced."""
        if not 0.0 < self.baseline_fraction <= 1.0:
            raise ValueError(
                f"baseline_fraction must be in (0, 1], got {self.baseline_fraction}"
            )
        if not 0.0 < self.baseline_use_fraction <= 1.0:
            raise ValueError(
                f"baseline_use_fraction must be in (0, 1], got {self.baseline_use_fraction}"
            )
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.n_finetune_segments < 0:
            raise ValueError(
                f"n_finetune_segments must be >= 0, got {self.n_finetune_segments}"
            )

        baseline_end = int(n * self.baseline_fraction)
        use_end = int(baseline_end * self.baseline_use_fraction)
        if use_end <= 0:
            raise ValueError(
                f"baseline slice is empty (n={n}, baseline_fraction={self.baseline_fraction}, "
                f"baseline_use_fraction={self.baseline_use_fraction})"
            )

        if self.n_finetune_segments > 0:
            remaining = n - baseline_end
            if remaining <= 0:
                raise ValueError(
                    f"no data left for fine-tuning (n={n}, baseline_fraction={self.baseline_fraction}); "
                    "lower baseline_fraction or set n_finetune_segments=0"
                )
            if remaining // self.n_finetune_segments <= 0:
                raise ValueError(
                    f"n_finetune_segments={self.n_finetune_segments} too large for "
                    f"{remaining} remaining timesteps"
                )


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--loader_batch_size", type=int, required=True)
        parser.add_argument(
            "--loader_num_workers",
            type=int,
            default=0 if platform.system() == "Windows" else 4,
        )

    @classmethod
    def from_config(cls, cfg: Namespace) -> "DataLoaderConfig":
        return cls(
            batch_size=cfg.loader_batch_size,
            num_workers=cfg.loader_num_workers,
        )

    def make_loader(self, dataset: TorchDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )


@dataclass
class Segment:
    """One training phase: a required training dataset and an optional validation dataset.

    Both are consumed by the trainer via model.compute_loss() — val supplies the
    early-stopping/checkpoint-selection loss, so it must be loss-shaped exactly like
    train (same batch format), not metric-shaped.

    Contract:
      train — yields single tensors (unsupervised) or (inputs, labels) tuples (supervised);
              passed to model.compute_loss().
      val   — same shape as train; passed to model.compute_loss() each epoch to compute
              the validation loss. May be None when the segment has no held-out tail.

    Note: this is NOT the dataset used to compute reported metrics. Post-training metric
    evaluation goes through the evaluator (model.predict_step()) on the separate
    get_val_eval_dataset() / get_test_dataset(), which are metric-shaped and yield
    (inputs, target) 2-tuples.
    """

    train: TorchDataset
    val: TorchDataset | None = None


class Dataset(Configurable, ABC):
    ARG_PREFIX = "dataset"
    _registry: ClassVar[dict[str, type["Dataset"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            Dataset._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def capabilities(self) -> set[DatasetCapability]:
        """Declare which optional features this dataset provides (test split, labels, val splits)."""
        ...

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        pass

    def get_train_segment(self) -> "Segment":
        """Return a single Segment spanning all training data (for single-pass training pipelines).
        Raises NotImplementedError for datasets that do not support training."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_train_segment()"
        )

    def get_train_eval_dataset(self) -> TorchDataset:
        """Return the full training series windowed with eval_stride for reference scoring.

        Used by pipelines to configure reference-based evaluators (ReferenceEvaluator).
        Must use eval_stride (not the training stride) so every timestep is covered.
        Yields single tensors — no labels, no targets.
        Raises NotImplementedError for datasets that do not support training."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_train_eval_dataset()"
        )

    def get_val_eval_dataset(self) -> TorchDataset:
        """Return the validation split windowed with eval_stride for post-training val eval.

        Mirrors the val time-range used in get_train_segment() but uses eval_stride=1
        so every timestep is covered. Only called when DatasetCapability.VAL is declared.
        Raises NotImplementedError for datasets that do not declare VAL capability."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_val_eval_dataset()"
        )

    def get_test_dataset(self) -> TorchDataset:
        """Return the test dataset. Only valid when TEST is in capabilities.

        Contract: must yield (inputs, target) 2-tuples so that model.predict_step(batch)
        can unpack inputs and pass the target through to the evaluator. The concrete shapes
        of inputs and target are task-specific and defined by the dataset/configurator pair.
        """
        raise NotImplementedError

    def analyze(self, analysis_dir: Path) -> None:
        """Write one-time dataset analysis artifacts (plots, CSVs) to analysis_dir. No-op by default."""


class PartitionedDataset(Dataset, ABC):
    """Dataset split into a baseline partition and N fine-tuning partitions."""

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_baseline_fraction", type=float, default=1.0)
        parser.add_argument(f"--{p}_baseline_use_fraction", type=float, default=1.0)
        parser.add_argument(f"--{p}_n_finetune_segments", type=int, default=0)
        parser.add_argument(f"--{p}_val_fraction", type=float, default=0.1)

    @classmethod
    def _split_config_from_cfg(
        cls, cfg: Namespace, prefix: str | None = None
    ) -> SplitConfig:
        """Extract a SplitConfig from the parsed namespace. Call this inside from_config."""
        p = prefix or cls.ARG_PREFIX
        return SplitConfig(
            baseline_fraction=getattr(cfg, f"{p}_baseline_fraction"),
            baseline_use_fraction=getattr(cfg, f"{p}_baseline_use_fraction"),
            n_finetune_segments=getattr(cfg, f"{p}_n_finetune_segments"),
            val_fraction=getattr(cfg, f"{p}_val_fraction"),
        )

    @abstractmethod
    def get_baseline(self) -> Segment: ...

    @abstractmethod
    def get_incremental_segments(self) -> list[Segment]:
        """Return n_finetune equal segments from the non-baseline remainder. Empty if n_finetune=0."""
        ...

    def get_baseline_val_eval_dataset(self) -> TorchDataset:
        """Eval-stride windowing over the baseline segment's held-out val slice.

        Used by the incremental pipeline for baseline/val, so the baseline is scored on
        its own validation data rather than the global tail. Raises NotImplementedError
        for datasets that do not support it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_baseline_val_eval_dataset()"
        )

    def get_incremental_val_eval_dataset(self) -> TorchDataset:
        """Eval-stride windowing over the union of every segment's held-out val slice
        (baseline + each finetune).

        Used by the incremental pipeline for merged/val, so the merged model is scored
        across all regimes and never on training data. Raises NotImplementedError for
        datasets that do not support it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_incremental_val_eval_dataset()"
        )


class TimeSeriesDataset(Dataset, ABC):
    """Intermediate contract for sliding-window time-series datasets with a fixed feature count."""

    @property
    @abstractmethod
    def n_features(self) -> int: ...

    @property
    @abstractmethod
    def window_len(self) -> int: ...


@runtime_checkable
class ImputationDataset(Protocol):
    """Structural protocol for datasets that declare the patch length used during fixed-mask eval.

    The patch_len must equal the model's patch_len so that masked token positions
    align with the model's tokenisation. Use isinstance(dataset, ImputationDataset)
    in the configurator to verify presence and then validate alignment.
    """

    @property
    def mask_patch_len(self) -> int: ...


@runtime_checkable
class ForecastDataset(Protocol):
    """Structural protocol for datasets that expose a forecast horizon.

    Use isinstance(dataset, ForecastDataset) to verify presence of forecast_len
    before accessing it — works at runtime and provides type-checker support.
    """

    @property
    def forecast_len(self) -> int: ...


@runtime_checkable
class ClassificationDataset(Protocol):
    """Structural protocol for datasets that expose a fixed number of target classes.

    Use isinstance(dataset, ClassificationDataset) to verify presence of n_classes
    before accessing it — works at runtime and provides type-checker support.
    """

    @property
    def n_classes(self) -> int: ...
