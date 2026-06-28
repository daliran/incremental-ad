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

    Contract:
      train — yields single tensors (unsupervised) or (inputs, labels) tuples (supervised);
              consumed by model.compute_loss().
      val   — yields (inputs, target) 2-tuples; consumed by model.predict_step().
              target is passed through to the evaluator unchanged (reconstruction
              target, forecast target, or a cheap placeholder like zeros for AD val).
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

        Used by pipelines to fit score-based thresholds (ReferenceScoredEvaluator).
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

        Contract: must yield (inputs, target) 2-tuples so that
        model.predict_step(batch) can unpack inputs and pass through targets.

        Task-specific formats:
          AD            — (window [W,F], anomaly_labels [W])
          Forecasting   — (full_window [W,F], future [H,F])
          Imputation    — (masked_window [W,F], (original [W,F], visible_idx [n_visible]))
          Classification — (window [W,F], label [])
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
