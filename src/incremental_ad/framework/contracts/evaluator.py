from abc import ABC, abstractmethod
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T")


class Evaluator(ABC, Generic[T]):

    @abstractmethod
    def update(self, outputs: T) -> None:
        """Accumulate model outputs for one batch. Called once per batch during an evaluation pass."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute and return the final metrics after all batches have been processed."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state. Called before each new evaluation pass."""
        ...

    def selection_metric(self) -> tuple[str, Literal["min", "max"]] | None:
        """Which of this evaluator's metrics ranks two models, and in which direction.

        ``None`` — the default — means *there is no usable signal here*, not merely that
        no default was picked. Callers that must choose between models are expected to
        refuse rather than fall back on an arbitrary key.

        The distinction is load-bearing for AD: ``AdValEvaluator`` reports reconstruction
        score statistics, but the AD test metrics are rank-based and (under
        ``threshold_strategy=oracle``, which sweeps its own threshold) invariant to any
        monotone rescaling of that score. Minimising val reconstruction error would
        therefore optimise a quantity with no established relationship to the metric
        actually reported, which looks rigorous while being arbitrary.
        """
        return None


@runtime_checkable
class ReferenceEvaluator(Protocol):
    """Optional capability: evaluator that configures its internal state from a reference
    pass — model outputs collected over an auxiliary dataset before the main eval pass."""

    def needs_reference(self) -> bool: ...
    def set_reference(self, outputs: np.ndarray) -> None: ...


@runtime_checkable
class DebugDataEvaluator(Protocol):
    """Optional capability: evaluator that exposes its last computed scores for debug visualization."""

    def debug_data(self) -> tuple[np.ndarray, np.ndarray, float] | None: ...


@runtime_checkable
class WandbChartsEvaluator(Protocol):
    """Optional capability: evaluator that can log native wandb charts."""

    def log_wandb_charts(self, step_name: str) -> None: ...
