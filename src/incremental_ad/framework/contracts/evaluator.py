from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

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
