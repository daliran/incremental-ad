import inspect
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, ClassVar, Self

import torch
from torch import Tensor

from incremental_ad.framework.contracts.configurable import Configurable
from incremental_ad.framework.contracts.dataset import Dataset
from incremental_ad.framework.contracts.debugger import Debugger
from incremental_ad.framework.contracts.evaluator import Evaluator


def _task_key(task: object) -> str:
    """Normalize a task identifier to its string key. Accepts a plain string or any
    enum-like object exposing .value (e.g. the project's Task enum), so the framework
    never depends on a concrete task taxonomy."""
    return getattr(task, "value", task)


class Model(Configurable, torch.nn.Module, ABC):
    ARG_PREFIX = "model"
    _registry: ClassVar[dict[str, type["Model"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            Model._registry[cls.__name__] = cls

    @abstractmethod
    def _build(self) -> None:
        """Construct all nn.Module layers. Always called by TaskModelConfigurator.configure()."""
        ...

    @abstractmethod
    def compute_loss(self, batch: Any) -> Tensor:
        """Compute and return the scalar training loss for one batch.
        batch is whatever the training DataLoader returns: a Tensor for self-supervised
        tasks, or a (inputs, targets) tuple for supervised tasks (forecasting, etc.).
        """
        ...

    @abstractmethod
    def score(self, inputs: Tensor) -> Tensor:
        """Pure inference: map input window(s) to the model's task output.

        The return shape is task-specific (e.g. per-window scores, class logits, or step
        predictions) and is interpreted by the matching evaluator.
        """
        ...

    def predict_step(self, batch: tuple[Tensor, Any]) -> tuple[Any, Any]:
        """Eval wrapper: unpack (inputs, target), score inputs, return (output, target).

        The target element travels unchanged to the evaluator — it carries whatever ground
        truth the task's evaluator needs (labels, targets, reconstruction references, ...).
        Override when the model needs the target to compute its output.
        """
        inputs, target = batch
        return self.score(inputs), target


class TaskModelConfigurator(Configurable, ABC):
    ARG_PREFIX = "configurator"
    # Keyed by (task_name, model_cls). task_name is a plain string so the framework
    # stays agnostic of the concrete task taxonomy (which lives in project/).
    _registry: ClassVar[dict[tuple[str, type], type["TaskModelConfigurator"]]] = {}

    # Default no-op implementations of the Configurable contract.
    # Configurators with CLI args override both; argument-free ones inherit these.
    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        pass

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        return cls()

    @classmethod
    def register(cls, task: object, model_cls: type):
        """Decorator that registers a configurator for a specific (task, Model) pair.
        task may be a string or an enum exposing .value (e.g. the project's Task)."""

        def decorator(configurator_cls: type) -> type:
            cls._registry[(_task_key(task), model_cls)] = configurator_cls
            return configurator_cls

        return decorator

    @classmethod
    def get(cls, task: object, model: object) -> type["TaskModelConfigurator"]:
        """Retrieve the configurator registered for the given task and model. Raises KeyError if incompatible."""
        return cls._registry[(_task_key(task), type(model))]

    @classmethod
    def lookup(cls, task: object, model_cls: type) -> type["TaskModelConfigurator"] | None:
        """Return the configurator for (task, model_cls), or None if none is registered."""
        return cls._registry.get((_task_key(task), model_cls))

    @classmethod
    def registered_tasks(cls) -> list[str]:
        """Sorted unique task names that have at least one registered configurator.
        Used to populate the --task CLI choices without hardcoding a taxonomy."""
        return sorted({task for task, _ in cls._registry})

    def configure(self, model: Model, dataset: Dataset) -> None:
        """Inject dataset-derived properties then build the model. Not overridable — customise
        _configure, which is also where each configurator asserts dataset compatibility
        (structural protocols and required capabilities)."""
        self._configure(model, dataset)
        model._build()

    def _configure(self, model: Model, dataset: Dataset) -> None:
        """Override to validate dataset compatibility (isinstance protocol checks and
        DatasetCapability requirements) and inject dataset-derived values into model config
        before _build() is called."""

    @abstractmethod
    def create_val_evaluator(self) -> Evaluator:
        """Instantiate the evaluator used during training validation.
        The pipeline calls evaluator.update(model.predict_step(batch)) for each val batch.
        """
        ...

    @abstractmethod
    def create_test_evaluator(self) -> Evaluator:
        """Instantiate the evaluator used for held-out test evaluation.
        The pipeline calls evaluator.update(model.predict_step(batch)) for each test batch.
        """
        ...

    def create_debugger(self) -> Debugger | None:
        """Return a debugger for post-eval visualization, or None to skip."""
        return None
