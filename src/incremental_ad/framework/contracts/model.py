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
from incremental_ad.framework.contracts.task import Task


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
        """Pure inference: map input window(s) to task output.

        Return shape depends on task:
          AD             → [B] scalar anomaly scores
          Classification → [B, n_classes] logits
          Forecasting    → [B, H, F] predictions
        """
        ...

    def predict_step(self, batch: tuple[Tensor, Any]) -> tuple[Any, Any]:
        """Eval wrapper: unpack (inputs, target), score inputs, return (scores, target).

        The target element travels unchanged to the evaluator — it carries anomaly labels
        for AD, forecast targets for forecasting, or (original, visible_idx) for imputation.
        Override when the model needs target to compute outputs (e.g. imputation).
        """
        inputs, target = batch
        return self.score(inputs), target


class TaskModelConfigurator(Configurable, ABC):
    ARG_PREFIX = "configurator"
    _registry: ClassVar[dict[tuple[Task, type], type["TaskModelConfigurator"]]] = {}

    # Default no-op implementations of the Configurable contract.
    # Configurators with CLI args (e.g. MaeTxAdConfigurator) override both.
    # Configurators with no args (e.g. MaeTxImputationConfigurator) inherit these.
    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        pass

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        return cls()

    @classmethod
    def register(cls, task: Task, model_cls: type):
        """Decorator that registers a configurator for a specific (Task, Model) pair."""

        def decorator(configurator_cls: type) -> type:
            cls._registry[(task, model_cls)] = configurator_cls
            return configurator_cls

        return decorator

    @classmethod
    def get(cls, task: Task, model: object) -> type["TaskModelConfigurator"]:
        """Retrieve the configurator registered for the given task and model. Raises KeyError if incompatible."""
        return cls._registry[(task, type(model))]

    def configure(self, model: Model, dataset: Dataset) -> None:
        """Inject dataset-derived properties then build the model. Not overridable — customise _configure."""
        self._configure(model, dataset)
        model._build()

    def _configure(self, model: Model, dataset: Dataset) -> None:
        """Override to inject dataset-derived values into model config before _build() is called."""

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
