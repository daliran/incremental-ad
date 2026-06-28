import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from incremental_ad.framework.contracts.configurable import Configurable
from incremental_ad.framework.contracts.dataset import Dataset
from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.framework.contracts.task import Task


@dataclass
class RunContext:
    """Shared execution state passed to every step in a pipeline run."""

    run_id: str
    run_dir: Path
    model: Model
    dataset: Dataset
    task: Task
    configurator: TaskModelConfigurator
    seed: int
    eval_seed: int

    def step_dir(self, step_name: str) -> Path:
        """Return (and create) the output directory for a named step within this run."""
        d = self.run_dir / step_name
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class StepResult:
    """Base result produced by any pipeline step. Concrete steps extend this with their own fields."""

    step_name: str
    started_at: datetime
    finished_at: datetime

    def write(self, out_dir: Path) -> None:
        """Write result.json into out_dir."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(self._to_json_dict(), indent=2))

    def _to_json_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }


class Pipeline(Configurable, ABC):
    ARG_PREFIX = "pipeline"
    _registry: ClassVar[dict[str, type["Pipeline"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            Pipeline._registry[cls.__name__] = cls

    @abstractmethod
    def run(self, context: RunContext) -> list[StepResult]:
        """Execute the ordered sequence of pipeline steps using the shared run context."""
        ...

    @staticmethod
    def _run_debugger(
        evaluator: Evaluator[Any], context: RunContext, model: Model, step_dir: Path
    ) -> None:
        debugger = context.configurator.create_debugger()
        if debugger is not None:
            debugger.run(model, context.dataset, evaluator, step_dir)
