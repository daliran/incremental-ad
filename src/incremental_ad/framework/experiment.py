import json
import logging
import os
import platform
import socket
import uuid
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import wandb

from incremental_ad.framework.contracts.dataset import Dataset
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.contracts.task import Task
from incremental_ad.framework.core.git import git_commit
from incremental_ad.framework.core.seed import set_seed
from incremental_ad.framework.core.tracking import init_wandb

log = logging.getLogger(__name__)


def _cfg_to_dict(cfg: Namespace) -> dict:
    """Serialize an argparse Namespace to a JSON-safe dict."""
    result = {}
    for k, v in vars(cfg).items():
        if isinstance(v, (int, float, bool, str, type(None))):
            result[k] = v
        elif isinstance(v, Enum):
            result[k] = v.value
        elif isinstance(v, list):
            result[k] = [
                i if isinstance(i, (int, float, bool, str, type(None))) else str(i)
                for i in v
            ]
        else:
            result[k] = str(v)
    return result


def _write_config(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(fields, indent=2), encoding="utf-8")


class Experiment:
    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        task: Task,
        pipeline: Pipeline,
        configurator: TaskModelConfigurator,
        seed: int,
        eval_seed: int,
        experiment_name: str = "default",
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.task = task
        self.pipeline = pipeline
        self.configurator = configurator
        self.seed = seed
        self.eval_seed = eval_seed
        self.experiment_name = experiment_name
        self._cfg: Namespace | None = None
        self.results: list[StepResult] = []

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--model", required=True)
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--task", required=True, choices=[t.value for t in Task])
        parser.add_argument("--pipeline", required=True)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--eval_seed", type=int, default=None)
        parser.add_argument("--experiment_name", required=True)

    @classmethod
    def from_config(cls, cfg: Namespace) -> "Experiment":
        task = Task(cfg.task)
        model_cls = Model._registry[cfg.model]
        
        configurator_cls = TaskModelConfigurator._registry.get((task, model_cls))

        if configurator_cls is None:
            raise RuntimeError(
                f"No configurator registered for (task={cfg.task}, model={cfg.model})."
            )
        
        model = model_cls.from_config(cfg)
        dataset = Dataset._registry[cfg.dataset].from_config(cfg)
        pipeline = Pipeline._registry[cfg.pipeline].from_config(cfg)
        experiment = cls(
            model=model,
            dataset=dataset,
            task=task,
            pipeline=pipeline,
            configurator=configurator_cls.from_config(cfg),
            seed=cfg.seed,
            eval_seed=cfg.eval_seed if cfg.eval_seed is not None else cfg.seed + 1,
            experiment_name=cfg.experiment_name,
        )
        experiment._cfg = cfg
        return experiment

    def run(self) -> None:
        set_seed(self.seed)

        slurm_id = os.environ.get("SLURM_JOB_ID")
        run_id = (
            slurm_id
            or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        runs_root = Path(os.environ.get("RUNS_ROOT", "runs"))
        run_dir = runs_root / self.experiment_name / run_id

        args = _cfg_to_dict(self._cfg) if self._cfg is not None else {}
        commit = git_commit()

        _write_config(
            run_dir,
            experiment_name=self.experiment_name,
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            python_version=platform.python_version(),
            git_commit=commit,
            seed=self.seed,
            eval_seed=self.eval_seed,
            model=type(self.model).__name__,
            dataset=type(self.dataset).__name__,
            task=self.task.value,
            pipeline=type(self.pipeline).__name__,
            args=args,
        )

        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        log.info(f"Run dir: {run_dir}")
        log.info(f"Experiment: {self.experiment_name}  run_id: {run_id}")
        log.info(
            f"Model: {type(self.model).__name__}  Dataset: {type(self.dataset).__name__}"
        )

        os.environ["WANDB_DIR"] = str(run_dir)

        init_wandb(
            experiment_name=self.experiment_name,
            run_id=run_id,
            config=args | {"git_commit": commit},
        )

        # configure() calls model._build()
        self.configurator.configure(self.model, self.dataset)

        analysis_dir = Path("analysis") / type(self.dataset).__name__
        self.dataset.analyze(analysis_dir)

        context = RunContext(
            run_id=run_id,
            run_dir=run_dir,
            model=self.model,
            dataset=self.dataset,
            task=self.task,
            configurator=self.configurator,
            seed=self.seed,
            eval_seed=self.eval_seed,
        )

        try:
            self.results = self.pipeline.run(context)
        finally:
            wandb.finish()
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
