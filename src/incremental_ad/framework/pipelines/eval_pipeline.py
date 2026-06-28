import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

import torch
import wandb

from incremental_ad.framework.contracts.dataset import (
    DataLoaderConfig,
    DatasetCapability,
)
from incremental_ad.framework.contracts.evaluator import (
    ReferenceScoredEvaluator,
    WandbChartsEvaluator,
)
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
from incremental_ad.framework.pipelines.standard_pipeline import EvalStepResult

log = logging.getLogger(__name__)


class EvalPipeline(Pipeline):
    """Load a checkpoint and run evaluation without any training."""

    def __init__(
        self,
        checkpoint_path: Path,
        runner: EvaluationRunner,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.runner = runner

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        DataLoaderConfig.add_args(parser)
        parser.add_argument(f"--{p}_checkpoint_path", type=str, required=True)
        EvaluationRunner.add_args(parser)

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            checkpoint_path=Path(getattr(cfg, f"{p}_checkpoint_path")),
            runner=EvaluationRunner.from_config(cfg),
        )

    def run(self, context: RunContext) -> list[StepResult]:
        log.info(f"Loading checkpoint from {self.checkpoint_path}")
        
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        
        context.model.load_state_dict(ckpt["model_state_dict"])
        # runner.run() moves the model to device before inference

        results: list[StepResult] = []
        step_dir = context.step_dir("eval")
        caps = context.dataset.capabilities

        if DatasetCapability.TEST in caps:
            evaluator = context.configurator.create_test_evaluator()

            reference_ds = (
                context.dataset.get_train_eval_dataset()
                if isinstance(evaluator, ReferenceScoredEvaluator)
                else None
            )

            log.info("[eval/test] Starting evaluation")
            started_at = datetime.now(timezone.utc)

            metrics = self.runner.run(
                context.model, evaluator, context.dataset.get_test_dataset(),
                reference_dataset=reference_ds,
                seed=context.eval_seed,
            )

            for k, v in metrics.items():
                log.info(f"  [eval/test] {k}: {v:.4f}")
            log.info("[eval/test] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"eval/test/{k}": v for k, v in metrics.items()}
                )

            if isinstance(evaluator, WandbChartsEvaluator):
                evaluator.log_wandb_charts("eval/test")

            result = EvalStepResult(
                step_name="eval/test",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                metrics=metrics,
                results_path=None,
            )

            result.write(step_dir / "test")
            results.append(result)
            
            self._run_debugger(evaluator, context, context.model, step_dir)

        return results
