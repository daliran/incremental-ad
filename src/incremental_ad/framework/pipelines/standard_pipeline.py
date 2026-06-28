import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

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
from incremental_ad.framework.trainers.standard_trainer import StandardTrainer

log = logging.getLogger(__name__)


@dataclass
class TrainStepResult(StepResult):
    final_train_loss: float
    best_train_loss: float
    final_val_loss: float | None
    best_val_loss: float | None
    best_epoch: int | None
    epochs_trained: int
    checkpoint_path: Path | None
    secondary_val_losses: dict[str, float] = field(default_factory=dict)

    def _to_json_dict(self) -> dict:
        d = super()._to_json_dict()
        d.update(
            {
                "final_train_loss": self.final_train_loss,
                "best_train_loss": self.best_train_loss,
                "final_val_loss": self.final_val_loss,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
                "epochs_trained": self.epochs_trained,
                "checkpoint_path": (
                    str(self.checkpoint_path) if self.checkpoint_path else None
                ),
                "secondary_val_losses": self.secondary_val_losses,
            }
        )
        return d


@dataclass
class EvalStepResult(StepResult):
    metrics: dict[str, float]
    results_path: Path | None

    def _to_json_dict(self) -> dict:
        d = super()._to_json_dict()
        d.update(
            {
                "metrics": self.metrics,
                "results_path": str(self.results_path) if self.results_path else None,
            }
        )
        return d


class StandardPipeline(Pipeline):

    def __init__(
        self, trainer: StandardTrainer, runner: EvaluationRunner
    ) -> None:
        self.trainer = trainer
        self.runner = runner

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        DataLoaderConfig.add_args(parser)
        StandardTrainer.add_args(parser, prefix="trainer")

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        trainer = StandardTrainer.from_config(cfg, prefix="trainer")
        return cls(
            trainer=trainer,
            runner=EvaluationRunner(DataLoaderConfig.from_config(cfg), device=trainer.device),
        )

    def run(self, context: RunContext) -> list[StepResult]:
        dataset = context.dataset
        model = context.model
        results: list[StepResult] = []
        step_dir = context.step_dir("train")

        # --- Train ---
        segment = dataset.get_train_segment()
        log.info("[train] Starting training")
        started_at = datetime.now(timezone.utc)

        summary = self.trainer.fit(
            model,
            segment,
            checkpoint_dir=step_dir / "checkpoints",
            step_name="train",
        )

        log.info(
            f"[train] Training complete — "
            f"best_val={summary.best_val_loss:.4f} epoch={summary.best_epoch}/{summary.epochs_trained}"
            if summary.best_val_loss is not None
            else f"[train] Training complete — {summary.epochs_trained} epochs"
        )

        train_result = TrainStepResult(
            step_name="train",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            final_train_loss=summary.final_train_loss,
            best_train_loss=summary.best_train_loss,
            final_val_loss=summary.final_val_loss,
            best_val_loss=summary.best_val_loss,
            best_epoch=summary.best_epoch,
            epochs_trained=summary.epochs_trained,
            checkpoint_path=(
                summary.checkpoint_path.relative_to(context.run_dir)
                if summary.checkpoint_path
                else None
            ),
            secondary_val_losses=summary.secondary_val_losses,
        )

        train_result.write(step_dir)
        results.append(train_result)

        caps = dataset.capabilities

        # --- Val eval ---
        if DatasetCapability.VAL in caps:
            evaluator = context.configurator.create_val_evaluator()
            eval_step = "train/val"
            log.info(f"[{eval_step}] Starting evaluation")
            started_at = datetime.now(timezone.utc)

            metrics = self.runner.run(
                model, evaluator, dataset.get_val_eval_dataset(), seed=context.eval_seed
            )

            for k, v in metrics.items():
                log.info(f"  [{eval_step}] {k}: {v:.4f}")
            log.info(f"[{eval_step}] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"{eval_step}/{k}": v for k, v in metrics.items()}
                )

            val_result = EvalStepResult(
                step_name=eval_step,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                metrics=metrics,
                results_path=None,
            )

            val_result.write(step_dir / "val")
            results.append(val_result)

        # --- Test eval ---
        if DatasetCapability.TEST in caps:
            evaluator = context.configurator.create_test_evaluator()

            reference_ds = (
                dataset.get_train_eval_dataset()
                if isinstance(evaluator, ReferenceScoredEvaluator)
                else None
            )

            eval_step = "train/test"
            log.info(f"[{eval_step}] Starting evaluation")
            started_at = datetime.now(timezone.utc)

            metrics = self.runner.run(
                model, evaluator, dataset.get_test_dataset(),
                reference_dataset=reference_ds,
                seed=context.eval_seed,
            )

            for k, v in metrics.items():
                log.info(f"  [{eval_step}] {k}: {v:.4f}")
            log.info(f"[{eval_step}] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"{eval_step}/{k}": v for k, v in metrics.items()}
                )

            if isinstance(evaluator, WandbChartsEvaluator):
                evaluator.log_wandb_charts(eval_step)

            test_result = EvalStepResult(
                step_name=eval_step,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                metrics=metrics,
                results_path=None,
            )

            test_result.write(step_dir / "test")
            results.append(test_result)
            self._run_debugger(evaluator, context, model, step_dir)

        return results
