import csv
import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from typing import Self

import wandb

from incremental_ad.framework.contracts.dataset import (
    DataLoaderConfig,
    DatasetCapability,
    PartitionedDataset,
)
from incremental_ad.framework.contracts.evaluator import ReferenceEvaluator
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
from incremental_ad.framework.pipelines.standard_pipeline import (
    EvalStepResult,
    TrainStepResult,
)
from incremental_ad.framework.trainers.standard_trainer import StandardTrainer

log = logging.getLogger(__name__)


class ContinualFineTuningPipeline(Pipeline):
    """Naive sequential fine-tuning — the continual-learning baseline for task arithmetic.

    The contrast with ``IncrementalTaskArithmeticPipeline`` is one line of code and the
    whole point of this pipeline: there, every segment is fine-tuned **from the frozen
    baseline**, which is what makes the task vectors tau_i = theta_i - theta_0 comparable
    and summable. Here the chain is sequential:

        theta_0 --seg 0--> theta_1 --seg 1--> theta_2 --seg 2--> theta_3

    each step starting from the *previous* model. That is what a practitioner does when new
    data arrives, and it is the thing task arithmetic has to beat. Note the two are not
    variants of one method: sequential models share no common base, so task arithmetic does
    not apply to them at all.

    **What it measures.** After finishing segment k the model is evaluated on *every*
    segment's held-out slice, producing the standard backward-transfer matrix

        a[k][i] = performance of the model after segment k, on segment i

    from which two scalars follow, both reported over the shard columns (``val_base`` is
    reported separately as it is the regime nobody fine-tunes on):

    - **ACC**  = mean_i a[n][i]  — final average performance across all regimes
    - **BWT**  = mean_{i<n} (a[n][i] - a[i][i])  — forgetting. Negative for a loss-shaped
      metric means later training *improved* earlier segments; positive means it hurt them.

    Because the val metrics here are losses (reconstruction error for AD, MSE for
    forecasting), **lower a[k][i] is better** and the sign of BWT reads accordingly. The
    written CSV carries raw values and the ratio to the baseline model, matching the
    convention of ``MergeDiagnosticsPipeline`` so the two are directly comparable.

    **L2-SP variant.** Set ``--finetune_trainer_reg_lambda > 0`` to anchor each sequential
    step to the *original* baseline theta_0 rather than to its immediate predecessor. That
    is a different use of the regulariser than in the task-arithmetic pipeline: there it
    bounds the magnitude of an independent task vector, here it constrains cumulative drift.
    Expect a stability/plasticity trade-off; the question is whether any lambda matches the
    merged model.
    """

    def __init__(
        self,
        trainer: StandardTrainer,
        finetune_trainer: StandardTrainer,
        runner: EvaluationRunner,
        anchor_to_baseline: bool = True,
    ) -> None:
        self.trainer = trainer
        self.finetune_trainer = finetune_trainer
        self.runner = runner
        self.anchor_to_baseline = anchor_to_baseline

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        DataLoaderConfig.add_args(parser)
        EvaluationRunner.add_args(parser)
        StandardTrainer.add_args(parser, prefix="trainer")
        StandardTrainer.add_args(parser, prefix="finetune_trainer")
        parser.add_argument(
            f"--{p}_anchor_to_baseline",
            type=_str_to_bool,
            default=True,
            help="when finetune_trainer_reg_lambda > 0, anchor the L2-SP penalty to the "
            "original baseline theta_0 (true) or to the immediately preceding model "
            "(false). Inert when reg_lambda = 0. True is the standard L2-SP formulation "
            "for continual learning: it bounds cumulative drift from the starting point "
            "rather than per-step drift.",
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        trainer = StandardTrainer.from_config(cfg, prefix="trainer")
        return cls(
            trainer=trainer,
            finetune_trainer=StandardTrainer.from_config(cfg, prefix="finetune_trainer"),
            runner=EvaluationRunner(DataLoaderConfig.from_config(cfg), device=trainer.device),
            anchor_to_baseline=getattr(cfg, f"{p}_anchor_to_baseline"),
        )

    # ── Evaluation helpers ────────────────────────────────────────────────────

    def _eval_columns(self, context: RunContext, n_segments: int) -> list[tuple[str, object]]:
        """Every held-out region, in report order: the base regime then each shard."""
        dataset = context.dataset
        assert isinstance(dataset, PartitionedDataset)
        if DatasetCapability.VAL not in dataset.capabilities:
            return []
        return [("val_base", dataset.get_baseline_val_eval_dataset())] + [
            (f"val_{i}", dataset.get_finetune_val_eval_dataset(i)) for i in range(n_segments)
        ]

    def _eval_test(self, context: RunContext) -> dict[str, float]:
        evaluator = context.configurator.create_test_evaluator()
        reference_dataset = (
            context.dataset.get_train_eval_dataset()
            if isinstance(evaluator, ReferenceEvaluator)
            else None
        )
        return self.runner.run(
            context.model,
            evaluator,
            context.dataset.get_test_dataset(),
            reference_dataset=reference_dataset,
            seed=context.eval_seed,
        )

    def _eval_all_columns(
        self, context: RunContext, columns: list[tuple[str, object]]
    ) -> dict[str, dict[str, float]]:
        evaluator_factory = context.configurator.create_val_evaluator
        return {
            name: self.runner.run(
                context.model, evaluator_factory(), dataset, seed=context.eval_seed
            )
            for name, dataset in columns
        }

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, context: RunContext) -> list[StepResult]:
        assert isinstance(context.dataset, PartitionedDataset), (
            f"{type(self).__name__} requires PartitionedDataset, "
            f"got {type(context.dataset).__name__}"
        )
        dataset = context.dataset
        model = context.model
        results: list[StepResult] = []

        segments = dataset.get_incremental_segments()
        if not segments:
            raise ValueError(
                "ContinualFineTuningPipeline needs at least one finetune segment; got 0. "
                "Set --dataset_n_finetune_segments > 0, or use StandardPipeline."
            )
        columns = self._eval_columns(context, len(segments))

        # step index -> {column -> {metric -> value}}; step 0 is the baseline itself.
        matrix: dict[int, dict[str, dict[str, float]]] = {}
        test_by_step: dict[int, dict[str, float]] = {}

        # --- Step 0: the baseline ---
        step_dir = context.step_dir("baseline")
        log.info("[baseline] Starting training")
        started_at = datetime.now(timezone.utc)
        global_step = 0

        summary = self.trainer.fit(
            model,
            dataset.get_baseline(),
            checkpoint_dir=step_dir / "checkpoints",
            step_name="baseline",
            step_offset=global_step,
        )
        global_step += summary.epochs_trained
        results.append(
            TrainStepResult(
                step_name="baseline",
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
            )
        )
        results[-1].write(step_dir)

        # theta_0 kept on CPU: it is the L2-SP anchor and the ratio reference for the matrix.
        # clone() is load-bearing: Tensor.cpu() returns *self* for a tensor already on CPU,
        # so without it the anchor aliases the live parameters and follows the model as it
        # is fine-tuned — making theta_0 track theta_t and the L2-SP penalty vanish.
        baseline_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        eval_started = datetime.now(timezone.utc)
        matrix[0] = self._eval_all_columns(context, columns)
        for name, metrics in matrix[0].items():
            log.info("  [baseline] %-9s %s", name,
                     "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if DatasetCapability.TEST in dataset.capabilities:
            test_by_step[0] = self._eval_test(context)
            results.append(
                EvalStepResult(
                    step_name="baseline/test",
                    started_at=eval_started,
                    finished_at=datetime.now(timezone.utc),
                    metrics=test_by_step[0],
                )
            )
            results[-1].write(step_dir / "test")

        # --- Sequential chain: each step continues from the previous model ---
        for index, segment in enumerate(segments):
            name = f"continual_{index}"
            step_dir = context.step_dir(name)
            log.info("[%s] Fine-tuning from the %s model", name,
                     "baseline" if index == 0 else f"continual_{index - 1}")
            started_at = datetime.now(timezone.utc)

            # The anchor is theta_0 by default (bounds cumulative drift), or the immediately
            # preceding model. Inert unless finetune_trainer.reg_lambda > 0.
            reference = (
                baseline_state
                if self.anchor_to_baseline
                else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

            summary = self.finetune_trainer.fit(
                model,
                segment,
                checkpoint_dir=step_dir / "checkpoints",
                step_name=name,
                step_offset=global_step,
                reference_state=reference,
            )
            global_step += summary.epochs_trained
            results.append(
                TrainStepResult(
                    step_name=name,
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
                )
            )
            results[-1].write(step_dir)

            # The whole point: score this step's model on EVERY region, not just its own.
            eval_started = datetime.now(timezone.utc)
            matrix[index + 1] = self._eval_all_columns(context, columns)
            for column, metrics in matrix[index + 1].items():
                log.info("  [%s] %-9s %s", name, column,
                         "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if DatasetCapability.TEST in dataset.capabilities:
                test_by_step[index + 1] = self._eval_test(context)
                results.append(
                    EvalStepResult(
                        step_name=f"{name}/test",
                        started_at=eval_started,
                        finished_at=datetime.now(timezone.utc),
                        metrics=test_by_step[index + 1],
                    )
                )
                results[-1].write(step_dir / "test")

        summary_started = datetime.now(timezone.utc)
        summary_metrics = self._write_outputs(
            context, matrix, test_by_step, columns, baseline_state, len(segments)
        )
        final = EvalStepResult(
            step_name="continual_summary",
            started_at=summary_started,
            finished_at=datetime.now(timezone.utc),
            metrics=summary_metrics,
        )
        final.write(context.step_dir("continual_summary"))
        results.append(final)
        for key, value in sorted(summary_metrics.items()):
            log.info("  [continual_summary] %s: %.4f", key, value)
        if wandb.run is not None:
            wandb.run.summary.update(
                {f"continual_summary/{k}": v for k, v in summary_metrics.items()}
            )
        return results

    # ── Output ────────────────────────────────────────────────────────────────

    def _write_outputs(
        self, context, matrix, test_by_step, columns, baseline_state, n_segments
    ) -> dict[str, float]:
        step_dir = context.step_dir("continual_summary")
        column_names = [name for name, _ in columns]
        shard_columns = [c for c in column_names if c != "val_base"]
        metrics = sorted({m for step in matrix.values() for col in step.values() for m in col})

        path = step_dir / "backward_transfer.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["after_step", "step_name", "column", "metric", "value", "ratio_to_baseline"]
            )
            for step in sorted(matrix):
                label = "baseline" if step == 0 else f"continual_{step - 1}"
                for column in column_names:
                    for metric in metrics:
                        value = matrix[step].get(column, {}).get(metric)
                        if value is None:
                            continue
                        anchor = matrix[0].get(column, {}).get(metric)
                        ratio = value / anchor if anchor else ""
                        writer.writerow([step, label, column, metric, value, ratio])
        log.info("[continual] Wrote %s", path)

        if test_by_step:
            test_path = step_dir / "test_by_step.csv"
            with test_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["after_step", "step_name", "metric", "value"])
                for step in sorted(test_by_step):
                    label = "baseline" if step == 0 else f"continual_{step - 1}"
                    for metric, value in sorted(test_by_step[step].items()):
                        writer.writerow([step, label, metric, value])
            log.info("[continual] Wrote %s", test_path)

        # ACC and BWT, per val metric. a[k][i] indexes shard columns, so the model after
        # segment k is step k+1 and shard i's "own step" is i+1.
        summary: dict[str, float] = {}
        last = n_segments
        for metric in metrics:
            def cell(step: int, column: str):
                return matrix.get(step, {}).get(column, {}).get(metric)

            finals = [cell(last, c) for c in shard_columns]
            if not all(v is not None for v in finals) or not finals:
                continue
            summary[f"{metric}/ACC"] = sum(finals) / len(finals)

            diffs = [
                cell(last, c) - cell(i + 1, c)
                for i, c in enumerate(shard_columns[:-1])
                if cell(last, c) is not None and cell(i + 1, c) is not None
            ]
            if diffs:
                # Loss-shaped metrics: positive BWT = later training made earlier shards worse.
                summary[f"{metric}/BWT"] = sum(diffs) / len(diffs)

            base_anchor = cell(0, "val_base")
            base_final = cell(last, "val_base")
            if base_anchor and base_final is not None:
                summary[f"{metric}/base_slice_ratio_final"] = base_final / base_anchor

            anchors = [cell(0, c) for c in shard_columns]
            if all(a for a in anchors):
                summary[f"{metric}/ACC_ratio_to_baseline"] = sum(
                    f / a for f, a in zip(finals, anchors)
                ) / len(finals)
        return summary


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")
