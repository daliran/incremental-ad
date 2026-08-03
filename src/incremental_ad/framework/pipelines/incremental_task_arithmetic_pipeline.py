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
from incremental_ad.framework.contracts.evaluator import (
    ReferenceEvaluator,
    WandbChartsEvaluator,
)
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.core.checkpoints import save_model_state
from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
from incremental_ad.framework.merging.task_vectors import (
    apply_task_vectors,
    merge_task_arithmetic,
    task_vector,
)
from incremental_ad.framework.pipelines.standard_pipeline import (
    EvalStepResult,
    TrainStepResult,
)
from incremental_ad.framework.trainers.standard_trainer import StandardTrainer

log = logging.getLogger(__name__)


class IncrementalTaskArithmeticPipeline(Pipeline):
    """
    1. Train on baseline  (uses trainer, registered under --trainer_*), then evaluate it
       (val on its own held-out slice, test on the global test set)
    2. For each finetune segment, reset to baseline weights, fine-tune
       (uses finetune_trainer, registered under --finetune_trainer_*), then evaluate it
       the same way (val on its own segment's held-out slice, test on the global test set)
    3. Merge via task arithmetic: theta = theta_base + scale * sum_i(theta_ft_i - theta_base)
    4. Evaluate the merged model (val on the union of every segment's held-out slice, test
       on the global test set)
    5. Optionally evaluate the same merge at further scales (extra_merge_scales), which
       needs no extra training — the task vectors are already in hand

    Two scale settings, deliberately distinct. ``merge_scale`` is the *primary* scale:
    the one this run commits to, producing the merged checkpoint and the merged/ results,
    and the one swept as a grid axis across trials. ``extra_merge_scales`` are *secondary*:
    evaluated but never materialised, purely to trace the scale-vs-metric curve.
    """

    def __init__(
        self,
        trainer: StandardTrainer,
        finetune_trainer: StandardTrainer,
        runner: EvaluationRunner,
        merge_scale: float = 1.0,
        extra_merge_scales: list[float] | None = None,
    ) -> None:
        self.trainer = trainer
        self.finetune_trainer = finetune_trainer
        self.runner = runner
        self.merge_scale = merge_scale
        self.extra_merge_scales = extra_merge_scales or []

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        DataLoaderConfig.add_args(parser)
        parser.add_argument(
            f"--{p}_merge_scale",
            type=float,
            default=1.0,
            help="the primary merge scale: produces merged/checkpoints and merged/ results.",
        )
        parser.add_argument(
            f"--{p}_extra_merge_scales",
            type=float,
            nargs="*",
            default=[],
            help="further scales to evaluate alongside the primary one. Evaluation only: "
            "no training, no checkpoints, and merged/ still comes from --{p}_merge_scale. "
            "Writes merged/merge_scale_curve.csv covering the primary scale and these. "
            "Because every point shares one set of checkpoints, the curve carries none of "
            "the run-to-run training noise a curve assembled from separate runs would."
            .replace("{p}", p),
        )
        StandardTrainer.add_args(parser, prefix="trainer")
        StandardTrainer.add_args(parser, prefix="finetune_trainer")

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        trainer = StandardTrainer.from_config(cfg, prefix="trainer")
        return cls(
            trainer=trainer,
            finetune_trainer=StandardTrainer.from_config(cfg, prefix="finetune_trainer"),
            runner=EvaluationRunner(DataLoaderConfig.from_config(cfg), device=trainer.device),
            merge_scale=getattr(cfg, f"{p}_merge_scale"),
            extra_merge_scales=getattr(cfg, f"{p}_extra_merge_scales"),
        )

    def run(self, context: RunContext) -> list[StepResult]:
        assert isinstance(
            context.dataset, PartitionedDataset
        ), f"{type(self).__name__} requires PartitionedDataset, got {type(context.dataset).__name__}"
        
        dataset = context.dataset
        results: list[StepResult] = []
        model = context.model

        incremental_segments = dataset.get_incremental_segments()

        if not incremental_segments:
            log.warning(
                "IncrementalTaskArithmeticPipeline: no finetune segments "
                "(n_finetune_segments=0). Running baseline-only — consider StandardPipeline instead."
            )

        secondary_loaders = {
            f"inc_{i}": self.runner.loader_config.make_loader(s.val, shuffle=False)
            for i, s in enumerate(incremental_segments)
            if s.val is not None
        } or None

        # --- Baseline ---
        baseline = dataset.get_baseline()
        baseline_step_dir = context.step_dir("baseline")

        log.info("[baseline] Starting training")
        started_at = datetime.now(timezone.utc)
        global_step = 0

        baseline_summary = self.trainer.fit(
            model,
            baseline,
            checkpoint_dir=baseline_step_dir / "checkpoints",
            secondary_loaders=secondary_loaders,
            step_name="baseline",
            step_offset=global_step,
        )

        global_step += baseline_summary.epochs_trained

        log.info(
            f"[baseline] Training complete — "
            f"best_val={baseline_summary.best_val_loss:.4f} "
            f"epoch={baseline_summary.best_epoch}/{baseline_summary.epochs_trained}"
            if baseline_summary.best_val_loss is not None
            else f"[baseline] Training complete — {baseline_summary.epochs_trained} epochs"
        )

        results.append(
            TrainStepResult(
                step_name="baseline",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                final_train_loss=baseline_summary.final_train_loss,
                best_train_loss=baseline_summary.best_train_loss,
                final_val_loss=baseline_summary.final_val_loss,
                best_val_loss=baseline_summary.best_val_loss,
                best_epoch=baseline_summary.best_epoch,
                epochs_trained=baseline_summary.epochs_trained,
                checkpoint_path=(
                    baseline_summary.checkpoint_path.relative_to(context.run_dir)
                    if baseline_summary.checkpoint_path
                    else None
                ),
                secondary_val_losses=baseline_summary.secondary_val_losses,
            )
        )

        results[-1].write(baseline_step_dir)

        # Store baseline state on CPU — avoids holding (N+1) model copies on GPU during the finetune loop.
        baseline_state = {k: v.cpu() for k, v in model.state_dict().items()}
        caps = context.dataset.capabilities

        # --- Baseline val eval ---
        if DatasetCapability.VAL in caps:
            val_evaluator = context.configurator.create_val_evaluator()
            baseline_val_step = "baseline/val"
            log.info(f"[{baseline_val_step}] Starting evaluation")
            started_at_eval = datetime.now(timezone.utc)

            baseline_val_metrics = self.runner.run(
                model, val_evaluator, dataset.get_baseline_val_eval_dataset(),
                seed=context.eval_seed,
            )

            for k, v in baseline_val_metrics.items():
                log.info(f"  [{baseline_val_step}] {k}: {v:.4f}")
            log.info(f"[{baseline_val_step}] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"{baseline_val_step}/{k}": v for k, v in baseline_val_metrics.items()}
                )

            baseline_val_result = EvalStepResult(
                step_name=baseline_val_step,
                started_at=started_at_eval,
                finished_at=datetime.now(timezone.utc),
                metrics=baseline_val_metrics,
            )

            baseline_val_result.write(baseline_step_dir / "val")
            results.append(baseline_val_result)

        # --- Baseline test eval (pre-fine-tuning reference) ---
        if DatasetCapability.TEST in caps:
            evaluator = context.configurator.create_test_evaluator()

            reference_ds = (
                dataset.get_train_eval_dataset()
                if isinstance(evaluator, ReferenceEvaluator)
                else None
            )

            baseline_test_step = "baseline/test"
            log.info(f"[{baseline_test_step}] Starting evaluation")
            started_at_eval = datetime.now(timezone.utc)

            metrics = self.runner.run(
                model, evaluator, context.dataset.get_test_dataset(),
                reference_dataset=reference_ds,
                seed=context.eval_seed,
            )

            for k, v in metrics.items():
                log.info(f"  [{baseline_test_step}] {k}: {v:.4f}")
            log.info(f"[{baseline_test_step}] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"{baseline_test_step}/{k}": v for k, v in metrics.items()}
                )

            if isinstance(evaluator, WandbChartsEvaluator):
                evaluator.log_wandb_charts(baseline_test_step)

            baseline_test_result = EvalStepResult(
                step_name=baseline_test_step,
                started_at=started_at_eval,
                finished_at=datetime.now(timezone.utc),
                metrics=metrics,

            )

            baseline_test_result.write(baseline_step_dir / "test")
            results.append(baseline_test_result)
            self._run_debugger(evaluator, context, model, baseline_step_dir)

        # --- Finetune segments ---
        ft_states: list[dict] = []

        for i, segment in enumerate(incremental_segments):
            step_name = f"finetune_{i}"
            step_dir = context.step_dir(step_name)

            model.load_state_dict(baseline_state)

            log.info(f"[{step_name}] Starting training")

            started_at = datetime.now(timezone.utc)

            ft_summary = self.finetune_trainer.fit(
                model,
                segment,
                checkpoint_dir=step_dir / "checkpoints",
                step_name=step_name,
                step_offset=global_step,
                reference_state=baseline_state,
            )

            global_step += ft_summary.epochs_trained

            log.info(
                f"[{step_name}] Training complete — "
                f"best_val={ft_summary.best_val_loss:.4f} "
                f"epoch={ft_summary.best_epoch}/{ft_summary.epochs_trained}"
                if ft_summary.best_val_loss is not None
                else f"[{step_name}] Training complete — {ft_summary.epochs_trained} epochs"
            )

            results.append(
                TrainStepResult(
                    step_name=step_name,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    final_train_loss=ft_summary.final_train_loss,
                    best_train_loss=ft_summary.best_train_loss,
                    final_val_loss=ft_summary.final_val_loss,
                    best_val_loss=ft_summary.best_val_loss,
                    best_epoch=ft_summary.best_epoch,
                    epochs_trained=ft_summary.epochs_trained,
                    checkpoint_path=(
                        ft_summary.checkpoint_path.relative_to(context.run_dir)
                        if ft_summary.checkpoint_path
                        else None
                    ),
                    secondary_val_losses=ft_summary.secondary_val_losses,
                )
            )

            results[-1].write(step_dir)

            # Store on CPU — N GPU copies would OOM on large models with many segments.
            ft_states.append({k: v.cpu() for k, v in model.state_dict().items()})

            # --- finetune_i/val (this segment's own held-out slice, mirrors baseline/val) ---
            if DatasetCapability.VAL in caps:
                val_evaluator = context.configurator.create_val_evaluator()
                ft_val_step = f"{step_name}/val"
                log.info(f"[{ft_val_step}] Starting evaluation")
                started_at_eval = datetime.now(timezone.utc)

                ft_val_metrics = self.runner.run(
                    model, val_evaluator, dataset.get_finetune_val_eval_dataset(i),
                    seed=context.eval_seed,
                )

                for k, v in ft_val_metrics.items():
                    log.info(f"  [{ft_val_step}] {k}: {v:.4f}")
                log.info(f"[{ft_val_step}] Evaluation complete")

                if wandb.run is not None:
                    wandb.run.summary.update(
                        {f"{ft_val_step}/{k}": v for k, v in ft_val_metrics.items()}
                    )

                ft_val_result = EvalStepResult(
                    step_name=ft_val_step,
                    started_at=started_at_eval,
                    finished_at=datetime.now(timezone.utc),
                    metrics=ft_val_metrics,
                )

                ft_val_result.write(step_dir / "val")
                results.append(ft_val_result)

            # --- finetune_i/test (this finetuned model on the global test set, before merging) ---
            if DatasetCapability.TEST in caps:
                evaluator = context.configurator.create_test_evaluator()

                reference_ds = (
                    dataset.get_train_eval_dataset()
                    if isinstance(evaluator, ReferenceEvaluator)
                    else None
                )

                ft_eval_step = f"{step_name}/test"
                log.info(f"[{ft_eval_step}] Starting evaluation")
                started_at_eval = datetime.now(timezone.utc)

                metrics = self.runner.run(
                    model, evaluator, context.dataset.get_test_dataset(),
                    reference_dataset=reference_ds,
                    seed=context.eval_seed,
                )

                for k, v in metrics.items():
                    log.info(f"  [{ft_eval_step}] {k}: {v:.4f}")
                log.info(f"[{ft_eval_step}] Evaluation complete")

                if wandb.run is not None:
                    wandb.run.summary.update(
                        {f"{ft_eval_step}/{k}": v for k, v in metrics.items()}
                    )

                if isinstance(evaluator, WandbChartsEvaluator):
                    evaluator.log_wandb_charts(ft_eval_step)

                ft_eval_result = EvalStepResult(
                    step_name=ft_eval_step,
                    started_at=started_at_eval,
                    finished_at=datetime.now(timezone.utc),
                    metrics=metrics,

                )

                ft_eval_result.write(step_dir / "test")
                results.append(ft_eval_result)
                self._run_debugger(evaluator, context, model, step_dir)

        # Kept so the extra-merge-scale curve can reuse the merge's own metrics as its
        # primary-scale row instead of paying for an identical extra evaluation pass.
        merged_val_metrics: dict[str, float] | None = None
        merged_test_metrics: dict[str, float] | None = None

        # --- Task arithmetic merge ---
        if ft_states:
            log.info(f"[merged] Merging {len(ft_states)} finetune models (scale={self.merge_scale})")
            merged_state = merge_task_arithmetic(
                baseline_state, ft_states, self.merge_scale
            )
            log.info("[merged] Merge complete")

            model.load_state_dict(merged_state)

            merged_dir = context.step_dir("merged")
            save_model_state(merged_dir / "checkpoints" / "best.pt", merged_state)

            # --- Merged val eval (sanity-check: does the merge hurt validation loss?) ---
            if DatasetCapability.VAL in caps:
                val_evaluator = context.configurator.create_val_evaluator()
                merged_val_step = "merged/val"
                log.info(f"[{merged_val_step}] Starting evaluation")
                started_at_eval = datetime.now(timezone.utc)

                merged_val_metrics = self.runner.run(
                    model, val_evaluator, dataset.get_merged_val_eval_dataset(),
                    seed=context.eval_seed,
                )

                for k, v in merged_val_metrics.items():
                    log.info(f"  [{merged_val_step}] {k}: {v:.4f}")
                log.info(f"[{merged_val_step}] Evaluation complete")

                if wandb.run is not None:
                    wandb.run.summary.update(
                        {f"{merged_val_step}/{k}": v for k, v in merged_val_metrics.items()}
                    )

                merged_val_result = EvalStepResult(
                    step_name=merged_val_step,
                    started_at=started_at_eval,
                    finished_at=datetime.now(timezone.utc),
                    metrics=merged_val_metrics,
    
                )

                merged_val_result.write(merged_dir / "val")
                results.append(merged_val_result)

        # --- Evaluate merged model (baseline was already evaluated above) ---
        if ft_states and DatasetCapability.TEST in caps:
            eval_step_name = "merged"
            eval_step_dir = context.step_dir(eval_step_name)
            evaluator = context.configurator.create_test_evaluator()

            reference_ds = (
                dataset.get_train_eval_dataset()
                if isinstance(evaluator, ReferenceEvaluator)
                else None
            )

            full_step = f"{eval_step_name}/test"
            log.info(f"[{full_step}] Starting evaluation")
            started_at = datetime.now(timezone.utc)

            metrics = self.runner.run(
                model, evaluator, context.dataset.get_test_dataset(),
                reference_dataset=reference_ds,
                seed=context.eval_seed,
            )

            for k, v in metrics.items():
                log.info(f"  [{full_step}] {k}: {v:.4f}")
            log.info(f"[{full_step}] Evaluation complete")

            if wandb.run is not None:
                wandb.run.summary.update(
                    {f"{full_step}/{k}": v for k, v in metrics.items()}
                )

            if isinstance(evaluator, WandbChartsEvaluator):
                evaluator.log_wandb_charts(full_step)

            merged_test_metrics = metrics

            result = EvalStepResult(
                step_name=full_step,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                metrics=metrics,

            )

            result.write(eval_step_dir / "test")
            results.append(result)

            self._run_debugger(evaluator, context, model, eval_step_dir)

        # --- Extra merge scales (evaluation only; runs last so a failure here cannot
        #     cost the merged/ artifacts already written above) ---
        if ft_states and self.extra_merge_scales:
            self._write_merge_scale_curve(
                context, model, baseline_state, ft_states,
                merged_val_metrics, merged_test_metrics,
            )

        return results

    def _write_merge_scale_curve(
        self,
        context: RunContext,
        model,
        baseline_state: dict,
        ft_states: list[dict],
        merged_val_metrics: dict[str, float] | None,
        merged_test_metrics: dict[str, float] | None,
    ) -> None:
        """Trace metrics against merge scale over the primary scale plus each extra one.

        The primary scale's row is the merge's own already-computed metrics — reused
        rather than recomputed, so it costs nothing and is the merged/ result by
        construction rather than a value that merely ought to match it.

        The task vectors are built once and re-applied per scale, so every point shares
        one set of checkpoints. That is what makes the curve's shape readable: it carries
        none of the run-to-run training noise a curve assembled from separate runs would.

        Rows go to a CSV rather than per-scale result.json files — at 19 AD metrics the
        latter would add hundreds of columns to the sweep-collection CSV, with a column
        set that shifts whenever the scale list changes.
        """
        dataset = context.dataset
        caps = dataset.capabilities
        evaluate_val = DatasetCapability.VAL in caps
        evaluate_test = DatasetCapability.TEST in caps
        if not (evaluate_val or evaluate_test):
            return

        rows: list[tuple[float, str, str, float]] = []
        if merged_val_metrics is not None:
            rows.extend((self.merge_scale, "val", k, v) for k, v in merged_val_metrics.items())
        if merged_test_metrics is not None:
            rows.extend((self.merge_scale, "test", k, v) for k, v in merged_test_metrics.items())

        # The primary scale is already covered above; asking for it again is harmless.
        scales = [s for s in dict.fromkeys(self.extra_merge_scales) if s != self.merge_scale]

        # Build the datasets once — re-windowing them per scale would dominate the runtime.
        val_dataset = dataset.get_merged_val_eval_dataset() if evaluate_val else None
        test_dataset = dataset.get_test_dataset() if evaluate_test else None
        reference_dataset = (
            dataset.get_train_eval_dataset()
            if evaluate_test
            and isinstance(context.configurator.create_test_evaluator(), ReferenceEvaluator)
            else None
        )

        task_vectors = [task_vector(baseline_state, ft) for ft in ft_states]

        log.info(
            f"[merge_scale_curve] Primary scale {self.merge_scale} reused; "
            f"evaluating {len(scales)} extra scale(s): {scales}"
        )

        for scale in scales:
            model.load_state_dict(apply_task_vectors(baseline_state, task_vectors, scale))

            if val_dataset is not None:
                metrics = self.runner.run(
                    model, context.configurator.create_val_evaluator(), val_dataset,
                    seed=context.eval_seed,
                )
                rows.extend((scale, "val", k, v) for k, v in metrics.items())

            if test_dataset is not None:
                metrics = self.runner.run(
                    model, context.configurator.create_test_evaluator(), test_dataset,
                    reference_dataset=reference_dataset,
                    seed=context.eval_seed,
                )
                rows.extend((scale, "test", k, v) for k, v in metrics.items())
                log.info(
                    f"  [merge_scale_curve] scale={scale}: "
                    + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )

        rows.sort(key=lambda row: (row[1], row[0]))
        curve_path = context.step_dir("merged") / "merge_scale_curve.csv"
        with curve_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["merge_scale", "split", "metric", "value"])
            writer.writerows(rows)

        # Leave the model at the primary scale, so evaluating the extras has no effect
        # on anything that runs afterwards.
        model.load_state_dict(
            apply_task_vectors(baseline_state, task_vectors, self.merge_scale)
        )
        log.info(f"[merge_scale_curve] Wrote {len(rows)} rows to {curve_path}")
