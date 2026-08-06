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

    ``select_merge_scale_on_val`` changes which of those is committed. Off (the default),
    the run commits to ``merge_scale`` chosen before training and the curve is purely
    observational — which means the good scale is only ever identified afterwards, in
    analysis, by reading the minimum off the curve's *test* column. The checkpoint on disk
    was then never built at the scale the numbers describe. On, the same candidate set is
    evaluated on the merged val union *before* the merge and its winner produces the
    merged checkpoint, so the checkpoint, the reported metrics and the scale all describe
    one model, and no test data enters the choice.
    """

    def __init__(
        self,
        trainer: StandardTrainer,
        finetune_trainer: StandardTrainer,
        runner: EvaluationRunner,
        merge_scale: float = 1.0,
        extra_merge_scales: list[float] | None = None,
        select_merge_scale_on_val: bool = False,
    ) -> None:
        self.trainer = trainer
        self.finetune_trainer = finetune_trainer
        self.runner = runner
        self.merge_scale = merge_scale
        self.extra_merge_scales = extra_merge_scales or []
        self.select_merge_scale_on_val = select_merge_scale_on_val

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
        parser.add_argument(
            f"--{p}_select_merge_scale_on_val",
            action="store_true",
            help="pick the merge scale from --{p}_merge_scale plus --{p}_extra_merge_scales "
            "by evaluating each on the merged val union, and build merged/ at the winner. "
            "Off by default, in which case --{p}_merge_scale is committed as given. "
            "Costs no extra compute: those val passes are ones the curve already performs. "
            "Requires a val evaluator that declares a selection_metric() — AD does not, "
            "since its test metrics are rank-based and blind to the scale."
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
            select_merge_scale_on_val=getattr(cfg, f"{p}_select_merge_scale_on_val"),
        )

    def run(self, context: RunContext) -> list[StepResult]:
        assert isinstance(
            context.dataset, PartitionedDataset
        ), f"{type(self).__name__} requires PartitionedDataset, got {type(context.dataset).__name__}"

        # Checked before a single epoch runs: an unusable selection signal is a config
        # error, and the alternative is discovering it after hours of training.
        if self.select_merge_scale_on_val:
            self._assert_can_select_merge_scale(context)

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

        # Store baseline state on CPU — avoids holding (N+1) model copies on GPU during the
        # finetune loop. clone() is load-bearing, not defensive: Tensor.cpu() returns *self*
        # for a tensor already on CPU, so without it this snapshot aliases the live
        # parameters and every later load_state_dict silently rewrites it. On CUDA .cpu()
        # copies and the bug hides; on a CPU run it drives every task vector to zero.
        baseline_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
            # clone() for the same reason as the baseline snapshot above: on CPU these
            # would otherwise all alias one live model and collapse to the last fine-tune.
            ft_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

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
        # Populated only when selecting: val at *every* candidate, which is exactly the
        # set of passes the curve would otherwise make later.
        val_metrics_by_scale: dict[float, dict[str, float]] | None = None
        selection_span: tuple[datetime, datetime] | None = None
        effective_scale = self.merge_scale

        # --- Task arithmetic merge ---
        if ft_states:
            if self.select_merge_scale_on_val:
                effective_scale, val_metrics_by_scale, selection_span = (
                    self._select_merge_scale(context, model, baseline_state, ft_states)
                )
                merged_val_metrics = val_metrics_by_scale[effective_scale]

            log.info(f"[merged] Merging {len(ft_states)} finetune models (scale={effective_scale})")
            merged_state = merge_task_arithmetic(
                baseline_state, ft_states, effective_scale
            )
            log.info("[merged] Merge complete")

            model.load_state_dict(merged_state)

            merged_dir = context.step_dir("merged")
            save_model_state(merged_dir / "checkpoints" / "best.pt", merged_state)

            # --- Merged val eval (sanity-check: does the merge hurt validation loss?) ---
            if DatasetCapability.VAL in caps:
                merged_val_step = "merged/val"

                if merged_val_metrics is None:
                    val_evaluator = context.configurator.create_val_evaluator()
                    log.info(f"[{merged_val_step}] Starting evaluation")
                    started_at_eval = datetime.now(timezone.utc)

                    merged_val_metrics = self.runner.run(
                        model, val_evaluator, dataset.get_merged_val_eval_dataset(),
                        seed=context.eval_seed,
                    )
                    finished_at_eval = datetime.now(timezone.utc)
                else:
                    # Already computed while selecting, on this same state and dataset.
                    # The step is credited with the selection pass's span rather than a
                    # zero duration: that pass is the work that produced these numbers.
                    log.info(
                        f"[{merged_val_step}] Reusing the selection pass at scale={effective_scale}"
                    )
                    assert selection_span is not None
                    started_at_eval, finished_at_eval = selection_span

                for k, v in merged_val_metrics.items():
                    log.info(f"  [{merged_val_step}] {k}: {v:.4f}")
                log.info(f"[{merged_val_step}] Evaluation complete")

                # Only when selecting, so a run without the flag keeps exactly the metric
                # set it has today. config.json records the *requested* scale, which with
                # selection on no longer identifies the model — hence reporting the chosen
                # one as a metric. collect_sweep_results captures every metric it finds,
                # so this reaches the sweep CSV with no collector change.
                val_result_metrics = dict(merged_val_metrics)
                if self.select_merge_scale_on_val:
                    val_result_metrics["merge_scale/selected"] = effective_scale

                if wandb.run is not None:
                    wandb.run.summary.update(
                        {f"{merged_val_step}/{k}": v for k, v in val_result_metrics.items()}
                    )

                merged_val_result = EvalStepResult(
                    step_name=merged_val_step,
                    started_at=started_at_eval,
                    finished_at=finished_at_eval,
                    metrics=val_result_metrics,
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
                context, model, baseline_state, ft_states, effective_scale,
                merged_val_metrics, merged_test_metrics, val_metrics_by_scale,
            )

        return results

    def _merge_scale_candidates(self) -> list[float]:
        """The primary scale plus every extra, de-duplicated, ascending.

        Ascending matters for selection: ties break towards the first element, so equal
        val metrics resolve to the smaller scale — the more conservative merge.
        """
        return sorted(dict.fromkeys([self.merge_scale, *self.extra_merge_scales]))

    def _assert_can_select_merge_scale(self, context: RunContext) -> None:
        """Reject a selection request that could not produce a meaningful choice."""
        p = self.ARG_PREFIX
        assert DatasetCapability.VAL in context.dataset.capabilities, (
            f"--{p}_select_merge_scale_on_val needs a val split to select on, but "
            f"{type(context.dataset).__name__} does not declare DatasetCapability.VAL."
        )

        candidates = self._merge_scale_candidates()
        assert len(candidates) > 1, (
            f"--{p}_select_merge_scale_on_val was requested but there is only one candidate "
            f"scale ({candidates[0]}), so there is nothing to select between. Pass the "
            f"grid via --{p}_extra_merge_scales, or drop the flag to commit to "
            f"--{p}_merge_scale as given."
        )

        selection = context.configurator.create_val_evaluator().selection_metric()
        assert selection is not None, (
            f"--{p}_select_merge_scale_on_val is not available for this task: "
            f"{type(context.configurator.create_val_evaluator()).__name__} declares no "
            f"selection_metric(), meaning its val metrics are not a basis for choosing "
            f"between models. Pass an explicit --{p}_merge_scale instead."
        )

    def _select_merge_scale(
        self,
        context: RunContext,
        model,
        baseline_state: dict,
        ft_states: list[dict],
    ) -> tuple[float, dict[float, dict[str, float]], tuple[datetime, datetime]]:
        """Choose the merge scale that wins on the merged val union, and record the evidence.

        Selection reads val only, and runs before the merge, so the scale that produces
        merged/checkpoints is the scale the merged/ metrics describe. Test is evaluated
        afterwards at the winner and never feeds back — the curve stays a full diagnostic
        trace while the choice stays clean.

        Returns the winner, every candidate's val metrics (the curve reuses these instead
        of recomputing them), and the span of the whole pass.
        """
        dataset = context.dataset
        metric, direction = context.configurator.create_val_evaluator().selection_metric()
        candidates = self._merge_scale_candidates()

        # Built once and re-applied per candidate, as in the curve: every point shares one
        # set of checkpoints, so the comparison carries no run-to-run training noise.
        task_vectors = [task_vector(baseline_state, ft) for ft in ft_states]
        val_dataset = dataset.get_merged_val_eval_dataset()

        log.info(
            f"[merge_scale_select] Selecting on merged val by {metric} ({direction}) over "
            f"{len(candidates)} candidate(s): {candidates}"
        )
        started_at = datetime.now(timezone.utc)

        val_metrics_by_scale: dict[float, dict[str, float]] = {}
        for scale in candidates:
            model.load_state_dict(apply_task_vectors(baseline_state, task_vectors, scale))
            metrics = self.runner.run(
                model, context.configurator.create_val_evaluator(), val_dataset,
                seed=context.eval_seed,
            )
            assert metric in metrics, (
                f"val evaluator declares selection metric {metric!r} but computed "
                f"{sorted(metrics)}"
            )
            val_metrics_by_scale[scale] = metrics
            log.info(f"  [merge_scale_select] scale={scale}: {metric}={metrics[metric]:.6f}")

        finished_at = datetime.now(timezone.utc)

        pick = min if direction == "min" else max
        selected = pick(candidates, key=lambda s: val_metrics_by_scale[s][metric])
        log.info(
            f"[merge_scale_select] Selected scale={selected} "
            f"({metric}={val_metrics_by_scale[selected][metric]:.6f})"
        )

        self._write_merge_scale_selection(context, metric, val_metrics_by_scale, selected)
        return selected, val_metrics_by_scale, (started_at, finished_at)

    def _write_merge_scale_selection(
        self,
        context: RunContext,
        metric: str,
        val_metrics_by_scale: dict[float, dict[str, float]],
        selected: float,
    ) -> None:
        """Write the selection's evidence, so the chosen scale is auditable and not just asserted."""
        path = context.step_dir("merged") / "merge_scale_selection.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["merge_scale", "metric", "value", "selected"])
            for scale in sorted(val_metrics_by_scale):
                writer.writerow(
                    [scale, metric, val_metrics_by_scale[scale][metric], int(scale == selected)]
                )
        log.info(f"[merge_scale_select] Wrote {path}")

    def _write_merge_scale_curve(
        self,
        context: RunContext,
        model,
        baseline_state: dict,
        ft_states: list[dict],
        primary_scale: float,
        merged_val_metrics: dict[str, float] | None,
        merged_test_metrics: dict[str, float] | None,
        val_metrics_by_scale: dict[float, dict[str, float]] | None = None,
    ) -> None:
        """Trace metrics against merge scale over the primary scale plus each extra one.

        The primary scale's row is the merge's own already-computed metrics — reused
        rather than recomputed, so it costs nothing and is the merged/ result by
        construction rather than a value that merely ought to match it. When selection
        ran, ``val_metrics_by_scale`` carries val at every candidate for the same reason:
        those passes have already happened, which is why selection is free.

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
        if val_metrics_by_scale:
            rows.extend(
                (scale, "val", k, v)
                for scale, metrics in val_metrics_by_scale.items()
                for k, v in metrics.items()
            )
        elif merged_val_metrics is not None:
            rows.extend((primary_scale, "val", k, v) for k, v in merged_val_metrics.items())
        if merged_test_metrics is not None:
            rows.extend((primary_scale, "test", k, v) for k, v in merged_test_metrics.items())

        # The primary scale is already covered above; asking for it again is harmless.
        scales = [s for s in dict.fromkeys(self.extra_merge_scales) if s != primary_scale]

        # Selection already evaluated val at every candidate, so re-running it here would
        # recompute identical numbers — that reuse is what makes selection cost nothing.
        evaluate_val_in_loop = evaluate_val and not val_metrics_by_scale

        # Build the datasets once — re-windowing them per scale would dominate the runtime.
        val_dataset = dataset.get_merged_val_eval_dataset() if evaluate_val_in_loop else None
        test_dataset = dataset.get_test_dataset() if evaluate_test else None
        reference_dataset = (
            dataset.get_train_eval_dataset()
            if evaluate_test
            and isinstance(context.configurator.create_test_evaluator(), ReferenceEvaluator)
            else None
        )

        task_vectors = [task_vector(baseline_state, ft) for ft in ft_states]

        log.info(
            f"[merge_scale_curve] Primary scale {primary_scale} reused; "
            f"evaluating {len(scales)} extra scale(s): {scales}"
            + ("" if evaluate_val_in_loop else " (test only — val came from selection)")
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
            apply_task_vectors(baseline_state, task_vectors, primary_scale)
        )
        log.info(f"[merge_scale_curve] Wrote {len(rows)} rows to {curve_path}")
