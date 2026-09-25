import csv
import json
import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

import wandb

from incremental_ad.framework.contracts.dataset import (
    DataLoaderConfig,
    DatasetCapability,
    PartitionedDataset,
)
from incremental_ad.framework.contracts.evaluator import ReferenceEvaluator
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.contracts.trainer import TrainSummary
from incremental_ad.framework.core.checkpoints import load_model_state
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
        lambda_source: str = "none",
        fixed_lambda: float = 1.0,
        one_over_t_counts_base: bool = True,
        fisher_batches: int = 64,
        fisher_batch_size: int | None = None,
        lambda_seed_from_base: bool = True,
        lambda_fisher_weighting: str = "unweighted",
        baseline_checkpoint: str | None = None,
    ) -> None:
        self.trainer = trainer
        self.finetune_trainer = finetune_trainer
        self.runner = runner
        self.anchor_to_baseline = anchor_to_baseline
        # Every field below is inert at its default: `lambda_source="none"` is the only gate
        # on the entire adaptive path, and nothing else reads these unless it is set.
        self.lambda_source = lambda_source
        self.fixed_lambda = fixed_lambda
        self.one_over_t_counts_base = one_over_t_counts_base
        self.fisher_batches = fisher_batches
        # 0 = every window once. `diagonal_fisher` reads None as "no cap".
        self._max_batches = fisher_batches if fisher_batches > 0 else None
        self.fisher_batch_size = fisher_batch_size
        self.lambda_seed_from_base = lambda_seed_from_base
        self.lambda_fisher_weighting = lambda_fisher_weighting
        self.baseline_checkpoint = baseline_checkpoint
        assert lambda_source in ("none", "became", "one_over_t", "fixed")
        assert 0.0 < fixed_lambda <= 1.0, f"fixed lambda must be in (0, 1], got {fixed_lambda}"

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

        parser.add_argument(
            f"--{p}_lambda_source", choices=["none", "became", "one_over_t", "fixed"],
            default="none",
            help="pull the chain back toward the accumulator after each period. 'none' is "
            "today's plain chain and takes no new code path at all. 'became' uses BECAME's "
            "closed-form coefficient (Eq. 20) in the SEQUENTIAL frame — which is not BECAME, "
            "because the published method also runs a gradient-projection stage; call it "
            "'adaptive-lambda sequential fine-tuning (BECAME's coefficient)'. 'one_over_t' is "
            "the fixed-schedule control. 'fixed' holds lambda constant.",
        )
        parser.add_argument(
            f"--{p}_fixed_lambda", type=float, default=1.0,
            help="lambda for --lambda_source fixed. 1.0 reproduces the plain chain exactly, "
            "which is what gate 1 of scripts/verify_adaptive_lambda.py checks.",
        )
        parser.add_argument(
            f"--{p}_one_over_t_counts_base", type=_str_to_bool, default=True,
            help="count the base model as task 1, so period 1 is task 2 and lambda = 1/2. "
            "Default true because the control must index tasks the way the adaptive rule "
            "does: with equal Fishers Eq. 20 gives lambda*_1 = 1/2, so starting the schedule "
            "at t=1 would confound the coefficient rule with the task indexing.",
        )
        parser.add_argument(
            f"--{p}_fisher_batches", type=int, default=64,
            help="batches per diagonal-Fisher estimate; matches the merging pipeline. 0 means "
            "a full pass over the period's training windows. The samples actually used are "
            "recorded per estimate in adaptive_lambdas.csv, whatever this is set to.",
        )
        parser.add_argument(
            f"--{p}_fisher_batch_size", type=int, default=None,
            help="batch size for Fisher estimation only; None keeps the training batch size, "
            "which is today's behaviour and what every published number used. Set 1 for the "
            "empirical Fisher: `diagonal_fisher` squares the gradient of a BATCH-MEAN loss, "
            "which at a converged minimum is minibatch noise scaling as 1/sqrt(B), so lambda's "
            "numerator is suppressed by a dataloader property rather than by the model. Note "
            "that --fisher_batches counts BATCHES, so lowering this lowers the sample count "
            "unless you raise that too.",
        )
        parser.add_argument(
            f"--{p}_lambda_seed_from_base", type=_str_to_bool, default=True,
            help="seed Lambda_0 with the base model's own Fisher. Algorithm 1 line 2 sets "
            "Lambda_1 = F_1(theta*_1) before the loop, and our base model plays that role. "
            "Without it lambda*_1 = 1 and period 1 enters whole, discarding the curvature of "
            "the model that has seen the most data.",
        )
        parser.add_argument(
            f"--{p}_lambda_fisher_weighting", choices=["unweighted", "data"],
            default="unweighted",
            help="'unweighted' is Algorithm 1 line 9 verbatim. 'data' scales each Fisher by "
            "its sample count, which is closer to the Laplace derivation when tasks are "
            "unequal in size — as they are here, where the base holds 50%% of the stream and "
            "a period may hold 10%%. Ablation, not a default.",
        )
        parser.add_argument(
            f"--{p}_baseline_checkpoint", type=str, default=None,
            help="load theta_0 from this checkpoint and skip step-0 training. Makes the "
            "comparison against an existing chain run PAIRED — same baseline, so the only "
            "difference is the pullback — which matters because GPU placement alone moves "
            "results by up to 18.8%% (EXPERIMENTS.md §3.2).",
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
            lambda_source=getattr(cfg, f"{p}_lambda_source"),
            fixed_lambda=getattr(cfg, f"{p}_fixed_lambda"),
            one_over_t_counts_base=getattr(cfg, f"{p}_one_over_t_counts_base"),
            fisher_batches=getattr(cfg, f"{p}_fisher_batches"),
            fisher_batch_size=getattr(cfg, f"{p}_fisher_batch_size"),
            lambda_seed_from_base=getattr(cfg, f"{p}_lambda_seed_from_base"),
            lambda_fisher_weighting=getattr(cfg, f"{p}_lambda_fisher_weighting"),
            baseline_checkpoint=getattr(cfg, f"{p}_baseline_checkpoint"),
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

    def _pullback(self, model, segment, step: int, accumulator, precision, step_dir):
        """One adaptive step: compute lambda, merge theta_hat into the accumulator, update Lambda.

        Called ONLY from inside `if self.lambda_source != "none"`. Returns
        ``(lambda_star, theta_star_state, precision)`` and leaves `model` holding theta*_t, so
        the chain continues from the merged model as §2's pseudocode requires.

        `step` is the period index counting from 1. The task index used by `one_over_t` is
        `step + 1` when the base counts as task 1 — see `--continual_one_over_t_counts_base`.
        """
        from incremental_ad.framework.merging.became import (
            accumulate_precision, became_lambda, diagonal_fisher,
        )

        keys = [k for k, v in accumulator.items() if v.is_floating_point()]
        # theta_hat_t, snapshotted off the live model. `.detach().cpu().clone()` and not
        # `.cpu()`: for a tensor already on CPU the latter returns *self*, so the snapshot
        # would alias the live parameters and follow them through the merge below. The same
        # bug is called out at the theta_0 anchor a few lines down in `run`.
        unconstrained = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        displacement = {k: unconstrained[k].double() - accumulator[k].double() for k in keys}

        task_index = step + 1 if self.one_over_t_counts_base else step
        one_over_t = 1.0 / task_index
        numerator = denominator = star_numerator = float("nan")
        hat_counts: dict = {}
        star_counts: dict = {}

        if self.lambda_source == "fixed":
            lam = self.fixed_lambda
        elif self.lambda_source == "one_over_t":
            lam = one_over_t
        else:
            loader = self.runner.loader_config.make_loader(
                segment.train, shuffle=True, batch_size=self.fisher_batch_size)
            fisher_hat = diagonal_fisher(model, loader, self.runner.device,
                                         max_batches=self._max_batches, counts=hat_counts)
            # Eq. 20 with Lambda_{t-1} passed as a one-element list; identical arithmetic to
            # passing every earlier Fisher, asserted in verify_adaptive_lambda.py.
            previous = [precision] if precision is not None else []
            lam = became_lambda(displacement, fisher_hat, previous)
            numerator, denominator = _quadratic_forms(displacement, fisher_hat, precision)

        assert 0.0 <= lam <= 1.0, f"lambda {lam} outside [0, 1] at period {step}"

        merged = {k: v.clone() for k, v in unconstrained.items()}
        for key in keys:
            merged[key] = ((1.0 - lam) * accumulator[key].double()
                           + lam * unconstrained[key].double()).to(accumulator[key].dtype)
        model.load_state_dict(merged)

        # Algorithm 1 line 9: Lambda_t = F_t(theta*_t) + Lambda_{t-1}. A SECOND Fisher pass,
        # at the MERGED point — not the one used for lambda, which is taken at theta_hat_t.
        # Using one pass for both would be a different algorithm.
        if self.lambda_source == "became":
            loader = self.runner.loader_config.make_loader(
                segment.train, shuffle=True, batch_size=self.fisher_batch_size)
            fisher_star = diagonal_fisher(model, loader, self.runner.device,
                                          max_batches=self._max_batches, counts=star_counts)
            weight = float(len(segment.train)) if self.lambda_fisher_weighting == "data" else 1.0
            # d^T F_t(theta*_t) d, the SAME quadratic form the numerator uses, on the same task,
            # the same data and the same d -- only the evaluation point differs. Its ratio to
            # `fisher_num` isolates at-a-minimum vs not-at-a-minimum with everything else held
            # constant, which is the candidate mechanism for the part of lambda's suppression
            # the batch-size artifact does not explain. Free: `fisher_star` is already computed
            # for Algorithm 1 line 9, so this adds a dot product and no forward pass.
            star_numerator, _ = _quadratic_forms(displacement, fisher_star, None)
            precision = accumulate_precision(precision, fisher_star, weight)

        log.info("[adaptive] period %d: lambda*=%.6f (1/t would be %.6f, t=%d)",
                 step, lam, one_over_t, task_index)
        # Every distance is measured where it is DEFINED, not wherever `model` happens to point
        # afterwards: `model` holds theta*_t by the time this returns, so reading the
        # unconstrained distance from it later would silently report the merged one.
        # ── Gate 6 ────────────────────────────────────────────────────────────────────────
        # theta*_t - theta*_{t-1} = lambda_t * (theta_hat_t - theta*_{t-1}) by construction, so
        # ||step|| must equal lambda_t * d_norm exactly. One identity that pins THREE things at
        # once: that d_norm measures the displacement it claims to, that the lambda written to
        # the CSV is the lambda actually applied, and that the interpolation runs in the right
        # direction. Gates 1-5 all passed while two distance columns were wrong in a way that
        # would have made P4 read 1.000 on every row — a pinned result indistinguishable from a
        # real finding. This is the check that catches that class.
        d_norm = _norm(displacement)
        step_norm = _distance(merged, accumulator)
        expected = lam * d_norm
        tolerance = 1e-6 * max(expected, 1.0)
        assert abs(step_norm - expected) <= tolerance, (
            f"period {step}: ||theta*_t - theta*_(t-1)|| = {step_norm:.9f} but lambda * d_norm = "
            f"{expected:.9f} (lambda={lam:.6f}, d_norm={d_norm:.9f}). The step taken is not the "
            f"step lambda describes — d_norm, the applied lambda or the interpolation direction "
            f"is wrong."
        )
        diagnostics = {
            "step": step,
            "lambda_star": lam,
            "one_over_t": one_over_t,
            "d_norm": d_norm,
            "step_norm": step_norm,
            "fisher_num": numerator,
            "fisher_den": denominator,
            "fisher_star_num": star_numerator,
            # Samples ACTUALLY used by each Fisher estimate (blank on the imposed-lambda paths).
            "fisher_hat_samples": hat_counts.get("samples", ""),
            "fisher_star_samples": star_counts.get("samples", ""),
            "fisher_seed_samples": self._seed_samples,
            "merged_dist_from_base": _distance(merged, self._theta_zero),
            "unconstrained_dist_from_base": _distance(unconstrained, self._theta_zero),
        }
        return merged, precision, diagnostics

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

        if self.baseline_checkpoint:
            # Paired comparison: point at the SAME baseline an existing chain run used, so the
            # only difference between plain and adaptive is the pullback. Without this, GPU
            # placement alone moves the baseline by up to 18.8% (EXPERIMENTS.md §3.2) and the
            # comparison measures the scheduler as much as the method.
            source = Path(self.baseline_checkpoint)
            log.info("[baseline] loading theta_0 from %s — step-0 training SKIPPED", source)
            model.load_state_dict(load_model_state(source))
            # The source is recorded in the step dir rather than through the context, which
            # has no extras channel; this keeps the provenance with the artefacts it explains.
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "baseline_source.json").write_text(
                json.dumps({"baseline_checkpoint": str(source)}, indent=2))
            summary = TrainSummary(
                final_train_loss=float("nan"), best_train_loss=float("nan"),
                final_val_loss=None, best_val_loss=None, best_epoch=0, epochs_trained=0,
                checkpoint_path=None,
            )
        else:
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

        # --- Adaptive-lambda state (inert unless --continual_lambda_source is set) ---
        # theta*_0 is the base model; Lambda_0 is seeded from its own Fisher, because
        # Algorithm 1 line 2 populates the precision matrix BEFORE the loop and our base model
        # plays the paper's theta*_1 (its theta_0 is a random init with no counterpart here).
        accumulator = baseline_state if self.lambda_source != "none" else None
        self._theta_zero = baseline_state      # fixed reference for the distance columns
        precision = None
        self._seed_samples = ""      # set below when Lambda_0 is seeded from the base
        lambda_rows: list[dict] = []
        unconstrained_matrix: dict[int, dict[str, dict[str, float]]] = {}
        if self.lambda_source == "became" and self.lambda_seed_from_base:
            from incremental_ad.framework.merging.became import (
                accumulate_precision, diagonal_fisher,
            )
            base_segment = dataset.get_baseline()
            loader = self.runner.loader_config.make_loader(
                base_segment.train, shuffle=True, batch_size=self.fisher_batch_size)
            log.info("[adaptive] seeding Lambda_0 from the base model's Fisher")
            weight = (float(len(base_segment.train))
                      if self.lambda_fisher_weighting == "data" else 1.0)
            seed_counts: dict = {}
            precision = accumulate_precision(
                None,
                diagonal_fisher(model, loader, self.runner.device,
                                max_batches=self._max_batches, counts=seed_counts),
                weight,
            )
            self._seed_samples = seed_counts["samples"]

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

            # ── adaptive-lambda pullback ────────────────────────────────────────────────
            # Everything here is behind the guard; with the default `none` the loop body is
            # exactly what it was. theta_hat_t is scored FIRST, while the model still holds
            # it — evaluating after the merge would need a reload and a second forward pass.
            if self.lambda_source != "none":
                unconstrained_matrix[index + 1] = self._eval_all_columns(context, columns)
                accumulator, precision, diagnostics = self._pullback(
                    model, segment, index + 1, accumulator, precision, step_dir
                )
                lambda_rows.append(diagnostics)

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
        if self.lambda_source != "none":
            self._write_adaptive_outputs(context, unconstrained_matrix, lambda_rows, columns)

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

    def _write_adaptive_outputs(self, context, unconstrained_matrix, lambda_rows, columns):
        """The two files the adaptive path adds. Neither has an existing consumer.

        `backward_transfer.csv` is deliberately NOT touched: it keeps today's schema and carries
        theta*_t, which is the chain's model and what §1.34 measures. Adding a `model` column to
        it would change a file `analysis/forgetting_report.py` parses, so theta_hat_t goes to a
        separate file with the same columns instead. A new file breaks nothing; a new column in
        a shared file breaks a consumer.
        """
        step_dir = context.step_dir("continual_summary")
        step_dir.mkdir(parents=True, exist_ok=True)
        metrics = sorted({m for step in unconstrained_matrix.values()
                          for col in step.values() for m in col})
        path = step_dir / "backward_transfer_unconstrained.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["after_step", "step_name", "column", "metric", "value"])
            for step in sorted(unconstrained_matrix):
                for column, _ in columns:
                    for metric in metrics:
                        value = unconstrained_matrix[step].get(column, {}).get(metric)
                        if value is not None:
                            writer.writerow([step, f"continual_{step - 1}", column, metric, value])

        # `fisher_star_num` is appended, never inserted: readers of this file index by name,
        # but a column added in the middle changes every diff of it for no reason.
        fields = ["step", "lambda_star", "one_over_t", "d_norm", "step_norm",
                  "fisher_num", "fisher_den",
                  "merged_dist_from_base", "unconstrained_dist_from_base",
                  "fisher_star_num", "fisher_hat_samples", "fisher_star_samples",
                  "fisher_seed_samples"]
        path = step_dir / "adaptive_lambdas.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(lambda_rows)
        log.info("[adaptive] wrote %s (%d step(s))", path, len(lambda_rows))

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


def _norm(state) -> float:
    """L2 norm over the floating-point tensors of a state dict, in float64."""
    import torch
    total = sum(float((v.double() ** 2).sum()) for v in state.values()
                if v.is_floating_point())
    return float(torch.tensor(total).sqrt())


def _distance(state, reference) -> float:
    """``||state - reference||_2`` over the shared floating-point tensors."""
    return _norm({k: v.detach().cpu().double() - reference[k].detach().cpu().double()
                  for k, v in state.items()
                  if v.is_floating_point() and k in reference})


def _quadratic_forms(displacement, fisher_new, precision):
    """``(d^T F_t d, d^T (F_t + Lambda) d)`` — lambda*'s numerator and denominator, for the CSV.

    Recomputed rather than returned from `became_lambda`, because §3.4 of the brief forbids
    changing that function's signature. The ratio is asserted against the logged lambda in
    `verify_adaptive_lambda.py`, so a divergence between the two shows up as a gate failure
    rather than as a quietly wrong diagnostic column.
    """
    numerator = denominator = 0.0
    for key in displacement:
        if key not in fisher_new:
            continue
        squared = displacement[key].detach().double().cpu() ** 2
        own = float((fisher_new[key] * squared).sum())
        other = float((precision[key] * squared).sum()) if precision and key in precision else 0.0
        numerator += own
        denominator += own + other
    return numerator, denominator


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")
