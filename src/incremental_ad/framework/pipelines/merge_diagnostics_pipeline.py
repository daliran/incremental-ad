import csv
import json
import logging
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Self, Sized, cast

from torch.utils.data import Dataset as TorchDataset

from incremental_ad.framework.contracts.dataset import (
    DataLoaderConfig,
    DatasetCapability,
    PartitionedDataset,
)
from incremental_ad.framework.contracts.evaluator import ReferenceEvaluator
from incremental_ad.framework.contracts.pipeline import Pipeline, RunContext, StepResult
from incremental_ad.framework.core.checkpoints import load_model_state
from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
from incremental_ad.framework.merging.task_vectors import (
    StateDict,
    apply_task_vectors,
    merge_task_arithmetic,
    task_vector,
)
from incremental_ad.framework.pipelines.standard_pipeline import EvalStepResult

log = logging.getLogger(__name__)

SOURCE_PIPELINE = "IncrementalTaskArithmeticPipeline"
STANDARD_PIPELINE = "StandardPipeline"

# Below this, |standard - base| is small enough that GRR is a ratio of two noise terms.
# 2% is the size of the reproducibility gap measured on ETTh1 (EXPERIMENTS.md §4: an
# exact same-seed repeat differed 2.1% MSE), so it is a floor, not a safety margin.
GRR_GAP_WARN_FRACTION = 0.02

# Args that must match the source run. Dataset args decide *which* regions get evaluated;
# model args decide *how* a checkpoint scores. Architecture mismatches surface anyway when
# load_state_dict rejects the shapes, but inference-behaviour args (n_eval_passes,
# mask_ratio, training_mode, instance_norm) change the numbers silently, leaving results
# that are not comparable with the source run's own.
COMPARED_ARG_PREFIXES = ("dataset_", "mae_tx_")

# Args a StandardPipeline run necessarily disagrees on: it trains on everything, so it
# records baseline_fraction=1.0 and n_finetune_segments=0. Requiring these to match would
# reject every valid Standard run. Everything else -- window_len, forecast_len,
# normalization, test_fraction and the whole model config -- still has to agree, or the
# reference model is being scored on data it was never trained for.
PARTITION_ARGS = frozenset({
    "dataset_baseline_fraction",
    "dataset_baseline_use_fraction",
    "dataset_n_finetune_segments",
})
BASE_ROW = "base"
MERGED_ROW = "merged"
STANDARD_ROW = "standard"
BASELINE_COLUMN = "val_base"
TEST_COLUMN = "test"


@dataclass(frozen=True)
class _Column:
    """One evaluation region: a name, the data, and which evaluator scores it."""

    name: str
    block: str  # "val" or "test"
    dataset: TorchDataset
    n_windows: int


class MergeDiagnosticsPipeline(Pipeline):
    """Post-hoc transfer matrix over a finished IncrementalTaskArithmeticPipeline run.

    Trains nothing: it reloads that run's checkpoints and cross-evaluates them.

    A finished run already records each specialist on *its own* shard's val slice and on
    the full test set — the matrix diagonal and the test column. What is missing, and
    what this fills in, is the off-diagonal: theta_0 + tau_i scored on shard *j*. Without
    it the diagonal is uninterpretable, because "theta_0 + tau_1 improved on shard 1"
    cannot be told apart from "it improved everywhere"; only improving *more* on its own
    shard than on the others demonstrates anything shard-specific was learned.

    Two column blocks, answering different questions:

    - **val slices** (`val_base`, `val_0` ... `val_{n-1}`) — did tau_i learn something
      specific to shard i? Scored with the *val* evaluator, so these are loss-shaped
      (reconstruction error for AD, whose training data carries no labels; MSE for
      forecasting), not detection-shaped. Reported as a ratio to the base model's value
      on the same column, so shards of differing difficulty are comparable and **lower is
      better** throughout.
    - **test** — does that translate into task performance? The full, unpartitioned test
      set with every task metric, identical for every row.

    Rows are the base model, each specialist, the merged model, and optionally a Standard
    model. Standard is scored on the test block only: it trains on everything but the
    global tail, so the interior val slices are inside its training data and its val cells
    would be training performance rather than held-out performance.
    """

    def __init__(
        self,
        source_run_dir: Path,
        runner: EvaluationRunner,
        standard_run_dir: Path | None = None,
        checkpoint_name: str = "best",
        merge_scales: list[float] | None = None,
        eval_seeds: list[int] | None = None,
        allow_config_mismatch: bool = False,
        compared_args: dict | None = None,
        curve_include_val: bool = False,
    ) -> None:
        self.source_run_dir = source_run_dir
        self.runner = runner
        self.standard_run_dir = standard_run_dir
        self.checkpoint_name = checkpoint_name
        self.merge_scales = merge_scales or []
        self.eval_seeds = eval_seeds or []
        self.curve_include_val = curve_include_val
        self.allow_config_mismatch = allow_config_mismatch
        # Captured in from_config: RunContext carries the built model and dataset, not the
        # args that built them, and validating against the source run needs the raw values.
        self.compared_args = compared_args or {}

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        DataLoaderConfig.add_args(parser)
        EvaluationRunner.add_args(parser)
        parser.add_argument(
            f"--{p}_source_run_dir", type=Path, required=True,
            help=f"a completed {SOURCE_PIPELINE} run directory to analyse.",
        )
        parser.add_argument(
            f"--{p}_standard_run_dir", type=Path, default=None,
            help="optional StandardPipeline run whose model is added as a reference row. "
            "Scored on the test block only -- see the class docstring.",
        )
        parser.add_argument(
            f"--{p}_checkpoint_name", choices=["best", "last"], default="best",
        )
        parser.add_argument(
            f"--{p}_merge_scales", type=float, nargs="*", default=[],
            help="merge scales to trace against the test metrics, e.g. 0.0 0.25 ... 1.5. "
            "Empty skips the curve. The source run's own scale is always included, and "
            "since every point reuses one set of checkpoints the curve carries none of "
            "the training noise a curve built from separate runs would. The shape is the "
            "diagnostic: a peak near 1.0 means the task vectors combine at full magnitude, "
            "a peak at 0.3-0.7 that they overlap and the full sum overshoots, and a "
            "monotone decline that any contribution is harmful.",
        )
        parser.add_argument(
            f"--{p}_eval_seeds", type=int, nargs="*", default=[],
            help="evaluation seeds. Defaults to the run's own eval_seed. Pass several to "
            "measure evaluation noise, which for AD is real: scoring averages "
            "--mae_tx_n_eval_passes random masks. Useful when val slices are small.",
        )
        parser.add_argument(
            f"--{p}_curve_include_val", action="store_true",
            help="also trace the per-shard val columns against the merge scale, not just "
            "test. Needed to select a scale without selecting on the test set, and to "
            "separate overshoot (the merged val curve descends to the diagonal) from "
            "irreducible interference (it plateaus above it). Off by default: it "
            "multiplies the per-scale cost by the val/test window ratio.",
        )
        parser.add_argument(
            f"--{p}_allow_config_mismatch", action="store_true",
            help="proceed even when dataset/model args differ from the source run. Unsafe: "
            "the val slices may not be the shards those checkpoints were trained on, or "
            "the checkpoints may be scored differently than the source run scored them.",
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            source_run_dir=getattr(cfg, f"{p}_source_run_dir"),
            runner=EvaluationRunner.from_config(cfg),
            standard_run_dir=getattr(cfg, f"{p}_standard_run_dir"),
            checkpoint_name=getattr(cfg, f"{p}_checkpoint_name"),
            merge_scales=getattr(cfg, f"{p}_merge_scales"),
            eval_seeds=getattr(cfg, f"{p}_eval_seeds"),
            allow_config_mismatch=getattr(cfg, f"{p}_allow_config_mismatch"),
            curve_include_val=getattr(cfg, f"{p}_curve_include_val"),
            compared_args={
                k: v for k, v in vars(cfg).items() if k.startswith(COMPARED_ARG_PREFIXES)
            },
        )

    # ── Loading and validation ────────────────────────────────────────────────

    def _read_config(self, run_dir: Path, expected_pipeline: str) -> dict:
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(f"{run_dir}: not a run directory (no config.json)")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("pipeline") != expected_pipeline:
            raise ValueError(
                f"{run_dir}: pipeline is {config.get('pipeline')!r}, "
                f"expected {expected_pipeline!r}"
            )
        return config

    def _check_config(
        self, run_dir: Path, config: dict, ignore: frozenset[str] = frozenset()
    ) -> None:
        """Every dataset and model arg must match this run's.

        This is the silent failure worth guarding hard. A different baseline_fraction,
        n_finetune_segments, val_fraction or window_len means the val slices evaluated
        here are simply not the shards those checkpoints were fine-tuned on; a different
        n_eval_passes, mask_ratio or training_mode means the checkpoints are scored
        differently than the run that produced them scored them. Either way every number
        would look plausible while being incomparable with its source.

        Applied to the Standard run too: its architecture would be caught by
        load_state_dict rejecting the shapes, but a different normalization or window
        would silently score it on data it was never trained for -- and it is the GRR
        denominator, so a wrong value there corrupts every recovery ratio.
        """
        run_args = config.get("args", {})
        current = self.compared_args
        mismatched = {
            key: (value, current.get(key))
            for key, value in run_args.items()
            if key.startswith(COMPARED_ARG_PREFIXES)
            and key not in ignore
            and key in current
            and current[key] != value
        }
        if not mismatched:
            return

        detail = "; ".join(
            f"{k}: run={s!r} current={c!r}" for k, (s, c) in sorted(mismatched.items())
        )
        if self.allow_config_mismatch:
            log.warning(
                "Args differ from %s (%s). Proceeding as --pipeline_allow_config_mismatch "
                "was set; these results may not be comparable with that run's own.",
                run_dir, detail,
            )
            return
        raise ValueError(
            f"args differ from {run_dir} ({detail}). Results would not be comparable with "
            "that run: dataset args change which regions are evaluated, model args change "
            "how the checkpoints score. Re-run with that run's args, or pass "
            "--pipeline_allow_config_mismatch."
        )

    def _finetune_dirs(self) -> list[Path]:
        directories = [
            d for d in self.source_run_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"finetune_\d+", d.name)
        ]
        # Numeric, not lexical: finetune_10 must not sort before finetune_2.
        return sorted(directories, key=lambda d: int(d.name.split("_")[1]))

    def _committed_merge_scale(self, source_config: dict) -> float:
        """The scale the source run's merged checkpoint was ACTUALLY built at.

        With ``--pipeline_select_merge_scale_on_val`` the run evaluates a grid and commits to
        the winner, so ``config.json``'s ``pipeline_merge_scale`` is the value that was *asked
        for*, not the one that was used. Reading the config alone therefore reconstructs a
        different model than the run produced — silently, and only for selecting runs.
        ``merged/val/result.json`` carries the value that won.
        """
        requested = source_config.get("args", {}).get("pipeline_merge_scale", 1.0)
        result = self.source_run_dir / "merged" / "val" / "result.json"
        if result.is_file():
            try:
                selected = json.loads(result.read_text()).get("metrics", {}).get("merge_scale/selected")
            except (OSError, ValueError):
                selected = None
            if selected is not None:
                if abs(float(selected) - float(requested)) > 1e-9:
                    log.info(
                        "[merge_diagnostics] source run selected merge_scale=%s on validation "
                        "(config requested %s); using the selected value",
                        selected, requested,
                    )
                return float(selected)
        return float(requested)

    def _load_rows(self, source_config: dict) -> dict[str, StateDict]:
        """Every model to put in the matrix, keyed by row name, in report order."""

        def checkpoint(directory: Path) -> Path:
            path = directory / "checkpoints" / f"{self.checkpoint_name}.pt"
            if not path.is_file():
                raise ValueError(f"missing checkpoint: {path}")
            return path

        base_state = load_model_state(checkpoint(self.source_run_dir / "baseline"))
        finetune_dirs = self._finetune_dirs()
        if not finetune_dirs:
            raise ValueError(f"{self.source_run_dir}: no finetune_* directories")

        expected = source_config.get("args", {}).get("dataset_n_finetune_segments")
        if expected is not None and len(finetune_dirs) != expected:
            raise ValueError(
                f"{self.source_run_dir}: found {len(finetune_dirs)} finetune dirs but "
                f"config.json records dataset_n_finetune_segments={expected}"
            )

        rows: dict[str, StateDict] = {BASE_ROW: base_state}
        ft_states = []
        for index, directory in enumerate(finetune_dirs):
            state = load_model_state(checkpoint(directory))
            ft_states.append(state)
            rows[f"ft_{index}"] = state

        # Recomputed rather than read from merged/checkpoints: it is bit-identical to what
        # the run wrote (verified across every run on disk), and this way the row exists
        # even for runs whose merged checkpoint was not kept.
        merge_scale = self._committed_merge_scale(source_config)
        rows[MERGED_ROW] = merge_task_arithmetic(base_state, ft_states, merge_scale)

        if self.standard_run_dir is not None:
            rows[STANDARD_ROW] = load_model_state(checkpoint(self.standard_run_dir / "train"))

        return rows

    # ── Columns ───────────────────────────────────────────────────────────────

    def _build_columns(self, context: RunContext, n_segments: int) -> list[_Column]:
        dataset = context.dataset
        assert isinstance(dataset, PartitionedDataset)
        caps = dataset.capabilities
        columns: list[_Column] = []

        if DatasetCapability.VAL in caps:
            val_datasets = [(BASELINE_COLUMN, dataset.get_baseline_val_eval_dataset())]
            val_datasets += [
                (f"val_{i}", dataset.get_finetune_val_eval_dataset(i))
                for i in range(n_segments)
            ]
            columns += [
                _Column(name, "val", ds, len(cast(Sized, ds))) for name, ds in val_datasets
            ]
        else:
            log.warning("Dataset declares no VAL capability - the transfer matrix will "
                        "have only the test block, which cannot show specialisation.")

        if DatasetCapability.TEST in caps:
            test_dataset = dataset.get_test_dataset()
            columns.append(
                _Column(TEST_COLUMN, "test", test_dataset, len(cast(Sized, test_dataset)))
            )

        return columns

    def _evaluate(self, context: RunContext, column: _Column, seed: int) -> dict[str, float]:
        if column.block == "val":
            return self.runner.run(
                context.model, context.configurator.create_val_evaluator(),
                column.dataset, seed=seed,
            )

        evaluator = context.configurator.create_test_evaluator()
        reference_dataset = (
            context.dataset.get_train_eval_dataset()
            if isinstance(evaluator, ReferenceEvaluator)
            else None
        )
        return self.runner.run(
            context.model, evaluator, column.dataset,
            reference_dataset=reference_dataset, seed=seed,
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, context: RunContext) -> list[StepResult]:
        assert isinstance(context.dataset, PartitionedDataset), (
            f"{type(self).__name__} requires PartitionedDataset, "
            f"got {type(context.dataset).__name__}"
        )

        started_at = datetime.now(timezone.utc)
        source_config = self._read_config(self.source_run_dir, SOURCE_PIPELINE)
        self._check_config(self.source_run_dir, source_config)
        if self.standard_run_dir is not None:
            self._check_config(
                self.standard_run_dir,
                self._read_config(self.standard_run_dir, STANDARD_PIPELINE),
                ignore=PARTITION_ARGS,
            )

        rows = self._load_rows(source_config)
        n_segments = sum(1 for name in rows if name.startswith("ft_"))
        columns = self._build_columns(context, n_segments)
        if not columns:
            raise ValueError("dataset exposes neither val slices nor a test set")

        seeds = self.eval_seeds or [context.eval_seed]
        step_dir = context.step_dir("merge_diagnostics")

        log.info(
            "[merge_diagnostics] %d model(s) x %d column(s) x %d seed(s) from %s",
            len(rows), len(columns), len(seeds), self.source_run_dir,
        )
        for column in columns:
            log.info("  column %-10s block=%-4s n_windows=%d", column.name, column.block, column.n_windows)

        # (row, column, metric, seed) -> value
        cells: dict[tuple[str, str, str, int], float] = {}
        for row_name, state in rows.items():
            context.model.load_state_dict(state)
            for column in columns:
                # Standard trained on everything but the global tail, so the interior val
                # slices sit inside its training data; only its test row is held out.
                if row_name == STANDARD_ROW and column.block == "val":
                    continue
                for seed in seeds:
                    metrics = self._evaluate(context, column, seed)
                    for metric, value in metrics.items():
                        cells[(row_name, column.name, metric, seed)] = value
            log.info("  [%s] evaluated", row_name)

        self._write_matrix(step_dir, rows, columns, seeds, cells)
        summary = self._summarise(rows, columns, seeds, cells)
        summary.update(
            self._write_merge_scale_curve(
                context, step_dir, source_config, rows, columns, seeds[0], cells
            )
        )
        self._write_source(step_dir, source_config, rows, columns)

        result = EvalStepResult(
            step_name="merge_diagnostics",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            metrics=summary,
        )
        result.write(step_dir)
        for key, value in sorted(summary.items()):
            log.info("  [merge_diagnostics] %s: %.4f", key, value)
        return [result]

    # ── Output ────────────────────────────────────────────────────────────────

    def _write_matrix(self, step_dir, rows, columns, seeds, cells) -> None:
        path = step_dir / "transfer_matrix.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "column", "block", "metric", "value",
                "ratio_to_base", "n_windows", "eval_seed",
            ])
            for column in columns:
                for seed in seeds:
                    metrics = sorted({m for (r, c, m, s) in cells if c == column.name and s == seed})
                    for metric in metrics:
                        base = cells.get((BASE_ROW, column.name, metric, seed))
                        for row_name in rows:
                            value = cells.get((row_name, column.name, metric, seed))
                            if value is None:
                                continue
                            ratio = value / base if base else ""
                            writer.writerow([
                                row_name, column.name, column.block, metric, value,
                                ratio, column.n_windows, seed,
                            ])
        log.info("[merge_diagnostics] Wrote %s", path)

    def _summarise(self, rows, columns, seeds, cells) -> dict[str, float]:
        """Derived scalars, per val metric, that separate the three failure modes.

        All are ratios to the base model on the same column, so 1.0 means "no different
        from the base" and, since val metrics are losses, below 1.0 means better.
        A specialist that learned something shard-specific shows diag < offdiag.
        """
        val_columns = [c.name for c in columns if c.block == "val" and c.name != BASELINE_COLUMN]
        val_metrics = sorted({m for (r, c, m, s) in cells if c in val_columns})
        seed = seeds[0]
        summary: dict[str, float] = {}

        for metric in val_metrics:
            def ratio(row_name: str, column: str) -> float | None:
                base = cells.get((BASE_ROW, column, metric, seed))
                value = cells.get((row_name, column, metric, seed))
                return value / base if base and value is not None else None

            diagonal: list[float] = []
            off_diagonal: list[float] = []
            for i, column in enumerate(val_columns):
                for j in range(len(val_columns)):
                    value = ratio(f"ft_{j}", column)
                    if value is not None:
                        (diagonal if i == j else off_diagonal).append(value)

            merged = [r for c in val_columns if (r := ratio(MERGED_ROW, c)) is not None]
            # ft_j on the baseline's own slice: how much adapting to a later shard costs
            # on the regime the base model was trained for.
            base_slice = [
                r for j in range(len(val_columns))
                if (r := ratio(f"ft_{j}", BASELINE_COLUMN)) is not None
            ]

            for label, values in [
                ("diag_ratio_mean", diagonal),
                ("offdiag_ratio_mean", off_diagonal),
                ("merged_ratio_mean", merged),
                ("base_slice_ratio_mean", base_slice),
            ]:
                if values:
                    summary[f"{metric}/{label}"] = sum(values) / len(values)

            if diagonal and off_diagonal:
                # Positive means each specialist helps more on its own shard than on the
                # others -- the evidence that anything shard-specific was learned at all.
                summary[f"{metric}/specialisation"] = (
                    sum(off_diagonal) / len(off_diagonal) - sum(diagonal) / len(diagonal)
                )

        summary.update(self._gap_recovery(rows, columns, seeds[0], cells))
        return summary

    def _gap_recovery(self, rows, columns, seed, cells) -> dict[str, float]:
        """Gap Recovery Ratio on the test block: how much of the distance from the frozen
        base to joint training the merge actually closed.

            GRR = (merged - base) / (standard - base)

        ~1 means merging recovers joint training; 0 means it is inert; <0 means it is
        destructive. The formula needs no sign handling -- for an error metric both
        differences are negative and the ratio stays positive, so it reads the same way
        for MSE as for AUROC.

        The gap itself is emitted next to every ratio, because GRR is only meaningful when
        the denominator clears the noise floor: where the base model already sits near the
        ceiling, GRR divides one noise term by another. That gate is what makes excluding
        a saturated dataset principled rather than post-hoc, so the number needed to apply
        it travels with the ratio rather than being left implicit.
        """
        if STANDARD_ROW not in rows:
            return {}

        recovery: dict[str, float] = {}
        for column in (c for c in columns if c.block == "test"):
            metrics = sorted({m for (r, c, m, s) in cells if c == column.name and s == seed})
            for metric in metrics:
                base = cells.get((BASE_ROW, column.name, metric, seed))
                merged = cells.get((MERGED_ROW, column.name, metric, seed))
                standard = cells.get((STANDARD_ROW, column.name, metric, seed))
                if base is None or merged is None or standard is None:
                    continue

                gap = standard - base
                recovery[f"{metric}/gap"] = gap
                if gap == 0:
                    continue
                recovery[f"{metric}/grr"] = (merged - base) / gap

                if base and abs(gap) < GRR_GAP_WARN_FRACTION * abs(base):
                    log.warning(
                        "[merge_diagnostics] %s: standard-vs-base gap is %.4g (%.1f%% of "
                        "base) - at or below the run-to-run noise floor, so its GRR of "
                        "%.3f is a ratio of two noise terms. Treat as uninformative.",
                        metric, gap, 100 * abs(gap) / abs(base), recovery[f"{metric}/grr"],
                    )

        return recovery

    def _write_merge_scale_curve(
        self, context, step_dir, source_config, rows, columns, seed, cells
    ) -> dict[str, float]:
        """Trace the test metrics against the merge scale, from the source run's weights.

        Answers the question the source run structurally cannot: a training run fixes its
        scale before training, so tracing the curve that way costs one full retrain per
        point *and* confounds the shape with seed-to-seed variance. Here every point
        shares one set of checkpoints, so the only thing varying is the scale.

        Two anchors cost nothing. alpha = 0 is bitwise the base model, and alpha = the
        source run's own scale is the merged model -- both already evaluated for the
        transfer matrix, so their rows are reused rather than recomputed, which also makes
        the curve exactly consistent with the matrix it sits beside.

        Test block by default. `--pipeline_curve_include_val` adds the per-shard val
        columns, which answers two things the test-only curve cannot:

        - **Selecting alpha honestly.** A curve over test metrics can only be read, not
          selected on -- picking the argmin of a test curve and then quoting that test
          value is selection on the reported number. The val columns give a selection
          signal that is independent of the test set.
        - **Separating overshoot from irreducible interference.** The merged val ratio at
          alpha = 1 mixes "we travelled too far along a good direction" with "these
          vectors genuinely cannot be combined." Tracing val against alpha splits them:
          if the merged curve descends to the diagonal, the whole cost was overshoot; if
          it plateaus above it, the residual is real interference.

        Off by default because it multiplies the per-scale cost by the val/test window
        ratio -- cheap on ETTh1, ~1.2x on SWaT and PSM.
        """
        if not self.merge_scales:
            return {}

        curve_columns = [c for c in columns if c.block == "test"]
        if self.curve_include_val:
            curve_columns = [c for c in columns if c.block == "val"] + curve_columns
        if not curve_columns:
            log.warning("[merge_scale_curve] no columns to trace - skipping")
            return {}

        source_scale = self._committed_merge_scale(source_config)
        reused = {0.0: BASE_ROW, source_scale: MERGED_ROW}

        # (scale, column name) -> {metric: value}
        curve: dict[tuple[float, str], dict[str, float]] = {}
        for scale, row_name in reused.items():
            for column in curve_columns:
                metrics = {
                    metric: value
                    for (r, c, metric, s), value in cells.items()
                    if r == row_name and c == column.name and s == seed
                }
                if metrics:
                    curve[(scale, column.name)] = metrics

        base_state = rows[BASE_ROW]
        task_vectors = [
            task_vector(base_state, rows[f"ft_{i}"])
            for i in range(sum(1 for name in rows if name.startswith("ft_")))
        ]
        scales = list(dict.fromkeys(self.merge_scales))
        pending = {
            scale: [c for c in curve_columns if (scale, c.name) not in curve]
            for scale in scales
        }

        log.info(
            "[merge_scale_curve] %d column(s) x %d scale(s); %d cell(s) reused from the "
            "matrix, %d to evaluate",
            len(curve_columns), len(scales), len(curve),
            sum(len(v) for v in pending.values()),
        )
        for scale in scales:
            if not pending[scale]:
                continue
            # One state load per scale, then every outstanding column for it.
            context.model.load_state_dict(apply_task_vectors(base_state, task_vectors, scale))
            for column in pending[scale]:
                curve[(scale, column.name)] = self._evaluate(context, column, seed)
            log.info("  [merge_scale_curve] scale=%s evaluated (%d column(s))",
                     scale, len(pending[scale]))

        blocks = {c.name: c for c in curve_columns}
        path = step_dir / "merge_scale_curve.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "merge_scale", "column", "block", "metric", "value", "n_windows", "eval_seed",
            ])
            for scale, column_name in sorted(curve, key=lambda k: (k[0], k[1])):
                column = blocks[column_name]
                for metric, value in curve[(scale, column_name)].items():
                    writer.writerow([
                        scale, column_name, column.block, metric, value, column.n_windows, seed,
                    ])
        log.info("[merge_scale_curve] Wrote %s", path)

        # Both extremes, because whether the optimum is the min or the max depends on
        # whether the metric is a loss or a score -- which the framework does not know.
        # Test keys stay unprefixed so existing readers and collect.py are unaffected;
        # val columns are namespaced by column name.
        summary: dict[str, float] = {}
        for column in curve_columns:
            points_by_metric: dict[str, list[tuple[float, float]]] = {}
            for (scale, column_name), metrics in curve.items():
                if column_name != column.name:
                    continue
                for metric, value in metrics.items():
                    points_by_metric.setdefault(metric, []).append((scale, value))
            prefix = "" if column.block == "test" else f"{column.name}/"
            for metric, points in sorted(points_by_metric.items()):
                summary[f"{prefix}{metric}/scale_at_min"] = min(points, key=lambda p: p[1])[0]
                summary[f"{prefix}{metric}/scale_at_max"] = max(points, key=lambda p: p[1])[0]
        return summary

    def _write_source(self, step_dir, source_config, rows, columns) -> None:
        (step_dir / "source.json").write_text(
            json.dumps(
                {
                    "source_run_dir": str(self.source_run_dir),
                    "source_run_id": source_config.get("run_id"),
                    "source_experiment_name": source_config.get("experiment_name"),
                    "source_git_commit": source_config.get("git_commit"),
                    "standard_run_dir": str(self.standard_run_dir) if self.standard_run_dir else None,
                    "checkpoint_name": self.checkpoint_name,
                    "merge_scale": self._committed_merge_scale(source_config),
                    "merge_scale_requested": source_config.get("args", {}).get("pipeline_merge_scale"),
                    "n_finetune_segments": source_config.get("args", {}).get("dataset_n_finetune_segments"),
                    "rows": list(rows),
                    "columns": [
                        {"name": c.name, "block": c.block, "n_windows": c.n_windows}
                        for c in columns
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
