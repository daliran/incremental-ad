# CLAUDE.md

Agent-oriented context for this repo. Full architecture is in [README.md](README.md) ("Framework internals") — read it for the deep dive; this file is the high-signal map plus the non-obvious gotchas that aren't visible in the code.

For the *research* concepts — task vectors, the transfer matrix, interference vs overshoot,
weight disentanglement, why the AD metrics can't see model change — read
[TASK_ARITHMETIC_NOTES.md](TASK_ARITHMETIC_NOTES.md). It explains the reasoning;
[EXPERIMENTS.md](EXPERIMENTS.md) remains the source of truth for numbers.

## What this is

Research codebase for **incremental anomaly detection on multivariate time series** via task arithmetic, generalized to 4 tasks (AD, forecasting, imputation, classification) on one MAE backbone. Entry point: `python -m incremental_ad.main --model … --dataset … --task … --pipeline …`.

## Architecture invariant (do not break)

- **`framework/`** = reusable batteries-included toolkit: generic contracts **plus** ready-to-use concrete implementations (trainer, evaluators incl. AD, splitting, sliding-window, pipelines, runner).
- **`project/`** = specific to THIS research: the `MaeTx`/`MaeTxClassifier` model, its task configurators, the AD debugger, the concrete datasets, `task.py`, experiment wiring.
- **`analysis/`** = post-hoc CLI entry points over finished runs (`geometry_report`, `diagnose`). Like `main.py`, neither framework nor project — they compose both. They emit measurements, never interpretations; the research reading of those numbers belongs in EXPERIMENTS.md.
- **The invariant is the dependency direction: `project` imports `framework`, NEVER the reverse.** That's what keeps the framework reusable — not "framework contains no domain words." Placement criterion = *reusability*; reusable code can be promoted `project → framework` with no call-site changes (everything goes through contracts + registries).
- Verify the boundary: `grep -rn "import incremental_ad.project" src/incremental_ad/framework` must be empty.

## Conventions

- **`Configurable`**: every pluggable component implements `add_args(parser, prefix)` + `from_config(cfg, prefix)`. Args are namespaced by `ARG_PREFIX` (e.g. `--mae_tx_patch_len`). `prefix` lets the same class be used twice (incremental pipeline uses `--trainer_*` and `--finetune_trainer_*`, no fallback between them).
- **Registries** via `__init_subclass__`: `Dataset`, `Model`, `Pipeline`, `Trainer` self-register by class name. `main.py` imports `project/` to populate them, then looks up by CLI string. Adding a component = create class + import in the relevant `__init__.py`; `main.py` never changes.
- **`TaskModelConfigurator`** is keyed by `(task_name: str, model_cls)`. Task is a **plain string** in the framework (`_task_key` normalizes the project's `Task` enum via `.value`); `--task` choices come from `registered_tasks()`. The `Task` enum lives in `project/task.py`.
- **Configurators** bridge a (task, model) pair: `_configure()` validates compatibility and injects dataset-derived values, then the framework calls `model._build()`. Models build NO nn layers in `__init__` — only in `_build()`, after the configurator injects `n_features`/`seq_len`/etc.

## Compatibility validation (two complementary mechanisms — keep both)

- **Structural protocols** (`ForecastDataset`/`ImputationDataset`/`ClassificationDataset`, all `@runtime_checkable`): checked with `isinstance` inside each configurator's `_configure`. Answers "does the dataset expose the typed property the task needs?" (`forecast_len`, `mask_patch_len`, `n_classes`). `isinstance` both checks AND type-narrows.
- **`DatasetCapability`** (`TEST`, `VAL`, `TEST_LABELS`): `TEST`/`VAL` gate pipeline stages; `TEST_LABELS` = "test set carries ground-truth labels". AD has no distinguishing property, so AD (and classification) assert `TEST_LABELS` **inline in `_configure`** next to the protocol asserts. There is intentionally NO `required_capabilities()` wrapper — all compat checks live in `_configure`.

## Non-obvious gotchas (these caused real bugs — don't regress)

- **`Segment.val` is loss-shaped, NOT metric-shaped.** The trainer feeds `Segment.val` to `model.compute_loss()` for early-stopping/checkpoint val loss, so it must have the SAME batch format as `train` (clean inputs). The metric evaluation is a SEPARATE path: `get_val_eval_dataset()` / `get_test_dataset()` → `model.predict_step()` → evaluator, yielding `(inputs, target)`. (Imputation once corrupted `Segment.val` with masked windows — bug; val must be clean `SlidingWindowDataset`.)
- **Split arithmetic is centralized** in `framework/datasets/splitting.py` (`baseline_range`, `finetune_ranges`, `all_segment_ranges`, `val_tail_split`). Don't re-inline it in datasets. `SplitConfig.validate(n)` is called in each partitioned dataset's `__init__` and must stay (guards empty/degenerate splits).
- **Incremental val semantics**: `baseline/val` uses the baseline's own val slice (`get_baseline_val_eval_dataset`); each `finetune_i/val` uses that segment's own val slice (`get_finetune_val_eval_dataset(i)`); `merged/val` uses the UNION of every segment's val slice (`get_merged_val_eval_dataset`) so the merge is checked across all regimes with no train leakage.
- **AD random-mask score = mean per-patch error over all `n_patches`** (`error_sum / error_counts.clamp(min=1)` then `.mean(-1)`). Do NOT "fix" the never-masked-token dilution; it's intentional ("use enough eval passes").
- **Forecasting masks exactly `forecast_patches`** (the horizon), gated on `inference_mode == FORECAST` in `_create_mask`. AD-causal masks the back half. Keep these two paths separate (the AD path must not reference `forecast_patches`).
- **`ReferenceEvaluator`** protocol (`needs_reference()` / `set_reference(outputs)`) is the generic "calibration pass" hook; the runner's `collect_reference_outputs()` runs `model.score` over an auxiliary dataset. AD percentile-threshold logic lives in `AdTestEvaluator.set_reference`.
- **Analysis output** goes to `$RUNS_ROOT/analysis/<Dataset>/` (shared across runs, `done.flag`-cached), NOT the per-run dir.
- **Task-vector code lives in `framework/merging/`** — `task_vectors.py` (`task_vector`, `apply_task_vectors`, `merge_task_arithmetic`) and `geometry.py`. There is no `_task_arithmetic_merge` in the pipeline any more. `merge_task_arithmetic` = `apply_task_vectors(base, [task_vector(...)], scale)`; the split exists because a scale sweep builds τ once and re-applies it. **The merge is bitwise reproducible from checkpoints** — recomputing every `merged/checkpoints/best.pt` under `runs/` from its baseline + finetunes gives `torch.equal` on every tensor, 47/47. Use that as the regression test for any change here; it is stronger than any unit test you could write.
- **Checkpoint I/O is confined to `framework/core/checkpoints.py`** (`save_model_state` / `load_model_state`). No `torch.load`/`torch.save` anywhere else. The payload is `{"model_state_dict", **metadata}`; readers must not assume any key beyond `model_state_dict` (a merged checkpoint carries no epoch/loss).
- **`MergeDiagnosticsPipeline` hard-errors on any `dataset_*`/`mae_tx_*` difference from the run it analyses** — dataset args change *which* regions get evaluated, model args change *how* checkpoints score, and either way results would silently stop being comparable. `PARTITION_ARGS` is exempted for the Standard reference run only, because `StandardPipeline` necessarily records `baseline_fraction=1.0`/`n_finetune_segments=0`. Drive it with `python -m incremental_ad.analysis.diagnose --source_run_dir …`, which reads the args back out of the source run's `config.json` rather than trusting a hardcoded copy.
- **`MaeTx` refuses a pretext task with fewer than 4 visible patches** (`_assert_pretext_non_degenerate`, called from `_build`). The count is **mode-aware** — `mask_ratio` is inert outside `RANDOM_MASK` (forecast masks the horizon, causal the back half, next-step one patch), so using it blindly rejects valid causal configs. This rejects 10/15 PSM and 4/9 SWaT `MODEL_SWEEP` cross trials, including the `patch_len=25` config EXPERIMENTS.md §2.3 reported as a winner. Both proven AD recipes sit at *exactly* 4 visible — no margin.
- **L2-SP regularization in `StandardTrainer`** (`reg_lambda`, `reg_exclude`, `reference_state` param on `fit()`): opt-in, no-op by default (`reg_lambda=0`, `reference_state=None`), so it never affects `StandardPipeline` or baseline training. `IncrementalTaskArithmeticPipeline` passes the baseline's weights as `reference_state` when fine-tuning each segment. Empirically found *harmful* on ETTh1 (monotonically worse merged MSE at every `reg_lambda > 0`, every `merge_scale` tried); on SWaT/PSM it made no measurable difference at any `merge_scale` (see `EXPERIMENTS.md`).
- **Grid search / hyperparameter sweeps are SLURM-only** — `slurm_grid_search/` (repo root, sibling to `scripts/`), not run locally anymore. `harness.py` has the generic `Sweep`/`submit_sweep`/`collect_sweep_results` machinery; `sweeps/{swat,psm,etth_forecast}.py` define per-dataset `MODEL_SWEEP` (architecture) + `TRAIN_STANDARD_SWEEP`/`TRAIN_INCREMENTAL_SWEEP` (training params, requires the architecture winner passed in via `--arch-args`). Submission (`submit.py`) never waits for a job — it fires one `sbatch` per trial and returns; collection (`collect.py`) is a separate, idempotent, re-runnable-any-time step that matches run directories back to trials via `config.json` and captures *every* metric found in any `result.json` (not a hand-picked subset — found on PSM that ranking by a single metric like `pa_f1` can disagree with window_f1/AUROC about whether a config helped). Everything generated (sbatch scripts, manifests, results CSVs) lands under `$SLURM_GRID_OUTPUT_ROOT` (default `$WORK/slurm_grid_search`), never inside the repo. Superseded the old local `scripts/_grid_search_harness.py` + `grid_search_ad.py`/`grid_search_etth_forecast.py` (deleted, along with their `scripts/grid_search_results/*.csv` output) — those don't scale to a real architecture search and don't parallelize on a single local GPU.

## Verify after changes

```bash
python -m py_compile $(find src -name "*.py" -not -path "*/__pycache__/*")
python -c "import incremental_ad.project.datasets, incremental_ad.project.models, incremental_ad.framework.trainers, incremental_ad.framework.pipelines, incremental_ad.framework.evaluators; print('OK')"
```
(`git ls-files 'src/**/*.py'` can return stale index paths after `git mv` — use `find` for compile globs.) Full end-to-end runs need HuggingFace dataset downloads (`thuml/Time-Series-Library`) / GPU, so they aren't run locally by default.

## Working-style notes

- Be conservative with anything that changes RNG, data splits, or metric definitions — prefer preserving established behavior unless explicitly asked, and surface behavior changes explicitly.
- Windows host, PowerShell primary; Bash tool available. `num_workers` defaults to 0 on Windows.
