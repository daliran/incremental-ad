# CLAUDE.md

Agent-oriented context for this repo. Full architecture is in [README.md](README.md) ("Framework internals") — read it for the deep dive; this file is the high-signal map plus the non-obvious gotchas that aren't visible in the code.

## What this is

Research codebase for **incremental anomaly detection on multivariate time series** via task arithmetic, generalized to 4 tasks (AD, forecasting, imputation, classification) on one MAE backbone. Entry point: `python -m incremental_ad.main --model … --dataset … --task … --pipeline …`.

## Architecture invariant (do not break)

- **`framework/`** = reusable batteries-included toolkit: generic contracts **plus** ready-to-use concrete implementations (trainer, evaluators incl. AD, splitting, sliding-window, pipelines, runner).
- **`project/`** = specific to THIS research: the `MaeTx`/`MaeTxClassifier` model, its task configurators, the AD debugger, the concrete datasets, `task.py`, experiment wiring.
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
- **Incremental val semantics**: `baseline/val` uses the baseline's own val slice (`get_baseline_val_eval_dataset`); `merged/val` uses the UNION of every segment's val slice (`get_incremental_val_eval_dataset`) so the merge is checked across all regimes with no train leakage.
- **AD random-mask score = mean per-patch error over all `n_patches`** (`error_sum / error_counts.clamp(min=1)` then `.mean(-1)`). Do NOT "fix" the never-masked-token dilution; it's intentional ("use enough eval passes").
- **Forecasting masks exactly `forecast_patches`** (the horizon), gated on `inference_mode == FORECAST` in `_create_mask`. AD-causal masks the back half. Keep these two paths separate (the AD path must not reference `forecast_patches`).
- **`ReferenceEvaluator`** protocol (`needs_reference()` / `set_reference(outputs)`) is the generic "calibration pass" hook; the runner's `collect_reference_outputs()` runs `model.score` over an auxiliary dataset. AD percentile-threshold logic lives in `AdTestEvaluator.set_reference`.
- **Analysis output** goes to `$RUNS_ROOT/analysis/<Dataset>/` (shared across runs, `done.flag`-cached), NOT the per-run dir.

## Verify after changes

```bash
python -m py_compile $(find src -name "*.py" -not -path "*/__pycache__/*")
python -c "import incremental_ad.project.datasets, incremental_ad.project.models, incremental_ad.framework.trainers, incremental_ad.framework.pipelines, incremental_ad.framework.evaluators; print('OK')"
```
(`git ls-files 'src/**/*.py'` can return stale index paths after `git mv` — use `find` for compile globs.) Full end-to-end runs need HuggingFace dataset downloads (`thuml/Time-Series-Library`) / GPU, so they aren't run locally by default.

## Working-style notes

- Be conservative with anything that changes RNG, data splits, or metric definitions — prefer preserving established behavior unless explicitly asked, and surface behavior changes explicitly.
- Windows host, PowerShell primary; Bash tool available. `num_workers` defaults to 0 on Windows.
