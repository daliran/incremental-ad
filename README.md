# incremental-ad

Research framework for **incremental anomaly detection on multivariate time series**. The core idea: a model trained on a baseline period can be incrementally adapted to new operating conditions via task arithmetic — without forgetting the original distribution.

The framework is general enough to support four task types on the same backbone: **anomaly detection**, **forecasting**, **imputation**, and **classification**.

---

## Setup

Activate the virtual environment and set the environment variables before running anything:

```bash
source .venv/bin/activate
```

| variable | purpose | default |
|----------|---------|---------|
| `RUNS_ROOT` | root directory for all run output | `runs/` |
| `WANDB_PROJECT` | wandb project name; leave unset to disable wandb | — |
| `WANDB_ENTITY` | wandb entity | wandb default |
| `WANDB_MODE` | `online` / `offline` / `disabled` | wandb default |
| `HF_HOME` | HuggingFace cache for dataset downloads | HF default |
| `SLURM_JOB_ID` | set by SLURM; used as `run_id` | timestamp + hex suffix |

---

## Tasks

| task | `--task` | supervision | eval metric |
|------|----------|-------------|-------------|
| Anomaly detection | `ad` | self-supervised | AUROC, F1, PA-F1 |
| Forecasting | `forecast` | self-supervised | MSE, MAE |
| Imputation | `imputation` | self-supervised | MSE, MAE at masked positions |
| Classification | `classification` | supervised | accuracy, macro-F1 |

---

## Running experiments

Every run is launched via `python -m incremental_ad.main` with a set of `--component_arg` flags. The entry point selects the model, dataset, task, and pipeline by name; each component declares and reads its own arguments. Full argument lists for each experiment are in the SLURM scripts and the VS Code launch configurations.

### Pipelines

**`StandardPipeline`** — single-phase training followed by evaluation. Use this for non-incremental baselines: train on all available data, then evaluate on val and test.

**`IncrementalTaskArithmeticPipeline`** — the main research pipeline. Trains a baseline model on the first fraction of the training data, then fine-tunes independently on each subsequent segment. The fine-tuned models are merged back into the baseline via task arithmetic:

```
θ_merged = θ_base + scale × Σᵢ (θ_ft_i − θ_base)
```

The scale is controlled by `--pipeline_merge_scale`. Each fine-tuned model contributes a *task vector* (the difference from baseline); these are summed and added back. Baseline and fine-tuned states are stored on CPU between phases so only the active model occupies GPU memory. Baseline, each fine-tune segment, and the merged model are all evaluated the same way — val (on their own segment's held-out slice) and test (on the global test set) — before/after the merge.

**`EvalPipeline`** — loads a saved checkpoint and runs evaluation only. Useful for re-evaluating a trained model with a different threshold strategy or for running the debugger visualizations. Pass `--pipeline_checkpoint_path` and the same model/dataset args used during training.

### SLURM scripts

Ready-to-submit scripts live in [`scripts/`](scripts/). Edit `PROJECT_ROOT` and `WORK` at the top of each script before submitting. Use [`scripts/debug.sh`](scripts/debug.sh) to open an interactive GPU session on the cluster.

| script | dataset | pipeline |
|--------|---------|----------|
| [`sbatch_mae_tx_swat_ad_standard.sh`](scripts/sbatch_mae_tx_swat_ad_standard.sh) | SWaT | StandardPipeline |
| [`sbatch_mae_tx_swat_ad_incremental.sh`](scripts/sbatch_mae_tx_swat_ad_incremental.sh) | SWaT | IncrementalTaskArithmeticPipeline |
| [`sbatch_mae_tx_swat_ad_eval.sh`](scripts/sbatch_mae_tx_swat_ad_eval.sh) | SWaT | EvalPipeline |
| [`sbatch_mae_tx_psm_ad_standard.sh`](scripts/sbatch_mae_tx_psm_ad_standard.sh) | PSM | StandardPipeline |
| [`sbatch_mae_tx_psm_ad_incremental.sh`](scripts/sbatch_mae_tx_psm_ad_incremental.sh) | PSM | IncrementalTaskArithmeticPipeline |
| [`sbatch_mae_tx_psm_ad_eval.sh`](scripts/sbatch_mae_tx_psm_ad_eval.sh) | PSM | EvalPipeline |
| [`sbatch_mae_tx_etth_forecast_standard.sh`](scripts/sbatch_mae_tx_etth_forecast_standard.sh) | ETTh1 | StandardPipeline |
| [`sbatch_mae_tx_etth_forecast_incremental.sh`](scripts/sbatch_mae_tx_etth_forecast_incremental.sh) | ETTh1 | IncrementalTaskArithmeticPipeline |
| [`sbatch_mae_tx_etth_forecast_eval.sh`](scripts/sbatch_mae_tx_etth_forecast_eval.sh) | ETTh1 | EvalPipeline |
| [`sbatch_mae_tx_etth_imputation_standard.sh`](scripts/sbatch_mae_tx_etth_imputation_standard.sh) | ETTh1 | StandardPipeline |
| [`sbatch_mae_tx_etth_imputation_eval.sh`](scripts/sbatch_mae_tx_etth_imputation_eval.sh) | ETTh1 | EvalPipeline |

### Local debugging

[`.vscode/launch.json`](.vscode/launch.json) contains debug configurations for all datasets, tasks, and pipelines — including synthetic datasets for quick smoke tests. `WANDB_MODE` is set to `disabled` in all local configs.

---

## Run outputs

Each run writes to `$RUNS_ROOT/<experiment_name>/<run_id>/`.

```
<run_dir>/
├── config.json             # full reproducibility record (all args + git commit)
├── run.log
│
├── baseline/               # IncrementalTaskArithmeticPipeline
│   ├── result.json
│   ├── checkpoints/
│   ├── val/result.json
│   └── test/result.json
├── finetune_0/ ... finetune_N/
│   ├── result.json
│   ├── checkpoints/
│   ├── val/result.json
│   └── test/result.json
├── merged/
│   ├── checkpoints/best.pt
│   ├── val/result.json
│   └── test/result.json
│
└── train/                  # StandardPipeline
    ├── result.json
    ├── checkpoints/
    ├── val/result.json
    └── test/result.json
```

Checkpoint paths inside `result.json` are stored relative to `run_dir` so run directories remain self-contained when moved.

Dataset analysis artifacts are written once per dataset to `$RUNS_ROOT/analysis/<DatasetName>/` and reused across runs and experiments (a `done.flag` short-circuits regeneration). The bundle — produced by the reusable `framework/datasets/analysis.py` on raw/unscaled data — includes `feature_stats.csv` (per-feature mean/std/min/max/percentiles, split into normal/anomaly with delta-z when labels exist), `anomaly_catalog.csv` (per-segment difficulty + top driving features, labelled datasets), per-feature time-series PDFs (`{train,test}_timeseries.pdf`, with ±2σ bands and anomaly shading), and z-score heatmaps (`{train,test}_heatmap.png`, sorted by temporal drift or anomaly detectability).

Wandb is opt-in: set `WANDB_PROJECT` to enable it. Training metrics use a `step_offset`-adjusted step counter so all phases share one x-axis. Final metrics from each eval step are written to `wandb.run.summary`. ROC and PR curves are logged as native wandb charts by the AD test evaluator.

---

---

## Framework internals

This section covers the design of the codebase — how the pieces fit together, the choices made, and how to extend it. `framework/` is a reusable, batteries-included toolkit: generic contracts **plus** off-the-shelf concrete implementations (trainer, evaluators, datasets building blocks) that any project can use as-is. `project/` holds what is specific to *this* research — the model under study, its configurators, and the experiment wiring.

### Codebase layout

```
src/incremental_ad/
├── framework/              # Reusable toolkit — contracts + ready-to-use implementations
│   ├── contracts/          # Abstract base classes (the pluggable interfaces)
│   ├── core/               # Stateless utilities (seed, device, git, wandb)
│   ├── datasets/           # Reusable building blocks (SlidingWindowDataset, splitting, analysis)
│   ├── evaluators/         # EvaluationRunner + reusable evaluators (AD, forecasting, imputation, classification)
│   ├── pipelines/          # Concrete pipeline implementations
│   ├── trainers/           # Concrete trainer implementations
│   └── experiment.py       # Orchestrator: wires components and runs
│
├── project/                # Specific to this project
│   ├── datasets/           # Swat, Psm, EtthForecastDataset, EtthImputationDataset,
│   │                       # TestForecastDataset, TestImputationDataset, TestClassificationDataset
│   ├── models/mae_tx/      # MaeTx + MaeTxClassifier, configurators, debugger
│   └── task.py             # Task enum (ad, forecast, imputation, classification)
│
└── main.py                 # CLI entry point
```

The invariant is the **dependency direction**: `project/` imports from `framework/`, never the reverse — that's what keeps the framework reusable. Within that rule, a concrete class lives in `framework/` if it's reusable beyond this project (it depends only on framework contracts) or in `project/` if it's bespoke; code can be *promoted* `project → framework` once it proves general, with no call-site changes since everything goes through contracts/registries. Adding a new model, dataset, or evaluator means adding files and importing them in the relevant `__init__.py` — `main.py` never changes.

### `Configurable` — the CLI contract

Every pluggable component implements two class methods:

- **`add_args(parser, prefix)`** — registers the component's CLI arguments into the shared parser. Every arg is prefixed with the component's `ARG_PREFIX` (e.g., `MaeTx` uses `mae_tx_`, so patch size is `--mae_tx_patch_len`).
- **`from_config(cfg, prefix)`** — reads those args back from the parsed namespace and constructs the component.

The `prefix` parameter handles the case where the same class is instantiated twice in one run. `IncrementalTaskArithmeticPipeline` uses `StandardTrainer` twice: once under `--trainer_*` (baseline) and once under `--finetune_trainer_*` (fine-tuning). Both sets of args are fully independent — there is no fallback or inheritance between them.

`main.py` uses two-pass argument parsing: a minimal pre-parser first identifies which model, dataset, task, and pipeline are selected, then the full parser is built with all component-specific arguments. This means every component declares and reads its own args; `main.py` never changes when new components are added.

### Registry pattern

Every concrete component class self-registers the moment it is defined, via `__init_subclass__`. `Dataset`, `Model`, `Pipeline`, `Trainer`, and `TaskModelConfigurator` each have a class-level `_registry` dict. `main.py` imports `project/` so the classes are defined (and registered), then looks them up by the string names passed on the CLI. No switch statements, no hard-coded lists — adding a new dataset means creating the class and importing it.

`TaskModelConfigurator` is keyed by a `(task_name, type[Model])` pair, where `task_name` is a plain string — the framework stays agnostic of the concrete task taxonomy (which lives in `project/task.py`). The right configurator is selected automatically for each task/model combination, and the valid `--task` choices are derived from the registered configurators rather than a hard-coded list.

### Component contracts

#### `Model`

Models must not construct any `nn.Module` layers in `__init__` — only configuration is stored there. `_build()` is called by `TaskModelConfigurator.configure()` after the dataset has injected its properties (`n_features`, `seq_len`, etc.). This ensures architecture dimensions are derived from the data at runtime, not hard-coded.

Training and inference are separated into two methods: `compute_loss(batch)` returns a scalar for the optimizer; `predict_step(batch)` returns a `(output, target)` 2-tuple consumed by the evaluator. Batch format depends on the task — plain tensors for unsupervised tasks, `(inputs, labels)` for supervised ones.

#### `Dataset` and `PartitionedDataset`

The dataset contract distinguishes between training-phase methods and evaluation-phase methods. Training-phase methods (`get_train_segment`, `get_baseline`, `get_incremental_segments`) return `Segment(train, val)` pairs windowed at the configurable `stride`, so early stopping conditions match training. Evaluation methods (`get_val_eval_dataset`, `get_train_eval_dataset`, `get_test_dataset`) always use stride=1 to ensure every timestep is covered.

`get_test_dataset()` yields `(inputs, target)` tuples. Batch formats by task:

| task | inputs | target |
|------|--------|--------|
| AD | `window [W, F]` | `anomaly_labels [W]` |
| Forecasting | `full_window [W+H, F]` | `future [H, F]` |
| Imputation | `masked_window [W, F]` | `(original [W, F], visible_idx [n_visible])` |
| Classification | `window [W, F]` | `label []` |

`PartitionedDataset` adds the incremental split: `--dataset_baseline_fraction` and `--dataset_baseline_use_fraction` control how much training data the baseline sees; `--dataset_n_finetune_segments` controls how many equal fine-tuning chunks are carved from the remainder.

Three `@runtime_checkable` protocols let configurators verify that a dataset exposes the properties they need (`forecast_len`, `mask_patch_len`, `n_classes`). A dataset only needs to implement the protocols relevant to the tasks it supports.

#### `Task`

`Task` is a string enum that lives in `project/task.py` — the framework never imports it. The configurator registry is keyed by the task's *string* value, so the set of tasks is a project concern and the framework just routes by name. A plain string works equally well as a key; the enum exists only for readable registration (`@TaskModelConfigurator.register(Task.AD, MaeTx)`).

#### `TaskModelConfigurator`

The bridge between a specific task and a specific model. Does four things:

1. **`_configure(model, dataset)`** — injects dataset-derived values into the model (`n_features`, `seq_len`, etc.) and sets task-specific model state (`inference_mode`, `forecast_patches`, `n_classes`). Always called before `model._build()`.
2. **`create_val_evaluator()`** — returns the evaluator for training-time validation.
3. **`create_test_evaluator()`** — returns the evaluator for final test evaluation.
4. **`create_debugger()`** — optionally returns a `Debugger` for post-eval visualizations (AD and forecasting; enabled via `--configurator_debug`).

`_configure()` is also where each configurator asserts dataset compatibility, in one place: the structural protocol checks (`isinstance(dataset, ForecastDataset)`, etc.) plus any `DatasetCapability` the task needs — e.g. AD and classification assert `TEST_LABELS` (the dataset's test set carries ground-truth labels). An incompatible (task, model, dataset) pairing fails fast here, before the model is built.

Registered configurators:

| configurator | task | model |
|-------------|------|-------|
| `MaeTxAdConfigurator` | `ad` | `MaeTx` |
| `MaeTxForecastingConfigurator` | `forecast` | `MaeTx` |
| `MaeTxImputationConfigurator` | `imputation` | `MaeTx` |
| `MaeTxClassificationConfigurator` | `classification` | `MaeTxClassifier` |

#### `StandardTrainer`

Owns device management for training: `model.to(device)` is called inside `fit()`. Pipelines and models are device-agnostic. Creates its own `DataLoader`s from `DataLoaderConfig` — pipelines pass the config, not pre-built loaders.

`secondary_loaders` are named val loaders evaluated every epoch but not used for early stopping. Used in `IncrementalTaskArithmeticPipeline` to track each fine-tuning segment's val loss during baseline training on one wandb chart. `step_offset` is added to the local epoch number when logging, giving all training phases a single monotonic x-axis.

`IncrementalTaskArithmeticPipeline` registers `StandardTrainer` twice: `--trainer_*` for the baseline and `--finetune_trainer_*` for fine-tuning. All args must be explicitly stated for both — there is no fallback.

#### `EvaluationRunner`

The counterpart to `StandardTrainer` for evaluation. Owns device management for inference: `model.to(device)` is called inside `run()`. Pipelines and models stay device-agnostic throughout.

`run()` resets the evaluator, optionally runs a reference pass over an auxiliary dataset for evaluators that configure themselves from it (via `ReferenceEvaluator` — e.g. AD threshold calibration), iterates the dataset, and returns the computed metrics. The evaluator retains its accumulated state after return so the caller can log wandb charts or run the debugger.

`EvalPipeline` exposes `--runner_device` via `EvaluationRunner.add_args`. `StandardPipeline` and `IncrementalTaskArithmeticPipeline` inherit the device from the trainer.

#### `Evaluator`

Pure metric accumulator: `reset()` before an eval pass, `update(outputs)` once per batch, `compute()` at the end. The evaluator knows nothing about devices, data loading, or model internals.

Three optional capability protocols are detected with `isinstance()` at call sites:

| protocol | purpose |
|----------|---------|
| `ReferenceEvaluator` | configure state from a reference pass (e.g. AD threshold from the training score distribution) |
| `WandbChartsEvaluator` | log native wandb ROC/PR curves after eval |
| `DebugDataEvaluator` | expose scores/labels/threshold to the debugger |

Evaluators by task:

| task | val evaluator | test evaluator |
|------|--------------|----------------|
| AD | `AdValEvaluator` — reconstruction error stats | `AdTestEvaluator` — AUROC, window/point/event F1, PA-F1 |
| Forecasting | `ForecastingEvaluator` — MSE/MAE/RMSE | same |
| Imputation | `ImputationEvaluator` — MSE/MAE/RMSE at masked positions | same |
| Classification | `ClassificationEvaluator` — accuracy, macro-F1 | same |

#### `Debugger`

Called after test evaluation when `create_debugger()` returns a non-None instance. `MaeTxAdDebugger` (AD) retrieves scores/labels via `DebugDataEvaluator.debug_data()` and writes a score timeline (normal/anomaly scatter), score distributions, an event-detection CSV (per segment: max/min score, margin to threshold, detected), and FP/FN/TP reconstruction samples. `MaeTxForecastDebugger` (forecasting) plots sample windows as context + actual-vs-predicted future, falling back to the val set when there is no test set. Both write to `step_dir/debug/` and are enabled with `--configurator_debug`.

---

### Models

#### `MaeTx` — masked autoencoder (AD, forecasting, imputation)

Windows are split into non-overlapping patches (`patch_len` timesteps × all features). The encoder processes only the *visible* tokens — the asymmetric MAE design where the heavy encoder sees a small subset while the lightweight decoder reconstructs all positions via learned mask tokens. Loss is MSE over masked patches only. When `patch_norm=True`, ground-truth patches are normalized per-patch before MSE, making the loss focus on shape rather than magnitude.

Three training modes (`--mae_tx_training_mode`):

| mode | visible tokens | masked tokens | use case |
|------|---------------|--------------|---------|
| `random_mask` | random `1-mask_ratio` fraction | remaining patches | AD, imputation |
| `causal_mask` | first half | second half (AD) / forecast horizon (forecasting) | forecasting |
| `next_step` | all but last | last patch | next-step prediction |

At inference, the mode set by the configurator determines how the model scores a window:

- **AD** (`random_mask`): `n_eval_passes` forward passes with independent random masks are run per batch. Per-patch errors accumulate across passes and are averaged — a Monte Carlo estimate of the expected reconstruction error. The final window score is the mean per-patch error. For `causal_mask`/`next_step`, a single deterministic pass is used.
- **Forecasting**: the context patches are visible; the model predicts the forecast-horizon patches (`forecast_len // patch_len`). Training masks exactly those horizon patches (causal), so the masked span seen during training matches what is predicted at inference. Only the forecast positions are extracted from the decoder output. With `--mae_tx_instance_norm` (default on, forecast-only) each window is normalized by its visible-context mean/std before encoding and the forecast is de-normalized afterwards — RevIN-style instance normalization that removes per-window level/trend shift (the dominant error on non-stationary series like ETTh1). It is leakage-free (stats come only from the context) and a no-op for AD/imputation.
- **Imputation**: the dataset applies a fixed deterministic mask per window (seeded by window index) and provides `visible_idx` alongside the masked input. The model encodes only visible patches and predicts at masked positions; the evaluator compares predictions vs. original only at those positions.

#### `MaeTxClassifier` — supervised classification

Extends the same encoder backbone with a mean-pooled linear classification head. No decoder is built — the model is strictly encoder-only, making it smaller than `MaeTx` for the same encoder configuration. Training is supervised with cross-entropy. Only encoder CLI args are required; decoder and masking args are not used.

---

### Datasets

The real datasets use a sliding window over a `[T, F]` time series, with configurable `window_len` and `stride`. Standard scaling is fitted on training data only.

| dataset | `--dataset` | features | anomaly rate |
|---------|------------|----------|--------------|
| SWaT | `Swat` | 51 | ~12% |
| PSM | `Psm` | 25 | ~28% |
| ETTh1 | `EtthForecastDataset` / `EtthImputationDataset` | 7 | — |

SWaT and PSM ship separate train/test splits with per-timestep anomaly labels. ETTh1 is a single continuous series with no separate test split and no labels, so `EtthForecastDataset`/`EtthImputationDataset` carve a chronological test set from the tail via `--dataset_test_fraction` (default 0.2; set `0` for train+val only), fitting the scaler on the train portion to avoid leakage.

For incremental (`PartitionedDataset`) runs, each fine-tune segment carves its own val tail, so `val_fraction × segment_size` must exceed `window_len` — otherwise the val loader is empty and training fails fast with a clear error (the SLURM/launch configs document the arithmetic inline).

Data is loaded from HuggingFace (`thuml/Time-Series-Library`) on first use. PSM has scattered NaN values (~2% of rows) that are forward- then backward-filled before scaling.

Four synthetic datasets (`TestForecastDataset`, `TestImputationDataset`, `TestClassificationDataset`) generate deterministic multivariate sine waves for end-to-end pipeline smoke tests without downloading anything.

---

### Extending the framework

#### Adding a new dataset

Subclass `TimeSeriesDataset` + `PartitionedDataset` (or just `Dataset` for non-incremental use), implement the required data-access methods, and import the class in `project/datasets/__init__.py`. The class self-registers on import — nothing else changes.

#### Adding a new model

Subclass `Model` (or `_MaeTxBase` to reuse the encoder backbone), create a `TaskModelConfigurator` subclass decorated with `@TaskModelConfigurator.register(Task.X, MyModel)`, and import both in `project/models/__init__.py`.

#### Adding a new task

Add a value to the `Task` enum in `project/task.py`, create a `TaskModelConfigurator` subclass for `(Task.NEW, SomeModel)`, and implement the task's evaluators — in `framework/evaluators/` if they're reusable (they only depend on the `Evaluator` contract), or in `project/` if bespoke. If the dataset needs to expose extra properties to the configurator, add a `@runtime_checkable` protocol to `framework/contracts/dataset.py`.

#### Adding a new pipeline

Subclass `Pipeline`, implement `add_args` (calling `DataLoaderConfig.add_args`, `StandardTrainer.add_args`, and `EvaluationRunner.add_args` as needed alongside any pipeline-specific args), `from_config`, and `run`. Import in `framework/pipelines/__init__.py`.
