# Incremental anomaly detection

Thesis project (AIE master degree) on **time-series anomaly detection** with
self-supervised reconstruction models, working toward an **incremental** setup
where a base model is adapted to new data regimes and the adaptations are merged.

The codebase is organized around three ideas:

- **operations** — the atomic units of work: `train`, `val_eval`, `train_slice_val_eval`, `test_eval` and `merge`. Each op is a pure function: it receives a model config and pre-built `DataLoader`s from the calling phase, and owns only the training/eval logic, wandb init, and file writes.
- **phases** — what you actually launch (one phase = one SLURM job); a phase
  orchestrates one or more operations and owns all dataset/loader construction
  (via `model_dataset_factory`). Implemented today: `pretrain`
  (train → train_slice_val_eval → test_eval), `incremental` (base pretrain →
  fine-tune × N → merge → train_slice_val_eval → test_eval), and `eval`
  (standalone, on-demand; supports `--split test` or `--split val`).
- **registries** — the available models and datasets, injected from `main.py`.

---

## Models

### Transformer MAE (`mae_tx`)

A masked auto-encoder over patched multivariate time series. Windows are split
into non-overlapping **patches** (`patch_len` timesteps × all features); a random
subset is masked and the model reconstructs the masked patches.

* **Architecture**
    * Fixed sequence length.
    * Learned encoder positional encoding.
    * Learned decoder positional encoding (different from the encoder's).
    * Learned decoder mask token.
    * Learned projection between encoder `d_model` and decoder `d_model`.
* **AE encoder/decoder**
    * Pre-norm instead of post-norm.
    * FFN dim = `4 * d_model`.
    * Encoder `d_model` > decoder `d_model` (asymmetric, MAE-style: the encoder
      only sees the *visible* tokens, the decoder sees all positions).
* **Training loss**
    * Computed only on the masked patches/tokens.
    * Optional patch-level normalization of the ground-truth patches
      (`patch_norm`) — makes the loss focus on shape, not magnitude.
    * MSE between predicted and ground-truth patches.
* **Peculiarity at eval time** — the anomaly score for a window is computed
  from the per-patch reconstruction errors obtained by running
  **`n_eval_passes`** forward passes, each with a *different random mask*, and
  averaging each patch's error over the passes it was masked in (weighted
  average). This Monte-Carlo style scoring reduces the variance of the
  single-mask estimate. The window score is the mean of the per-patch errors.

  Eval is run with dataset **stride = 1** (one window per timestep), which the
  metrics rely on.

Config (`--mae-tx-*`): `patch-len`, `encoder-embed-dim`, `encoder-layers`,
`encoder-heads`, `decoder-embed-dim`, `decoder-layers`, `decoder-heads`,
`mask-ratio`, `patch-norm`, `n-eval-passes`.

---

## Datasets

### SWaT (`swat`)

Secure Water Treatment testbed (51 features), loaded from the HuggingFace
`thuml/Time-Series-Library` dataset. The `train` split is normal-only; the
`test` split carries `Normal/Attack` labels.

* **Normalization** — `standard` (per-feature, fit on train) or `none`.
* **Windowing** — sliding windows of `window-len`, advanced by `stride`.
* **Validation** — the last `val-ratio` fraction of the train series is held out
  (temporal split, no shuffling).
* **Eval** — windowing uses a separate `eval-stride` (set it to `1`, which the
  metrics rely on). Training windows at `stride`; `build_for_eval` windows at
  `eval-stride`.

Config (`--swat-*`): `window-len`, `stride`, `normalization`, `val-ratio`,
`eval-stride`.

### PSM (`psm`)

Pooled Server Metrics dataset (25 features), loaded from the HuggingFace
`thuml/Time-Series-Library` dataset. Unlike SWaT, data and labels are in
separate configs (`PSM-data` / `PSM-label`); the label split is named
`test_label` with a binary `label` column (0 = normal, 1 = anomaly).
The train split contains scattered NaN values (~2 % of rows) that are
forward-filled then backward-filled before scaling.

* **Normalization** — `standard` (per-feature, fit on train) or `none`.
* **Windowing / Validation / Eval** — same scheme as SWaT.

Config (`--psm-*`): `window-len`, `stride`, `normalization`, `val-ratio`,
`eval-stride`.

---

## Run identity

Every run is identified by four coordinates:

| coordinate     | source                               | example         |
| -------------- | ------------------------------------ | --------------- |
| `experiment`   | `--experiment` (manual namespace)    | `mae_tx_swat`   |
| `phase`        | `--phase` (the launched stage)       | `pretrain` |
| `run_id`       | `SLURM_JOB_ID`, or a local timestamp | `43069`         |
| `run_tag`      | `--run-tag` (optional label)         | `p95`           |

`run_label = <run_tag>_<run_id>` if a tag is given, else `<run_id>`.

---

## Phases & operations

`main.py` dispatches on `--phase`:

### `pretrain` — train, then val eval, then test eval (one job)
Trains the model on the full train set, evaluates reconstruction quality on the
training val set, then evaluates anomaly detection on the test set — all in one
job/folder. Both evals window at the dataset's `eval-stride` config (e.g. `--swat-eval-stride`).

```bash
python -m incremental_ad.main \
    --phase pretrain --experiment mae_tx_swat \
    --dataset swat --model mae_tx \
    --swat-window-len 100 --swat-stride 50 --swat-normalization standard --swat-val-ratio 0.15 --swat-eval-stride 1 \
    --mae-tx-patch-len 10 --mae-tx-encoder-embed-dim 256 --mae-tx-encoder-layers 2 \
    --mae-tx-encoder-heads 2 --mae-tx-decoder-embed-dim 128 --mae-tx-decoder-layers 1 \
    --mae-tx-decoder-heads 2 --mae-tx-patch-norm false --mae-tx-mask-ratio 0.80 --mae-tx-n-eval-passes 30 \
    --train-seed 42 --train-epochs 300 --train-patience 30 --train-batch-size 64 \
    --train-optimizer adamw --train-weight-decay 1e-2 --train-learning-rate 1e-4 \
    --train-grad-clip 0.5 --train-scheduler cosine --train-warmup-ratio 0.1 --train-checkpoint-interval 0 \
    --val-eval-seed 0 --val-eval-batch-size 512 \
    --test-eval-seed 0 --test-eval-batch-size 512 \
    --test-eval-threshold-strategy oracle --test-eval-threshold-percentile 99
```

### `eval` — standalone, on-demand
Evaluates an existing checkpoint. The checkpoint is always passed explicitly;
experiment/phase are read back from it. Use `--split test` for anomaly detection
metrics or `--split val` for reconstruction quality on the full training val set.

```bash
# Test eval (AD metrics, e.g. to try a different threshold)
python -m incremental_ad.main \
    --phase eval --run-tag p95 --checkpoint <path>/checkpoints/best.pt \
    --dataset swat --split test \
    --swat-window-len 100 --swat-stride 1 --swat-normalization standard --swat-val-ratio 0.15 --swat-eval-stride 1 \
    --test-eval-seed 0 --test-eval-batch-size 512 \
    --test-eval-threshold-strategy oracle --test-eval-threshold-percentile 95

# Val eval (reconstruction quality)
python -m incremental_ad.main \
    --phase eval --run-tag val_reconstruction --checkpoint <path>/checkpoints/best.pt \
    --dataset swat --split val \
    --swat-window-len 100 --swat-stride 1 --swat-normalization standard --swat-val-ratio 0.15 --swat-eval-stride 1 \
    --val-eval-seed 0 --val-eval-batch-size 512
```

### `incremental` — the whole pipeline in one job
On a static dataset: pretrain a base on the first `--partial-ratio` of the train
series, fine-tune it on each of the `--n-finetune` remaining equal chunks, merge
the fine-tunings into the base via task arithmetic
(`θ_pre + --merge-scale·Σ(θ_FTᵢ − θ_pre)`), then evaluate the merged model — all in
one job. Outputs land in `base/`, `ft_0/`…`ft_{N-1}/`, `merged/` under one run dir
and share one wandb group. The base pretrain uses `--train-*`; the fine-tunings use
a separate `--finetune-*` config.

During base pretraining, the train portion of each ft split is passed as a
**secondary val loader**. This logs `loss/ft_0/train`, `loss/ft_1/train`, … to
wandb every epoch alongside `loss/train` and `loss/val`, so you can watch how
the base model generalises to unseen future data as training progresses. These
curves do not affect early stopping or checkpointing.

`--base-data-ratio` controls what fraction of the base's allocated slice
(`--partial-ratio`) the base model actually trains on. The FT chunks are always
anchored at `partial_ratio × n` regardless of this value — only the base sees less
data. `--base-data-ratio 1.0` is the baseline (full allocation); e.g. `0.4` with
`--partial-ratio 0.5` on 100 k timesteps gives the base 20 k timesteps while the
FTs still start at 50 k.

See [`scripts/sbatch_mae_tx_swat_incremental.sh`](scripts/sbatch_mae_tx_swat_incremental.sh)
for the full argument list.

### `model_dataset_factory` — the three building blocks

All dataset and model construction goes through three functions:

| Function | Returns | Stride used |
|---|---|---|
| `build_model(model_name, dataset_name, model_cfg, dataset_cfg)` | `BaseModel` | — |
| `build_datasets(dataset_name, dataset_cfg, train_slice, …)` | `(train_ds, val_ds)` | `dataset_cfg.stride` |
| `build_eval_datasets(dataset_name, dataset_cfg, split)` | `(train_ds, eval_ds)` | `dataset_cfg.eval_stride` |

`build_datasets` accepts any `train_slice` (`"full"`, `"base"`, `"ft_<i>"`).
`build_eval_datasets` always loads the full train series and returns either the
val tail or the test set based on `split`. Phases call these before invoking
ops; ops never touch the factory for datasets.

### Operations

- **train** (`trainer.py`) — AdamW, `cosine`/`constant` scheduler with linear
  warmup, gradient clipping, early stopping on primary val loss; writes
  `best.pt`/`last.pt` (plus periodic `epoch_*.pt` if
  `--train-checkpoint-interval > 0`). Accepts an optional
  `secondary_val_loaders` dict: each entry is evaluated every epoch and logged
  to wandb under `loss/<key>` without affecting checkpointing or early stopping.
- **train_slice_val_eval** (`val_evaluator.py`) — reconstruction quality on the val
  tail of the specific training slice used during training; bundled inside
  `pretrain`/`incremental` to inspect what the trainer actually saw. Writes
  `train_slice_val_eval_info.json` + `val_reconstruction.json`.
- **val_eval** (`val_evaluator.py`) — reconstruction quality on the full training val
  series at `eval_stride`; used by the `eval` phase on-demand. Writes
  `val_eval_info.json` + `val_reconstruction.json`.
- **test_eval** (`test_evaluator.py`) — anomaly detection metrics at **window**,
  **point**, **point-adjusted** and **event** level, saved to `test_results.json`
  (plus ROC/PR curves to wandb). Receives both a `train_loader` (for threshold
  fitting) and an `eval_loader` (test set). Thresholding via `oracle` or
  `train_percentile` (`--test-eval-threshold-percentile`).
- **merge** (`ops/merge.py`) — task arithmetic over the base + fine-tuned
  checkpoints: clones the base checkpoint and swaps in
  `θ_pre + scale·Σ(θ_FTᵢ − θ_pre)`.

(A fine-tuning is just **train** with the base weights loaded as the starting
point; the `incremental` phase orchestrates the lot.)

### Roadmap
The `incremental` phase targets a **static** dataset. A production /
continual-learning phase (real timestamps, day-by-day fine-tuning, selectively
merging accumulated adaptations) will be a separate phase with its own dataset and
logic — the operations (`train`/`eval`/`merge`) and the slicing helpers are reused.

---

## Filesystem — provenance

Output root is `$RUNS_ROOT` (defaults to `experiments/`; set to `$WORK/experiments`
on the cluster). **One job → one immutable folder**, keyed by what produced it.

### File layout

```
experiments/<experiment>/<phase>/<run_label>/
    phase_config.json                     # global identity + provenance + phase-specific params
    wandb/                                # local wandb files

    # produced by train + train_slice_val_eval (bundled inside pretrain/incremental):
    config.json                           # dataset/model/train config for this op
    checkpoints/best.pt, last.pt
    train_slice_val_eval_info.json        # checkpoint + eval config (slice-specific val eval)
    val_reconstruction.json               # reconstruction score statistics

    # produced by a bundled test_eval op:
    eval_info.json                        # checkpoint section + eval section
    test_results.json                     # AD metrics (window/point/event)

    # produced by val_eval (standalone eval phase, --split val):
    val_eval_info.json                    # checkpoint section + eval section
    val_reconstruction.json               # reconstruction score statistics
```

For the `incremental` phase, each sub-step gets its own sub-directory:

```
experiments/mae_tx_swat/
    pretrain/100/                          # pretrain job: train → val_eval → test_eval
        phase_config.json
        config.json
        checkpoints/
        train_slice_val_eval_info.json
        val_reconstruction.json
        eval_info.json
        test_results.json

    eval/p95_200/                          # standalone re-eval job
        phase_config.json                  # includes split + checkpoint path
        val_eval_info.json                 # (--split val) checkpoint + eval section
        val_reconstruction.json            # (--split val) reconstruction stats
        eval_info.json                     # (--split test) checkpoint + eval section
        test_results.json                  # (--split test) AD metrics

    incremental/300/                       # incremental pipeline job
        phase_config.json                  # includes partial_ratio, n_finetune, merge_scale
        base/                              # partial pretrain + train_slice_val_eval
            config.json
            checkpoints/
            train_slice_val_eval_info.json
            val_reconstruction.json
        ft_0/  ft_1/  ft_2/               # fine-tune steps, same layout as base/
        merged/                            # merged model + train_slice_val_eval + test_eval
            checkpoints/
            train_slice_val_eval_info.json
            val_reconstruction.json
            eval_info.json
            test_results.json
```

### Config hierarchy

Two levels, no duplication between them:

| level | file | contents |
|---|---|---|
| phase | `phase_config.json` | global identity (experiment, phase, run_id), provenance (host, git commit, timestamp), phase-specific params (partial_ratio, n_finetune, merge_scale, …) |
| op | `config.json` / `val_eval_info.json` / `eval_info.json` | dataset + model + op-specific config; eval files split into a `checkpoint` section (what the model was trained with) and an `eval` section (what this eval run used) |

Cross-phase references are explicit checkpoint paths (an eval or a future
fine-tuning is given the `--checkpoint` to read).

---

## wandb — analysis

`project = $WANDB_PROJECT`, `entity = $WANDB_ENTITY`.

- **group** = `<experiment>/<phase>/<group_run_id>` — **one group per trained
  model and all of its evaluations**. `group_run_id` is the *producing* job's id:
  a training run uses its own id; an eval uses the id of the checkpoint it
  evaluates, so re-evals re-join the model's group.
- **job_type** = the operation (`train` / `val_eval` / `test_eval`).
- **name** = `<op>[_<run_tag>]-<run_id>` — uses *this* run's own id.
- **config** = dataset/model/train|eval configs + experiment/phase/op/run_tag/slurm
  id (so you can also group/filter dynamically in the UI).

Example: a pretrain job `100` and a later threshold re-eval `200` (`--run-tag p95`):

```
mae_tx_swat/pretrain/100        <- group (one model)
  ├─ train-100                    (job_type train)
  ├─ train_slice_val_eval-100     (job_type train_slice_val_eval, bundled)
  ├─ test_eval-100                (job_type test_eval, bundled)
  └─ test_eval_p95-200            (job_type test_eval, standalone job 200, re-joined the group)
```

An `incremental` job `300` puts all its sub-runs in one group
`mae_tx_swat/incremental/300`: `train_base-300`, `train_slice_val_eval_base-300`,
`train_ft_0-300`, `train_slice_val_eval_ft_0-300` …
`train_slice_val_eval_merged-300`, `test_eval_merged-300`.

**The one asymmetry (by design):** the standalone eval's *folder* lives under its
own job (`eval/p95_200`, provenance), while its *wandb group* points at the model
it evaluated (`pretrain/100`, analysis). The filesystem is keyed by *which
job wrote the bytes*; wandb is grouped by *which model is being studied*. They
coincide for a producing job and diverge only for a standalone eval.

---

## Analysis

`analysis/analyze_dataset.py` produces a set of offline diagnostics for a
given dataset. Outputs land in `debug/dataset/<name>/` by default.

| file | contents |
|---|---|
| `stats.csv` | Per-feature descriptive stats for train, test-normal, test-anomaly, and anomaly Δz |
| `anomaly_events.csv` | Contiguous anomaly segments ranked by hardness (lowest mean \|z\| first) |
| `train_timeseries.pdf` | Multi-page PDF of the train signal per feature with ±2σ reference lines |
| `timeseries.pdf` | Multi-page PDF of the test signal with anomaly shading and ±2σ reference lines |
| `train_heatmap.png` | Features × time z-score heatmap of the train set; features sorted by temporal drift (useful for choosing split boundaries) |
| `anomaly_heatmap.png` | Features × time z-score heatmap of the test set; features sorted by anomaly detectability |

```bash
python analysis/analyze_dataset.py --dataset swat
python analysis/analyze_dataset.py --dataset psm --out-dir debug/psm_run1
```

Supported datasets: `swat`, `psm`. Adding a new one requires implementing
`load_<name>() -> DatasetBundle` and registering it in `LOADERS`.

---

## Environment

| variable        | purpose                              | default        |
| --------------- | ------------------------------------ | -------------- |
| `RUNS_ROOT`     | output root for run folders          | `experiments`  |
| `WANDB_PROJECT` | wandb project                        | —              |
| `WANDB_ENTITY`  | wandb entity                         | —              |
| `WANDB_MODE`    | `online` / `offline` / `disabled`    | wandb default  |
| `HF_HOME`       | HuggingFace cache (dataset download) | HF default     |
| `SLURM_JOB_ID`  | set by SLURM; becomes the `run_id`   | timestamp      |

Cluster submission scripts live in [`scripts/`](scripts/):

| script | dataset | phase |
|---|---|---|
| [`sbatch_mae_tx_swat_pretrain.sh`](scripts/sbatch_mae_tx_swat_pretrain.sh) | SWaT | pretrain + eval |
| [`sbatch_mae_tx_swat_incremental.sh`](scripts/sbatch_mae_tx_swat_incremental.sh) | SWaT | pretrain + fine-tune + merge + eval |
| [`sbatch_mae_tx_swat_eval.sh`](scripts/sbatch_mae_tx_swat_eval.sh) | SWaT | standalone eval |
| [`sbatch_mae_tx_psm_pretrain.sh`](scripts/sbatch_mae_tx_psm_pretrain.sh) | PSM | pretrain + eval |
| [`sbatch_mae_tx_psm_incremental.sh`](scripts/sbatch_mae_tx_psm_incremental.sh) | PSM | pretrain + fine-tune + merge + eval |
| [`sbatch_mae_tx_psm_eval.sh`](scripts/sbatch_mae_tx_psm_eval.sh) | PSM | standalone eval |

Local debug configs (with `WANDB_MODE=disabled`) are in
[`.vscode/launch.json`](.vscode/launch.json).