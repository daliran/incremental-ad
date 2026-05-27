# Incremental anomaly detection

Thesis project (AIE master degree) on **time-series anomaly detection** with
self-supervised reconstruction models, working toward an **incremental** setup
where a base model is adapted to new data regimes and the adaptations are merged.

The codebase is organized around three ideas:

- **operations** — the atomic units of work: `train`, `test_eval`, `val_eval` and `merge`.
- **phases** — what you actually launch (one phase = one SLURM job); a phase
  orchestrates one or more operations. Implemented today: `pretrain`
  (train → val_eval → test_eval), `incremental` (pretrain → fine-tune → merge →
  val_eval → test_eval), and `eval` (standalone, on-demand; supports `--split
  test` or `--split val`).
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
* **Peculiarity at eval time** — the anomaly score for a window is the mean
  reconstruction error obtained by running **`n_eval_passes`** forward passes,
  each with a *different random mask*, and averaging the per-token errors
  (weighted by how often each token was masked). This Monte-Carlo style scoring
  reduces the variance of the single-mask estimate. Eval is run with dataset
  **stride = 1** (one window per timestep), which the metrics rely on.

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
job/folder. Both evals window at the dataset's `--swat-eval-stride`.

```bash
python -m incremental_ad.main \
    --phase pretrain --experiment mae_tx_swat \
    --dataset swat --model mae_tx \
    --swat-window-len 100 --swat-stride 50 --swat-normalization standard --swat-val-ratio 0.15 --swat-eval-stride 1 \
    --mae-tx-patch-len 10 --mae-tx-encoder-embed-dim 256 --mae-tx-encoder-layers 2 \
    --mae-tx-encoder-heads 2 --mae-tx-decoder-embed-dim 128 --mae-tx-decoder-layers 1 \
    --mae-tx-decoder-heads 2 --mae-tx-patch-norm false --mae-tx-mask-ratio 0.90 --mae-tx-n-eval-passes 10 \
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
a separate `--finetune-*` config. See
[`scripts/sbatch_mae_tx_incremental.sh`](scripts/sbatch_mae_tx_incremental.sh) for
the full argument list.

### Operations

- **train** (`trainer.py`) — AdamW, `cosine`/`constant` scheduler with linear
  warmup, gradient clipping, early stopping on val loss; writes `best.pt`/`last.pt`
  (plus periodic `epoch_*.pt` if `--train-checkpoint-interval > 0`).
- **val_eval** (`val_evaluator.py`) — reconstruction quality on a val set;
  saves score statistics to `val_reconstruction.json`. The bundled version
  (inside `pretrain`/`incremental`) uses the training slice's own val tail; the
  standalone `eval` phase uses the full training val at `eval_stride`.
- **test_eval** (`test_evaluator.py`) — anomaly detection metrics at **window**,
  **point**, **point-adjusted** and **event** level, saved to `test_results.json`
  (plus ROC/PR curves to wandb). Thresholding via `oracle` or `train_percentile`
  (`--test-eval-threshold-percentile`).
- **merge** (`ops/merge.py`) — task arithmetic over the base + fine-tuned
  checkpoints: clones the base checkpoint and swaps in
  `θ_pre + scale·Σ(θ_FTᵢ − θ_pre)`.

(A fine-tuning is just **train** with `--train-slice ft_<i>` and the base weights
loaded as the starting point; the `incremental` phase orchestrates the lot.)

### Roadmap
The `incremental` phase targets a **static** dataset. A production /
continual-learning phase (real timestamps, day-by-day fine-tuning, selectively
merging accumulated adaptations) will be a separate phase with its own dataset and
logic — the operations (`train`/`eval`/`merge`) and the slicing helpers are reused.

---

## Filesystem — provenance

Output root is `$RUNS_ROOT` (defaults to `experiments/`; set to `$WORK/experiments`
on the cluster). **One job → one immutable folder**, keyed by what produced it:

```
experiments/<experiment>/<phase>/<run_label>/
    checkpoints/best.pt, last.pt          # produced by train
    config.json                           # train run provenance snapshot
    eval_info.json                        # eval run provenance snapshot
    val_reconstruction.json               # val_eval reconstruction stats
    test_results.json                     # test_eval AD metrics
    wandb/                                # local wandb files
```

Concretely:

```
experiments/mae_tx_swat/
    pretrain/100/                          # the pretrain job (train + bundled eval)
    eval/p95_200/                          # a standalone re-eval (its own job)
    incremental/300/                       # the incremental pipeline job
        base/  ft_0/ ft_1/ ft_2/ merged/   # base, fine-tunings, merged model + eval
```

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
  ├─ train-100        (job_type train)
  ├─ val_eval-100     (job_type val_eval, bundled)
  ├─ test_eval-100    (job_type test_eval, bundled)
  └─ test_eval_p95-200  (job_type test_eval, standalone job 200, re-joined the group)
```

An `incremental` job `300` puts all its sub-runs in one group
`mae_tx_swat/incremental/300`: `train_base-300`, `val_eval_base-300`,
`train_ft_0-300`, `val_eval_ft_0-300` … `val_eval_merged-300`,
`test_eval_merged-300`.

**The one asymmetry (by design):** the standalone eval's *folder* lives under its
own job (`eval/p95_200`, provenance), while its *wandb group* points at the model
it evaluated (`pretrain/100`, analysis). The filesystem is keyed by *which
job wrote the bytes*; wandb is grouped by *which model is being studied*. They
coincide for a producing job and diverge only for a standalone eval.

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
[`sbatch_mae_tx_pretrain.sh`](scripts/sbatch_mae_tx_pretrain.sh) (pretrain + eval),
[`sbatch_mae_tx_incremental.sh`](scripts/sbatch_mae_tx_incremental.sh) (pretrain +
fine-tune + merge + eval), and
[`sbatch_mae_tx_eval.sh`](scripts/sbatch_mae_tx_eval.sh) (standalone eval). Local
debug configs (with `WANDB_MODE=disabled`) are in
[`.vscode/launch.json`](.vscode/launch.json).