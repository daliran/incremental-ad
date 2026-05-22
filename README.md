# Incremental anomaly detection

Thesis project (AIE master degree) on **time-series anomaly detection** with
self-supervised reconstruction models, working toward an **incremental** setup
where a base model is adapted to new data regimes and the adaptations are merged.

The codebase is organized around three ideas:

- **operations** — the atomic units of work: `train` and `eval`.
- **phases** — what you actually launch (one phase = one SLURM job); a phase
  orchestrates one or more operations. Implemented today: `pretrain_full`
  (train → eval, bundled) and `eval` (standalone, on-demand).
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
* **Eval** — uses `stride = 1`; for `pretrain_full` this stride is forced
  internally regardless of the training stride.

Config (`--swat-*`): `window-len`, `stride`, `normalization`, `val-ratio`.

---

## Run identity

Every run is identified by four coordinates:

| coordinate     | source                               | example         |
| -------------- | ------------------------------------ | --------------- |
| `experiment`   | `--experiment` (manual namespace)    | `mae_tx_swat`   |
| `phase`        | `--phase` (the launched stage)       | `pretrain_full` |
| `run_id`       | `SLURM_JOB_ID`, or a local timestamp | `43069`         |
| `run_tag`      | `--run-tag` (optional label)         | `p95`           |

`run_label = <run_tag>_<run_id>` if a tag is given, else `<run_id>`.

---

## Phases & operations

`main.py` dispatches on `--phase`:

### `pretrain_full` — train, then eval (one job)
Trains the model on the full train set and immediately evaluates the trained
checkpoint, all in one job/folder. The bundled eval reuses the training dataset
config with stride forced to 1.

```bash
python -m incremental_ad.main \
    --phase pretrain_full --experiment mae_tx_swat \
    --dataset swat --model mae_tx \
    --swat-window-len 100 --swat-stride 50 --swat-normalization standard --swat-val-ratio 0.15 \
    --mae-tx-patch-len 10 --mae-tx-encoder-embed-dim 256 --mae-tx-encoder-layers 2 \
    --mae-tx-encoder-heads 2 --mae-tx-decoder-embed-dim 128 --mae-tx-decoder-layers 1 \
    --mae-tx-decoder-heads 2 --mae-tx-patch-norm false --mae-tx-mask-ratio 0.90 --mae-tx-n-eval-passes 10 \
    --train-seed 42 --train-epochs 300 --train-patience 30 --train-batch-size 64 \
    --train-optimizer adamw --train-weight-decay 1e-2 --train-learning-rate 1e-4 \
    --train-grad-clip 0.5 --train-scheduler cosine --train-warmup-ratio 0.1 --train-checkpoint-interval 0 \
    --eval-seed 0 --eval-split test --eval-batch-size 512 \
    --eval-threshold-strategy oracle --eval-threshold-percentile 99
```

### `eval` — standalone, on-demand
Evaluates an existing checkpoint (e.g. to try a different threshold). The
checkpoint is always passed explicitly; experiment/phase are read back from it.

```bash
python -m incremental_ad.main \
    --phase eval --run-tag p95 --checkpoint <path>/checkpoints/best.pt \
    --dataset swat \
    --swat-window-len 100 --swat-stride 1 --swat-normalization standard --swat-val-ratio 0.15 \
    --eval-seed 0 --eval-split test --eval-batch-size 512 \
    --eval-threshold-strategy oracle --eval-threshold-percentile 95
```

### Operations

- **train** (`trainer.py`) — AdamW, `cosine`/`constant` scheduler with linear
  warmup, gradient clipping, early stopping on val loss; writes `best.pt`/`last.pt`
  (plus periodic `epoch_*.pt` if `--train-checkpoint-interval > 0`).
- **eval** (`evaluator.py`):
    - `--eval-split val` → score statistics, saved to `val_scores.npy`.
    - `--eval-split test` → anomaly metrics at **window**, **point**,
      **point-adjusted** and **event** level, saved to `test_results.json` (plus
      ROC/PR curves to wandb). Thresholding via `oracle` or `train_percentile`
      (`--eval-threshold-percentile`).

### Roadmap (not yet implemented)
`pretrain_partial` (train on a restricted slice + prepare the data split),
`finetune` (continue from a base checkpoint on a data chunk — loads weights only,
fresh optimizer/seed, no RNG restore), and `merge` (combine fine-tunings into a
unified model via task arithmetic, then eval).

---

## Filesystem — provenance

Output root is `$RUNS_ROOT` (defaults to `experiments/`; set to `$WORK/experiments`
on the cluster). **One job → one immutable folder**, keyed by what produced it:

```
experiments/<experiment>/<phase>/<run_label>/
    checkpoints/best.pt, last.pt     # produced by train
    config.json                      # train run provenance snapshot
    eval_info.json                   # eval run provenance snapshot
    test_results.json                # eval metrics (or val_scores.npy for --eval-split val)
    wandb/                           # local wandb files
```

Concretely:

```
experiments/mae_tx_swat/
    pretrain_full/100/               # the pretrain job (train + bundled eval)
    eval/p95_200/                    # a standalone re-eval (its own job)
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
- **job_type** = the operation (`train` / `eval`).
- **name** = `<op>[_<run_tag>]-<run_id>` — uses *this* run's own id.
- **config** = dataset/model/train|eval configs + experiment/phase/op/run_tag/slurm
  id (so you can also group/filter dynamically in the UI).

Example: a pretrain job `100` and a later threshold re-eval `200` (`--run-tag p95`):

```
mae_tx_swat/pretrain_full/100        <- group (one model)
  ├─ train-100        (job_type train)
  ├─ eval-100         (job_type eval, bundled in the same job)
  └─ eval_p95-200     (job_type eval, standalone job 200, re-joined the group)
```

**The one asymmetry (by design):** the standalone eval's *folder* lives under its
own job (`eval/p95_200`, provenance), while its *wandb group* points at the model
it evaluated (`pretrain_full/100`, analysis). The filesystem is keyed by *which
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
[`sbatch_pretrain_full.sh`](scripts/sbatch_pretrain_full.sh) (train + eval) and
[`sbatch_eval.sh`](scripts/sbatch_eval.sh) (standalone eval). Local debug configs
(with `WANDB_MODE=disabled`) are in [`.vscode/launch.json`](.vscode/launch.json).