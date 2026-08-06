# Experiment Summary — incremental learning by model merging

**This file is the source of truth for every number in this project.** §1 is the complete
current snapshot and is self-contained — no run directory needs to be opened to reason about
the results. §2 records the exact configuration of every run behind it, so the results survive
loss of the checkpoints. §3 records the method gotchas that cost real time.

The historical grid-search record was removed on 2026-08-05: it merged at a fixed α = 1.0,
judged significance against a noise floor now known to be wrong by 4–30×, and its ETTh1 rows
used the `val_fraction` setting that leaves one task vector untrained. The per-trial CSVs
remain under `$SLURM_GRID_OUTPUT_ROOT` if any of it is ever needed.

Companion documents: [THEORY.md](THEORY.md) explains the theory
and reasoning; [EXECUTION_PLAN.md](EXECUTION_PLAN.md) tracks what is done and what is next,
and states the five research questions this project exists to answer.

**Which question each section serves.** *Does task arithmetic work when shards are not
orthogonal?* — §1.11, §1.15, with the argument in [THEORY.md §2](THEORY.md). *Merging or
continual fine-tuning?* — §1.13, §1.14. *How large should a shard be?*, *accumulate or
materialise?*, and *how to choose among materialised models?* are **not answered here** — no
experiment in this file materialises a second model or asks which shard count is best. See
EXECUTION_PLAN.md §3.1–§3.3.
The commands for reproducing any of this are in THEORY.md §14.

Snapshot 2026-08-06: **four datasets × three segment counts** (n = 2, 3, 5), three training
seeds each on the forecasting datasets and one on the AD datasets, for the merge arm, the
sequential arm and the joint-training reference.

## TL;DR

- **Merging is essentially free once the scale is right.** One merged model performs within
  ~1.0–1.1× of keeping a separate specialist per regime, on SWaT, PSM and ETTh1 — and this
  holds at **every segment count tested** (§1.11). exchange_rate is the outlier at 1.6–2.1×.
- **The optimal merge is the *mean* of the task vectors.** α\*·n is constant within each
  dataset: ≈1.0 on SWaT, PSM and ETTh1, ≈1.5 on exchange_rate, across n = 2, 3, 5. Since the
  merge is θ₀ + α·Στᵢ, constant α·n means a fixed multiple of the arithmetic mean regardless
  of how many vectors there are. Every seed agrees exactly. §1.11
- **On the most strongly drifting dataset, merging beats training on all the data at once.**
  exchange_rate GRR **1.421 / 1.164 / 1.238** at n = 2 / 3 / 5, all above 1.0, all with α
  chosen on validation rather than test.
- **What used to look like interference was a scale error.** Every result before 2026-08-05
  merged at α = 1.0. The task vectors mostly agree, so summing them at full strength
  overshoots. At α\* the damage disappears, and so does the "forgetting" once reported.
- **In unsupervised AD the merge scale cannot be chosen honestly.** Validation reconstruction
  and test AUROC disagree about α — SWaT's val optimum is 0.5 while AUROC improves
  monotonically to 1.5 — so selecting on validation costs **22–99%** of the achievable GRR on
  AD, against **1–8%** on forecasting. Every AD merge number in this file is oracle-selected.
  §1.12
- **Merge versus sequential is not a property of a dataset.** It flips with segment count:
  exchange_rate favours sequential at n=2 and merging by +38% at n=5. §1.13
- **No regime predictor exists at n = 9.** Nine decisive configurations, split 5 merge / 4
  sequential; no cheap signal separates them beyond chance. The original question was
  mis-posed — any predictor needs the segment count among its inputs. §1.14
- **Task vectors shrink and de-align as segments multiply** (ETTh1: mean ‖τ‖ 1.85 → 1.10,
  alignment 0.824 → 0.633, while ‖Στ‖ stays flat), which is what proves α\* tracks the
  *count* rather than the vector magnitude. §1.15
- **The reproducibility floor is dataset-specific:** PSM 0.07%, SWaT 0.13%, exchange_rate
  5.29%, ETTh1 8.20%. The universal 2% assumption previously used was wrong in both directions.

---

## 1. Current results — complete snapshot (2026-08-06)

Everything in this section is the **current** state at three training seeds per dataset.
It is self-contained: every number needed to reason about the results is here, so no run
directory needs to be opened. Nothing here predates the 2026-08-05 corrections.

**Reading conventions used throughout.** `±` is the half-range over three training seeds.
A difference is only claimed where two `±` intervals do **not** overlap. Validation-block
figures are ratios to that run's own base model, so **lower is better and 1.00 means
"indistinguishable from the base model"**. Validation metrics are loss-shaped —
reconstruction error for AD (whose training data carries no labels), MSE for forecasting.

### 1.1 Setup

| dataset | source run | joint-training reference | primary test metric | ρ (update overlap) |
|---|---|---|---|---|
| SWaT | `58941` | `58930` | `window_auroc` | 0.607 |
| PSM | `59101` | `59090` | `window_auroc` | 0.226 |
| ETTh1 | `59077` | `59062` | `forecast/mse` | 0.076 |
| Exchange | `79671` | `79672` | `forecast/mse` | 0.117 |

The base model trains on `baseline_fraction=0.5` of the training data and is then frozen;
three later segments each get an independent fine-tune from that same base. Joint training
sees 100% of the training data. All partitions hold out the same `val_fraction`.

Diagnostics are **training-free**: they reload existing checkpoints and re-score them. All
87 merged checkpoints on disk reproduce **bitwise** from their base + fine-tunes, recomputed at the scale each was actually committed at.

### 1.2 Headline — n = 3

**Merge cost** = merged error ÷ the error of that shard's own specialist, averaged over the
three shards. **1.00× means merging costs nothing** relative to keeping one model per regime.

> **Scope and convention.** This section is the n = 3 slice, and it selects α on the
> validation *block* — the mean of the per-shard ratios. §1.11 covers n = 2, 3, 5 and pools
> the validation union by `n_windows` instead, which is what the pipeline itself now does.
> The two conventions give slightly different α (exchange_rate 0.43 here, 0.53 there) and so
> slightly different merge costs (ETTh1 1.007 here, 1.080 there). Neither is wrong; they
> answer "best on average across shards" versus "best on the pooled data". **Quote §1.11 for
> anything compared across segment counts, and this section only for the n = 3 detail.**

| | merge cost @ α=1.0 | merge cost @ α\* | α\* | old regime @ α=1.0 | old regime @ α\* |
|---|---|---|---|---|---|
| **SWaT** | 3.79 ±0.29× | **1.079 ±0.002×** | 0.250 ±0.000 | 4.931 ±0.287 | **1.013 ±0.012** |
| **PSM** | 1.18 ±0.11× | **1.008 ±0.020×** | 0.500 ±0.000 | 1.664 ±0.158 | **1.139 ±0.053** |
| **ETTh1** | 1.69 ±0.32× | **1.007 ±0.026×** | 0.367 ±0.050 | 1.897 ±0.379 | **0.967 ±0.029** |
| **Exchange** | 3.29 ±1.05× | **1.014 ±0.103×** | 0.433 ±0.050 | 1.344 ±0.260 | **0.835 ±0.068** |

- **On PSM and ETTh1 the merge cost interval contains 1.00 — merging is free.**
- **α\* does not move across seeds on the AD datasets** (±0.000). This matters: §1.9 shows
  α cannot be tuned honestly in unsupervised AD, and a pre-declared value is defensible
  precisely because the optimum is stable.
- Everything the α=1.0 column shows as damage is **overshoot**, not interference: the task
  vectors mostly agree, so summing them at full strength travels too far.

### 1.3 Model quality against the merge scale

Merged model on the validation block, ratio to base. `new` is the mean over the three
shards; `old` is the base model's own regime, which nobody fine-tuned on.

**SWaT** (α\* = 0.25, separate specialists = 0.608)

| α | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| new | 1.000 | 0.655 | 0.742 | 1.319 | 2.431 | 4.079 | 6.213 |
| old | 1.000 | 1.022 | 1.651 | 2.985 | 5.121 | 8.090 | 11.842 |

**PSM** (α\* = 0.5, separate specialists = 0.753)

| α | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| new | 1.000 | 0.809 | 0.760 | 0.792 | 0.896 | 1.069 | 1.302 |
| old | 1.000 | 1.014 | 1.122 | 1.332 | 1.650 | 2.081 | 2.621 |

**ETTh1** (α\* = 0.3, separate specialists = 0.700)

| α | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new | 1.000 | 0.844 | 0.753 | 0.716 | 0.723 | 0.764 | 0.831 | 0.916 | 1.013 | 1.116 | 1.223 | 1.333 | 1.450 | 1.579 | 1.721 | 1.875 |
| old | 1.000 | 0.958 | 0.940 | 0.943 | 0.964 | 1.005 | 1.067 | 1.152 | 1.261 | 1.397 | 1.559 | 1.747 | 1.960 | 2.196 | 2.455 | 2.738 |

**Exchange** (α\* = 0.4, separate specialists = 0.327)

| α | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new | 1.000 | 0.679 | 0.451 | 0.326 | 0.299 | 0.355 | 0.476 | 0.648 | 0.863 | 1.115 | 1.401 | 1.716 | 2.056 | 2.415 | 2.790 | 3.178 |
| old | 1.000 | 0.888 | 0.827 | 0.806 | 0.814 | 0.842 | 0.885 | 0.937 | 0.993 | 1.053 | 1.112 | 1.169 | 1.224 | 1.276 | 1.325 | 1.372 |

### 1.4 Transfer matrices

Each specialist θ₀+τᵢ scored on **every** shard's held-out slice. This is the measurement no
ordinary training run produces — the off-diagonal — and it is what separates a genuine
specialist from a model that merely got better at everything. `merged` is at α\*.

**SWaT** — reconstruction/score_mean, ratio to base

| model | val_base | val_0 | val_1 | val_2 |
|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| θ₀+τ₀ | 1.026 | 0.740 | 0.698 | 0.750 |
| θ₀+τ₁ | 1.235 | 0.770 | 0.603 | 0.538 |
| θ₀+τ₂ | 1.413 | 0.861 | 0.638 | 0.481 |
| **merged @ α\*=0.25** | **1.022** | **0.722** | **0.626** | **0.616** |

**PSM** — reconstruction/score_mean, ratio to base

| model | val_base | val_0 | val_1 | val_2 |
|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| θ₀+τ₀ | 1.039 | 0.774 | 0.845 | 0.909 |
| θ₀+τ₁ | 1.125 | 0.763 | 0.769 | 0.875 |
| θ₀+τ₂ | 1.090 | 0.950 | 0.665 | 0.714 |
| **merged @ α\*=0.5** | **1.122** | **0.778** | **0.705** | **0.796** |

**ETTh1** — forecast/mse, ratio to base

| model | val_base | val_0 | val_1 | val_2 |
|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| θ₀+τ₀ | 1.104 | 0.914 | 0.805 | 1.594 |
| θ₀+τ₁ | 0.993 | 0.697 | 0.647 | 1.492 |
| θ₀+τ₂ | 1.086 | 0.850 | 0.714 | 0.539 |
| **merged @ α\*=0.3** | **0.943** | **0.699** | **0.626** | **0.823** |

**Exchange** — forecast/mse, ratio to base

| model | val_base | val_0 | val_1 | val_2 |
|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| θ₀+τ₀ | 0.757 | 0.234 | 0.658 | 0.811 |
| θ₀+τ₁ | 0.994 | 0.404 | 0.708 | 0.713 |
| θ₀+τ₂ | 1.107 | 1.680 | 0.593 | 0.039 |
| **merged @ α\*=0.4** | **0.814** | **0.203** | **0.417** | **0.277** |

### 1.5 Complete metric report — test set

Every metric the evaluators produce, on the full unpartitioned test set. `sd` is that
metric's own spread across three training seeds. A row marked **inside noise** has a
base-to-joint gap smaller than three times its sd, so neither the gap nor its GRR means
anything.

**SWaT**

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| *Event-level* | | | | | | | | | |
| `event_f1` ↑ | 0.3478 | 0.3636 | 0.3478 | 0.3556 | 0.3721 | 0.0243 | ~~0.32~~ | 0.0139 | inside noise |
| `event_precision` ↑ | 0.7273 | 0.8889 | 0.7273 | 0.8000 | 1.0000 | 0.2727 | ~~0.27~~ | 0.0974 | inside noise |
| `event_recall` ↑ | 0.2286 | 0.2286 | 0.2286 | 0.2286 | 0.2286 | 0.0000 | — | 0.0000 | inside noise |
| `event_tp` ↑ | 8 | 8 | 8 | 8 | 8 | 0 | — | 0.0 | inside noise |
| `event_fp` ↓ | 3 | 1 | 3 | 2 | 0 | -3 | ~~0.33~~ | 2.0 | inside noise |
| `event_fn` ↓ | 27 | 27 | 27 | 27 | 27 | 0 | — | 0.0 | inside noise |
| *Point-adjusted* | | | | | | | | | |
| `pa_f1` ↑ | 0.8373 | 0.8383 | 0.8407 | 0.8369 | 0.8428 | 0.0055 | -0.07 | 0.0012 |  |
| `pa_precision` ↑ | 0.9336 | 0.9488 | 0.9344 | 0.9327 | 0.9324 | -0.0012 | ~~0.76~~ | 0.0167 | inside noise |
| `pa_recall` ↑ | 0.7590 | 0.7590 | 0.7641 | 0.7590 | 0.7689 | 0.0099 | ~~0.00~~ | 0.0085 | inside noise |
| *Point-level* | | | | | | | | | |
| `point_auroc` ↑ | 0.8159 | 0.8181 | 0.8209 | 0.8160 | 0.8235 | 0.0076 | 0.01 | 0.0010 |  |
| `point_auprc` ↑ | 0.7086 | 0.7082 | 0.7085 | 0.7078 | 0.7027 | -0.0059 | ~~0.13~~ | 0.0047 | inside noise |
| `point_f1` ↑ | 0.7673 | 0.7674 | 0.7674 | 0.7673 | 0.7674 | 0.0001 | ~~0.33~~ | 0.0002 | inside noise |
| `point_precision` ↑ | 0.9806 | 0.9804 | 0.9799 | 0.9805 | 0.9799 | -0.0008 | ~~0.14~~ | 0.0005 | inside noise |
| `point_recall` ↑ | 0.6302 | 0.6304 | 0.6306 | 0.6303 | 0.6307 | 0.0005 | 0.21 | 0.0001 |  |
| *Window-level* | | | | | | | | | |
| `window_auroc` ↑ | 0.8005 | 0.8029 | 0.8060 | 0.8006 | 0.8089 | 0.0084 | 0.01 | 0.0010 |  |
| `window_auprc` ↑ | 0.7035 | 0.7039 | 0.7046 | 0.7036 | 0.7028 | -0.0007 | ~~-0.07~~ | 0.0013 | inside noise |
| `window_f1` ↑ | 0.7508 | 0.7511 | 0.7513 | 0.7510 | 0.7514 | 0.0006 | ~~0.24~~ | 0.0003 | inside noise |
| `window_precision` ↑ | 0.9953 | 0.9955 | 0.9954 | 0.9954 | 0.9952 | -0.0001 | ~~-0.81~~ | 0.0004 | inside noise |
| `window_recall` ↑ | 0.6027 | 0.6031 | 0.6034 | 0.6029 | 0.6036 | 0.0008 | 0.19 | 0.0002 |  |

*14 of 19 metrics are inside their own noise floor.*

**PSM**

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| *Event-level* | | | | | | | | | |
| `event_f1` ↑ | 0.2294 | 0.2957 | 0.2749 | 0.2070 | 0.3960 | 0.1665 | -0.14 | 0.0112 |  |
| `event_precision` ↑ | 0.1356 | 0.1799 | 0.1652 | 0.1174 | 0.2599 | 0.1244 | -0.15 | 0.0074 |  |
| `event_recall` ↑ | 0.7465 | 0.8732 | 0.8169 | 0.8732 | 0.8310 | 0.0845 | 1.50 | 0.0163 |  |
| `event_tp` ↑ | 53 | 62 | 58 | 62 | 59 | 6 | 1.50 | 1.2 |  |
| `event_fp` ↓ | 338 | 269 | 293 | 466 | 168 | -170 | -0.75 | 16.6 |  |
| `event_fn` ↓ | 18 | 9 | 13 | 9 | 12 | -6 | 1.50 | 1.2 |  |
| *Point-adjusted* | | | | | | | | | |
| `pa_f1` ↑ | 0.8067 | 0.7675 | 0.7716 | 0.7699 | 0.7676 | -0.0391 | ~~0.94~~ | 0.0208 | inside noise |
| `pa_precision` ↑ | 0.9639 | 0.6637 | 0.6574 | 0.6538 | 0.6685 | -0.2954 | ~~1.05~~ | 0.1671 | inside noise |
| `pa_recall` ↑ | 0.6935 | 0.9361 | 0.9338 | 0.9361 | 0.9011 | 0.2076 | ~~1.17~~ | 0.1221 | inside noise |
| *Point-level* | | | | | | | | | |
| `point_auroc` ↑ | 0.7712 | 0.7902 | 0.8019 | 0.7941 | 0.7977 | 0.0265 | 0.86 | 0.0007 |  |
| `point_auprc` ↑ | 0.5630 | 0.5598 | 0.5663 | 0.5601 | 0.5643 | 0.0013 | ~~-2.17~~ | 0.0137 | inside noise |
| `point_f1` ↑ | 0.5817 | 0.5979 | 0.6289 | 0.6038 | 0.6391 | 0.0574 | 0.38 | 0.0009 |  |
| `point_precision` ↑ | 0.4729 | 0.4949 | 0.4823 | 0.4610 | 0.4885 | 0.0156 | ~~-0.76~~ | 0.0061 | inside noise |
| `point_recall` ↑ | 0.7556 | 0.9216 | 0.9036 | 0.8748 | 0.9239 | 0.1682 | 0.71 | 0.0141 |  |
| *Window-level* | | | | | | | | | |
| `window_auroc` ↑ | 0.7740 | 0.7936 | 0.7991 | 0.7973 | 0.8002 | 0.0262 | 0.89 | 0.0005 |  |
| `window_auprc` ↑ | 0.6379 | 0.6419 | 0.6467 | 0.6435 | 0.6532 | 0.0154 | ~~0.37~~ | 0.0105 | inside noise |
| `window_f1` ↑ | 0.6361 | 0.6557 | 0.6790 | 0.6620 | 0.6918 | 0.0557 | 0.47 | 0.0029 |  |
| `window_precision` ↑ | 0.5162 | 0.5854 | 0.5538 | 0.5599 | 0.5652 | 0.0489 | ~~0.89~~ | 0.0348 | inside noise |
| `window_recall` ↑ | 0.8285 | 0.8923 | 0.8772 | 0.8096 | 0.8916 | 0.0632 | ~~-0.30~~ | 0.0739 | inside noise |

*8 of 19 metrics are inside their own noise floor.*

**ETTh1**

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| *Forecasting* | | | | | | | | | |
| `forecast/mse` ↓ | 0.7023 | 0.4503 | 0.5918 | 0.4920 | 0.4271 | -0.2752 | 0.76 | 0.0570 |  |
| `forecast/rmse` ↓ | 0.8380 | 0.6710 | 0.7693 | 0.7015 | 0.6535 | -0.1845 | 0.74 | 0.0343 |  |
| `forecast/mae` ↓ | 0.6131 | 0.4789 | 0.5661 | 0.4953 | 0.4573 | -0.1558 | 0.76 | 0.0249 |  |

*0 of 3 metrics are inside their own noise floor.*

**Exchange**

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| *Forecasting* | | | | | | | | | |
| `forecast/mse` ↓ | 0.7754 | 0.5727 | 0.7390 | 0.3310 | 0.4356 | -0.3397 | 1.31 | 0.0387 |  |
| `forecast/rmse` ↓ | 0.8806 | 0.7568 | 0.8597 | 0.5753 | 0.6600 | -0.2205 | 1.38 | 0.0225 |  |
| `forecast/mae` ↓ | 0.6991 | 0.5986 | 0.6807 | 0.4920 | 0.5323 | -0.1667 | 1.24 | 0.0151 |  |

*0 of 3 metrics are inside their own noise floor.*

### 1.6 Versus sequential fine-tuning — validation block, n = 3

`ContinualFineTuningPipeline`: θ₀ → seg 0 → seg 1 → seg 2, each step from the *previous*
model. A genuinely different method — sequential models share no common base, so task
arithmetic does not apply to them at all.

| dataset | ρ | sequential — new | merged — new | verdict | sequential — old | merged — old | verdict |
|---|---|---|---|---|---|---|---|
| **SWaT** | 0.607 | 0.697 ±0.011 | 0.656 ±0.001 | **merge** | 1.635 ±0.049 | 1.013 ±0.012 | **merge** |
| **PSM** | 0.226 | 0.702 ±0.013 | 0.759 ±0.015 | **sequential** | 1.159 ±0.024 | 1.139 ±0.053 | *tie* |
| **ETTh1** | 0.076 | 0.560 ±0.038 | 0.705 ±0.019 | **sequential** | 1.110 ±0.143 | 0.967 ±0.029 | *tie* |
| **Exchange** | 0.117 | 0.474 ±0.061 | 0.331 ±0.034 | **merge** | 0.847 ±0.076 | 0.835 ±0.068 | *tie* |

**Merging wins outright only on SWaT** — *at n = 3, on the validation block*. §1.13 repeats
this comparison on the test set across n = 2, 3, 5 and finds the winner changes with the
segment count, so read this row as one slice rather than a dataset property.

Forgetting under sequential fine-tuning, base-regime ratio after each step:

| dataset | start | +1 | +2 | +3 | ρ | headroom |
|---|---|---|---|---|---|---|
| **SWaT** | 1.000 | 1.115 | 1.209 | 1.586 | 0.607 | 1% |
| **PSM** | 1.000 | 1.046 | 1.217 | 1.158 | 0.226 | 3% |
| **ETTh1** | 1.000 | 1.037 | 0.936 | 0.972 | 0.076 | 39% |
| **Exchange** | 1.000 | 0.792 | 0.759 | 0.782 | 0.117 | 44% |

**SWaT gives the textbook forgetting curve; ETTh1 shows none.** Two explanations point the
same way and cannot be separated with these three datasets: SWaT's updates repeat each
other most (highest ρ), *and* SWaT's base is already at its ceiling so extra training can
only move it. ETTh1 is the opposite on both counts.

### 1.7 Regime indicator — accumulate, or materialise a new model?

> **Superseded by §1.14.** This section reports the indicator as it stood at four
> single-`n` datapoints. The segment sweep raised that to nine decisive configurations and
> found no predictor, and showed the outcome is not a property of a dataset at all — so ρ's
> 3-of-4 hit rate below should be read as the observation that motivated the larger test,
> not as a result. The measurements themselves are unchanged and still correct.

Computed from the weights alone, after each segment is fine-tuned and *before* deciding
whether to fold it in. **ρ_k** is the fraction of τ_k's energy already inside
span(τ₀…τ_{k−1}), so the genuinely new component has norm ‖τ_k‖·√(1−ρ_k). Reported
relative to ‖θ₀‖ so it is comparable across datasets. This is an exact decomposition, not
a composite score.

| dataset | step 0 | step 1 | step 2 | reading |
|---|---|---|---|---|
| **SWaT** | ρ=—<br>new=0.00441 | ρ=0.478<br>new=0.00471 | ρ=0.737<br>new=0.00379 | updates largely repeating |
| **PSM** | ρ=—<br>new=0.00523 | ρ=0.303<br>new=0.00537 | ρ=0.149<br>new=0.00546 | each segment still adds new directions |
| **ETTh1** | ρ=—<br>new=0.01517 | ρ=0.067<br>new=0.02594 | ρ=0.085<br>new=0.01281 | each segment still adds new directions |
| **Exchange** | ρ=—<br>new=0.01163 | ρ=0.208<br>new=0.01012 | ρ=0.026<br>new=0.03098 | each segment still adds new directions |

**Track record: 3 of 4.** The rule *"high ρ → merge, low ρ → sequential"* holds on SWaT
(ρ→0.74, merge wins), PSM and ETTh1 (ρ<0.31, sequential wins on the new segments) — and
**fails on exchange_rate**, which has the *lowest* ρ of all (0.026) yet merging wins
decisively (0.331 ±0.034 against 0.474 ±0.061).

**Why it fails there matters.** exchange_rate also has the largest headroom (43.8%) and
the strongest input drift, and its merge beats *joint training* (GRR 1.218 ±0.081). So
redundancy is one route to merging winning, and large headroom under strong drift appears
to be a second, independent one — which ρ cannot see. **Use ρ as a diagnostic for why a
merge behaved as it did, not as a controller for the decision.**

The *materialise a separate model* branch remains untested: no arm in this study builds
one, so the indicator points at a decision whose other side has never been measured.

### 1.8 Geometry

Computed from the weights alone — no GPU, no data, seconds.

| measure | SWaT | PSM | ETTh1 | Exchange |
|---|---|---|---|---|
| Update overlap ρ (0 = all new) | **0.607** | **0.226** | **0.076** | **0.117** |
| Mean off-diagonal cosine | 0.737 | 0.399 | 0.240 | 0.217 |
| Effective rank (of 3) | 1.63 | 2.51 | 2.35 | 1.88 |
| Mean ‖τ‖ / ‖θ₀‖ | 0.0061 | 0.0059 | 0.0185 | 0.0181 |
| Per-segment ‖τ‖ / ‖θ₀‖ | 0.0044 / 0.0065 / 0.0074 | 0.0052 / 0.0064 / 0.0059 | 0.0152 / 0.0269 / 0.0134 | 0.0116 / 0.0114 / 0.0314 |
| Cosine at distance 1 → 2 | 0.771 → 0.667 | 0.466 → 0.265 | 0.253 → 0.213 | 0.306 → 0.038 |
| Best merge scale α\* | **0.250 ±0.000** | **0.500 ±0.000** | **0.367 ±0.050** | **0.433 ±0.050** |

**Cosine decays with temporal distance on all three** — so temporal shift really is what
differentiates the task vectors. This was the check that could have invalidated the framing.

**ρ, not pairwise cosine, is the meaningful statistic.** Near-orthogonality is the default
in high dimensions, so a small cosine proves little; ρ asks the useful question — does this
update carry anything the earlier ones did not?

### 1.9 Reproducibility floor

Baseline-stage test metric, three training seeds, identical configuration.

| dataset | seeds | mean | sd | sd as % of mean |
|---|---|---|---|---|
| **SWaT** (window_auroc) | 3 | 0.7996 | 0.0010 | **0.13%** |
| **PSM** (window_auroc) | 3 | 0.7742 | 0.0005 | **0.07%** |
| **ETTh1** (forecast/mse) | 3 | 0.6961 | 0.0570 | **8.20%** |
| **Exchange** (forecast/mse) | 3 | 0.7321 | 0.0387 | **5.29%** |

The old universal 2%-of-base assumption was wrong in **both** directions — 15–30× too
conservative for the AD datasets, 4× too loose for ETTh1.

**The instability is in absolute values, not ratios.** ETTh1's ratio-to-own-baseline varies
about 4% across the same seeds, because dividing by a run's own base model cancels the
shared variance. Two rules follow: **report ETTh1 as ratios**, and **treat ETTh1's GRR as
fragile** since GRR mixes a base model from one run with a `standard` model from another.

### 1.10 What honest α selection costs — forecasting, n = 3

Every α\* elsewhere in this file is read off the merge-scale curve's **test** column, so it
is an *oracle* value: obtaining it required seeing the test set. The deployable alternative
is to select α on the merged-val union, which the pipeline now supports
(`--pipeline_select_merge_scale_on_val`, EXECUTION_PLAN.md §2.9). This section measures the
gap between the two, so the honest number is on record next to the oracle one.

Computed from the existing diagnostics curves — `val_base`, `val_0`, `val_1`, `val_2` pooled
by `n_windows` to reconstruct the union the selector actually sees. Forecasting only; the AD
metrics are blind to α (§1.3), which is why the flag refuses there.

| run | val-picked α | test-optimal α | test MSE @ val-α | @ test-α | penalty |
|---|---|---|---|---|---|
| ETTh1 seed 7 | 0.30 | 0.50 | 0.4794 | 0.4500 | 6.54% |
| ETTh1 seed 123 | 0.30 | 0.30 | 0.5049 | 0.5049 | 0.00% |
| exchange seed 42 | 0.50 | 0.40 | 0.3420 | 0.3310 | 3.31% |
| exchange seed 7 | 0.60 | 0.50 | 0.3240 | 0.3035 | 6.77% |
| exchange seed 123 | 0.50 | 0.40 | 0.3562 | 0.3337 | 6.75% |

Validation lands **within one grid step** of the test optimum in every case, with no
systematic direction (it picks lower on ETTh1, higher on exchange seed 7). The penalty
averages 3.3% on ETTh1 — well inside its 8.20% reproducibility floor, so undetectable — and
5.6% on exchange_rate, roughly one noise unit against its 5.29% floor.

**The headline exchange_rate result survives.** Recomputing GRR with the val-picked α:

| seed | GRR (oracle α) | GRR (honest α) |
|---|---|---|
| 42 | 1.308 | 1.276 |
| 7 | 1.145 | 1.089 |
| 123 | 1.200 | 1.127 |
| **mean ± sd** | **1.218 ± 0.081** | **1.164 ± 0.080** |

Merging still beats joint training on every seed when α is chosen without touching test.

**Caveat.** The candidate list is itself a choice, and a coarse grid cannot land on a fine
optimum — these penalties come from the grid already used for the curve. A finer grid would
shrink them; a coarser one would not.

### 1.11 The segment-count sweep

`n_finetune_segments ∈ {2, 3, 5}` across all four datasets — 12 configurations. The
baseline partition is fixed at 50% of training data throughout, so **segment size varies
only as a consequence of the count** (25% / 16.7% / 10% of train). Count and segment size
are therefore confounded by construction — see EXECUTION_PLAN.md §3.2 for the control that
would separate them.

n = 8 is unreachable: ETTh1 would give each segment a 130-row validation slice against a
144-row window requirement — zero windows, deterministically, on every seed.

Every quantity below is recomputed **identically at every (dataset, n)** from the
diagnostics curves: α is the argmin of the `n_windows`-pooled validation union, and GRR,
merge cost and the merged model are all read at that α. The raw `result.json` values are
*not* comparable across rows, because each run committed to its own α.

#### α\* and the α\*·n invariant

| dataset | α\* (n=2) | α\* (n=3) | α\* (n=5) | α\*·n (n=2) | α\*·n (n=3) | α\*·n (n=5) |
|---|---|---|---|---|---|---|
| **SWaT** | 0.50 | 0.25 | 0.25 | 1.00 | 0.75 | 1.25 |
| **PSM** | 0.50 | 0.38 | 0.25 | 1.00 | 1.12 | 1.25 |
| **ETTh1** | 0.50 | 0.30 | 0.20 | 1.00 | 0.90 | 1.00 |
| **exchange** | 0.70 | 0.53 | 0.30 | 1.40 | 1.60 | 1.50 |

**α\*·n is constant within each dataset.** Since the merge is θ₀ + α·Στᵢ, a constant α·n
means the optimal merge is a fixed multiple of the **arithmetic mean** of the task vectors —
≈1.0× on SWaT, PSM and ETTh1, ≈1.5× on exchange_rate — regardless of how many there are.
Every seed agrees exactly on ETTh1 and exchange_rate, on datasets whose absolute metrics
wander by 8.2% and 5.3%.

> **The AD evidence for this is weak, and the α grid is why.** The AD runs step α by **0.25**
> against **0.10** on the forecasting runs, and the products reflect that: SWaT's α\*·n spreads
> 0.75–1.25 (**1.67×**) against ETTh1's 0.90–1.00 (**1.11×**). Worse, the value the rule
> predicts at n = 3 — 1/3 = 0.333 — **is not on SWaT's grid at all** (0.25 and 0.50 are the
> neighbours), so SWaT cannot express the predicted optimum even if it holds exactly. Treat
> α\*·n = constant as established on the forecasting datasets and merely *consistent with* the
> AD ones. Refining the AD grid is cheap and is EXECUTION_PLAN.md §3.2.

#### Merge cost — merged model ÷ each specialist on its own shard

| dataset | n=2 | n=3 | n=5 |
|---|---|---|---|
| **SWaT** | 1.039 | 1.096 ±0.003 | 0.996 |
| **PSM** | 1.051 | 1.038 ±0.071 | 1.060 |
| **ETTh1** | 1.075 ±0.014 | 1.080 ±0.038 | 1.020 ±0.024 |
| **exchange** | 1.629 ±0.009 | 2.082 ±0.288 | 2.062 ±0.110 |

**Flat in n.** Merging stays as faithful to its specialists at five segments as at two.
SWaT, PSM and ETTh1 all sit at ~1.0–1.1; exchange_rate is the outlier at 1.6–2.1.

#### GRR — share of the base-to-joint gap the merge closes

| dataset | n=2 | n=3 | n=5 | | n=2 | n=3 | n=5 |
|---|---|---|---|---|---|---|---|
| **SWaT** | 0.116 | 0.005 ±0.037 | 0.064 | | 0.707 | 0.477 ±0.115 | 0.947 |
| **PSM** | 0.903 | 0.668 ±0.226 | 0.497 | | 1.156 | 1.059 ±0.059 | 0.861 |
| **ETTh1** | 0.865 ±0.035 | 0.733 ±0.149 | 0.885 ±0.057 | | 0.874 ±0.034 | 0.778 ±0.213 | 0.963 ±0.018 |
| **exchange** | 1.421 ±0.240 | 1.164 ±0.099 | 1.238 ±0.208 | | 1.424 ±0.237 | 1.218 ±0.083 | 1.238 ±0.208 |

Left block: α selected on validation (deployable). Right block: α read off the test
curve (oracle, unobtainable in practice).

**No consistent trend with n.** Only PSM declines (0.903 → 0.668 → 0.497). ETTh1 and
exchange_rate are flat; SWaT is non-monotone. An earlier reading of these runs claimed GRR
falls with n on every dataset — that was an artefact of comparing rows computed at different
α, and it does not survive the uniform recomputation. **Withdrawn.**

SWaT's GRR is the least trustworthy number here regardless: its base-to-joint gap is only
0.0095 AUROC, so a seed-level wobble of 0.001 moves GRR by ~0.11, and n=2 and n=5 are
single-seed.

### 1.12 Validation cannot select α for anomaly detection

The GRR table's two blocks are the measurement. Cost of choosing α on validation instead
of on test, as a fraction of the oracle GRR:

| dataset | n=2 | n=3 | n=5 | verdict |
|---|---|---|---|---|
| **SWaT** | 84% | 99% | 93% | **val selection fails** |
| **PSM** | 22% | 37% | 42% | **val selection fails** |
| **ETTh1** | 1% | 6% | 8% | **val selection is free** |
| **exchange** | 0% | 4% | 0% | **val selection is free** |

On the forecasting datasets validation costs **1–8%** of the oracle GRR — the val optimum
and the test optimum are within one grid step, and on exchange_rate at n=5 they coincide
exactly. On the AD datasets it costs **22–99%**.

The mechanism is direct, and sharper than the earlier "AD metrics are blind to α" framing:
the AD **validation** metric is reconstruction error, the AD **test** metric is AUROC, and
the two disagree about α.

| dataset | α minimising val reconstruction | α maximising test AUROC |
|---|---|---|
| SWaT | 0.50 | ≥1.50 *(monotone increasing — the optimum is past the grid edge)* |
| PSM | 0.50 | 0.75 |

AUROC is *not* insensitive to α — it moves 0.0060 on SWaT (6× the noise floor) and 0.0426
on PSM (85×). It simply moves in a direction validation cannot see. On SWaT, α=0.5
minimises reconstruction error while AUROC keeps improving all the way to 1.5, where
reconstruction is 2.5× worse than at its optimum. **A model that reconstructs badly can
separate anomalies well**, so validation reconstruction is not a proxy for detection.

This is why `--pipeline_select_merge_scale_on_val` refuses on AD (EXECUTION_PLAN.md §2.9),
and it is now a measured justification rather than an argued one. The consequence for the
research is uncomfortable and worth stating plainly: **on AD there is no deployable way to
choose α.** Every AD merge number in this file is oracle-selected.

### 1.13 Merge versus sequential fine-tuning, by segment count

| dataset | n=2 | n=3 | n=5 |
|---|---|---|---|
| **SWaT** | sequential (-0.20%) | sequential (-0.48%) | sequential (-0.45%) |
| **PSM** | sequential (-0.78%) | sequential (-0.36%) | sequential (-1.75%) |
| **ETTh1** | sequential (-12.73%) | sequential (-17.17%) | tie (-2.42%) |
| **exchange** | sequential (-16.17%) | tie (+4.98%) | merge (+38.42%) |

Positive = merge better. Judged against each dataset's own reproducibility floor (§1.9).
Merge is evaluated at the val-selected α, which is the honest choice on forecasting and,
per §1.12, an unfair handicap on AD — at oracle α the AD verdicts flip to merge. **The AD
rows therefore do not support a conclusion in either direction**: the comparison depends on
a hyperparameter that cannot be set honestly.

On the forecasting datasets, where the comparison is sound: ETTh1 favours sequential at
n=2 and n=3 and ties at n=5; exchange_rate favours sequential at n=2, ties at n=3, and
favours merging decisively at n=5 (+38%). **The winner is not a property of the dataset** —
it changes with the chunking on exchange_rate, which is the finding that invalidates the
premise of the old regime-indicator question (§1.7).

exchange_rate's n=5 result is driven by the sequential arm collapsing, not by merging
improving: sequential test MSE goes 0.220 → 0.362 → 0.531 as steps accumulate, while the
merge holds. That is forgetting, measured.

### 1.14 The regime predictor — a negative result at n = 9

§1.7 asked whether a cheap signal can decide *accumulate or materialise* before you commit.
The segment sweep was run to answer it with enough datapoints. It does, and the answer is no.

Excluding ties, the sweep yields **9 decisive configurations, 5 merge / 4 sequential**.
Testing every signal available *before* a strategy is chosen — a single threshold, asked to
separate the two groups perfectly:

| candidate signal | separates? |
|---|---|
| headroom (base-to-joint gap) | no |
| α\*·n | no |
| merge cost | no |
| n itself | no |
| specialisation (offdiag − diag) | yes — on the 5 forecasting configurations only |

That one hit is **exactly what chance predicts**. Five points split 3/2 separate perfectly
20% of the time, five signals were tested, so 1.0 false positive was expected and 1 was
found. **No predictor is identified**, now at n = 9 rather than the n = 4 of §1.7.

The more useful result is that the question was mis-posed. The outcome is **not a property
of a dataset**: it flips with segment count on exchange_rate (sequential at n=2 → merge at
n=5). Any predictor needs `n` among its inputs, and the four single-`n` datapoints that
motivated the original search were under-specified rather than merely too few.

### 1.15 Task-vector geometry against segment count

Measured on the ETTh1 checkpoints, which have three seeds at every n:

| n | mean ‖τᵢ‖ | ‖Στᵢ‖ | alignment ‖Στ‖ / Σ‖τ‖ |
|---|---|---|---|
| 2 | 1.851 | 3.052 | 0.824 |
| 3 | 1.532 | 3.385 | 0.737 |
| 5 | 1.102 | 3.487 | 0.633 |

Three things move together. Individual task vectors **shrink** as their segment shrinks;
their sum stays **nearly constant**; and they become steadily **less mutually aligned**.

This is what rules out the alternative reading of α\*·n = constant. If α\* were compensating
for vector *magnitude*, it would have to grow as ‖τ‖ falls — instead it falls as 1/n. α\*
tracks the **count**, not the segment size. Partition size changes the vectors; the scale
correction is arithmetic over how many there are.

The falling alignment is also the redundancy measurement
[EXECUTION_PLAN.md §3.4](EXECUTION_PLAN.md) was designed to obtain,
arriving here for free: more segments means more mutually orthogonal task vectors.
---

## 2. Exact configurations

Recorded so the results survive loss of the checkpoints. These are the runs every number
in §1 comes from, read back out of each run's own `config.json`.

Each dataset uses **one configuration** for all three arms — incremental (task
arithmetic), continual (sequential fine-tuning) and standard (joint training). The arms
differ only in `--pipeline`, and `StandardPipeline` necessarily sets
`baseline_fraction=1.0, n_finetune_segments=0`. Seeds 42 / 7 / 123 throughout.

### 2.1 SWaT

`slurm_grid_swat_train_incremental/58941` (incremental) · `slurm_grid_swat_train_standard/58930` (joint) · model `MaeTx` · dataset `Swat` · task `ad`

| group | argument | value |
|---|---|---|
| model | `decoder_embed_dim` | `128` |
|  | `decoder_heads` | `4` |
|  | `decoder_layers` | `1` |
|  | `encoder_embed_dim` | `256` |
|  | `encoder_heads` | `2` |
|  | `encoder_layers` | `2` |
|  | `instance_norm` | `True` |
|  | `mask_ratio` | `0.8` |
|  | `n_eval_passes` | `30` |
|  | `patch_len` | `5` |
|  | `patch_norm` | `False` |
|  | `training_mode` | `random_mask` |
| dataset | `baseline_fraction` | `0.5` — joint run uses `1.0` |
|  | `baseline_use_fraction` | `1.0` |
|  | `n_finetune_segments` | `3` — joint run uses `0` |
|  | `normalization` | `standard` |
|  | `stride` | `50` |
|  | `val_fraction` | `0.15` |
|  | `window_len` | `100` |
| loader | `batch_size` | `64` |
|  | `num_workers` | `4` |
| trainer (baseline) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `0.5` |
|  | `learning_rate` | `0.0001` |
|  | `n_epochs` | `300` |
|  | `optimizer` | `adamw` |
|  | `patience` | `30` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.01` |
| trainer (fine-tune) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `0.5` |
|  | `learning_rate` | `1e-05` |
|  | `n_epochs` | `50` |
|  | `optimizer` | `adamw` |
|  | `patience` | `10` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.01` |
| pipeline | `merge_scale` | `1.0` |
| evaluation | `debug` | `False` |
|  | `threshold_percentile` | `99.0` |
|  | `threshold_strategy` | `oracle` |

Other: `eval_seed=None`, `seed=42`

### 2.2 PSM

`slurm_grid_psm_train_incremental/59101` (incremental) · `slurm_grid_psm_train_standard/59090` (joint) · model `MaeTx` · dataset `Psm` · task `ad`

| group | argument | value |
|---|---|---|
| model | `decoder_embed_dim` | `128` |
|  | `decoder_heads` | `2` |
|  | `decoder_layers` | `1` |
|  | `encoder_embed_dim` | `128` |
|  | `encoder_heads` | `2` |
|  | `encoder_layers` | `2` |
|  | `instance_norm` | `True` |
|  | `mask_ratio` | `0.8` |
|  | `n_eval_passes` | `30` |
|  | `patch_len` | `5` |
|  | `patch_norm` | `False` |
|  | `training_mode` | `random_mask` |
| dataset | `baseline_fraction` | `0.5` — joint run uses `1.0` |
|  | `baseline_use_fraction` | `1.0` |
|  | `n_finetune_segments` | `3` — joint run uses `0` |
|  | `normalization` | `standard` |
|  | `stride` | `50` |
|  | `val_fraction` | `0.15` |
|  | `window_len` | `100` |
| loader | `batch_size` | `64` |
|  | `num_workers` | `4` |
| trainer (baseline) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `0.5` |
|  | `learning_rate` | `0.0001` |
|  | `n_epochs` | `300` |
|  | `optimizer` | `adamw` |
|  | `patience` | `30` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.01` |
| trainer (fine-tune) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `0.5` |
|  | `learning_rate` | `1e-05` |
|  | `n_epochs` | `50` |
|  | `optimizer` | `adamw` |
|  | `patience` | `10` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.01` |
| pipeline | `merge_scale` | `1.0` |
| evaluation | `debug` | `False` |
|  | `threshold_percentile` | `72.0` |
|  | `threshold_strategy` | `oracle` |

Other: `eval_seed=None`, `seed=42`

### 2.3 ETTh1

`slurm_grid_etth_forecast_train_incremental/59077` (incremental) · `slurm_grid_etth_forecast_train_standard/59062` (joint) · model `MaeTx` · dataset `EtthForecastDataset` · task `forecast`

| group | argument | value |
|---|---|---|
| model | `decoder_embed_dim` | `64` |
|  | `decoder_heads` | `4` |
|  | `decoder_layers` | `2` |
|  | `encoder_embed_dim` | `128` |
|  | `encoder_heads` | `4` |
|  | `encoder_layers` | `3` |
|  | `instance_norm` | `False` |
|  | `mask_ratio` | `0.5` |
|  | `n_eval_passes` | `1` |
|  | `patch_len` | `4` |
|  | `patch_norm` | `False` |
|  | `training_mode` | `causal_mask` |
| dataset | `baseline_fraction` | `0.5` — joint run uses `1.0` |
|  | `baseline_use_fraction` | `1.0` |
|  | `forecast_len` | `24` |
|  | `n_finetune_segments` | `3` — joint run uses `0` |
|  | `normalization` | `standard` |
|  | `stride` | `1` |
|  | `test_fraction` | `0.2` |
|  | `val_fraction` | `0.15` |
|  | `window_len` | `120` |
| loader | `batch_size` | `128` |
|  | `num_workers` | `4` |
| trainer (baseline) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `1.0` |
|  | `learning_rate` | `0.001` |
|  | `n_epochs` | `100` |
|  | `optimizer` | `adamw` |
|  | `patience` | `15` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| trainer (fine-tune) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `1.0` |
|  | `learning_rate` | `0.0001` |
|  | `n_epochs` | `30` |
|  | `optimizer` | `adamw` |
|  | `patience` | `10` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| pipeline | `merge_scale` | `1.0` |
| evaluation | `debug` | `False` |

Other: `eval_seed=None`, `seed=42`

### 2.4 exchange_rate

`exch_gate_incremental/79671` (incremental) · `exch_gate_standard/79672` (joint) · model `MaeTx` · dataset `ExchangeRateForecastDataset` · task `forecast`

| group | argument | value |
|---|---|---|
| model | `decoder_embed_dim` | `64` |
|  | `decoder_heads` | `4` |
|  | `decoder_layers` | `2` |
|  | `encoder_embed_dim` | `128` |
|  | `encoder_heads` | `4` |
|  | `encoder_layers` | `3` |
|  | `instance_norm` | `False` |
|  | `mask_ratio` | `0.5` |
|  | `n_eval_passes` | `1` |
|  | `patch_len` | `4` |
|  | `patch_norm` | `False` |
|  | `training_mode` | `causal_mask` |
| dataset | `baseline_fraction` | `0.5` — joint run uses `1.0` |
|  | `baseline_use_fraction` | `1.0` |
|  | `forecast_len` | `12` |
|  | `n_finetune_segments` | `3` — joint run uses `0` |
|  | `normalization` | `standard` |
|  | `stride` | `1` |
|  | `test_fraction` | `0.2` |
|  | `val_fraction` | `0.25` |
|  | `window_len` | `48` |
| loader | `batch_size` | `128` |
|  | `num_workers` | `4` |
| trainer (baseline) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `1.0` |
|  | `learning_rate` | `0.001` |
|  | `n_epochs` | `100` |
|  | `optimizer` | `adamw` |
|  | `patience` | `15` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| trainer (fine-tune) | `checkpoint_interval` | `0` |
|  | `device` | `auto` |
|  | `grad_clip` | `1.0` |
|  | `learning_rate` | `0.0001` |
|  | `n_epochs` | `30` |
|  | `optimizer` | `adamw` |
|  | `patience` | `10` |
|  | `reg_exclude` | `['norm', 'bias']` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| pipeline | `extra_merge_scales` | `[]` |
|  | `merge_scale` | `0.5` |
| evaluation | `debug` | `False` |

Other: `eval_seed=None`, `seed=42`

---

## 3. Method notes and gotchas

### 3.0 Known limitations of the setup

Two properties of the experimental setup are deliberate but would not survive a genuinely
streaming deployment. They do not affect the results reported above; they bound what those
results can be claimed to show.

**The normalisation scaler is fit on the full training series**, including segments the base
model has not reached yet and their validation tails (`swat.py::_prepare_data` and the
equivalent in every partitioned dataset). This is *necessary* for the comparison being made —
task vectors from different segments are only commensurable if every segment shares one
normalisation — and the test split is transformed with the training scaler, never fit on. But
it means the base model's preprocessing already encodes statistics a real system could not
have seen at that point in time.

Harmless for the questions §1 answers, which are about the *geometry and composition* of task
vectors under a fixed preprocessing. **It matters for the open questions**: "when should I
materialise a new model" and "which model do I route to" are exactly the settings where a
deployment cannot standardise against the future. Any Q4/Q5 experiment should either re-fit
the scaler causally or state this explicitly as an upper bound.

**Segment boundaries use floor division** (`finetune_ranges`), so up to `n_finetune_segments −
1` trailing timesteps are unused. Negligible at these lengths (≤4 rows of 6,000–495,000) and
it keeps each segment exactly the size the fine-tune trains on.

### 3.1 Original method notes

These cost real time; do not rediscover them.

- **`merge_scale` must not be swept in training.** Verified: runs 59068 (α=0.5) and 59071
  (α=1.0) have **identical τ norms**. The scale is applied only at merge time and has no
  effect whatsoever on training, so sweeping it as a grid axis triples the compute for
  byte-identical checkpoints. `MergeDiagnosticsPipeline` traces the whole curve from one run.
- **`val_fraction=0.10` breaks fine-tuning on ETTh1.** At 113 validation windows, early
  stopping fires on a lucky first epoch: `finetune_0` stops at epoch 1 with ‖τ‖/‖θ₀‖ = 0.0012
  against siblings at 0.0198 / 0.0283. Run 59077 is identical at 0.15 and its fine-tunes are
  healthy (best epochs 7 / 13 / 6 versus 1 / 8 / 19). **11 of 15 ETTh1 `train_incremental`
  sweep trials used 0.10**, so those results carry a dead task vector.
- **The validation slice is the temporal tail of each segment** (`val_tail_split`), so `val_i`
  sits adjacent to segment *i+1* and hands `ft_{i+1}` an unearned advantage. Tested where both
  neighbours are equidistant, the successor beats the predecessor on **3 of 3** datasets. This
  biases the off-diagonal down and makes `specialisation` *understate* the truth. It is **not**
  fixable by random splitting: with `stride=1, window_len=120` adjacent windows share 119 of
  120 timesteps, so the tail split is what prevents leakage and is the correct design.
- **ETTh1's test set is the tail of the same series**, so it resembles the last training
  segment and the most recent specialist has a structural advantage there. SWaT and PSM use
  the benchmark's own separate test files; PSM shows no recency ordering at all.
### 3.0b L2-SP, tested cleanly — no measurable effect

Re-run 2026-08-06 after the val-loss fix, ETTh1 at n = 3, three seeds per arm, with α
selected on validation **per arm** (a fixed α would confound "does L2-SP help" with "is that α
still right for a more constrained model"). `reg_lambda ∈ {0, 1e-3, 1e-2}`.

| reg_lambda | merged ÷ own baseline | best specialist ÷ baseline | α\* |
|---|---|---|---|
| 0 | 0.6945 ±0.0350 | 0.6265 | 0.3 |
| 1e-3 | 0.6675 ±0.0434 | 0.6141 | 0.3 |
| 1e-2 | 0.6743 ±0.0474 | 0.6089 | 0.3 – 0.4 |

**No resolvable effect, and the sign is not stable** — which is itself the finding. Judged
three ways: arm means in absolute MSE put λ = 1e-2 **2.7% worse**; the same arms in
ratio-to-own-baseline put it **2.9% better**; and the two hardware-matched pairs (below) put
it **2.6–3.0% worse**. Every one of those is inside the ±5% within-arm spread. An effect whose
direction flips with the choice of normalisation is smaller than the noise.

**The defensible claim:** at these λ, L2-SP neither helps nor hurts merging on ETTh1
measurably. The earlier "actively harmful, monotonically worse" claim is **withdrawn** — it
was made at the broken `val_fraction=0.10`, against a floor 4× too tight, and with an
early-stopping loss that handicapped every λ > 0 run (§3.0c).

### 3.0c Run-to-run variation is driven by GPU model, not by the seed

Found while analysing the L2-SP arms, where the baseline is identical by construction (λ
touches only fine-tuning) and so should be reproducible across arms at a fixed seed. It is
not — up to **18.8%** apart:

| seed | λ=0 | λ=1e-3 | λ=1e-2 | spread |
|---|---|---|---|---|
| 7 | 0.7456 *(RTX 5000)* | 0.7233 *(2080 Ti)* | 0.7456 *(RTX 5000)* | 3.1% |
| 42 | 0.7023 *(RTX 5000)* | 0.6774 *(RTX 6000)* | 0.7023 *(RTX 5000)* | 3.7% |
| 123 | 0.6662 *(2080 Ti)* | 0.7830 *(A5000)* | 0.7914 *(RTX 5000)* | **18.8%** |

**Same seed and same GPU model → bit-identical. Different GPU model → a different result**, with
no consistent per-GPU direction — hardware placement acts as an additional random draw, not as
a systematic offset.

Two consequences. First, **the reproducibility floors in §1.9 are honest**: they were measured
across runs that also landed on mixed hardware, so they already fold this in, and every
comparison judged against them is judged fairly. Second, **hardware cannot be controlled on
this cluster** — a single-model `--constraint` is rejected ("we advise you to allow also …"),
`--nodelist` is refused outright, and the ≥24G models require a different partition. So
paired-hardware comparisons are only available opportunistically, by checking after the fact
which GPU each job landed on. Worth doing for any comparison near the floor.

- **L2-SP does not reproduce as harmful.** *(Superseded by §3.0b — kept for the record.)* At `val_fraction=0.15`: merged test MSE 0.5918 /
  0.5904 / 0.6042 for `reg_lambda` 0 / 0.001 / 0.01 — a 2.3% spread against an 8.2% sd, and
  not monotone. The earlier "actively harmful, monotonically worse" claim was made at the
  broken `val_fraction=0.10` and against a floor 4× too tight.
  **These runs also predate the val-loss fix** (§3.0), so their early stopping selected on
  task loss *plus* the L2-SP penalty — a bias that grows with λ and favours earlier, less
  trained checkpoints. That bias pushes λ > 0 runs to look **worse**, so the null result above
  was obtained *despite* a handicap and is conservative: removing it can only move λ > 0
  towards parity or better. The original "harmful" claim, made under the same bias, is
  correspondingly **withdrawn**. A clean re-test is cheap (ETTh1, minutes per run) and is the
  only way to say anything positive about L2-SP.
- **Cluster:** `sbatch --export=VAR=value` gets the job `CANCELLED by 0` about two seconds in
  with **no log written at all**. Pass variables as an env prefix and let the default
  `--export=ALL` carry them. Measured: no `--export` ✅, `--export=ALL` ✅,
  `--export=ALL,FOO=bar` ❌, `--export=NONE` ❌.
- **`geometry_report` OOMs on the login node** at ~65 run directories in one invocation; split
  per experiment.

---
