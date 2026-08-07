# Experiment Summary — incremental learning by model merging

**The setting.** A model is trained on history and deployed; every period a new batch arrives
and the model must be brought up to date, with *some* history retainable but not all. The
question is whether task arithmetic — keep only what each period changed and add the deltas back
— is a viable update strategy on time series, where shards are aligned rather than disjoint.
[EXECUTION_PLAN.md §1](EXECUTION_PLAN.md) states the full question set.

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

Snapshot 2026-08-07: **six datasets × three segment counts** (n = 2, 3, 5), three training
seeds each on the forecasting datasets and one on the AD datasets, for the merge variant, the
sequential variant and the joint-training reference.

## TL;DR

- **Merging is essentially free once the scale is right.** One merged model performs within
  ~1.0–1.1× of keeping a separate specialist per regime, on SWaT, PSM and ETTh1 — and this
  holds at **every segment count tested** (§1.11). exchange_rate is the outlier at 1.6–2.1×.
- **Start from the mean of the task vectors.** In the deployment parameterisation — fixed base
  model, n shards tiling the data that arrived after it — α\*·n ≈ 1.0 on SWaT, ETTh1, ETTh2 and ETTm2, and
  ≈ 1.5 on exchange_rate. **This is an empirical regularity, not a law.** It does not survive a
  fixed-shard control on exchange_rate, nor a prefix-merge design on either dataset, and no
  experiment can settle it: `baseline + n × shard = total` means the count is never identifiable
  in isolation. §1.18
- **Merging is worth two to four periods of retained history — dataset-dependent.** Against
  retraining θ₀ on the last W periods, merging beats small budgets everywhere; the crossover is
  **W = 3** on ETTh1 and exchange_rate and **W = 5** on ETTh2. Its value is delivering a few
  periods' worth of accuracy while storing **no data**, only deltas. §1.21, §1.23
- **Whether old data hurts is dataset-specific, and drift does not predict it.** A 3-period
  window beats using all history on exchange_rate (+26%) — but on ETTh2, which has nearly the
  same drift statistic, *more data is monotonically better* and joint training wins outright.
  §1.21, §1.23
- **Shard size, not drift, drives most cross-dataset differences — but not all.**
  exchange_rate (6,071 rows) behaves unlike ETTh2 and ETTm2, which share its drift but have 2×
  and 9× the data; three of its four distinctive behaviours fail to reproduce. Retention
  claims ("under strong drift, X") become "on small shards, X". **The exception is routing
  headroom, which orders by drift** — 6.1% on ETTh1 against 81.0% on ETTh2 at identical shard
  size. §1.16, §1.23, §1.24
- **The one general result about continual fine-tuning: it degrades as steps chain.** On ETTm2
  it goes 0.0920 → 0.1839 from n = 3 to n = 5 while merging holds; on exchange_rate
  0.220 → 0.362 → 0.531. **Merging overtakes it not by improving but by not collapsing.** §1.24
- **Recency is not relevance.** Excluding the trivial last regime, the newest specialist is the
  best model for a regime in **1 case out of 16** — which regime the data belongs to is what
  decides, not how recent the model is. §1.20
- **Accumulated merges generalise forward.** Scored on shards no task vector has touched, the
  merge beats the base model in 7 of 8 cases with no decay as vectors accumulate. This is the
  direct evidence for the case merging exists for. §1.19
- **On the most strongly drifting dataset, merging beats training on all the data at once.**
  exchange_rate GRR **1.421 / 1.164 / 1.238** at n = 2 / 3 / 5, all above 1.0, all with α
  chosen on validation rather than test.
- **What used to look like interference was a scale error.** Every result before 2026-08-05
  merged at α = 1.0. The task vectors mostly agree, so summing them at full strength
  overshoots. At α\* the damage disappears, and so does the "forgetting" once reported.
- **In unsupervised AD the merge scale cannot be chosen honestly.** Validation reconstruction
  and test AUROC disagree about α — SWaT's val optimum is 0.5 while AUROC improves
  monotonically to 1.5 — so selecting on validation costs **84–99% on SWaT (but only 5–19% on
PSM)** of the achievable GRR on
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

## 0. Methods and definitions

Everything in this file compares a small number of update strategies. They are easy to conflate,
so they are defined precisely here and referred to by these names throughout.

### 0.1 The setting

A series is split once, chronologically and never shuffled:

```
|<---------------- training portion (80%) ---------------->|<-- test (20%) -->|
|<--- base: first 50% --->|<-- period 1 --><-- 2 --> ... <-- n -->|
```

The **base model θ₀** trains on the first 50% of the training portion and is then frozen. The
remainder is cut into `n` equal **periods** (called *shards* or *segments* elsewhere in the
codebase — same thing). The test set is the final 20% of the whole series and is never used for
any training or model selection. Each period holds out its own **validation slice** — the last
`val_fraction` of it, temporally — which is what "val_i" means in every table below.

### 0.1b The six datasets

| dataset | task | rows (train) | features | window / horizon | drift 3-way | drift 5-way | shard @ n=5 | reproducibility floor | headroom |
|---|---|---|---|---|---|---|---|---|---|
| **SWaT** | anomaly detection | 495,000 | 51 | 100 / — | 0.112 | — | 49,500 | 0.13% | 1.1% — saturated |
| **PSM** | anomaly detection | 132,481 | 25 | 100 / — | 0.465 | — | 13,248 | 0.07% | 3.4% — saturated |
| **ETTh1** | forecasting | 13,936 | 7 | 120 / 24 | 0.309 | 0.412 | 1,393 | 8.20% | 38–43% |
| **exchange_rate** | forecasting | 6,071 | 8 | 48 / 12 | 0.769 | 0.833 | **607** | 5.29% | 43.8% |
| **ETTh2** | forecasting | 13,936 | 7 | 120 / 24 | — | 0.753 | 1,393 | 6.7% | 83.8% |
| **ETTm2** | forecasting | 55,744 | 7 | 120 / 24 | — | 0.752 | **5,574** | 14.1% | 86.6% |

**Drift** is the model-free segment-shift statistic — the standard deviation of per-period
feature means, relative to the overall standard deviation — computed before any training.
⚠️ **The two drift columns are different measurements and must not be compared across.** The
3-way screen ([EXECUTION_PLAN.md §2.5](EXECUTION_PLAN.md)) and the 5-way screen
([EXECUTION_PLAN.md §2.18](EXECUTION_PLAN.md)) segment
the series differently — ETTh1 reads 0.309 on the first and 0.412 on the second. Only the
*ranking* is stable. Re-measuring every dataset on one segmentation is open work; until then,
**compare within a column, never across**. The AD datasets were never put through the 5-way
screen, which is why the last two rows have no 3-way value and the first two no 5-way value.

**Shard @ n=5** is the training rows each task vector is estimated from at the standard 5-way
split — the axis that turned out to explain nearly every cross-dataset difference (§1.24).
exchange_rate's 607 rows and ETTm2's 5,574 are the two poles of that comparison.
**Headroom** is the base-to-joint gap: how much a frozen base on 50% of the data loses against
training on 100%. *Both AD datasets are saturated*: a base model on half the data is already
within 1.1% and 3.4% of one trained on everything, so **no update strategy can show much on
them**, and their results should be read as "nothing separates" rather than as evidence.

The two forecasting datasets carry the informative results. exchange_rate is the most strongly
drifting dataset available and is where merging looks best; ETTh1 is mildly drifting and is
where it looks worst. That contrast drives most of the conclusions below.

All four come from `thuml/Time-Series-Library`. Every model is the same MAE transformer
(`MaeTx`); per-dataset architecture and training settings are in §2.

### 0.2 The update strategies

| name | what it does each period | stores | starts from |
|---|---|---|---|
| **base** | nothing — θ₀ frozen | — | — |
| **specialist** `ft_i` | fine-tune a copy of θ₀ on period *i* alone | one model per period | θ₀ |
| **model merging** (task arithmetic) | keep τᵢ = θᵢ − θ₀ from each specialist; serve θ₀ + α·Στᵢ | **weight deltas, no data** | θ₀ |
| **window retrain** | fine-tune a **fresh copy of θ₀** on the last W periods pooled together | **W periods of raw data** | **θ₀ every time** |
| **continual fine-tuning** | keep training the *current* model on the newest period | one model, no data | **the previous model** |
| **joint / full retrain** | train on 100% of the training portion at once | all data | scratch |

Three distinctions matter and are easy to lose:

- **Window retrain is not continual fine-tuning.** It restarts from θ₀ every time, so nothing
  compounds across updates — there is no forgetting chain. Continual fine-tuning starts from the
  *previous* model, which is exactly why its error accumulates.
- **Window retrain is not merging.** It uses raw retained data; merging uses only weight deltas
  and stores no data at all. That difference is the entire practical trade (§1.21).
- **A "specialist" is a window retrain with W = 1.** The two names are used where each reads
  more naturally, but `ft_i` and a one-period window retrain on period *i* are the same object.

### 0.3 The merge scale α

The merged model is θ₀ + α·Στᵢ. **α = 0 is the untouched base model**; α = 1 adds every task
vector at full strength. **α\*** denotes the α selected on validation data. Writing α = k/n
turns the merge into θ₀ + k·mean(τ), so **α·n is the multiple of the *average* task vector** —
which is why that product appears throughout.

### 0.4 Metrics used throughout

| term | definition | direction |
|---|---|---|
| **ratio to base** | a model's error on a slice ÷ θ₀'s error on the same slice | lower better; 1.00 = same as base |
| **merge cost** | merged error ÷ the error of that period's own specialist | 1.00 = merging is free |
| **GRR** | (merged − base) ÷ (joint − base): share of the base-to-joint gap the merge closes | 1.00 = as good as full retraining; >1 = better |
| **transfer matrix** | every model scored on every period's validation slice | see §1.4 |
| **specialisation** | mean off-diagonal − mean diagonal of that matrix | positive = specialists are period-specific |
| **ρ (subspace overlap)** | fraction of a new task vector's energy already inside the span of its predecessors | 0 = entirely new, 1 = nothing new |
| **forward transfer** | a merge of periods 0…k−1 scored on period k, which none of them trained on | §1.19 |
| **reproducibility floor** | run-to-run spread of the same configuration; the threshold a difference must clear | §1.9 |

**Primary test metrics:** `window_auroc` for the AD datasets (SWaT, PSM), `forecast/mse` for
the forecasting datasets (ETTh1, exchange_rate, ETTh2, ETTm2). AD also reports
point/event/point-adjusted families — see §1.5, and the caveat on `pa_f1` there.

**Percentage convention — "A beats B by p%" always means p = (B − A) / B**, i.e. the
improvement is expressed as a fraction of **B, the alternative being compared against**. The
reverse denominator inflates every figure (a halving reads as 50% one way and 100% the other),
so it is fixed here once. ⚠️ Three published figures in §1.23–§1.24 used the reverse denominator
and were corrected on 2026-08-07: ETTm2's "old data helps by 51%" is **34%**, ETTh2's "merging
beats W=3 by 37%" is **27%**, and "merging loses to W=5 by 24%" is **32%**. The raw errors those
were computed from were correct and unchanged, as were every conclusion's direction; only the
magnitudes moved.

### 0.5 What was run

| study | § | configurations | jobs |
|---|---|---|---|
| Diagnostics on the headline runs | 1.2–1.10 | 4 datasets × 3 seeds | training-free |
| Segment-count sweep | 1.11–1.15 | 4 datasets × n ∈ {2,3,5} | 48 |
| Routing headroom | 1.16 | 4 datasets at n = 5 | training-free |
| n = 1 baseline | 1.17 | 4 datasets | 8 |
| Fixed-shard-size control | 1.18 | 2 datasets × n ∈ {2,3,5} | 18 |
| Prefix merges / forward transfer | 1.18–1.19, 1.22 | 2 datasets × 5 prefixes | 6 (training-free merges) |
| Recency vs regime | 1.20 | 4 datasets at n = 5 | training-free |
| Sliding-window retrain | 1.21 | 4 datasets × W ∈ {1,2,3} | 24 |
| L2-SP clean test | 3.0b | ETTh1 × λ ∈ {0,1e-3,1e-2} | 9 |

Seeds: **3 on the forecasting datasets, 1 on the AD datasets** unless stated. AD carries fewer
because its reproducibility floors are 0.07–0.13% against 5–8% on forecasting.

---

### 0.6 How every number here is computed — and how to recompute it

**This section exists so that any number in this file can be re-derived without asking the
author what was meant.** Three published figures were found in August 2026 to use a different
denominator, a different aggregation order, or a different source experiment than the prose
implied — each defensible alone, none reproducible afterwards, because the calculation lived in
an ad-hoc script that no longer existed. Every quantity below is now produced by a committed
entry point under `src/incremental_ad/analysis/`, and the definitions are stated as executable
rules rather than prose.

#### The three entry points

| script | reads | produces |
|---|---|---|
| `analysis/results_audit.py` | `config.json` + `result.json` | per-experiment metric means/sd, reproducibility floor, headroom, committed α, GRR |
| `analysis/routing_report.py` | `transfer_matrix.csv` | routing headroom, merge cost, specialisation |
| `analysis/geometry_report.py` | checkpoints | cosine, ρ, effective rank, principal angles, ‖τ‖ |

Regenerate everything (one CPU job, ~4 minutes over 415 runs):

```bash
python -m incremental_ad.analysis.results_audit  --runs_root $RUNS_ROOT --out $OUT
python -m incremental_ad.analysis.routing_report $RUNS_ROOT/*_diagnostics/*/ --out $OUT/routing
python -m incremental_ad.analysis.geometry_report <incremental run dirs> --out $OUT/geometry
```

`geometry_report` accepts only `IncrementalTaskArithmeticPipeline` runs that have a full
checkpoint set, and exits on anything else — filter the list before passing it.

#### Definitions, fixed once

- **Aggregation.** Every quantity comes from the per-seed *mean*. Ratios are taken of the means,
  not averaged over per-seed ratios. GRR is the documented exception (below).
- **Percentages.** "A beats B by p%" is **always** p = (B − A) / B for an error metric — a
  fraction of **B, the alternative**. The reverse denominator inflates every figure; it produced
  three of the errors found in August 2026.
- **Direction** is read from the metric name (`auroc`, `auprc`, `f1`, `precision`, `recall`,
  `accuracy` are higher-better; everything else is an error), never assumed per dataset.
- **Reproducibility floor** = sample sd ÷ mean of the baseline-stage test metric over the seeds
  of **one** experiment. Comparable only within an experiment — see the warning in §1.9, where
  independent estimates of the same configuration differ by up to 3.3×.
- **Headroom** = (base − joint) / base, the share of the base model's error that training on
  everything removes.
- **GRR** = (base − merged) / (base − joint). The joint model lives in a different experiment,
  matched on dataset and every `mae_tx_*`/`dataset_*` argument except the partition arguments a
  joint run must differ on. Because base and joint come from different runs, GRR is reported
  **two ways**: `grr` (ratio of means) and `grr_paired` (mean of per-seed ratios over the seeds
  present in all three). They differ by up to 0.07; where they differ, prefer `grr_paired` and
  treat the gap as the uncertainty.
- **Merge cost** = mean over periods of merged(val_i) ÷ ft_i(val_i) — per period, on that
  period's *own* slice, so it needs the transfer matrix. Computing it from test-set numbers
  gives a plausible-looking but different quantity (0.56–0.92 instead of ≈1.05).
- **Specialisation** = mean off-diagonal − mean diagonal of the transfer matrix.
- **Routing headroom** = distance from the merged model to the transfer matrix's column optimum,
  as a fraction of that optimum. Seeds are averaged **before** the optimum is taken; taking the
  optimum per seed and averaging after lets a different seed win each column and biases the
  oracle low. Only comparable between runs whose merged model sits at a *selected* α — a run
  merged at α = 1 is measuring overshoot, so `routing_report` emits `merge_scale` alongside
  every row to make that visible.
- **Honest-α cost** (§1.12) = 1 − GRR(α_val) / GRR(α_oracle), where α_val is the argmin of the
  mean `reconstruction/score_mean` over the per-period validation columns and α_oracle the argmax
  of the test metric, both read off one merge-scale curve. The validation statistic must be the
  same across every row; mixing `score_mean` with `score_p99` is what corrupted the old PSM row.

#### What was verified, and what could not be

Recomputed and **matching** the published values: the ETTh1 task-vector norms (1.851 / 1.532 /
1.102 against 1.85 / 1.53 / 1.10), SWaT's ρ (0.608 against 0.607), the α\*·n products
(ETTh1 1.00/1.00, exchange_rate 1.40/1.50, ETTh2 0.97/0.80/0.75), exchange_rate's 5.29%
reproducibility floor, ETTh1's GRR, and **all three SWaT rows** of §1.12.

Recomputed and **corrected**: the PSM rows of §1.12, the routing table (§1.16), and three
percentages in §1.23–§1.24.

**Not reproducible from anything on disk**, and flagged in place rather than restated: the AD
routing headroom (§1.16 — the per-regime columns carry no detection metric, only reconstruction
statistics, so a per-regime oracle cannot be formed at all) and ETTh1's published 8.20%
reproducibility floor (its third seed's run is not identifiable among the surviving
experiments; the value is plausible and conservative but cannot be re-derived).


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

> **On `pa_f1` — point-adjusted F1.** Under point adjustment, a whole anomalous segment counts
> as detected if *any single point* inside it is flagged. This is known to overestimate
> detection severely: **Kim et al., "Towards a Rigorous Evaluation of Time-series Anomaly
> Detection" (AAAI 2022, arXiv:2109.05257)** show a *random* anomaly score can reach
> state-of-the-art PA-F1, so rankings built on it are unreliable. Replacement metrics proposed
> since include PA%K (same paper), range-based precision/recall (Tatbul et al., NeurIPS 2018),
> affiliation precision/recall (Huet et al., KDD 2022) and VUS-ROC/PR.
>
> This project follows their recommendation: **threshold-free metrics (AUROC/AUPRC) are the
> primary ones, and `pa_f1` is reported only for comparability with the literature that still
> uses it.** That is also why §1.5 notes PA-F1 often moving opposite to AUROC — a model firing
> more broadly scores better on PA and worse on everything else. *None of this critique is a
> contribution of this work; the citations are.* **Verify the references against the sources
> before they go into the thesis** — they are recorded here from prior knowledge, not fetched.


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


**ETTh2** — n = 3, three seeds. `etth2_merge_n3` against `etth2_gate_base` (base) and
`etth2_gate_joint` (joint); α = 1 read off the merge-scale curve in
`etth2_merge_n3_diagnostics`.

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| `forecast/mse` ↓ | 0.8415 | 0.1771 | 0.6346 | 0.2153 | 0.1362 | 0.7052 | 0.89 | 0.0184 |  |
| `forecast/rmse` ↓ | 0.9170 | 0.4208 | 0.7659 | 0.4637 | 0.3688 | 0.5481 | 0.83 | 0.0200 |  |
| `forecast/mae` ↓ | 0.6741 | 0.3168 | 0.6386 | 0.3625 | 0.2690 | 0.4051 | 0.77 | 0.0047 |  |

Merge-scale curve, `forecast/mse` on test (mean of 3 seeds):

| α | 0 | 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mse | 0.912 | 0.566 | 0.293 | 0.226 | **0.193** | 0.233 | 0.311 | 0.384 | 0.450 | 0.512 | 0.573 | 0.635 | 0.700 | 0.772 | 0.853 | 0.945 | 1.047 |

Curve minimum at **α = 0.3** (test-oracle); the pipeline selected α on validation — see §1.23/§1.24 for the α\* actually committed.


**ETTm2** — n = 3, three seeds. `ettm2_merge_n3` against `ettm2_gate_base` (base) and
`ettm2_gate_joint` (joint); α = 1 read off the merge-scale curve in
`ettm2_merge_n3_diagnostics`.

| metric | base | best specialist | merged @ α=1 | merged @ α\* | joint | gap | GRR | sd | |
|---|---|---|---|---|---|---|---|---|---|
| `forecast/mse` ↓ | 0.5608 | 0.0989 | 1.1266 | 0.1121 | 0.0750 | 0.4858 | 0.92 | 0.0130 |  |
| `forecast/rmse` ↓ | 0.7477 | 0.3145 | 1.0588 | 0.3345 | 0.2739 | 0.4738 | 0.87 | 0.0196 |  |
| `forecast/mae` ↓ | 0.5382 | 0.2307 | 0.8435 | 0.2586 | 0.1940 | 0.3442 | 0.81 | 0.0177 |  |

Merge-scale curve, `forecast/mse` on test (mean of 3 seeds):

| α | 0 | 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mse | 0.579 | 0.310 | 0.153 | 0.119 | **0.101** | 0.116 | 0.170 | 0.248 | 0.345 | 0.483 | 0.728 | 1.127 | 1.655 | 2.281 | 2.980 | 3.736 | 4.533 |

Curve minimum at **α = 0.3** (test-oracle); the pipeline selected α on validation — see §1.23/§1.24 for the α\* actually committed.

**Note the α = 1 column.** On ETTm2 merging at α = 1 gives `forecast/mse` **1.1266 — twice as
bad as the base model's 0.5608**, and worse than doing nothing at all. This is the overshoot of
§0.3 in its clearest form: summing three aligned task vectors instead of averaging them steps
three times too far. The same merge at α\* = 0.267 scores 0.1121. Any result reported at α = 1
is measuring the scale error, not the method.

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

The *materialise a separate model* branch remains untested: no variant in this study builds
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
| **ETTh2** (forecast/mse) | 3 | 0.8415 | 0.0567 | **6.74%** |
| **ETTm2** (forecast/mse) | 3 | 0.5608 | 0.0791 | **14.11%** |

> ⚠️ **The floor is itself a noisy estimate, and this limits every "outside the floor" claim.**
> Several experiment groups train the *same* baseline configuration at three seeds, so each
> gives an independent estimate of the same quantity. They disagree substantially:
>
> | dataset | independent 3-seed estimates | ratio | published |
> |---|---|---|---|
> | ETTh1 | 3.16 / 4.55 / 7.23 / 8.87% | 2.8× | 8.20% |
> | exchange_rate | 1.85 / 2.62 / 5.73 / 6.02% | 3.3× | 5.29% |
> | ETTh2 | 2.88 / 5.71 / 6.23 / 6.72 / 6.74% | 2.3× | 6.74% |
> | ETTm2 | 12.13 / 14.01 / 14.11 / 14.45 / 17.34% | 1.4× | 14.11% |
>
> A standard deviation from three samples has very wide confidence bounds, so this is expected
> rather than a defect — but it means the floor is a **rough threshold, not a significance
> test**, and a difference near the floor should be treated as undecided rather than decided.
> The published values happen to sit at or near the *top* of their ranges on ETTh1,
> exchange_rate and ETTh2, which is the conservative direction: using them makes a difference
> *harder* to call real, so no published claim is inflated by this. Differences comfortably
> outside the range (ETTm2's retention results, ETTh2's headroom) are unaffected. Verified
> 2026-08-07; ETTm2's tight spread is the most trustworthy of the six.

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
| **PSM** | **0.75** | **0.50** | **0.50** | **1.50** | **1.50** | **2.50** |
| **ETTh1** | 0.50 | 0.30 | 0.20 | 1.00 | 0.90 | 1.00 |
| **exchange** | 0.70 | 0.53 | 0.30 | 1.40 | 1.60 | 1.50 |

**α\*·n looks constant within each dataset.** Since the merge is θ₀ + α·Στᵢ, a constant α·n
would mean the optimal merge is a fixed multiple of the **arithmetic mean** of the task vectors
— ≈1.0× on SWaT, PSM and ETTh1, ≈1.5× on exchange_rate — regardless of how many there are.

> **Two corrections to how this was first reported (2026-08-06).**
>
> **The apparent exact seed agreement was a grid artefact.** On the 0.1 grid all three seeds
> returned the same α\*. Re-measured at 0.05 they disagree by one step. α\* is not
> seed-invariant; the coarse grid was hiding its spread.
>
> **The error bars swamp the effect.** Quantisation alone contributes ±(grid × n), which at
> n = 5 on a 0.1 grid is **±0.50** — a 50% uncertainty on the quantity whose constancy is the
> claim. "1.00 ± 0.50" is not evidence of a constant. The numbers below are the measurements;
> they are **not** yet a verified invariant. §1.18.

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

### 1.12 Validation cannot select α on SWaT — but it can on PSM

Cost of choosing α on validation instead of on test, as a fraction of the oracle GRR.
**Recomputed 2026-08-07** under one fixed definition (§0.6): α_val is the argmin of the mean
`reconstruction/score_mean` over the per-period validation columns, α_oracle the argmax of test
`window_auroc`, both read off the same merge-scale curve.

| dataset | n=2 | n=3 | n=5 | verdict |
|---|---|---|---|---|
| **SWaT** | 84% | 99% | 93% | **val selection fails** |
| **PSM** | **6%** | **19%** | **5%** | **val selection is adequate** |
| **ETTh1** | 1% | 6% | 8% | **val selection is free** |
| **exchange** | 0% | 4% | 0% | **val selection is free** |

> ⚠️ **The PSM row is corrected; it previously read 22% / 37% / 42%.** All three SWaT rows
> reproduce exactly from the runs, so the method is confirmed — but no single validation signal
> reproduces the old PSM row. Searching all five `reconstruction/score_*` statistics against
> three column subsets, PSM n=2's published 0.903 is only reachable with `score_p99` or
> `score_std`, while the SWaT rows require `score_mean`; and PSM n=5's published 0.497 is not
> reproduced by **any** of the fifteen combinations (closest 0.578). The old row therefore mixed
> validation signals across datasets. Under the single consistent definition PSM's cost is
> **5–19%**, three to eight times smaller than published.

On the forecasting datasets validation costs **1–8%** of the oracle GRR — the val optimum
and the test optimum are within one grid step, and on exchange_rate at n=5 they coincide
exactly. **On AD the result splits**: catastrophic on SWaT (84–99%), mild on PSM (5–19%).

**This narrows the claim.** "Validation cannot select α for anomaly detection" is too strong —
it is what SWaT shows, not what AD shows. The defensible statement is that **validation
selection on AD is unreliable rather than impossible**: it can cost almost everything (SWaT) or
almost nothing (PSM), and *nothing observable without labels tells you which case you are in*.
For a practitioner that is still a blocker — an unpredictable method is not a usable one — but
it is a weaker and more honest claim than the original, and it means a labelled calibration set
buys much less on PSM than [THEORY.md §11.4](THEORY.md) assumed.

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

exchange_rate's n=5 result is driven by the sequential variant collapsing, not by merging
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
### 1.16 Routing versus merging — is a router worth building?

You end up holding several models: the base, one specialist per period, and the merge. Which
do you use on new data? The transfer matrix answers the prior question directly, because its
column minimum *is* the best possible router — pick, for every regime, whichever model turns
out best on it. No real router beats that ceiling, so the distance from the merged model to it
bounds what routing could ever gain.

Measured at n = 5, ratio-to-base on each regime's held-out slice, with the merged model taken
at its **validation-selected α**. Recomputed 2026-08-07 by
`python -m incremental_ad.analysis.routing_report` — see the warning below.

| dataset | drift (5-way) | shard @ n=5 | n=2 | n=3 | n=5 |
|---|---|---|---|---|---|
| exchange_rate | 0.833 | 607 | +15.9% | +13.0% / +23.0% | **+106.9%** |
| **ETTh2** | 0.753 | 1,393 | — | **+108.3%** | **+81.0%** |
| **ETTm2** | 0.752 | 5,574 | — | **+65.3%** | **+66.0%** |
| ETTh1 | 0.412 | 1,393 | +5.6% | *(α=1 only)* | **+6.1%** |
| SWaT / PSM | — | — | **not derivable** | not derivable | not derivable |

*(exchange_rate has two independent n=3 groups, hence two values. ETTh1's only n=3 diagnostics
merged at α = 1, which measures overshoot rather than routing — its merge cost is 1.84 against
1.02–1.07 at a selected α — so it is excluded rather than quoted.)*

> ⚠️ **These figures replace the ones published before 2026-08-07** (SWaT +6.2%, PSM +7.8%,
> ETTh1 +7.0%, exchange_rate +102.3% / +318.0%). Those were computed ad hoc with no script of
> record and could not be re-derived: a faithful recomputation reproduces ETTh1 to within a
> point but misses exchange_rate by 5 points and its newest-specialist column by 157. The
> definition is now pinned in `analysis/routing_report.py`, which also fixes the aggregation
> order (average over seeds *first*, then take the column optimum — choosing per seed and
> averaging after lets a different seed win each column and biases the oracle low).

> ⚠️ **The two AD rows are withdrawn rather than recomputed.** On AD the transfer matrix's
> per-regime columns carry only `reconstruction/score_*` — `window_auroc` exists on the `test`
> column alone, because the interior validation slices have no labels. A per-regime oracle
> router therefore **cannot be computed on AD at all** from these runs, and the only per-regime
> signal that does exist is exactly the one §1.12 shows is blind to detection quality. The
> previously published +6.2% / +7.8% cannot be derived from the stated method. This does not
> weaken the practical advice — it strengthens it: on AD routing is not merely unattractive,
> it is **unmeasurable**, which is the same blocker as §3.2.

**Routing headroom tracks drift, not shard size — but the evidence is a separation, not a
clean ordering.**

> ⚠️ An earlier version of this section reported a perfect four-way ordering by drift. That
> held **only at n = 5**. Adding every other segment count shows exchange_rate is unstable
> across n — 13–23% at n = 2 and 3, but 106.9% at n = 5 — so it cannot anchor an ordering, and
> the n = 5 agreement was partly coincidence.

What survives is a clean **separation**, stable at every segment count measured:

| | drift | routing headroom, across all n |
|---|---|---|
| ETTh1 | 0.412 | **5.6 – 6.1%** — consistently negligible |
| ETTm2 | 0.752 | **65.3 – 66.0%** — consistently large |
| ETTh2 | 0.753 | **81.0 – 108.3%** — consistently large |
| exchange_rate | 0.833 | 13.0 – 106.9% — **erratic** |

The decisive comparison is **ETTh1 against ETTh2**: identical row count, feature count, sampling
frequency, sensor type, window sizing *and* shard size, differing only in station and therefore
in drift (0.412 vs 0.753). Headroom differs by a factor of thirteen to nineteen, and it does so
at every n. ETTm2 — nine times exchange_rate's data, four times ETTh1's shard size — still shows
66%, eleven times ETTh1's. **Scarcity cannot produce that; drift can.** But the relationship is
a threshold rather than a dose-response: low-drift data has no headroom, high-drift data has a
lot, and how much is not predictable from the drift statistic.

This is the one place where the scarcity explanation of §1.24 does *not* apply, and the
distinction matters: *old data hurting*, *the retention crossover* and *elevated α\*·n* track
**shard size**; *routing headroom* tracks **drift**. They are different behaviours with
different drivers, and the blanket restatement "every 'under strong drift, X' becomes 'on small
shards, X'" is too broad — see §1.24.

**When is a router worth it, then?** Only on genuinely drifting data, and only if selection can
be made without labels. On low-drift forecasting (ETTh1) a perfect router recovers 6%, which
does not justify storing n models plus selection logic that can itself be wrong. On strongly
drifting forecasting it recovers 66–107%, which does.

**Merging beats always-using-the-newest-specialist on all four forecasting datasets** — so if
you keep one model, the merge is the right one, not the most recent.

> **Method note.** The first run of this analysis reported merged at +573% on SWaT. The AD
> segment-sweep runs inherit `merge_scale=1.0` from the `noisefloor_*` configs, so at n = 5 that
> is α·n = 5 — a fivefold overshoot, and the stored `merged` row was a wrecked model. Rebuilding
> that row from the curve at α\* is what produces the table above. Any comparison involving a
> merged model must fix α first.

### 1.17 The n = 1 baseline — does merging beat *not splitting the data at all*?

One fine-tune on the entire post-baseline block. No partitioning, no merging, same total data.
This is the null hypothesis for the whole enterprise in a static setting, and it had never been
run. Merged values at α\*; `n=1` is that run's `finetune_0` on the global test set.

| dataset | n = 1 (unsplit) | best merge | difference | floor |
|---|---|---|---|---|
| SWaT | 0.7994 | 0.8013 | +0.24% | 0.13% |
| PSM | 0.7898 | 0.7977 | +1.01% | 0.07% |
| **ETTh1** | **0.4134** | 0.4517 | **−9.25%** | 8.20% |
| exchange_rate | 0.2788 | 0.2554 | +8.38% | 5.29% |

**Merging is not an accuracy win over training on the same data unsplit.** On ETTh1 the single
fine-tune beats every merge by more than the noise floor. On the AD pair merging wins by margins
that clear their very tight floors but are practically negligible (0.24%, 1.01%). Only
exchange_rate — the most strongly drifting dataset — shows merging ahead by a margin that
matters.

**What this means for the thesis.** Merging's justification is **the streaming constraint**, not
accuracy: data arrives over time and cannot be retained, so the unsplit fine-tune is not
available. That is a perfectly good justification, and it is now stated with a number instead of
assumed. Where merging *does* beat the unsplit baseline is exactly where the regimes differ most
— consistent with §1.16, and with exchange_rate being the only dataset that also beats joint
training.

It also sharpens [THEORY.md §6.6](THEORY.md): if the optimal merge is always k·mean(τ), then at large n
you are averaging n noisy estimates of largely one drift direction — merging may be **denoising a
single update** rather than composing distinct knowledge. ETTh1 losing to n = 1 is what that
would look like.
### 1.18 The α\*·n regularity — what it is, and what it is not

This has been amended three times. Here is the whole thing in one place.

#### The observation

Selecting α on validation and multiplying by the number of shards gives a product that barely
moves within a dataset:

| dataset | n=2 | n=3 | n=5 |
|---|---|---|---|
| SWaT | 1.00 | 0.75 | 1.25 |
| PSM | 1.00 | 1.12 | 1.25 |
| ETTh1 | 1.00 | 0.90 | 1.00 |
| exchange_rate | 1.40 | 1.60 | 1.50 |

Since the merge is θ₀ + α·Στᵢ, writing α = k/n makes it θ₀ + k·**mean**(τ): a constant α·n
means the best merge is a fixed multiple of the *average* task vector, whatever the count.
That is a genuinely different regime from image classification, where the published scaling
coefficients give a product that **grows** with task count.

#### Three tests, and the claim narrowing each time

**1. Fixed-shard-size control.** Hold the shard at 15% of train and let the baseline absorb the
difference (`baseline_fraction` 0.70 / 0.55 / 0.25), 3 seeds, 0.05 α grid.

| dataset | baseline fixed *(original)* | shard fixed *(control)* |
|---|---|---|
| ETTh1 | 1.00 / 0.90 / 1.00 | 1.00 / 1.10 / 1.08 — flat |
| exchange_rate | 1.40 / 1.60 / 1.50 | **0.97 / 1.25 / 1.58 — grows** |

exchange_rate's endpoints separate beyond both seed spread and quantisation. **The strong form
— α\*·n as a constant of the dataset — is falsified here.**

**2. Prefix merges.** From one n = 5 run, merge τ₀, then τ₀+τ₁, up to all five — base model and
shard size both fixed, only the count varying. 3 seeds, 0.05 grid.

| | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| ETTh1 α\*·k | 0.48 | 0.57 | 0.85 | 0.87 | 0.83 |
| exchange_rate α\*·k | 1.20 | 0.97 | 1.10 | 1.53 | 1.50 |

Rises then plateaus, 1.6–1.8× spread. **Not constant here either.** (ETTh1 k=1 is unreliable —
seeds gave 0.30, 0.40, 0.75.)

**3. Why no test can settle it.** On a fixed series, `baseline + n × shard = total`. Fix any
two and the third must move, so **every possible design confounds the count with something**:

| design | held fixed | confounded with count |
|---|---|---|
| original sweep | baseline, total coverage | shard size |
| fixed-shard control | shard size, total | baseline size |
| prefix merges | baseline, shard size | total coverage |

**"Does α\* depend on the count alone" is not identifiable on a fixed dataset.** More datasets
do not help — each one reproduces the same algebraic constraint. This is why the invariant
appears under one parameterisation and dissolves under the others.

#### Measurement caveats that apply to every number above

- **The apparent exact seed agreement was a grid artefact.** On the 0.1 grid all seeds returned
  the same α\*; at 0.05 they differ by one step.
- **Quantisation contributes ±(grid × n)** — ±0.50 at n = 5 on a 0.1 grid.
- **The optimum itself is sharp enough to test**: 15–51% validation penalty between α·n = 1.0
  and 1.5 against floors of 8.2% and 5.3%. The limit was resolution and identifiability, never
  the sharpness of the minimum.

#### What to claim

> **In the deployment parameterisation — a fixed base model, with n shards tiling the data that
> arrived after it — α\*·n ≈ 1 on SWaT, ETTh1, ETTh2 and ETTm2; ≈1.5 on exchange_rate; 1.5–2.5 on PSM.** Starting
> from the mean of the task vectors is therefore a good default, and better than α = 1.

State it as an empirical regularity in that setting. **Do not state it as a law**, do not claim
it isolates the count, and do not claim seed-exactness.

### 1.19 Forward transfer — does an accumulated merge help on a regime it has never seen?

Each prefix scored on `val_k`, the first shard it has **not** seen, at the α selected only on
the shards it *has* seen. Ratio to the base model on that same shard; below 1.0 means the merge
beats the base on genuinely unseen data.

| dataset | k=1 → val_1 | k=2 → val_2 | k=3 → val_3 | k=4 → val_4 |
|---|---|---|---|---|
| ETTh1 | 0.931 | 0.863 | 0.908 | 0.843 |
| exchange_rate | 0.468 | **1.037** | 0.779 | 0.542 |

**Seven of eight cases beat the base model on data no task vector has touched, and there is no
decay as vectors accumulate** — ETTh1's best is at k = 4, exchange_rate's second best at k = 4.

This is the first direct evidence for the case merging is actually *for*: not merely retaining
old regimes, but generalising forward to the next one. It also bounds the "shard starvation"
story — accumulating more, smaller vectors did not degrade forward transfer over this range.

The exchange_rate k = 2 exception (1.037, no better than base) is worth keeping rather than
smoothing: `val_2` is also the shard where §1.16 found the merged model furthest from the
per-regime oracle. Something about that regime resists both merging and routing.

**What this does not answer:** when a *freshly materialised* model would have been better. That
comparator needs training, and is the open half of Q4 (EXECUTION_PLAN.md §3.2).


### 1.20 Recency or regime? Which specialist is actually best where

If the newest data were always the most relevant, the newest specialist would win every regime
and there would be nothing to route between. It does not. Best per-regime model at n = 5,
across all four datasets (20 regime-columns):

| the winner is… | count |
|---|---|
| the **matching** specialist (trained on that regime) | 10 / 20 |
| the **newest** specialist | 5 / 20 — but 4 of those are the last regime, where newest *is* matching |
| some other specialist | 9 / 20 |

**Excluding the trivial last regime, the newest specialist wins 1 of 16.** Recency does not
determine relevance: *which regime the data belongs to* does. That is the empirical basis for
treating old and new regimes as different targets rather than assuming the latest model
supersedes the rest, and it is why routing has headroom at all (§1.16).

**A caveat that inflates the "other" bucket.** The winner is frequently `ft_{i+1}` on column
`val_i` — the *next* specialist rather than the matching one. That is the tail-adjacency
artefact in §3: each validation slice is the temporal tail of its shard, so it sits immediately
before the next shard and the next specialist trains on data adjacent to it. So regime-matching
is *understated* here, and recency is if anything flattered.
### 1.21 Merging versus retraining on a window — how much history is merging worth?

The production constraint is *some* history retainable, not none. So the honest comparator is
neither full retraining nor pure merging but **retrain θ₀ on the last W periods**.

> **Terminology.** `W` counts **periods of retained data** — not to be confused with
> `window_len`, which is the model's input length in timesteps (120 on ETTh1). They are
> unrelated. A *condition* below (W=1, W=2, …) is one experimental variant; everything except
> the amount of retained data is held identical.

**How the split is made.** The base model θ₀ always trains on the first 50% of the training
portion; the test set is the final 20% of the series and is never touched by any condition.
Only the fine-tuning range moves. On ETTh1 (17,420 rows total → 13,936 for training):

| condition | trains on rows | = which part of the stream |
|---|---|---|
| base θ₀ | 0 – 6,968 | first 50%, the "history" |
| W=1 | 12,542 – 13,936 | most recent period |
| W=2 | 11,148 – 13,936 | 2 most recent periods |
| W=3 | 9,755 – 13,936 | 3 most recent periods |
| W=5 | 6,968 – 13,936 | everything after the base |

Implemented with existing knobs — `baseline_fraction = 1 − 0.1W` moves the fine-tune anchor and
`baseline_use_fraction = 0.5/(1 − 0.1W)` keeps the baseline training on exactly the first 50%.
**Verified: θ₀ trains on identical rows in every condition**, so the comparison isolates one
variable. Test set, 3 seeds on forecasting, 1 on AD.

| dataset | base | W=1 | W=2 | W=3 | W=5 (all) | best merge |
|---|---|---|---|---|---|---|
| SWaT | 0.8013 | 0.8027 | 0.8028 | 0.8027 | 0.7994 | 0.8013 |
| PSM | 0.7746 | 0.7870 | 0.7924 | 0.7952 | 0.7898 | 0.7977 |
| ETTh1 | 0.6920 | 0.4899 | 0.4300 | **0.3913** | 0.4134 | 0.4517 |
| exchange_rate | 0.6985 | 0.5908 | 0.3949 | **0.2053** | 0.2788 | 0.2554 |

Gap to the best merge (positive = the window wins; `*` = outside that dataset's floor):

| dataset | W=1 | W=2 | W=3 | W=5 |
|---|---|---|---|---|
| SWaT | +0.2%\* | +0.2%\* | +0.2%\* | −0.2%\* |
| PSM | −1.3%\* | −0.7%\* | −0.3%\* | −1.0%\* |
| ETTh1 | −8.5%\* | +4.8% | **+13.4%\*** | +8.5%\* |
| exchange_rate | −131.3%\* | −54.6%\* | **+19.6%\*** | −9.2%\* |

#### The answer: merging is worth about two periods of retained history

On both datasets with real headroom the crossover is at **W = 3**. Merging comfortably beats
retaining one or two periods — by 131% and 55% on exchange_rate — and loses to retaining three,
by 13% and 20%.

**This is the quantitative form of the claim the project has been making qualitatively.**
Merging's value is not accuracy; it is that it delivers roughly the accuracy of a
two-period window while storing **no data at all**, only weight deltas. Whether that is a good
trade is now a question about your retention budget rather than about the method:

> **Keep ≤ 2 periods of history → merge. Keep ≥ 3 → retrain on the window.**

On the saturated AD datasets nothing separates: SWaT moves 0.2% in total, and PSM favours
merging at every W by 0.3–1.3%, margins that clear its very tight 0.07% floor but mean little.

#### Joint training is not the ceiling — which reframes GRR

A 3-period window beats **full joint training** on both informative datasets:

| dataset | base | joint (100% of training data) | W = 3 window | W=3 vs joint |
|---|---|---|---|---|
| ETTh1 | 0.6930 | 0.4271 | **0.3913** | **8.4% better** |
| exchange_rate | 0.7321 | 0.3957 | **0.2053** | **48.1% better** |

**GRR is defined as (merged − base) ÷ (joint − base)** — progress toward joint. So GRR = 1.00
does *not* mean "as good as you can do"; it means "as good as a target that is itself beatable
on a drifting series". **This deflates the headline that merging beats joint training on
exchange_rate** (GRR 1.42): impressive against joint, but a plain windowed retrain beats joint
by 48% there. Read every GRR in this file as *normalised progress toward a specific baseline*,
useful for comparing merges to each other, and not as a fraction of the achievable.

It also means **"headroom" (the base-to-joint gap) understates what is achievable** under drift:
on exchange_rate the gap is 46% of base, while W = 3 improves on base by 72%.

#### A second finding: on *small shards*, a recent window beats all history

> ⚠️ **This section originally claimed the effect orders by measured drift. ETTh2 and ETTm2
> falsified that** — see §1.23 and §1.24. The measurements below stand; the drift explanation
> does not. Read the corrected reading at the end of this subsection.

W = 3 beats W = 5 on exchange_rate (0.2053 vs 0.2788, +26% and outside the floor) and on PSM
(+0.7%, outside its floor); ETTh1 favours W = 3 too but inside the floor.

**Why more data can be worse.** The test set is the *end* of the series, and the oldest
incremental data (ETTh1 rows 6,968–9,755) is furthest from it in time. Training on it pulls the
model toward patterns that no longer hold — where the effect appears, older data is not merely
redundant but **actively misleading**.

The four datasets available when this was first written ordered by measured drift, which looked
like the explanation:

| dataset | segment-shift drift ([EXECUTION_PLAN.md §2.5](EXECUTION_PLAN.md)) | W=3 vs W=5 |
|---|---|---|
| exchange_rate | 0.769 | **+26%** — real |
| PSM | 0.465 | +0.7% — real against its 0.07% floor |
| ETTh1 | 0.309 | +5.3% — inside its 8.20% floor, a tie |
| SWaT | 0.112 | ~0 — all within noise |

*(Drift figures are the model-free segment-shift statistic computed before any training; the
plan records exchange_rate 0.769, ETTh1 0.309, SWaT 0.112, traffic 0.090, weather 0.369, and
PSM 0.465 from the same sweep.)*

**Why that ordering was not the explanation.** Two datasets added afterwards carry
exchange_rate's drift with more data, and both break the ordering:

| dataset | drift (**5-way** segment-shift) | training rows | old data |
|---|---|---|---|
| exchange_rate | 0.833 | 6,071 | **hurts (+26% for W=3)** |
| ETTh2 | 0.753 | 13,936 | helps — W=5 monotonically best (§1.23) |
| ETTm2 | 0.752 | 55,744 | **helps by 34%** (§1.24) |

⚠️ **The two tables above use different segmentations and their numbers are not
interchangeable** — the first is the original screen, the second the 5-way measure on which
ETTh2 and ETTm2 were added (exchange_rate is 0.769 on the first, 0.833 on the second). Each
table is internally consistent; comparisons *across* them are not valid. Re-measuring every
dataset on one segmentation is open work (EXECUTION_PLAN.md §3).

At essentially identical drift the sign of the effect flips with **shard size**, not with drift.
The corrected reading: **on thin shards a recent window beats all history; once shards are large
enough, more history is better.** "Retrain on everything" is the wrong default only when each
period contributes few rows — which is the same scarcity axis that governs every other
cross-dataset difference in this project (§1.24).

Note this does not contradict §1.20. **Recency of *training data* matters** — recent windows
beat old ones. **Recency of a *model* does not determine which regime it serves best** — the
matching specialist beats the newest one 10 times out of 20. Those are different quantities.

### 1.22 Accumulate or materialise, on the next unseen period

Both variants already existed: `merge(τ₀…τ_{k−1})` is the accumulation, and `ft_{k−1}` — a
specialist fine-tuned on the most recent shard alone — **is** a freshly materialised model. Both
scored on `val_k`, which neither has trained on. Ratio to the base model on that shard.

| | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| ETTh1 accumulate | 0.931 | 0.863 | 0.908 | **0.843** |
| ETTh1 materialise | 0.915 | 0.888 | 0.911 | 0.962 |
| | tie | tie | tie | **accumulate** |
| exchange accumulate | 0.468 | 1.037 | **0.779** | 0.542 |
| exchange materialise | **0.431** | **0.843** | 0.851 | **0.317** |
| | materialise | materialise | accumulate | materialise |

**The rule is drift-dependent.** On mildly drifting ETTh1 accumulation is at least as good
throughout and pulls ahead by k = 4. On strongly drifting exchange_rate the freshest specialist
usually predicts the *next* regime better than any accumulation of past ones.

So: **materialise when drift is strong enough that recency beats accumulation on the next
period** — and that is measurable every period at no cost, because a fresh specialist is trained
each period anyway. Compare it against the accumulated merge on the newest held-out slice; when
the fresh model starts winning, branch.
### 1.23 ETTh2 — the drift explanation does not survive

ETTh2 was chosen as the closest available controlled experiment for drift: identical rows
(13,936), features (7), sampling frequency, sensor type and window sizing (120/24) to ETTh1,
from a different transformer station, with **roughly double the drift** (0.753 vs 0.412 on the
5-way screen). Every drift-dependent claim in this file had a directional prediction. **Most of
them failed.**

Base 0.8415, joint 0.1362 — **83.8% headroom**, the largest of any dataset here, against a 6.7%
seed spread. Three seeds throughout.

| | W=1 | W=2 | W=3 | W=5 | joint |
|---|---|---|---|---|---|
| window retrain | 0.4475 | 0.3644 | 0.2952 | **0.1632** | **0.1362** |

| | n=2 | n=3 | n=5 |
|---|---|---|---|
| merging (α on val) | 0.2612 | **0.2153** | 0.2669 |
| α\*·n | 0.97 | 0.80 | 0.75 |
| continual fine-tuning | 0.2332 | **0.1970** | 0.2495 |

#### What replicated

- **Merging beats small retention budgets.** Best merge 0.2153 against W=1 (0.4475) and W=2
  (0.3644) — the same shape as ETTh1 and exchange_rate.
- **α\*·n is order 1** (0.97 / 0.80 / 0.75), consistent with SWaT and ETTh1 and unlike
  exchange_rate's ≈1.5. It drifts downward rather than staying flat, which is one more reason
  not to call it a constant (§1.18).

#### What did not

**1. "A recent window beats all history" fails.** On ETTh2 more data is monotonically better:
W=1 → W=5 improves all the way, and joint (0.1362) beats every window. Old data does **not**
hurt here — and ETTh2 is the *higher-drift* member of the ETTh pair, so this is the opposite of
what the drift story predicts.

**2. The retention crossover moves.** Merging loses to W=5 by 32%, but beats W=3 by 27%. So
merging is worth **3–4 periods** here against ≈2 on ETTh1 and exchange_rate. The crossover is
dataset-dependent, not a constant of the method.

**3. Merging loses to continual fine-tuning at every n** (−12.0%, −9.3%, −7.0%) — the ETTh1
pattern, not the exchange_rate one, despite ETTh2 having nearly exchange_rate's drift.

#### ETTh2 diagnostics — transfer matrix, n = 3

Ratio to base, mean of 3 seeds, `forecast/mse`. Rows are models, columns the slice each is
scored on. Added 2026-08-07 (EXECUTION_PLAN.md §2.21); the earlier ETTh2 write-up had none.

| model | val_base | val_0 | val_1 | val_2 | test |
|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **merged** | 0.955 | 0.353 | **0.162** | 0.578 | **0.236** |
| ft_0 | 1.267 | **0.427** | 0.749 | 1.612 | 0.390 |
| ft_1 | 0.966 | **0.231** | 0.121 | 0.661 | 0.195 |
| ft_2 | 1.260 | 0.802 | **0.103** | **0.190** | 0.395 |
| joint | — | — | — | — | 0.149 |

**GRR 0.898, specialisation 0.447** (mean off-diagonal 0.693 − mean diagonal 0.246). The
diagonal is well below the off-diagonal on every column, so each task vector carries real
period-specific signal rather than a uniform improvement — the same structure as ETTh1, and the
precondition for merging to mean anything. Note `ft_0` *hurts* on `val_base` (1.267) and on
`val_2` (1.612): a specialist is actively worse than the base model away from its own period,
which is what the merge has to average away.

#### The consequence: drift, as measured, does not predict behaviour

| | exchange_rate | **ETTh2** | ETTh1 |
|---|---|---|---|
| drift (5-way segment-shift) | 0.833 | **0.753** | 0.412 |
| training rows | 6,071 | **13,936** | 13,936 |
| does old data hurt? | **yes**, +26% | **no** — more is better | mild, inside floor |
| merge vs continual | **merge** at n=5 | **continual** at every n | continual at n=2,3 |
| retention crossover | W=3 | **W=5** | W=3 |

**exchange_rate and ETTh2 have almost the same drift statistic and behave oppositely; ETTh2
behaves like ETTh1, which has half the drift.** What ETTh2 shares with ETTh1 and not with
exchange_rate is **data volume** — 13,936 rows against 6,071.

This is the first direct evidence on the scarcity confound flagged in §3.0: **exchange_rate's
distinctive results are more likely about small shards than about strong drift.** Every claim
in this file of the form *"under strong drift, X"* should be read as *"on exchange_rate, X"*
until ETTm2 (same drift as exchange_rate, 9× the data) settles it.

**A proposed explanation, tested and rejected.** The natural guess is that segment-shift misses
the *shape* of the drift: it measures how much period means differ, not whether they **trend**.
Old data would be stale only if the series moves away and stays away; if it oscillates, old
periods still cover regimes that recur. That would have explained everything — exchange_rate
progressive, ETTh2 oscillating.

**It is wrong.** Measuring the distance from each training period's mean to the *test* period's
mean (standardised on training data only) gives essentially the same structure on all four:

| dataset | base | p0 | p1 | p2 | p3 | p4 | nearest | spread |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 0.55 | 0.58 | 0.79 | 0.54 | **0.25** | 0.36 | p3 | 3.1× |
| ETTh2 | 1.20 | 1.33 | 0.76 | **0.55** | 0.62 | 0.95 | p2 | 2.4× |
| ETTm2 | 1.20 | 1.33 | 0.76 | **0.55** | 0.63 | 0.95 | p2 | 2.4× |
| exchange_rate | 1.44 | 1.61 | 1.32 | 1.13 | **0.65** | 0.98 | p3 | 2.5× |

**None is progressive.** In every case the test period is nearest to a *middle* period, not the
last, and the spread between nearest and furthest is 2.4–3.1× everywhere. exchange_rate is not
more trend-like than ETTh2 — their profiles are almost interchangeable. So a trend-aware drift
measure would **not** separate them either, and the earlier speculation that one would is
**withdrawn**.

What remains unexplained is therefore *why* old data hurts on exchange_rate and helps on ETTh2.
The leading hypothesis is still **data volume** (§3.0), and it is untested until ETTm2 —
which, note, has an almost identical distance profile to ETTh2 while carrying four times the
data, making the ETTh2/ETTm2 pair a second, tighter test of volume with regime structure held
nearly fixed.
### 1.24 ETTm2 — scarcity, not drift, explains exchange_rate

ETTm2 was run to settle one question: **exchange_rate carries the project's most distinctive
results and is by far its smallest dataset.** ETTm2 has essentially exchange_rate's drift (0.752
vs 0.833) with **nine times the training data** (55,744 rows vs 6,071), and — usefully, from the
shape test above — an almost identical period-to-test distance profile to ETTh2, so regime
structure is held nearly fixed while volume varies.

Base 0.5608, joint 0.0750 — **86.6% headroom**. Note the **14.1% seed spread on the base**, the
widest of any dataset here; differences below that are not resolvable. Three seeds throughout.

| | W=1 | W=2 | W=3 | W=5 | joint |
|---|---|---|---|---|---|
| window retrain | 0.1589 | 0.1459 | 0.1092 | **0.0724** | 0.0750 |

| | n=2 | n=3 | n=5 |
|---|---|---|---|
| merging (α on val) | 0.1240 | **0.1121** | 0.1385 |
| α\*·n | 0.77 | 0.80 | 0.75 |
| continual fine-tuning | 0.0952 | **0.0920** | 0.1839 |

#### The verdict on exchange_rate

| | exchange_rate | ETTh2 | **ETTm2** | ETTh1 |
|---|---|---|---|---|
| drift | 0.833 | 0.753 | **0.752** | 0.412 |
| training rows | 6,071 | 13,936 | **55,744** | 13,936 |
| n=5 shard | 607 rows | 1,393 | **5,574** | 1,393 |
| old data hurts? | **yes, +26%** | no | **no, −34%** | no |
| retention crossover | **W=3** | W=5 | **W=5** | W=3 |
| α\*·n | **≈1.5** | 0.75–0.97 | **0.75–0.80** | ≈1.0 |
| merge beats continual | at n=5 | never | **at n=5** | never |

**Three of exchange_rate's four distinctive behaviours fail to reproduce** at the same drift
with 2× and 9× the data: old data hurting, the early crossover, and the elevated α\*·n. On
ETTm2 old data does not merely fail to hurt — using all of it is **34% better** than a 3-period
window, and joint training is essentially the ceiling.

**So exchange_rate's distinctiveness is mostly a small-data effect, not a strong-drift effect.**
Claims of the form *"under strong drift, X"* about **retention and merge behaviour** — old data
hurting, the crossover point, elevated α\*·n — are restated as *"on small shards, X"*. With
607-row shards each task vector is estimated from very little; that is the regime where
merging's averaging helps most and where old data is most likely to be out-of-regime.

> ⚠️ **One exception, and it is not a small one: routing headroom.** An earlier version of this
> section restated *every* drift claim as a shard-size claim. That is too broad. §1.16 measures
> routing headroom on all four forecasting datasets and finds it orders by **drift**, not shard
> size: ETTm2, with nine times exchange_rate's data and four times ETTh1's shard size, still
> shows 66.0% headroom against ETTh1's 6.1%. The decisive case is ETTh1 vs ETTh2 — identical in
> every respect including shard size, differing only in station and drift — where headroom
> differs by a factor of thirteen. **Retention behaviour tracks shard size; routing headroom
> tracks drift.** Check which of the two a claim is about before restating it.

#### ETTm2 diagnostics — transfer matrix, n = 3

Ratio to base, mean of 3 seeds, `forecast/mse`. Added 2026-08-07 (EXECUTION_PLAN.md §2.21).

| model | val_base | val_0 | val_1 | val_2 | test |
|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **merged** | 0.857 | 0.283 | 0.219 | 0.381 | **0.195** |
| ft_0 | 0.739 | **0.223** | 0.781 | 2.555 | 0.318 |
| ft_1 | 1.039 | 0.191 | **0.163** | 0.524 | 0.175 |
| ft_2 | 1.372 | 1.152 | 0.138 | **0.205** | 0.321 |
| joint | — | — | — | — | 0.132 |

**GRR 0.927, specialisation 0.694** — the highest specialisation of any dataset measured, and
higher than ETTh2's 0.447 despite ETTm2's shards being four times larger. That is consistent
with §1.16's finding that *regime distinctness*, not shard size, is what makes specialists
differ from one another. `ft_0` reaching 2.555 on `val_2` is the extreme case: the first
period's specialist is two and a half times **worse** than the base model on the last period.

#### The one behaviour that does reproduce, and why

**Merge overtakes continual at n = 5** on ETTm2 (+24.7%) as it does on exchange_rate. But the
mechanism is visible in the numbers and it is not a dataset property: **continual degrades from
0.0920 at n = 3 to 0.1839 at n = 5** — forgetting compounding over more sequential steps — while
the merge is roughly flat (0.1121 → 0.1385). Merging wins there because its competitor
collapses, which is the same mechanism recorded on exchange_rate (0.220 → 0.362 → 0.531).

**This is a property of continual fine-tuning, not of any dataset**, and it is the one
genuinely general finding in the merge-versus-continual comparison: *the more update steps you
chain, the worse continual fine-tuning gets, while merging holds.*
---

## 2. Exact configurations

Recorded so the results survive loss of the checkpoints. These are the runs every number
in §1 comes from, read back out of each run's own `config.json`.

Each dataset uses **one configuration** for all three variants — incremental (task
arithmetic), continual (sequential fine-tuning) and standard (joint training). The variants
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

### 2.5 ETTh2

`etth2_merge_n3/80551` (incremental) · `etth2_gate_joint/80533` (joint) · model `MaeTx` · dataset `Etth2ForecastDataset` · task `forecast`

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
|  | `reg_exclude` | `2 values, bias–norm` |
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
|  | `reg_exclude` | `2 values, bias–norm` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| pipeline | `extra_merge_scales` | `25 values, 0.0–1.2` |
|  | `merge_scale` | `1.0` |
|  | `select_merge_scale_on_val` | `True` |

### 2.6 ETTm2

`ettm2_merge_n3/80588` (incremental) · `ettm2_gate_joint/80570` (joint) · model `MaeTx` · dataset `Ettm2ForecastDataset` · task `forecast`

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
|  | `reg_exclude` | `2 values, bias–norm` |
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
|  | `reg_exclude` | `2 values, bias–norm` |
|  | `reg_lambda` | `0.0` |
|  | `scheduler` | `cosine` |
|  | `warmup_ratio` | `0.1` |
|  | `weight_decay` | `0.0001` |
| pipeline | `extra_merge_scales` | `25 values, 0.0–1.2` |
|  | `merge_scale` | `1.0` |
|  | `select_merge_scale_on_val` | `True` |

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

**exchange_rate is simultaneously the most important dataset here and the smallest.** It
carries the strongest results — merging beats joint training (GRR > 1 at every n), routing has
+102% headroom, materialising beats accumulating — and it has only **6,071 training rows**, so
at n = 5 each shard is 607 rows with 92 validation windows. **Every one of those conclusions is
confounded with data scarcity**, because the same configuration that gives it the strongest
drift also gives it the thinnest shards. Nothing in this file separates the two.

The clean test exists and has not been run: **ETTm2 has essentially the same drift (0.752 vs
0.833 on the 5-way measure) with 55,744 training rows — nine times as much.** If exchange_rate's
behaviour reproduces there, it is drift; if it disappears, it was scarcity. Until then, read
every exchange_rate-specific claim as "strong drift *or* small shards".

**Segment boundaries use floor division** (`finetune_ranges`), so up to `n_finetune_segments −
1` trailing timesteps are unused. Negligible at these lengths (≤4 rows of 6,000–495,000) and
it keeps each segment exactly the size the fine-tune trains on.

### 3.1 L2-SP, tested cleanly — no measurable effect

Re-run 2026-08-06 after the val-loss fix, ETTh1 at n = 3, three seeds per variant, with α
selected on validation **per variant** (a fixed α would confound "does L2-SP help" with "is that α
still right for a more constrained model"). `reg_lambda ∈ {0, 1e-3, 1e-2}`.

| reg_lambda | merged ÷ own baseline | best specialist ÷ baseline | α\* |
|---|---|---|---|
| 0 | 0.6945 ±0.0350 | 0.6265 | 0.3 |
| 1e-3 | 0.6675 ±0.0434 | 0.6141 | 0.3 |
| 1e-2 | 0.6743 ±0.0474 | 0.6089 | 0.3 – 0.4 |

**No resolvable effect, and the sign is not stable** — which is itself the finding. Judged
three ways: variant means in absolute MSE put λ = 1e-2 **2.7% worse**; the same variants in
ratio-to-own-baseline put it **2.9% better**; and the two hardware-matched pairs (below) put
it **2.6–3.0% worse**. Every one of those is inside the ±5% within-variant spread. An effect whose
direction flips with the choice of normalisation is smaller than the noise.

**The defensible claim:** at these λ, L2-SP neither helps nor hurts merging on ETTh1
measurably. The earlier "actively harmful, monotonically worse" claim is **withdrawn** — it
was made at the broken `val_fraction=0.10`, against a floor 4× too tight, and with an
early-stopping loss that handicapped every λ > 0 variant (§3.2).

### 3.2 Run-to-run variation is driven by GPU model, not by the seed

Found while analysing the L2-SP variants, where the baseline is identical by construction (λ
touches only fine-tuning) and so should be reproducible across variants at a fixed seed. It is
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

- **L2-SP does not reproduce as harmful.** *(Superseded by §3.1 — kept for the record.)* At `val_fraction=0.15`: merged test MSE 0.5918 /
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

### 3.3 Method notes and gotchas that cost real time

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

