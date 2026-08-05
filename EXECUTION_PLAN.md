# Execution plan — what has been run, and what to run next

The execution log: **what was done, what it settled, and what is next.** Findings appear here
in one line; the numbers behind them live elsewhere.

| file | holds |
|---|---|
| **EXECUTION_PLAN.md** *(this file)* | what to do, and what has been done |
| [EXPERIMENTS.md](EXPERIMENTS.md) | full detailed results and observations |
| [THEORY.md](THEORY.md) | the concepts and reasoning, written to be studied |
| [artifact](https://claude.ai/code/artifact/b9478404-0c3d-4629-a43b-0aedb1ee24bf) | the same results, made visual |
| [CLAUDE.md](CLAUDE.md) | code map and repo invariants |

**Status legend:** ✅ done · 🟡 partly · ⬜ not started · ❌ dropped (with reason)

Last updated 2026-08-05. Four datasets, three training seeds each.

---

## 1. Where the project stands

Merging is essentially free once the merge scale is set correctly; it beats joint training
outright on the most strongly drifting dataset; and against sequential fine-tuning it is a
trade rather than a win. Numbers in [EXPERIMENTS.md §1](EXPERIMENTS.md).

The binding constraint is no longer method quality — it is **how few informative datasets
there are**. SWaT and PSM are saturated (base within 1.1% and 3.4% of joint training); only
ETTh1 and exchange_rate have real headroom.

---

## 2. Done

### 2.1 Diagnostics from existing checkpoints ✅

`MergeDiagnosticsPipeline` (transfer matrix, merge-scale curve, GRR) and `geometry_report`
(cosine, per-tensor cosine, principal angles, effective rank, subspace overlap ρ, norms,
cosine-vs-distance). Training-free — reloads checkpoints and re-scores.

**Settled:** the task vectors carry real, segment-specific signal on every dataset, ruling out
redundancy and degenerate fine-tuning as global explanations.

### 2.2 The merge scale ✅ — the single biggest correction

Added `--pipeline_curve_include_val` to trace the **validation block** against α, not only the
test metrics. Test-only was insufficient: the AD metrics are structurally blind to model
quality.

**Settled:** every earlier result merged at α = 1.0, which overshoots. What had been written
up as *interference* and *forgetting* was almost entirely a scale error. At the
validation-selected α\* the merge-cost intervals contain 1.00 on three of four datasets.

### 2.3 Continual fine-tuning baseline ✅

`ContinualFineTuningPipeline` — sequential chain, backward-transfer matrix, ACC/BWT, optional
L2-SP anchored to θ₀.

**Settled:** a stability/plasticity trade, not a win for either. Merging wins outright on SWaT
and exchange_rate; on PSM and ETTh1 sequential adapts better and the old-regime difference is
inside seed noise. An earlier single-seed claim that merging protects the base regime on *all*
datasets was **withdrawn** after replication.

### 2.4 Reproducibility floor ✅ — 3 seeds × 4 datasets

**Settled:** the floor is dataset-specific — PSM 0.07%, SWaT 0.13%, exchange_rate 5.29%,
ETTh1 8.20%. The universal 2% assumption was wrong in both directions (15–30× too conservative
for AD, 4× too loose for ETTh1). It rescued SWaT's headline GRR while correctly disqualifying
most of its individual metrics. **ETTh1's instability is in absolute values only** — report it
as ratios.

### 2.5 exchange_rate added ✅ — the most informative dataset

Chosen by measuring drift across every implemented dataset first (exchange_rate 0.769
segment-shift vs ETTh1 0.309, SWaT 0.112, traffic 0.090), then gated on a two-job headroom
check before spending more.

**Settled:** 43.8% headroom and **GRR 1.218 ±0.081 — merging beats training on all the data at
once**, which no other dataset shows. Config note: the shipped script's `val_fraction=0.1`
would have given 33 validation windows and broken fine-tuning exactly as on ETTh1; 0.25 used
instead.

### 2.6 Regime indicator ✅ built · ❌ does not generalise

ρ per merge step plus the exact orthogonal remainder ‖τ_k‖·√(1−ρ_k). Reported in
EXPERIMENTS.md §1.7, derived in [THEORY.md §4.4](THEORY.md).

**Settled:** right on 3 of 4 datasets, **wrong on exchange_rate** — lowest ρ of all, most
decisive merge win. Worse, a search over 10 candidate predictors found **2** that separate the
outcome groups, where chance alone predicts **3.3** at n = 4. **No predictor is identified,
and none can be at this sample size.** Use ρ as a diagnostic for explaining a result, never as
a controller. See §3.1 for what it would take.

### 2.7 Bugs and confounds found ✅

- `merge_scale` is a **no-op in training** — sweeping it triples compute for byte-identical
  checkpoints. Remove it from `TRAIN_INCREMENTAL_SWEEP` (§3.5).
- `val_fraction=0.10` breaks ETTh1 fine-tuning (113 validation windows; one task vector never
  trains). Affected 11 of 15 sweep trials.
- The validation slice is the temporal **tail** of each segment, so it sits adjacent to the
  next one and inflates the off-diagonal — `specialisation` therefore *understates* the truth.
  Not fixable by random splitting: adjacent windows share 119 of 120 timesteps.
- L2-SP's "actively harmful on ETTh1" result **does not reproduce** at `val_fraction=0.15`.
- The historical grid-search record was removed from EXPERIMENTS.md — fixed α, a wrong noise
  floor, the broken `val_fraction`. Per-trial CSVs remain under `$SLURM_GRID_OUTPUT_ROOT`.

### 2.8 Dropped, with reasons ❌

- **Held-out future regime carve-out** — the test set already *is* unseen data, and the
  merge-vs-newest-specialist comparison already exists in the test column. *(Narrow exception
  kept as §3.6: on AD the test metrics are blind, so an unlabelled held-out slice is the only
  way to see forward transfer there.)*
- **Pairwise merges** — with merge cost at ~1.00× there is almost no interference left to
  decompose.
- **`traffic`** — drifts *less than SWaT* (0.090 vs 0.112). Nothing to show.
- **Merge-order permutation** — plain task arithmetic is commutative; identical by
  construction. Revisit only if a sequential merging method is implemented.

---

## 3. Next

Ordered by value per unit of compute.

### 3.1 Enough datapoints to test a regime predictor ⬜ — top priority

At n = 4 no predictor can be established: a random one separates a 2/2 split **1 time in 3**.
Significance needs **n ≥ 8** configurations where the merge-vs-sequential outcome differs.

**The cheap way to get there is not more datasets — it is more configurations per dataset.**
Sweep `n_finetune_segments ∈ {2, 3, 5, 8}` across the four datasets: 16 configurations, each
yielding one (ρ, headroom, outcome) triple. The 5-segment runs already exist for all four.

Each configuration needs: incremental + standard + continual + diagnostics. Cheap on
exchange_rate and ETTh1 (minutes), expensive on SWaT (~1.5 h each).

**Register predictions before running.** Current candidates, none supported: ρ (redundancy
route), headroom, overshoot cost at α = 1, effective rank.

This also answers §5.4 — whether α\* falls as the number of segments grows.

### 3.2 `weather` as a fifth dataset ⬜

Drift 0.369, comparable to ETTh1, 21 features and 3× the rows. Already implemented and
registered. **Gate it** — one incremental + one standard, check headroom, commit the full arm
only if the gap is real. Check the window arithmetic first: `val_fraction` must give ≳150
validation windows.

### 3.3 Separate redundancy from headroom ⬜

They co-vary across all four datasets, so both explain everything equally well. The
random-vs-chronological split control is the cheapest attack: IID shards give **high
redundancy with headroom preserved**, which no existing dataset does.

Hold `baseline_fraction`, `n_finetune_segments` and windows-per-segment fixed. Arm A =
contiguous chronological segments (current). Arm B = shards drawn IID from the same pooled
region. Compare merge cost, α\*, ρ and the continual comparison.

**Registered prediction:** Arm B should show high ρ with α\* → 1/n, and merging should win
there. If it does not, the redundancy route is wrong.

### 3.4 Honest α selection inside the training pipeline ⬜

`--pipeline_merge_scale` is fixed before training, so every run commits to a guess. The
pipeline already builds a merged-val dataset spanning all regimes
(`get_merged_val_eval_dataset`); have it sweep α on that and use the winner for `merged/`.

**Forecasting only.** For AD there is no honest signal (EXPERIMENTS.md §1.9), so the flag
should refuse to auto-select and require an explicit α — defensible because α\* does not move
across seeds on either AD dataset (±0.000).

### 3.5 Remove `merge_scale` from `TRAIN_INCREMENTAL_SWEEP` ⬜

Verified no-op in training; currently triples compute for identical checkpoints.

### 3.6 AD forward transfer ⬜ — narrow

On SWaT and PSM the test metrics cannot see model quality, so *"does the merge generalise to
an unseen regime"* is unanswerable there. A 5-segment run merged over τ₀…τ₃ and evaluated on
the untouched `val_4` would show it on the reconstruction block. `58949` (SWaT) is clean;
`59109` (PSM) and `59079` (ETTh1) have an unbalanced vector.

Needs one flag: `--pipeline_merge_exclude_last N`.

### 3.7 BECAME — adaptive coefficient only ⬜

Derives its merging coefficient rather than tuning it, sidestepping the AD α problem. Skip the
gradient-projection first stage (class-incremental machinery; does not transfer). Log λ\* per
step alongside ρ — a second candidate regime signal, and one that might see the route ρ is
blind to.

### 3.8 OPCM ⬜ — scope as memory, not accuracy

Merge cost is already ~1.00×, so a better merging algorithm competes for a few percent on
something that is not the bottleneck. Its real selling point is **continual merging** — folding
in one model at a time without storing every task vector.

**Registered prediction:** OPCM projects out the component of an incoming vector lying in the
span of the accumulated update. On SWaT that shared component *is* the signal (ρ = 0.607), so
OPCM should underperform plain task arithmetic there and be more competitive on exchange_rate
(ρ = 0.117).

### 3.9 More AD datasets ⬜ — low priority

SMD / MSL / SMAP would broaden a side where both current datasets are saturated, but need a
custom loader like `swat.py` rather than the `HfSeriesForecastDataset` base. Every AD result is
handicapped by metrics that cannot see model quality, so forecasting datasets buy more per
unit of work.

---

## 4. Operational knowledge — do not rediscover this

**Cluster**

- **`sbatch --export=VAR=value` gets the job `CANCELLED by 0`** about two seconds in, with
  **no log written at all**. Measured: no `--export` ✅, `--export=ALL` ✅,
  `--export=ALL,FOO=bar` ❌, `--export=FOO=bar` ❌, `--export=NONE` ❌. Pass variables as an env
  prefix and let the default `--export=ALL` carry them:
  ```bash
  SOURCE_RUN=$W/runs/exp/59077 STANDARD_RUN=$W/runs/std/59062 \
      sbatch scripts/sbatch_merge_diagnostics.sh
  ```
- **The session scratchpad under `/tmp` is node-local** and invisible to compute nodes. Put
  helper files on `$WORK`; `scripts/sbatch_run_command.sh` runs a pre-generated command file
  from there.
- **`geometry_report` OOMs on the login node** at ~65 run directories; split per experiment.
- Heavy work goes through SLURM, never the login node.

**Generating runs**

Regenerate commands from a finished run's own `config.json` rather than retyping them — the
same trick `analysis/diagnose.py` uses. Every replication arm in §2 was produced that way, and
none drifted from its source configuration.

**Before committing to a dataset**

1. **Measure drift** (segment shift / KS between segments) — cheap, model-free, no training.
2. **Check the window arithmetic**: `val_windows = int(segment_len × val_fraction) −
   window_len + 1`. Below ~120 windows, early stopping fires on noise and a task vector dies.
3. **Gate on headroom** — one incremental + one standard run. If the base-to-joint gap is
   small the dataset cannot show anything, and you stop having spent minutes.

**Invariants to re-check after any merging change**

- All merged checkpoints recompute **bitwise** from baseline + fine-tunes.
- The α curve at the run's own scale reproduces the matrix `merged` test row exactly; at α = 0
  it reproduces `base` exactly.
- Hand-computed GRR matches `result.json`.

**Regenerating the reports**

`extract.py` reads every number from the run directories into JSON, `build_page.py` renders the
artifact from it, `gen_experiments.py` emits EXPERIMENTS.md §1. Re-run all three after new
results land — no figure is transcribed by hand.

---

## 5. Open questions

1. **Why does merging beat joint training on exchange_rate?** GRR 1.218 ±0.081 is the most
   surprising result here and has only a hypothesis behind it (ensembling across regimes).
2. **Is there any cheap signal that predicts merge-vs-sequential?** ρ is 3-of-4 and no
   candidate survives a chance-level check at n = 4. §3.1.
3. **Can redundancy and headroom be separated?** They co-vary across all four datasets. §3.3.
4. **Does α\* decrease as the number of segments grows?** Direct evidence on accumulating
   interference; 5-segment runs already exist. Falls out of §3.1.
5. **Is the residual interference real or inside the noise?** Needs per-metric floors applied
   to the validation block, not just the test block.
6. **Does merging beat the newest specialist on an unseen regime, for AD?** §3.6.
