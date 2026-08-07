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

Last updated 2026-08-07. Six datasets (four with real headroom) × three segment counts
(n = 2, 3, 5). ETTh2/ETTm2 added for the drift-versus-scarcity test (§2.19, §2.20) and given
full diagnostics in §2.21.

---

## 1. The setting and the research questions

### The production setting this is for

A model θ₀ is trained on history and deployed. Every period — day, week, month — a new batch of
data arrives and the model should be brought up to date. **Some history can be retained, but not
all of it** (retention limits, volume, cost). Three decisions recur:

1. **How is the model updated?** Retrain on a window, fine-tune continually, or merge task
   vectors into the base.
2. **One model served, or several?**
3. **When does accumulating stop** and a new specialist get materialised?

The thesis question is whether **task arithmetic** — keep only what each period changed,
τᵢ = θᵢ − θ₀, and add the deltas back — is a viable answer, on **time series**, where shards are
not disjoint the way image-classification classes are but either near-identical (stationary) or
progressively drifting.

### The questions, and where each stands

| # | question | status |
|---|---|---|
| **Q0** | Is retraining on a retained window enough, making continual methods unnecessary? | **answered** — §2.16: merging is worth ≈2 periods of retained history. Keep ≤2 → merge; keep ≥3 → retrain |
| **Q1** | Does task arithmetic work when shards are *not* orthogonal? | **answered** — §2.11, §2.15; it works, and non-orthogonality sets the scale rather than breaking it |
| **Q2** | Merging, or continual fine-tuning? | **answered as a trade** — §2.11; neither dominates, the winner flips with shard count, and they fail differently |
| **Q3** | How large should a shard be — i.e. what update cadence? | **open** — §3.1; evidence leans to fewer, bigger shards |
| **Q4** | Accumulate, or materialise a new model? When? | **answered** — §2.15, §2.17: accumulation holds on unseen shards; materialise when drift makes recency beat accumulation, measurable free each period |
| **Q5** | With several models, which one do you serve? | **answered, with a caveat** — §2.12, §2.21: serve the merge on low-drift data (router recovers 6.1%); under strong drift a perfect router recovers 66–107%, so routing pays *if* it can be built. On AD it cannot be measured at all |

**Q0 is the one that decides the thesis's scope**, and it was implicit until the production
framing made it explicit. Merging's justification is *not* accuracy — a single fine-tune on the
unsplit data beats it on ETTh1 (§2.13). Its justification is that the unsplit fine-tune is
unavailable when data cannot be retained. But "cannot retain *all*" is not "cannot retain
*any*", so the honest baseline is a **sliding window** — and that gap has now been measured:
**merging is worth about two periods of retained history** (§2.16). Keep ≤ 2 → merge; keep ≥ 3 →
retrain on the window. That is the sharpest statement the project makes, and it turns the
question from "does merging work" into "what is your retention budget".

**Q1 is the contribution.** In image classification task vectors for different classes are
near-orthogonal because the tasks are disjoint. Time-series shards are not: subspace overlap ρ
runs 0.607 (SWaT) to 0.076 (ETTh1), and cosine decays with temporal distance. Non-orthogonality
does not break merging — **it sets the scale**: average rather than sum. And accumulated merges
**generalise forward**, beating the base on shards no task vector has seen in 7 of 8 cases
(§2.15), which is the case merging exists for.

**Q2's answer is a mechanism, not a leaderboard.** Nine decisive configurations split 5 / 4,
with no signal separating them beyond chance. They degrade for *different* reasons — sequential
forgets (exchange_rate 0.220 → 0.362 → 0.531 as steps accumulate), merging starves as shards
shrink — and which loses first depends on the chunking.

**Recency does not determine relevance.** Excluding the trivial last regime, the newest
specialist is the best model for a regime in **1 case out of 16** ([EXPERIMENTS.md §1.20](EXPERIMENTS.md)).
Which regime data belongs to is what matters — which is why old and new regimes are distinct
targets, and why routing has any headroom at all.

### What limits the work now

1. **A measurement limit, not a data limit.** The count-vs-size confound was tested (§2.14) and
   the strong α\*·n claim did not survive. What remains under-powered is the scoped form: at a
   0.1 α grid, quantisation alone gives ±0.50 at n = 5. The fix is resolution — and the clean
   isolation of *count* is training-free, falling out of §3.2.
2. **A blocker, and it may scope the thesis.** On AD the merge scale cannot be chosen without
   test access ([EXPERIMENTS.md §1.12](EXPERIMENTS.md)). Q3, Q4 and Q5 all need a signal that tracks quality
   on unlabelled data — the same signal that does not exist. **Either those questions are
   scoped to forecasting, or §3.3 has to be solved first.**
3. **Still few informative datasets.** SWaT and PSM are saturated (base within 1.1% and 3.4%
   of joint training); only ETTh1 and exchange_rate have real headroom.

**The practical decision rule** — merge, route, or keep fine-tuning — is consolidated in
[THEORY.md §11](THEORY.md), including what a labelled AD calibration set would and would not buy.

The regime-predictor question is **closed as a negative result**: at nine decisive
configurations no cheap signal separates merge-wins from sequential-wins, and the outcome is
not a property of a dataset in the first place.

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

**Settled:** 43.8% headroom and **merging beats training on all the data at once** — GRR
1.421 / 1.164 / 1.238 at n = 2 / 3 / 5, all above 1.0 with α chosen on validation, which no
other dataset shows. Config note: the shipped script's `val_fraction=0.1`
would have given 33 validation windows and broken fine-tuning exactly as on ETTh1; 0.25 used
instead.

### 2.6 Regime indicator ✅ built · ❌ does not generalise

ρ per merge step plus the exact orthogonal remainder ‖τ_k‖·√(1−ρ_k). Reported in
EXPERIMENTS.md §1.7, derived in [THEORY.md §5.4](THEORY.md).

**Settled, then superseded by §2.11.** At four single-`n` datapoints ρ was right on 3 of 4 and
wrong on exchange_rate. The segment sweep raised the sample to nine decisive configurations and
found **no predictor at all**, and — more importantly — that the outcome is not a property of a
dataset: it flips with the segment count. Use ρ to explain a result, never as a controller.

### 2.7 Bugs and confounds found ✅

- `merge_scale` is a **no-op in training** — sweeping it triples compute for byte-identical
  checkpoints. Removed from `TRAIN_INCREMENTAL_SWEEP` (§2.10).
- `val_fraction=0.10` breaks ETTh1 fine-tuning (113 validation windows; one task vector never
  trains). Affected 11 of 15 sweep trials.
- The validation slice is the temporal **tail** of each segment, so it sits adjacent to the
  next one and inflates the off-diagonal — `specialisation` therefore *understates* the truth.
  Not fixable by random splitting: adjacent windows share 119 of 120 timesteps.
- L2-SP's "actively harmful on ETTh1" result **does not reproduce** at `val_fraction=0.15`.
- **L2-SP re-tested cleanly: no measurable effect** (EXPERIMENTS.md §3.1). Three seeds per
  variant on ETTh1 with α selected per variant; the effect is inside the within-variant spread and its
  *sign flips* depending on whether you read absolute MSE, ratio-to-own-baseline, or
  hardware-matched pairs. Both the old "harmful" claim and its rejection are settled as
  "no resolvable difference".
- **Run-to-run variation is GPU-driven, not seed-driven** (EXPERIMENTS.md §3.2). Same seed +
  same GPU model → bit-identical; different GPU → up to **18.8%** apart on ETTh1, with no
  consistent per-GPU direction. The [EXPERIMENTS.md §1.9](EXPERIMENTS.md) floors already fold this in (they were measured on
  mixed hardware), so they remain the right yardstick. Hardware **cannot** be pinned here:
  single-model `--constraint` is rejected, `--nodelist` is refused, ≥24G needs another
  partition. Match hardware post hoc from the job logs for any comparison near the floor.
- **The early-stopping val loss included the L2-SP penalty** (fixed 2026-08-06). Selection was
  therefore biased toward earlier, less-trained checkpoints whenever `reg_lambda > 0`, on top
  of the intended regularisation, and the val number scaled with λ so cross-λ comparison was
  confounded. **26 runs used λ > 0** — all in the discarded grid searches and
  `etth_vf15_regsweep`; **no §1 headline number derives from any of them**, verified by
  checking `reg_lambda` across all 223 runs on disk. The bias ran *against* λ > 0, so the
  "not harmful" null survives as conservative while the original "harmful" claim does not.
- **State snapshots aliased the live model on CPU runs.** `{k: v.cpu() for k, v in
  model.state_dict().items()}` does not copy when the tensor is already on CPU —
  `Tensor.cpu()` returns *self* — so the snapshot tracked the model through every later
  `load_state_dict`. Effect on a CPU run: all task vectors collapse to zero, every merge
  scale scores identically, and the merged model is silently just the baseline. On the
  continual pipeline it made the L2-SP anchor follow θ_t, so the penalty vanished. Found by
  the α-selection smoke test, which reorders the merge ahead of the eval loop and exposed it.
  Fixed with `.detach().cpu().clone()` at all four sites. **No published result is affected**
  — every real run is GPU, where `.cpu()` copies; re-verified by recomputing all **87**
  merged checkpoints from their baseline + fine-tunes, 87/87 bitwise identical.
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

### 2.9 Honest α selection inside the training pipeline ✅

`--pipeline_select_merge_scale_on_val` (off by default) makes the *existing* candidate grid
decisive instead of observational. Before: `--pipeline_merge_scale` was committed ahead of
training and the good α was only ever identified afterwards, in analysis, by reading the
minimum off the curve's **test** column — so the checkpoint on disk was never the model the
reported numbers described. Now the same candidate set (`{merge_scale} ∪ extra_merge_scales`)
is evaluated on the merged-val union *before* the merge, and its winner builds `merged/`.

- **Costs no extra compute.** Those val passes are ones the curve already performed; they
  just run earlier and get used. The curve CSV is unchanged.
- **Refuses on AD, before training.** `AdValEvaluator` declares no `selection_metric()`, so
  the run dies in the first second rather than after 1.5 h of SWaT. Defensible because α\*
  does not move across seeds on either AD dataset (±0.000).
- **Also refuses** a single-candidate grid (nothing to select between) and a dataset with no
  `VAL` capability.
- New `merged/merge_scale_selection.csv` (one row per candidate) makes the choice auditable;
  `merge_scale/selected` is added to the `merged/val` metrics, so `collect_sweep_results`
  picks it up with no collector change. `config.json` still records the *requested* α.
- **Priced from existing data before building it** (EXPERIMENTS.md §1.10): val lands within
  one grid step of the test optimum every time, costing 3.3% mean on ETTh1 and 5.6% on
  exchange_rate. exchange_rate's GRR stays above 1.0 on all three seeds — 1.164 ± 0.080
  honest, against 1.218 ± 0.081 oracle. **Merging still beats joint training.**

Verified end-to-end on ETTh1: the checkpoint is bitwise the merge at the selected scale, the
curve at that scale reproduces the `merged/` results, and the selection CSV matches the
curve's val column. Both refusal paths fire before any epoch runs.

### 2.10 `merge_scale` removed from `TRAIN_INCREMENTAL_SWEEP` ✅

It was a cross axis in all three sweeps, training bitwise-identical checkpoints three times
over for a parameter applied after training. **15 → 9 trials each, 18 jobs saved.**

- **Forecasting** (`etth_forecast`): the axis is replaced by per-trial selection —
  `--pipeline_extra_merge_scales 0.0…1.0` plus `--pipeline_select_merge_scale_on_val` in
  `_INCREMENTAL_EXTRA_ARGS`, so every trial reports a deployable α (§2.9) instead of three
  oracle variants of one model.
- **AD** (`psm`, `swat`): selection refuses there, so α is pinned to the established α\* —
  PSM 0.5, SWaT 0.25 — which does not move across seeds.

Verified by rendering each sweep's first trial into a full command line (architecture
injected as `--arch-args` does at submit time) and parsing it with the real `main.py` parser:
all three parse, ETTh1 with 11 candidates and selection on, PSM/SWaT with a single pinned α.

### 2.11 The segment-count sweep ✅ — 12 configurations, 48 jobs

`n_finetune_segments ∈ {2, 3, 5}` × 4 datasets × {merge, sequential} + diagnostics. Submitted
and collected 2026-08-06; 32 training jobs then 16 training-free diagnostics, all completed.
Full numbers in [EXPERIMENTS.md §1.11–§1.15](EXPERIMENTS.md).

**n = 8 is unreachable** — ETTh1 would give each segment a 130-row validation slice against a
144-row window requirement. Checked against the real split arithmetic *before* submitting.

What it settled:

- **α\*·n is constant for a fixed baseline** (≈1.0 on SWaT/PSM/ETTh1, ≈1.5 on exchange_rate).
  **The stronger claim — a constant of the dataset — is falsified** by the fixed-shard control:
  under a varying baseline exchange_rate grows 0.97 → 1.58 (§2.14). Two measurement caveats:
  the apparent exact seed agreement was 0.1-grid quantisation, and quantisation contributes
  ±0.50 at n = 5.
- **Merge cost is flat in n** — ~1.0–1.1 on three datasets, 1.6–2.1 on exchange_rate.
- **Validation cannot select α on AD.** Val reconstruction and test AUROC disagree about α
  (SWaT: val optimum 0.5, AUROC monotone increasing to 1.5). Selecting on val costs 84–99% (SWaT) / 5–19% (PSM) of
  the achievable GRR on AD versus 1–8% on forecasting. **Every AD merge number in the project
  is oracle-selected**, and that is now measured rather than argued.
- **Merge-vs-sequential is not a dataset property** — it flips with n on exchange_rate.
- **No regime predictor at n = 9** (§2.6).
- **Withdrawn:** an intermediate reading that GRR falls with n on every dataset. That came
  from comparing rows computed at different α; under uniform recomputation only PSM declines.

**Methodological note worth keeping.** Every command was generated from a published run's own
`config.json` and parsed with the real `main.py` parser before submission, and each diagnostics
job was matched to a *compatible* joint run rather than merely a same-seed one — four datasets
have leftover architecture-sweep runs at the same seed that would have failed the config guard.
Both checks caught real errors at zero cluster cost.

### 2.12 Routing versus merging ✅ — use the merge (Q5)

The transfer matrix's column minimum is the best possible router, so the distance from the
merged model to it bounds what routing could ever gain. At n = 5, with merged taken at α\*.
⚠️ **Superseded by §2.21**, which recomputed every figure from a pinned definition and added
ETTh2/ETTm2: **+6.1% (ETTh1), +66.0% (ETTm2), +81.0% (ETTh2), +106.9% (exchange_rate)**, with
the AD figures withdrawn as not computable. The original reading below was based on the four
datasets then available.

**Settled:** a *perfect* router recovers ≤7% **on low-drift data** — not worth storing n
models and building selection logic that can be wrong. Under strong drift it recovers far more,
and §2.21 shows the split is drift, not shard size. Merging also beats
always-use-the-newest-specialist on all four. Routing only has real headroom under strong drift
(exchange_rate), which is the same place merging beats joint training. On AD it is doubly
unattractive: little headroom *and* no way to select without labels (§3.4).

Full numbers: [EXPERIMENTS.md §1.16](EXPERIMENTS.md). Cost: no cluster time — the measurement
was already on disk.

### 2.13 The n = 1 baseline ✅ — merging is not an accuracy win (Q1/Q2)

One fine-tune on the whole post-baseline block, unsplit. The null hypothesis the project had
never run. Best merge versus n = 1: **SWaT +0.24%, PSM +1.01%, ETTh1 −9.25%, exchange_rate
+8.38%** against floors of 0.13 / 0.07 / 8.20 / 5.29%.

**Settled, and it reframes the headline claim.** Merging does **not** beat training on the same
data unsplit — ETTh1 loses outright, the AD wins clear their floors but are practically
negligible, and only exchange_rate wins by a margin that matters. **Merging's justification is
the streaming constraint, not accuracy**: the unsplit fine-tune is unavailable when data arrives
over time and cannot be retained. Now stated with a number rather than assumed.

Where merging *does* win is where regimes differ most — consistent with §2.12 and with
exchange_rate being the only dataset that also beats joint training.

[EXPERIMENTS.md §1.17](EXPERIMENTS.md). 8 jobs, minutes each on the forecasting datasets.
### 2.14 The fixed-shard-size control ✅ — the strong α\*·n claim is falsified

Hold the shard at 15% of train at every n and let the baseline absorb the difference
(`baseline_fraction = 1 − 0.15n` → 0.70 / 0.55 / 0.25). ETTh1 and exchange_rate, 3 seeds, α on
a 0.05 grid. 18 jobs, minutes each. [EXPERIMENTS.md §1.18](EXPERIMENTS.md).

**Settled — and it cost the headline theoretical claim some reach.** ETTh1 stays flat
(1.00 / 1.10 / 1.08) even though its baseline degrades nearly twofold across the variants;
exchange_rate grows (0.97 / 1.25 / 1.58) with the endpoints separated after accounting for both
seed spread and grid quantisation. **α\*·n is therefore not a constant of the dataset.** The
surviving claim is *for a fixed baseline* — still the operationally relevant one, since in
deployment the base is fixed.

Two measurement findings that matter more than the result:

- **The apparent exact seed agreement on α\* was a 0.1-grid artefact.** At 0.05 the seeds
  differ by one step. Every "±0.000 across seeds" statement about α\* is withdrawn.
- **Quantisation contributes ±(grid × n)** — at n = 5 on a 0.1 grid that is **±0.50**, a 50%
  uncertainty on the quantity whose constancy was being claimed. The validation minimum is
  sharp enough to test the hypothesis (15–51% penalty between α·n = 1.0 and 1.5, against floors
  of 8.2% and 5.3%), so **the limit is resolution, not the data or the number of datasets.**

**Do not add datasets to shore this up** — at a 0.1 grid they would add imprecise points and
make the claim look better supported without testing it better. The clean isolation of *count*
(prefix merges, base and shard size both fixed) is training-free and falls out of §3.2.
### 2.15 Prefix merges ✅ — forward transfer works; the α\*·n question is unidentifiable

`--pipeline_prefix_merges` on the diagnostics pipeline (opt-in, off by default): merge τ₀…τ_k
for every prefix across the α grid, on the validation columns. Training-free — the n = 5
checkpoints already exist. 6 jobs, ~9 min each.
[EXPERIMENTS.md §1.18–§1.19](EXPERIMENTS.md).

**Settled — forward transfer (the accumulate half of Q4).** Each prefix scored on the first
shard it has *not* seen, at the α chosen only on shards it has: the merge beats the base model
in **7 of 8 cases**, with **no decay as vectors accumulate** (ETTh1's best is at k = 4). This is
the first direct evidence for the case merging exists for — generalising forward, not just
retaining. It also bounds the shard-starvation story over this range.

**Settled — and it closes the α\*·n question, negatively.** α\*·k is *not* constant under this
design either (ETTh1 0.48 → 0.83, exchange_rate 1.20 → 1.50). More importantly, chasing it
exposed why: on a fixed series `baseline + n × shard = total`, so fixing any two forces the
third to move and **the count can never be isolated**. Three designs, three different
confounds, no fourth available. **Do not spend more jobs on this** — the regularity stands as an
empirical statement in the deployment parameterisation and nothing further is establishable.

**Still open:** the materialise comparator. We now know accumulating keeps helping on unseen
shards up to k = 4, but not when a freshly built model would have been better. That needs
training and a comparator design (§3.2).
### 2.16 The sliding-window baseline ✅ — Q0 answered: merging is worth ~2 periods of history

**`W` counts periods of retained data**, not to be confused with `window_len` (the model's input
length in timesteps). θ₀ held byte-identical while fine-tuning on the last W periods only (existing knobs:
`baseline_fraction = 1 − 0.1W`, `baseline_use_fraction = 0.5/(1 − 0.1W)`). 24 jobs, all four
datasets. [EXPERIMENTS.md §1.21](EXPERIMENTS.md).

**Settled — and it is the quantitative form of the whole thesis claim.** On both datasets with
real headroom the crossover is at **W = 3**: merging beats retaining one or two periods
(by 131% and 55% on exchange_rate) and loses to retaining three (by 13–20%). AD is saturated and
separates nothing.

> **Keep ≤ 2 periods of history → merge. Keep ≥ 3 → retrain on the window.**

Merging's value is therefore *not* accuracy but **delivering roughly a two-period window's
accuracy while storing no data at all, only weight deltas.** Whether that trade is worth taking
is a question about the retention budget, not about the method — which is a far more useful
thing to be able to tell a practitioner than "merging works".

**Second finding: on small shards, a recent window beats all history.** W = 3 beats W = 5 on
exchange_rate by 26% and on PSM by 0.7%, both outside their floors, and it explains why the
n = 1 baseline (§2.13) was not the ceiling it looked like. ⚠️ **The original wording attributed
this to drift; §2.19–2.20 falsified that.** ETTh2 (0.753) and ETTm2 (0.752) carry
exchange_rate's drift (0.833, same 5-way measure) with 2× and 9× its data, and on both, old
data *helps* — by 34% on ETTm2.
The sign of the effect tracks **shard size, not drift**: thin shards favour a recent window,
large shards favour all history. "Retrain on everything" is the wrong default only under data
scarcity.

### 2.17 Accumulate or materialise ✅ — the trigger is measurable for free (Q4)

Both variants were already on disk: `merge(τ₀…τ_{k−1})` is accumulation, and `ft_{k−1}` — a
specialist on the most recent shard alone — is a freshly materialised model. Scored on the
shard neither has seen. [EXPERIMENTS.md §1.22](EXPERIMENTS.md).

**Settled: the rule is drift-dependent.** On mildly drifting ETTh1 accumulation ties or wins
throughout and pulls ahead at k = 4. On strongly drifting exchange_rate the freshest specialist
usually predicts the next regime better.

**The operational trigger:** materialise when drift is strong enough that recency beats
accumulation on the next period — measurable every period at **no extra cost**, because a fresh
specialist is trained each period anyway. Compare it to the accumulated merge on the newest
held-out slice; branch when the fresh model starts winning. This is the same
performance-triggered rule the dynamic-weighted-ensemble literature arrives at
([THEORY.md §15](THEORY.md)).
### 2.18 Dataset screen ✅ — better candidates than `weather` exist

Model-free drift screen over every remaining `thuml/Time-Series-Library` dataset, the same
method that picked exchange_rate over traffic. One CPU job.

| dataset | rows | features | drift | KS |
|---|---|---|---|---|
| exchange_rate | 6,071 | 8 | **0.833** | 0.641 |
| **ETTh2** | 13,936 | 7 | **0.753** | 0.616 |
| **ETTm2** | 55,744 | 7 | **0.752** | 0.616 |
| ETTh1 | 13,936 | 7 | 0.412 | 0.298 |
| ETTm1 | 55,744 | 7 | 0.410 | 0.299 |
| weather | 42,157 | 21 | 0.382 | 0.407 |
| national_illness | 773 | 7 | 0.297 | 0.456 |
| electricity | 21,044 | 321 | 0.284 | 0.187 |
| traffic | 14,036 | 862 | 0.106 | 0.078 |

> **Not comparable to the older drift figures.** This screen segments 5 ways; §2.5's numbers
> segment 3 ways, so ETTh1 reads 0.412 here against 0.309 there. The *ranking* is unchanged.
> **Re-measure both consistently before either table goes in the thesis.**

**Two candidates beat `weather`, which is only a replication of the ETTh1 drift regime:**

- **ETTh2 — a near-controlled experiment for drift.** Same rows, features, sensor type and
  source as ETTh1, different transformer station, and **drift 0.753 against 0.412**. Everything
  the project claims is drift-dependent (merging wins under strong drift, routing has headroom
  under strong drift, materialise under strong drift) becomes testable with size, dimensionality
  and domain held fixed.
- **ETTm2 — separates drift from scarcity.** Same drift as exchange_rate with **9× the data**.
  This addresses a real confound in the current results (EXPERIMENTS.md §3.0): exchange_rate
  carries the strongest conclusions *and* has the thinnest shards, and nothing so far tells
  those apart.

**Next step, awaiting a decision:** gate both on headroom first — 2 jobs each — since drift alone
is not sufficient (PSM has drift 0.465 and is still saturated). `weather` is deprioritised.
### 2.19 ETTh2 ✅ — the drift explanation does not survive

Implemented `Etth2ForecastDataset` (two class attributes on the shared HF base) and ran the full
replication: headroom gate, sliding window W ∈ {1,2,3}, n = 1, merging at n ∈ {2,3,5} with α
selection, continual at n ∈ {2,3,5}. 36 jobs, 3 seeds.
[EXPERIMENTS.md §1.23](EXPERIMENTS.md).

Chosen as the closest available **controlled experiment for drift**: identical rows, features,
frequency, sensor type and window sizing to ETTh1, different station, ~double the drift. It
passed the gate with **83.8% headroom**, the largest of any dataset here.

**What replicated:** merging beats small retention budgets; α\*·n is order 1 (0.97 / 0.80 / 0.75).

**What did not, and this is the important part:**

- **"A recent window beats all history" fails.** On ETTh2 more data is monotonically better and
  joint training wins outright — the opposite of exchange_rate, despite ETTh2 having nearly the
  same drift statistic.
- **The retention crossover moves** to W = 5 (from W = 3 on ETTh1 and exchange_rate), so it is
  dataset-dependent rather than a constant of the method.
- **Continual beats merging at every n** (−12.0 / −9.3 / −7.0%) — the ETTh1 pattern, not the
  exchange_rate one.

**Consequence: the segment-shift drift statistic does not predict which strategy wins.**
exchange_rate (0.833) and ETTh2 (0.753) behave oppositely; ETTh2 matches ETTh1 (0.412). What
ETTh2 shares with ETTh1 and not exchange_rate is **data volume** — 13,936 rows against 6,071.
That is the first direct evidence on the scarcity confound (EXPERIMENTS.md §3.0): **every
"under strong drift, X" claim should be read as "on exchange_rate, X"** until ETTm2 — same drift
as exchange_rate, 9× the data — settles it. **ETTm2 is now the highest-value remaining run.**

Why the statistic misses it: segment-shift measures how much period means *differ*, not whether
they *trend*. Old data is stale only if the series moves away and stays away; large but
oscillating variation leaves old periods still relevant. A trend-aware measure would likely
predict better, and none is implemented.
### 2.20 ETTm2 ✅ — scarcity, not drift. The confound is resolved.

Implemented `Ettm2ForecastDataset` and ran the full replication — gate, window W ∈ {1,2,3},
n = 1, merge n ∈ {2,3,5} with α selection, continual n ∈ {2,3,5}. 36 jobs, 3 seeds.
[EXPERIMENTS.md §1.24](EXPERIMENTS.md). Headroom 86.6%; note the wide **14.1% base seed spread**.

ETTm2 holds drift at exchange_rate's level (0.752 vs 0.833) with **9× the data**, and has an
almost identical period-to-test distance profile to ETTh2 — so regime structure is nearly fixed
while volume varies.

**Settled: exchange_rate's distinctiveness is a small-data effect.** Three of its four
distinctive behaviours fail to reproduce — old data hurting (ETTm2: using everything is 34%
*better* than a 3-period window), the early W = 3 crossover (ETTm2: W = 5), and the elevated
α\*·n (ETTm2: 0.75–0.80, not ≈1.5). **Retention claims of the form "under strong drift, X"
are restated as "on small shards, X" — with one exception: routing headroom orders by drift,
not size (§2.21).**

**The one behaviour that does reproduce has its own mechanism.** Merge overtakes continual at
n = 5 on ETTm2 (+24.7%) as on exchange_rate — but because **continual collapses** (0.0920 →
0.1839 from n = 3 to n = 5) while the merge holds. That is a property of chaining update steps,
not of a dataset, and it is the one general result in the merge-versus-continual comparison:
**the more steps you chain, the worse continual gets; merging wins by not collapsing.**

**Also settled, negatively:** the drift *shape* hypothesis. Measuring each period's distance to
the test period gives near-identical profiles on all four forecasting datasets, with the test
period nearest a *middle* period every time — so no drift statistic available here, trend-aware
or not, distinguishes the cases. Recorded in EXPERIMENTS.md §1.23.

### 2.21 Diagnostics for ETTh2/ETTm2 ✅ — and routing headroom is drift, not size

ETTh2 and ETTm2 had been run for one question only (drift or scarcity?) and were never given
the diagnostics every other dataset has. Twelve jobs closed the gap: `MergeDiagnosticsPipeline`
at n = 3 and n = 5, three seeds each, joint run matched per seed for the GRR reference
(all twelve pairs verified compatible on every `dataset_*`/`mae_tx_*` argument before
submission). This produced their transfer matrices, merge-scale curves, GRR and specialisation
— [EXPERIMENTS.md §1.4](EXPERIMENTS.md), §1.16, §1.23, §1.24.

**The result overturns a claim made earlier the same day.** §2.20 concluded that *every*
"under strong drift, X" statement should be restated as "on small shards, X". Routing headroom
is the exception, and a large one:

| dataset | drift (5-way) | shard @ n=5 | merged vs oracle router |
|---|---|---|---|
| exchange_rate | 0.833 | 607 | **+106.9%** |
| **ETTh2** | 0.753 | 1,393 | **+81.0%** |
| **ETTm2** | 0.752 | 5,574 | **+66.0%** |
| ETTh1 | 0.412 | 1,393 | **+6.1%** |

The ordering is drift's; shard size gets it wrong (it would put ETTm2 last). The decisive case
is **ETTh1 vs ETTh2** — identical rows, features, frequency, window sizing *and* shard size,
differing only in station and drift — where headroom differs by a factor of thirteen. So:
**retention behaviour tracks shard size, routing headroom tracks drift.** Both were corrected
in place rather than removed.

**Two integrity problems surfaced and are fixed:**

- **The §1.16 routing table had no script of record.** It could not be re-derived — a faithful
  recomputation reproduced ETTh1 to within a point but missed exchange_rate by 5 points and its
  newest-specialist column by 157. The definition is now pinned in
  `analysis/routing_report.py` (a new post-hoc entry point alongside `diagnose` and
  `geometry_report`), and every published figure regenerated from it.
- **The AD routing numbers are withdrawn, not recomputed.** On AD the transfer matrix's
  per-regime columns carry only `reconstruction/score_*`; `window_auroc` exists on the `test`
  column alone, because interior validation slices have no labels. A per-regime oracle router
  is therefore **not computable on AD at all**, so the published SWaT +6.2% / PSM +7.8% cannot
  follow from the stated method. This reinforces §3.2 rather than weakening it.

### 2.22 Full recomputation audit ✅ — every published scalar re-derived from the runs

Three ad-hoc errors found on 2026-08-07 made it clear that no published number could be trusted
without a script of record. Every quantity in EXPERIMENTS.md was therefore recomputed from the
415 run directories on disk, and the calculations committed as three entry points —
`analysis/results_audit.py` (new), `analysis/routing_report.py` (new) and the existing
`analysis/geometry_report.py`. The definitions are written up in
[EXPERIMENTS.md §0.6](EXPERIMENTS.md), which is the section to hand to anyone auditing a number.

**Verified correct — reproduced from the runs:**

- ETTh1 task-vector norms **1.851 / 1.532 / 1.102** against published 1.85 / 1.53 / 1.10, and
  the derived "alignment" figures 0.824 / 0.738 / 0.633 = ‖Στ‖ ÷ Σ‖τ‖.
- SWaT subspace overlap **ρ = 0.608** against published 0.607.
- α\*·n: ETTh1 1.00 / 1.00, exchange_rate 1.40 / 1.50, ETTh2 0.97 / 0.80 / 0.75,
  ETTm2 0.77 / 0.80 / 0.75.
- exchange_rate's **5.29%** reproducibility floor, exactly.
- Merge cost: ETTh1 **1.02–1.07** against published "~1.05×", exchange_rate **1.63–2.04**
  against published "1.6–2.1×".
- Specialisation, independently matching the pipeline's own values (ETTh2 0.447, ETTm2 0.694).
- **All three SWaT rows** of the honest-α table (0.116 / 0.707 / 84%, 0.006 / 0.483 / 99%,
  0.064 / 0.947 / 93%).

**Corrected:**

- **The PSM rows of §1.12 — a headline claim.** No single validation signal reproduces the
  published 22% / 37% / 42%: PSM n=2 needs `score_p99` while the SWaT rows need `score_mean`,
  and PSM n=5's value is not reproduced by any of fifteen signal × column-subset combinations.
  Under one consistent definition PSM costs **6% / 19% / 5%**. So *"validation cannot select α
  for anomaly detection"* is what **SWaT** shows, not what AD shows — the honest claim is that
  val selection on AD is **unreliable, not impossible**, and nothing observable without labels
  says which case you are in.
- **The routing ordering (§1.16), for the second time.** The clean four-way ordering by drift
  holds only at n = 5; across all segment counts exchange_rate is erratic (13–107%). What
  survives is a stable *separation* — ETTh1 5.6–6.1% at every n against ETTh2 81–108% and
  ETTm2 65–66% — anchored by the controlled ETTh1/ETTh2 pair.
- Three percentages in §1.23–§1.24 (wrong denominator), and the §1.9 floor caveat.

**A bug in the new audit tooling itself,** caught by hand-checking one cell: `routing_report`
built its column list from `(model, column)` keys without deduplicating, so it saw each column
once per model and mispaired every specialist in the merge-cost and specialisation loops.
Routing headroom was unaffected (the duplication weights each column equally) but merge cost
read 0.93 where the correct value is 1.74. Fixed, and the corrected output now matches both a
hand computation and the pipeline's independently-computed specialisation.

**Still not derivable from anything on disk**, flagged in place rather than restated: AD routing
headroom (no per-regime detection metric exists — the columns carry only reconstruction
statistics) and ETTh1's published 8.20% floor (its third seed's run is not identifiable among
the surviving experiments).

---

## 3. Next

Ordered by value per unit of compute, and mapped to the questions in §1. Q1 and Q2 are
answered; everything below serves Q3, Q4, Q5, or the blocker that gates all three.

### 3.1 Shard size — is there a rule? 🟡 **DEFERRED** (Q3)

> **Why deferred.** This was the top question when merging looked like the primary strategy.
> §2.16 scoped merging to the **≤ 2 retained periods** case, so you are merging two or three
> vectors and the choice of n is narrow by construction. Second-order now. Revisit only if a
> better merging method (§3.6) pushes the retention crossover up.

The sweep varied the shard *count* and never asked which count is best. The existing GRR data
already leans: on PSM (0.903 → 0.497) and exchange_rate (1.421 → 1.238), **fewer and bigger
shards win**, consistent with the starvation mechanism — bigger shards make better specialists,
and merging faithfully reproduces whatever it is given. The competing pressure is that a shard
spanning two regimes learns their average.

**The rule to test: a shard should be as large as possible without spanning a regime change.**
Within-shard versus between-shard drift is measurable model-free and before any training (the
`drift.py` statistics). Sweep n, compute that ratio, and check whether the n minimising it is
the n maximising GRR. If they coincide, this is the project's first *usable* rule — a
shard-size criterion computable in advance.

**Also folded in here: refine the AD α grid.** The AD curves step α by 0.25 against 0.10 on
forecasting, so SWaT's α\*·n spreads 1.67× against ETTh1's 1.11× — and 1/3, the value the rule
predicts at n = 3, is not on SWaT's grid at all. The α\*·n evidence is therefore established on
forecasting and only *consistent with* AD. Re-running the AD curves at 0.05 steps needs **only
the validation columns**, not the expensive test pass, so it is a fraction of the 84-minute
full diagnostics.

**Folded in here: the count-versus-size control.** α\*·n = constant was measured with the
baseline fixed at 50%, so shard size fell as 1/n and the two are confounded. Holding size fixed
while varying n needs only existing knobs: `baseline_fraction = 1 − n·S`, giving 0.8 / 0.7 / 0.5
for n = 2 / 3 / 5 at S = 10% of train. **Registered prediction:** if α\*·n stays constant under
fixed shard size, the count explanation is confirmed and "the optimal merge is the mean task
vector" becomes a clean claim; if α\* tracks size instead, the invariant is an artefact of the
sweep design.

Cost: ~18 forecasting jobs plus diagnostics, all minutes-scale.

### 3.2 Can α — or any quality signal — be chosen honestly on AD? ⬜ — the blocker

[EXPERIMENTS.md §1.12](EXPERIMENTS.md) measured the problem: validation reconstruction and test AUROC point to different α, so
val selection costs 84–99% of achievable GRR on SWaT but only 5–19% on PSM, against 1–8% on
forecasting — unreliable rather than impossible (§2.22). **Q3, Q4 and Q5
all need the same missing thing**: a signal that tracks detection quality on unlabelled data.
If §3.1 also fails on AD, that is two independent symptoms of one cause.

Worth one focused attempt before conceding. Detection depends on the score *distribution*, not
its level, and mean reconstruction throws exactly that away. The curves already record
`reconstruction/score_{std,p50,p95,p99}` — check whether any spread statistic has its optimum
where AUROC does. **Analysis of finished runs, not new training.**

**A second and probably stronger route: synthetic anomaly injection.** Inject known corruptions
into held-out *normal* validation data — point outliers, level shifts, sensor freezes, amplitude
scaling — and compute AUROC against those synthetic labels. That yields a validation signal
measuring **detection rather than reconstruction**, without any real labels, which is exactly
the quantity [EXPERIMENTS.md §1.12](EXPERIMENTS.md) shows is missing.

It is directly falsifiable: real labels exist on test, so measure whether the α chosen by
synthetic-AUROC matches the oracle α there. If it does on PSM, the blocker is solved and Q3–Q5
unblock on AD. **Caveat to state rather than assume:** it will only select well if the synthetic
corruptions resemble real ones, which for SWaT's subtle process attacks is doubtful — and that
is itself measurable.

If neither route works, the honest conclusion is that unsupervised AD cannot tune the merge
scale or route between models, and the thesis scopes Q3–Q5 to forecasting while AD carries
plain task arithmetic with a pre-declared α (→ §3.5).

### 3.3 BECAME — adaptive coefficient only ⬜

Derives its merging coefficient rather than tuning it, which **sidesteps §3.4 entirely**: if α
cannot be selected honestly on AD, a method that computes it from the vectors is not a
refinement but the only way to run AD honestly. Skip the gradient-projection first stage
(class-incremental machinery; does not transfer). Log λ\* per step alongside ρ, and check it
against the measured α\* ≈ 1/n — a derivation that reproduces the empirical rule would be
strong corroboration for both.

### 3.4 More datasets ✅ **DONE for now** — ETTh2 (§2.19) and ETTm2 (§2.20) both run

**ETTh2 is done (§2.19) and it falsified the drift explanation**, which promotes ETTm2 from
"third datapoint" to the run that settles the project's main remaining ambiguity.

exchange_rate carries the most distinctive results in the project *and* is by far the smallest
dataset. ETTh2 has nearly exchange_rate's drift with twice the data and behaves like ETTh1
instead — pointing at **data volume, not drift**, as the driver. **ETTm2 has exchange_rate's
drift (0.752 vs 0.833) with 55,744 rows — nine times as much.** If exchange_rate's behaviour
reproduces there it is drift; if it disappears it was scarcity. Gate on headroom first (2 jobs).

*(Superseded plan for `weather` — drift 0.382 makes it a replication of the ETTh1 regime.)*

### 3.4b `weather` 🟡 **DEFERRED** — superseded by ETTh2/ETTm2

Drift 0.369, comparable to ETTh1, 21 features and 3× the rows. Already implemented and
registered. **Gate it** — one incremental + one standard, check headroom, commit the full variant
only if the gap is real. Check the window arithmetic first: `val_fraction` must give ≳150
validation windows, and at n = 5 that constraint bites (§2.11).

### 3.5 Separate redundancy from headroom 🟡 **DEFERRED**

> **Why deferred.** A mechanism question that does not change any decision in the rule of
> [THEORY.md §11](THEORY.md). Worth doing for a mechanistic chapter, not for the conclusions.

They co-vary across all four datasets, so both explain everything equally well. The
random-vs-chronological split control is the cheapest attack: IID shards give **high redundancy
with headroom preserved**, which no existing dataset does. Hold `baseline_fraction`,
`n_finetune_segments` and windows-per-shard fixed; Variant A = contiguous chronological shards
(current), Variant B = shards drawn IID from the same pooled region.

**Registered prediction:** Variant B should show high ρ with α\* → 1/n, and merging should win
there. §2.11 already found α\*·n ≈ 1 on chronological splits, so the α\* half is partly
answered — what remains is whether merging *wins* under high redundancy.

### 3.6 OPCM ⬜ — orthogonal projection, and a sharp prediction from our own geometry

**Correction (2026-08-06):** this item previously described OPCM as "scope as memory, not
accuracy". That was wrong about the mechanism. OPCM's method is **orthogonal projection**: when
folding a new task vector into the accumulated merge, it keeps only the component orthogonal to
what is already there, discarding the part lying in the existing span. Not storing every vector
is a *consequence* of merging continually, not the point of the method — the point is to stop
repeated updates pulling the merge the same way.

That makes it far more interesting here than the old framing suggested, because **this project
has measured exactly the quantity OPCM acts on.**

**Registered predictions, all from measurements already in hand:**

1. **Where the shared component *is* the signal, OPCM should lose.** On SWaT ρ = 0.607 — most of
   each new vector already lies in the accumulated span, so orthogonalising discards most of it.
   Plain task arithmetic should win there, and OPCM should be competitive on exchange_rate
   (ρ = 0.117) where the vectors are nearly new each time.
2. **OPCM's advantage should widen with the shard count**, because alignment falls as shards get
   finer (0.824 → 0.633 from n = 2 to n = 5, [EXPERIMENTS.md §1.15](EXPERIMENTS.md)).
3. **It is a different composition rule from the one we measured as optimal.**
   [EXPERIMENTS.md §1.18](EXPERIMENTS.md) finds the
   best plain merge is ≈ the *mean* of the task vectors; orthogonalisation is not averaging. On
   aligned time-series shards — which is the regime this whole project is about, unlike the
   near-orthogonal image tasks these methods were designed for — those two rules disagree
   strongly, and which wins is genuinely open.

**Judge it against the retention crossover, not merge cost.** Merge cost is already ~1.05× and
flat in n, so a few percent there changes nothing anyone decides. The question that matters is
whether OPCM moves the crossover in §2.16 — *does it make merging worth more than two periods of
retained history?* That reframing is itself a contribution: it is how these methods should be
benchmarked for a streaming setting.

### 3.7 More AD datasets 🟡 **DEFERRED** — but this is a scope decision, not a priority call

> **The tension to settle with your supervisor.** This repository is `incremental-ad`, yet every
> informative result is *forecasting*. SWaT and PSM are saturated — a frozen base on half the
> data is within 1.1% and 3.4% of full training — so **no update strategy can demonstrate
> anything on them**, and on top of that α cannot be selected honestly there (§3.2). Two honest
> routes: **(a)** reframe the scope to time series generally, with AD as a case where blind
> metrics and a missing α signal make the question harder — which is itself a finding; or
> **(b)** add a drifting, non-saturated AD dataset (SMD / MSL / SMAP, each needing a custom
> loader like `swat.py`) so the AD side becomes real. Deferred as engineering, not as a
> question — the choice should be deliberate rather than discovered at writing time.

SMD / MSL / SMAP would broaden a side where both current datasets are saturated, but need a
custom loader like `swat.py` rather than the `HfSeriesForecastDataset` base. Every AD result is
handicapped by §3.4, so forecasting datasets buy more per unit of work until that is resolved.

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
- **Nodes `aldo`, `giacomo`, `giovanni` kill any job.** They carry `gpu_RTXPro6000B_96G` —
  Blackwell, sm_120 — and the pinned `torch 2.5.1+cu121` ships no kernel for it. The job
  allocates, looks healthy for ~30 s, then dies at the first CUDA launch with *"no kernel
  image is available for execution on the device"*. Five of the first 32 segment-sweep jobs
  were lost to this before it was diagnosed. `scripts/sbatch_run_command.sh` now carries a
  `--constraint` naming every *other* GPU type; drop it only once torch is rebuilt for sm_120.
- **You cannot pin the GPU model.** `--constraint=gpu_2080Ti_11G` alone is rejected ("we
  advise you to allow also …"), `--nodelist` is refused ("you can't request specific nodes"),
  and any model with ≥24G VRAM demands the `boost_usr_prod` partition — use that only when the
  job genuinely needs the memory. Since results are bit-identical within a GPU model and
  differ across models (§2.7), the only way to compare at hardware parity is to read the GPU
  out of each job log afterwards and compare matched pairs.
- **`--exclude` is rejected by cluster policy** (*"using --exclude is deprecated. Use
  --constraint instead"*), and SLURM constraints have no negation — so excluding a node means
  listing every acceptable GPU feature positively. `scontrol update` on a queued job is also
  denied (`Access/permission denied`), and it silently drops `Partition`/`Account` when it
  does run: to change a submitted job's placement you must `scancel` and resubmit.

**Generating runs**

Regenerate commands from a finished run's own `config.json` rather than retyping them — the
same trick `analysis/diagnose.py` uses. Every replication variant in §2 was produced that way, and
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

1. **Why does merging beat joint training on exchange_rate?** GRR above 1.0 at every segment
   count (1.421 / 1.164 / 1.238) is the most surprising result here and has only a hypothesis
   behind it (ensembling across regimes).
2. ~~Is there any cheap signal that predicts merge-vs-sequential?~~ **Closed as a negative
   result** — §2.6, [EXPERIMENTS.md §1.14](EXPERIMENTS.md). No signal separates nine decisive
   configurations beyond chance, and the outcome is not a property of a dataset.
3. ~~Does α\* decrease as the number of segments grows?~~ **Answered** — §2.11: α\*·n is
   constant, so α\* falls exactly as 1/n.
4. **Can redundancy and headroom be separated?** They co-vary across all four datasets. §3.7.
5. **Is the residual interference real or inside the noise?** Needs per-metric floors applied
   to the validation block, not just the test block.
6. **Does merging beat the newest specialist on an unseen regime, for AD?** §3.3 — the
   forward-transfer variant, which is the AD-side view of *accumulate or materialise*.
7. ~~Does merging beat a single fine-tune on all the post-baseline data?~~ **Answered — no.**
   §2.13: merging is not an accuracy win over the unsplit fine-tune; its justification is the
   streaming constraint.
