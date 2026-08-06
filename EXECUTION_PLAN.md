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

Last updated 2026-08-06. Four datasets × three segment counts (n = 2, 3, 5) — 12
configurations, 48 jobs.

---

## 1. The research questions, and where each one stands

The thesis asks whether task arithmetic can evolve a model over time as data accumulates, on
**time series** — where shards are not disjoint the way image-classification classes are, but
either near-identical (stationary) or progressively drifting. Five questions follow from that,
and the plan below is organised around them rather than around methods.

| # | question | status |
|---|---|---|
| **Q1** | Does task arithmetic work when shards are *not* orthogonal? | **answered** — §2.11, and the theory in [THEORY.md §2](THEORY.md) and [§6.6](THEORY.md) |
| **Q2** | Merging, or continual fine-tuning? | **answered as a trade** — §2.11; neither dominates, and the winner depends on shard count |
| **Q3** | How large should a shard be? | **open** — §3.2. The sweep varied the *count*; it never asked which count is best |
| **Q4** | Accumulate forever, or materialise a new model? When? | **open, no evidence at all** — §3.3. No arm here materialises a second model |
| **Q5** | With several materialised models, which one do you use? | **open, no evidence at all** — §3.1 |

**Q1 is the contribution.** In image classification, task vectors for different classes are
near-orthogonal because the tasks are disjoint. Time-series shards are not: subspace overlap ρ
runs from 0.607 (SWaT) to 0.076 (ETTh1), and cosine between task vectors **decays with temporal
distance** (0.737 at distance 1 on SWaT against 0.095 on ETTh1). The finding is that
non-orthogonality does not break merging — **it sets the scale**. α\*·n = constant is precisely
the correction for summing n partially-aligned vectors: average rather than add. Alignment also
falls as shards get finer (0.824 → 0.633 from n = 2 to n = 5), which is why one rule holds at
every count.

**Q2's answer is a mechanism, not a leaderboard.** Nine decisive configurations split 5 merge /
4 sequential, with no signal separating them beyond chance. The two approaches degrade for
*different* reasons — sequential forgets (exchange_rate test MSE 0.220 → 0.362 → 0.531 as steps
accumulate), merging starves as each shard shrinks — and which one loses first depends on the
chunking. That is a stronger claim than "method X wins".

### What limits the work now

1. **A confound.** Count and shard size were varied together, so α\*·n = constant has an
   untested alternative explanation (§3.2).
2. **A blocker, and it may scope the thesis.** On AD the merge scale cannot be chosen without
   test access ([EXPERIMENTS.md §1.12](EXPERIMENTS.md)). Q3, Q4 and Q5 all need a signal that tracks quality
   on unlabelled data — the same signal that does not exist. **Either those questions are
   scoped to forecasting, or §3.4 has to be solved first.**
3. **Still few informative datasets.** SWaT and PSM are saturated (base within 1.1% and 3.4%
   of joint training); only ETTh1 and exchange_rate have real headroom.

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
- **State snapshots aliased the live model on CPU runs.** `{k: v.cpu() for k, v in
  model.state_dict().items()}` does not copy when the tensor is already on CPU —
  `Tensor.cpu()` returns *self* — so the snapshot tracked the model through every later
  `load_state_dict`. Effect on a CPU run: all task vectors collapse to zero, every merge
  scale scores identically, and the merged model is silently just the baseline. On the
  continual pipeline it made the L2-SP anchor follow θ_t, so the penalty vanished. Found by
  the α-selection smoke test, which reorders the merge ahead of the eval loop and exposed it.
  Fixed with `.detach().cpu().clone()` at all four sites. **No published result is affected**
  — every real run is GPU, where `.cpu()` copies; re-verified by recomputing all **71**
  merged checkpoints from their baseline + fine-tunes, 71/71 bitwise identical.
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

- **α\*·n is constant** within each dataset (≈1.0 on SWaT/PSM/ETTh1, ≈1.5 on exchange_rate).
  The optimal merge is a fixed multiple of the **mean** task vector, at any segment count.
  Confirmed by geometry: ‖τᵢ‖ shrinks with n while ‖Στᵢ‖ stays flat, so α tracks the *count*,
  not the magnitude.
- **Merge cost is flat in n** — ~1.0–1.1 on three datasets, 1.6–2.1 on exchange_rate.
- **Validation cannot select α on AD.** Val reconstruction and test AUROC disagree about α
  (SWaT: val optimum 0.5, AUROC monotone increasing to 1.5). Selecting on val costs 22–99% of
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

---

## 3. Next

Ordered by value per unit of compute, and mapped to the questions in §1. Q1 and Q2 are
answered; everything below serves Q3, Q4, Q5, or the blocker that gates all three.

### 3.1 Router — which materialised model do you use? ⬜ — do this first (Q5)

**No evidence exists on this question at all**, and the experiment needs no GPU: the transfer
matrix *is* the measurement. Rows are models, columns are regimes, so the column minimum is
the oracle router. The question is whether a cheap router recovers it.

Candidate router: score each available model on a short *recent* window and take the best.
Compare against (a) the column oracle, (b) always using the merged model, (c) always using the
newest specialist. The n = 5 diagnostics give six models × six regimes per dataset, already on
disk.

**Registered prediction:** on forecasting the router recovers most of the oracle, because
validation error and test error are the same quantity. On AD it fails, for the same reason α
selection fails ([EXPERIMENTS.md §1.12](EXPERIMENTS.md)) — reconstruction does not track detection. If that holds, it is the
second independent consequence of one root cause, which strengthens §3.4 considerably.

Cost: an afternoon of analysis, no cluster time.

### 3.2 Shard size — is there a rule? ⬜ (Q3)

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

**Folded in here: the count-versus-size control.** α\*·n = constant was measured with the
baseline fixed at 50%, so shard size fell as 1/n and the two are confounded. Holding size fixed
while varying n needs only existing knobs: `baseline_fraction = 1 − n·S`, giving 0.8 / 0.7 / 0.5
for n = 2 / 3 / 5 at S = 10% of train. **Registered prediction:** if α\*·n stays constant under
fixed shard size, the count explanation is confirmed and "the optimal merge is the mean task
vector" becomes a clean claim; if α\* tracks size instead, the invariant is an artefact of the
sweep design.

Cost: ~18 forecasting jobs plus diagnostics, all minutes-scale.

### 3.3 Accumulate or materialise, and when? ⬜ (Q4)

**The genuine gap.** Nothing in this project ever starts a second model, so "when to
materialise" has no evidence behind it — only the observation that merging degrades as shards
starve.

Design: from the n = 5 runs, merge τ₀…τ_j for increasing j and evaluate on segment j+1's
held-out slice — forward transfer onto a regime no vector in the merge has seen. Compare
against a model materialised on recent data only. **The materialisation point is where the
accumulated merge stops beating the fresh model on new data.**

Needs one flag: `--pipeline_merge_exclude_last N`. The n = 5 checkpoints already exist for every
dataset, so the merge side is training-free; only the freshly-materialised comparators need
training.

This also subsumes the old "AD forward transfer" item: on AD the test metrics cannot see model
quality, so an unlabelled held-out slice is the only way to observe forward transfer there.

### 3.4 Can α — or any quality signal — be chosen honestly on AD? ⬜ — the blocker

[EXPERIMENTS.md §1.12](EXPERIMENTS.md) measured the problem: validation reconstruction and test AUROC point to different α, so
val selection costs 22–99% of achievable GRR on AD against 1–8% on forecasting. **Q3, Q4 and Q5
all need the same missing thing**: a signal that tracks detection quality on unlabelled data.
If §3.1 also fails on AD, that is two independent symptoms of one cause.

Worth one focused attempt before conceding. Detection depends on the score *distribution*, not
its level, and mean reconstruction throws exactly that away. The curves already record
`reconstruction/score_{std,p50,p95,p99}` — check whether any spread statistic has its optimum
where AUROC does. **Analysis of finished runs, not new training.**

If none works, the honest conclusion is that unsupervised AD cannot tune the merge scale or
route between models, and the thesis scopes Q3–Q5 to forecasting while AD carries plain task
arithmetic with a pre-declared α (→ §3.5).

### 3.5 BECAME — adaptive coefficient only ⬜

Derives its merging coefficient rather than tuning it, which **sidesteps §3.4 entirely**: if α
cannot be selected honestly on AD, a method that computes it from the vectors is not a
refinement but the only way to run AD honestly. Skip the gradient-projection first stage
(class-incremental machinery; does not transfer). Log λ\* per step alongside ρ, and check it
against the measured α\* ≈ 1/n — a derivation that reproduces the empirical rule would be
strong corroboration for both.

### 3.6 `weather` as a fifth dataset ⬜

Drift 0.369, comparable to ETTh1, 21 features and 3× the rows. Already implemented and
registered. **Gate it** — one incremental + one standard, check headroom, commit the full arm
only if the gap is real. Check the window arithmetic first: `val_fraction` must give ≳150
validation windows, and at n = 5 that constraint bites (§2.11).

### 3.7 Separate redundancy from headroom ⬜

They co-vary across all four datasets, so both explain everything equally well. The
random-vs-chronological split control is the cheapest attack: IID shards give **high redundancy
with headroom preserved**, which no existing dataset does. Hold `baseline_fraction`,
`n_finetune_segments` and windows-per-shard fixed; Arm A = contiguous chronological shards
(current), Arm B = shards drawn IID from the same pooled region.

**Registered prediction:** Arm B should show high ρ with α\* → 1/n, and merging should win
there. §2.11 already found α\*·n ≈ 1 on chronological splits, so the α\* half is partly
answered — what remains is whether merging *wins* under high redundancy.

### 3.8 OPCM ⬜ — scope as memory, not accuracy

Merge cost is already ~1.0–1.1× and flat in n, so a better merging algorithm competes for a few
percent on something that is not the bottleneck. Its real selling point is **continual
merging** — folding in one model at a time without storing every task vector, which is directly
relevant to Q4.

**Registered prediction:** OPCM projects out the component of an incoming vector lying in the
span of the accumulated update. On SWaT that shared component *is* the signal (ρ = 0.607), so
OPCM should underperform plain task arithmetic there and be more competitive on exchange_rate
(ρ = 0.117). [EXPERIMENTS.md §1.15](EXPERIMENTS.md) adds a second prediction: alignment falls as
n grows, so OPCM's advantage should widen with the shard count.

### 3.9 More AD datasets ⬜ — low priority

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
- **`--exclude` is rejected by cluster policy** (*"using --exclude is deprecated. Use
  --constraint instead"*), and SLURM constraints have no negation — so excluding a node means
  listing every acceptable GPU feature positively. `scontrol update` on a queued job is also
  denied (`Access/permission denied`), and it silently drops `Partition`/`Account` when it
  does run: to change a submitted job's placement you must `scancel` and resubmit.

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

1. **Why does merging beat joint training on exchange_rate?** GRR above 1.0 at every segment
   count (1.421 / 1.164 / 1.238) is the most
   surprising result here and has only a hypothesis behind it (ensembling across regimes).
2. **Is there any cheap signal that predicts merge-vs-sequential?** ρ is 3-of-4 and no
   candidate survives a chance-level check at n = 4. §3.1.
3. **Can redundancy and headroom be separated?** They co-vary across all four datasets. §3.3.
4. **Does α\* decrease as the number of segments grows?** Direct evidence on accumulating
   interference; 5-segment runs already exist. Falls out of §3.1.
5. **Is the residual interference real or inside the noise?** Needs per-metric floors applied
   to the validation block, not just the test block.
6. **Does merging beat the newest specialist on an unseen regime, for AD?** §3.6.
