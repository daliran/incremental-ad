# Execution Plan — status, results, and what's next

Living plan. Supersedes the original `phases_1_4_plan.md`. **Self-contained on purpose**:
written so that a session with no prior context can pick up from here without re-deriving
anything.

- [EXPERIMENTS.md](EXPERIMENTS.md) — source of truth for numbers. If this file disagrees, that one wins.
- [TASK_ARITHMETIC_NOTES.md](TASK_ARITHMETIC_NOTES.md) — the concepts and why the measurements mean what they mean.
- [PHASE1_RUNBOOK.md](PHASE1_RUNBOOK.md) — operational procedure for the diagnostics.
- [CLAUDE.md](CLAUDE.md) — code map and repo invariants.

**Status legend:** ✅ done · 🟡 partly done · ⬜ not started · ❌ dropped (with reason)

Last updated 2026-08-05. **All figures are at three training seeds per dataset**; `±` is the
half-range and differences are only claimed where intervals do not overlap.

[EXPERIMENTS.md §1](EXPERIMENTS.md) is the complete self-contained data snapshot — this file
carries only what is needed to decide what to do next.

> **Replication complete (2026-08-05).** All three datasets now have **three training seeds**
> for the continual arm and for the diagnostics, plus Standard-pipeline seeds. Every figure
> below carries a ± half-range, and differences are only claimed where intervals do not
> overlap.

---

## 0. Where the project actually stands

Read this before planning anything.

### The headline result

> **Task arithmetic merges incrementally fine-tuned models at essentially no cost — within
> 1–8% of keeping a separate specialist per regime, with no measurable forgetting — provided
> the merge scale α is set correctly. What earlier looked like interference was almost
> entirely a scale error.**

**3 training seeds per dataset; ± is the half-range.**

| | merge cost @ α=1.0 | merge cost @ α\* | α\* | forgetting @ α\* |
|---|---|---|---|---|
| SWaT | 4.00× | **1.079 ±0.002** | 0.250 ±0.000 | **1.013 ±0.012** |
| PSM | 1.19× | **1.008 ±0.020** | 0.500 ±0.000 | 1.139 ±0.053 |
| ETTh1 | 1.75× | **1.007 ±0.026** | 0.367 ±0.050 | **0.967 ±0.029** |

On PSM and ETTh1 the merge cost is **statistically indistinguishable from 1.00**. α\* is
perfectly stable across seeds on both AD datasets, which matters given fact 3: a pre-declared
α is defensible there because the optimum does not move.

*Merge cost* = merged error ÷ that shard's own specialist, averaged over shards; 1.00 = free.
*Forgetting* = merged ratio on `val_base`. α\* is selected on the mean of the shard val
slices, **never on test**.

### Versus joint training (GRR)

| | base (50% data) | joint training | gap | GRR @ α\* |
|---|---|---|---|---|
| SWaT (AUROC) | 0.8005 | 0.8089 | 1.0% of base | *inside noise — 0.13% floor* |
| PSM (AUROC) | 0.7740 | 0.8002 | 3.4% of base | **0.857 ±0.031** |
| ETTh1 (MSE) | 0.7023 | 0.4271 | 39.2% of base | 0.767 ±0.141 |

### Seven facts that drive everything else

1. **ETTh1 drifts; SWaT and PSM barely do.** One fact, many symptoms: ETTh1's base→joint gap
   is 39% of base while SWaT's is 1.1%; ETTh1's specialists differ hugely on test
   (0.7355 → 0.4503) while SWaT's differ by 0.004; only ETTh1 shows a recency gradient.
2. **AD detection metrics cannot see model quality.** AUROC/AUPRC are rank-based and the runs
   use `threshold_strategy=oracle`, which sweeps its own threshold — so both are invariant to
   any monotone rescaling of scores. On SWaT a 5× change in reconstruction moved AUROC by
   +0.008. The val block is the only instrument that sees anything on SWaT.
3. **Consequence: in unsupervised AD the merge scale cannot be tuned honestly.** Val measures
   reconstruction, test measures detection, they come apart (on PSM val-selected α makes test
   *worse*: GRR 0.958 → 0.890), and AD training data has no labels by construction so there is
   no third signal. **Use a fixed, pre-declared α for AD; select on val for forecasting.**
4. **Entanglement concentrates on the newest shard** (SWaT 1.28×, PSM 1.11×, ETTh1 1.53×;
   ~1.0 elsewhere). Mechanical: α\* < 1 scales *every* vector down, and the last segment sits
   furthest from θ₀.
5. **Merging is not for maximising performance on the newest data.** If you only serve the
   latest regime, keep the latest specialist. Merging earns its place when one model must
   serve *all* regimes without a replay buffer.
6. **The reproducibility floor is dataset-specific, and the old 2% assumption was wrong both
   ways** (§5.3, measured over 3 seeds): PSM **0.07%** sd, SWaT **0.13%**, ETTh1 **8.20%**.
   So PSM's result is ~50 sd and rock solid; SWaT's GRR is marginal rather than meaningless;
   ETTh1's "real shortfall" is withdrawn. **The instability is in absolute metrics only** —
   ETTh1 ratios vary 3.8%. Report ETTh1 as ratios, and treat its GRR as fragile since it
   mixes models from two runs.
7. **Versus sequential fine-tuning, merging wins outright only on SWaT** (§4.1, 3 seeds).
   There it is better on both new and old data (0.656 ±0.001 vs 0.697 ±0.011; 1.013 ±0.012 vs
   1.635 ±0.049). On PSM and ETTh1 sequential adapts better to the new periods and the
   old-regime difference is **inside seed noise**. An earlier claim that merging protects the
   base regime on all three is **withdrawn**. SWaT is also the dataset whose updates repeat
   each other most (ρ = 0.607), which is the pattern: **merging's advantage appears where
   sequential training would drift the same way repeatedly.** Forgetting under sequential
   fine-tuning tracks the same quantity: SWaT ends 59% worse on the base regime, ETTh1 shows
   none.

---

## 1. Phase 1 — diagnostics from existing checkpoints ✅ COMPLETE

No training. Everything reused checkpoints already on disk.

### 1.1 Per-segment specialist evaluation ✅

Built as `MergeDiagnosticsPipeline`; exactly the Part A (adaptation matrix on per-segment val
slices) + Part B (detection columns on the unpartitioned test set) structure the original plan
specified. The plan's instruction **not** to partition the test set was followed.

Results at α = 1.0 (the source runs' own scale):

| | `specialisation` | `diag_ratio_mean` | `offdiag_ratio_mean` |
|---|---|---|---|
| SWaT | +0.101 | 0.608 | 0.709 |
| PSM | +0.082 | 0.753 | 0.835 |
| ETTh1 (59071) | +0.076 | 0.732 | 0.808 |

**Verdict against the plan's decision table:** strong diagonal everywhere (0.61–0.75, far from
1.0), positive specialisation everywhere → the task vectors carry real, segment-specific
signal. **Redundancy and degenerate fine-tuning are both ruled out as the global explanation.**

### 1.2 Merge-scale curves ✅ — and extended beyond the plan

The plan asked for AUROC and pa_f1 traced against α. **That is not sufficient**, because those
metrics are blind (fact 2 above). Added `--pipeline_curve_include_val` to trace the **val
block** — actual model quality — against α. That extension is what found the overshoot and is
the most important change to the original plan.

Curve shapes against the plan's interpretation table: all three peak in 0.25–0.50, i.e.
"vectors overlap, sum overshoots". **But the plan then files that under partial
disentanglement, which conflates two separable things** — see §7.

### 1.3 Geometry ✅

All seven measurements from the plan exist in `framework/merging/geometry.py`: global cosine,
per-tensor cosine, principal angles, effective rank, sequential overlap ρ, norms, cosine vs
temporal distance.

| | mean off-diag cosine | subspace overlap ρ | effective rank (of 3) | mean ‖τ‖/‖θ₀‖ |
|---|---|---|---|---|
| SWaT | 0.737 | **0.607** | 1.63 | 0.0061 |
| PSM | 0.399 | 0.226 | 2.51 | 0.0059 |
| ETTh1 | 0.240 | 0.076 | 2.35 | 0.0185 |

*ETTh1 is run 59077 (`val_fraction=0.15`, healthy fine-tunes). The superseded 59071 at 0.10
had a dead task vector, which depressed both its cosine and its rank — a good illustration of
why effective rank must be read alongside the norms.*

**Cosine decays with temporal distance on all three** (SWaT 0.771→0.667, PSM 0.466→0.265,
ETTh1 0.121→0.042). The plan flagged this as the check that could invalidate the framing —
**it passed**.

Two readings worth carrying forward:

- **Pairwise cosine in ℝ^D is a weak statistic** — near-orthogonality is the default in high
  dimensions. Subspace overlap ρ is the meaningful one, and it says SWaT's fine-tunes are 61%
  redundant with each other.
- **Effective rank conflates direction and magnitude.** ETTh1's 1.88 is low because one vector
  was *dead*, not because two were parallel. Always read it with the norms.

### 1.4 Merge-order permutation ❌ deferred to Phase 4

Correct as written: plain TA is commutative, so permutations are identical by construction.

---

## 2. Phase 2 — controls

### 2.1 Random vs chronological split ⬜ NOT DONE — highest-priority control

Unchanged in motivation, and **more important than the plan assumed**: we now know the
temporal structure also affects *evaluation* (the tail-test-split effect, EXPERIMENTS.md §2), not just training.

One refinement: the plan predicts Arm B (IID shards) should show *high cosine but
non-destructive* overlap. We can now sharpen that — predict high **ρ** (subspace overlap), not
just high cosine, and predict α\* → 1/n rather than → 1.0, since with fully redundant vectors
the sum is n copies of one direction.

### 2.2 SWaT saturation curve 🟡 partly superseded

The claim is now **directly measured**: frozen base on 50% of the data reaches AUROC 0.8005;
joint training on 100% reaches 0.8089. **The entire prize available to any adaptation method
is 0.0084**, inside the reproducibility floor. That is stronger than the fraction×epochs grid
the plan proposed.

Still worth producing the grid as a *figure* for the thesis, and the plan's placement advice
holds — put it **before** the merging results so SWaT reads as a designed control rather than
a rationalisation.

### 2.3 n_finetune_segments sweep 🟡 runs exist, never diagnosed

Runs on disk at n ∈ {2, 3, 5} for all three datasets; **n = 8 missing**; none diagnosed with
the current tooling. The plan's confound handling (fixed-total vs fixed-per-segment data) is
still right.

New reason to run it: α\* should *decrease* as n grows if interference accumulates. We have
α\* for n = 3 only. This is now a cheap, high-information sweep because the diagnostics are
training-free.

---

## 3. Phase 3 — tokenisation confound

### 3.1 Matched-token PSM grid ⬜ not run

The assert the plan asked for **is implemented** (`_assert_pretext_non_degenerate` in `MaeTx`,
called from `_build`). It is mode-aware — `mask_ratio` is inert outside `RANDOM_MASK` — and it
rejects 10/15 PSM and 4/9 SWaT `MODEL_SWEEP` cross trials.

### 3.2 Reclassify p25 results 🟡

PSM p25 runs exist and were diagnosed (78071 covers p5; p25 runs 58957–58965 were not
re-diagnosed). Note the guard now **rejects** p25 at window 100, so those runs cannot be
reproduced without changing the window.

---

## 4. Phase 4 — baselines and methods

### 4.1 Continual fine-tuning arm ✅ DONE (2026-08-05)

Implemented as `ContinualFineTuningPipeline`. θ₀ → seg 0 → seg 1 → seg 2, each step from the
*previous* model, evaluated after every step on every region; writes `backward_transfer.csv`,
`test_by_step.csv`, and ACC / BWT. L2-SP via the existing `reference_state`, anchored to θ₀ by
default (`--pipeline_anchor_to_baseline`).

**Result — a stability/plasticity split, not a winner.** Val block, ratio to each run's own
baseline, TA taken at α\*:

**3 training seeds per dataset; ± is the half-range. A difference is only claimed when the
two intervals do not overlap.**

| dataset | ρ | sequential — new | merged — new | verdict | sequential — old | merged — old | verdict |
|---|---|---|---|---|---|---|---|
| SWaT | 0.607 | 0.697 ±0.011 | **0.656 ±0.001** | **merge** | 1.635 ±0.049 | **1.013 ±0.012** | **merge** |
| PSM | 0.226 | **0.702 ±0.013** | 0.759 ±0.015 | **sequential** | 1.159 ±0.024 | 1.139 ±0.053 | *tie* |
| ETTh1 | 0.076 | **0.560 ±0.038** | 0.705 ±0.019 | **sequential** | 1.110 ±0.143 | 0.967 ±0.029 | *tie* |

- **Merging wins outright only on SWaT** — better on both new and old data, neither interval
  overlapping.
- On PSM and ETTh1, **sequential adapts better to the new periods** and the old-regime
  difference is **inside seed noise**.
- **Forgetting under sequential FT tracks collinearity**: SWaT (ρ = 0.607) gives the textbook
  curve 1.115 → 1.209 → **1.586**; ETTh1 (ρ = 0.076) shows none (BWT = −0.031).

> **Merging's advantage appears exactly where the task vectors are most collinear** — the
> opposite of the naive expectation that redundancy makes merging pointless.

An earlier single-seed draft claimed task arithmetic protects the base regime on *all three*.
**Withdrawn** — with error bars it does so demonstrably only on SWaT.

Full detail: EXPERIMENTS.md §1.6. **Open:** the L2-SP sweep on the CL arm has not been run.

### 4.2 OPCM ⬜

The plan's **registered prediction is now quantitatively supported** and should be stated with
the numbers: OPCM projects out the component of an incoming vector that lies in the span of
the accumulated update. Here the shared component *is* the signal — subspace overlap is 0.607
on SWaT and 0.226 on PSM — so OPCM should underperform plain TA on the AD datasets, and be
more competitive on ETTh1 where ρ = 0.076.

**Expected accuracy headroom is small**: merge cost is already 1.01–1.08×, so a better merging
algorithm competes for 1–8%. The GRR shortfall does *not* come from merging — it comes from
the specialists not containing what joint training learns. The real reason to run OPCM is its
**streaming property** (merge one model at a time, don't store every task vector), which is a
memory argument, not an accuracy one. Scope it that way.

### 4.3 BECAME — adaptive coefficient only ⬜

Unchanged. The frame-mismatch discussion (option (a) recommended) still applies.

One addition: BECAME derives its coefficient rather than tuning it, which **side-steps the
AD α-tuning problem in fact 3**. That is a genuinely strong argument for it in this project —
possibly stronger than its accuracy.

### 4.4 λ\* trajectory logging ⬜

Unchanged. Feeds the materialisation question.

### 4.5 Merge-order permutation ⬜

Unchanged; requires 4.2 and 4.3.

---

## 5. New items discovered during Phase 1

### 5.1 All-regime test evaluation ⬜ — a *second* question, not a fix

**The tail split is not a bug.** For forecasting it reproduces the deployment condition: you
always predict forward. Keep it as the primary evaluation. What it does mean is that the
current test number answers one specific question, and it isn't the one merging is built for.

Three deployment scenarios, three correct evaluations:

| deployment | right evaluation | what the data says |
|---|---|---|
| **Forward-only** — only ever predict the next period | tail test *(current)* | **keep the newest specialist.** ETTh1 `ft_2` = 0.4503 beats merged 0.4920 |
| **Serve all regimes** — any regime may reappear | held-out slices from every segment | merging wins: merge cost 1.02× |
| **Unknown regime** — cannot tell which regime the next data belongs to | held-out *unseen* regime (§5.2) | **unmeasured** |

**The third row is the crux.** Keeping the newest specialist is only correct *if the next data
resembles the newest training data*. In production that is an assumption you usually cannot
verify — serving n specialists requires a router, and a router needs to know the regime.

> **Merging's value proposition is routing-free insurance**, not peak accuracy on the freshest
> data. The newest specialist wins there. Merging's claim is that it does not fall apart when
> the incoming data does *not* match the most recent regime.

ETTh1 already shows the risk that insures against: `ft_0` scores **0.7355** on test, *worse
than the frozen base* (0.7023). A specialist tuned to the wrong regime is actively harmful.

**Action:** report tail-test as primary, add all-regime as a clearly-labelled second column.
Scope of the recency effect: **ETTh1 only** — PSM shows no recency ordering (its merge beats
every specialist) and SWaT's ordering is inside the noise floor.

### 5.2 Held-out future regime ⬜ — **the key experiment**, promoted

Per §5.1 this is what actually tests merging's value proposition.

Add a segment and **do not** fine-tune on it. Evaluate base / each specialist / merged / joint
on it. Asks whether merging accumulates *transferable* knowledge or merely stores regimes.
Nothing run so far answers this, and it is arguably the question deployment cares about.

### 5.3 Measure the noise floor ✅ DONE (2026-08-05) — **and it changed §0's verdicts**

Three training seeds per dataset, identical config. The universal 2% assumption was wrong in
**both** directions:

| | sd across seeds | range | assumed |
|---|---|---|---|
| PSM window_auroc | **0.07%** | 0.13% | 2.0% |
| SWaT window_auroc | **0.13%** | 0.25% | 2.0% |
| ETTh1 forecast/mse | **8.20%** | 16.3% | 2.0% |

- **PSM's "matches joint training" is far stronger than stated** — a 3.4% gap against a 0.07%
  floor, ~50 sd.
- **SWaT's GRR is marginal, not meaningless** — 1.1% gap against 0.13%. The blanket "noise
  divided by noise" dismissal in EXPERIMENTS §8 was too harsh.
- **ETTh1's "REAL 5.2% shortfall" is withdrawn** — inside an 8.2% sd.

**The instability is in *absolute* metrics only.** ETTh1's ratio-to-own-baseline varies 3.8%
across the same seeds. **Report ETTh1 as ratios; GRR is fragile there** because it mixes
models from two different runs.

### 5.4 Fix `val_fraction` on ETTh1 ✅ diagnosed, ⬜ not propagated

`val_fraction = 0.10` gives 113-window val slices, and early stopping fires on a lucky epoch-1
reading: `finetune_0` stops at epoch 1 with ‖τ‖/‖θ₀‖ = 0.0012 against siblings at 0.0198 /
0.0283. Run 59077 is identical except `val_fraction = 0.15` and its fine-tunes are healthy
(best epochs 7 / 13 / 6 versus 1 / 8 / 19).

**11 of 15 ETTh1 `train_incremental` sweep trials used 0.10**, so those runs carry a dead task
vector. The two regularised arms were re-run at 0.15 (EXPERIMENTS.md §2); the rest of that
sweep was retired rather than repeated.

The original plan predicted this exactly ("count the windows before interpreting anything") —
promote it from caveat to precondition.

### 5.5 Honest α selection inside the training pipeline ⬜

`--pipeline_merge_scale` is fixed before training, so every result used whatever was guessed.
The pipeline already builds a merged-val dataset spanning all regimes
(`get_merged_val_eval_dataset`). Having it sweep α on that and use the winner for `merged/`
would make future runs correct by construction — **for forecasting**. For AD see fact 3: there
is no honest signal, so the flag should refuse to auto-select and require an explicit α.

---

## 6. Corrections to the original plan

Anything below contradicts `phases_1_4_plan.md`; this file wins.

**6.1 §1.2 must trace the val block.** Test-only curves cannot show the overshoot on AD.

**6.2 §1.1's decision rule is wrong at α = 1.0.** *"If merged is worse, that is genuine
interference"* — at α = 1.0 that is mostly overshoot. The rule holds only with merged
evaluated at α\*.

**6.3 Overshoot ≠ entanglement.** The plan's §1.2 table files "peak at 0.3–0.7" under partial
disentanglement. They are separable:

> total merge cost = **overshoot** (curable by scaling α) + **irreducible interference**
> (what remains at α\*)

Measured split: overshoot dominates everywhere; interference is 1–8%.

**6.4 Entanglement is a comparison, not a cell value.** It is merged ÷ specialist on the same
shard, at α\*. A cell can look excellent and still be entangled — SWaT `val_2` reads 0.616
(38% better than base) while its specialist alone reaches 0.481, so the merge gave up 28%.

**6.5 §1.1's selection-bias caveat has an opposite-signed twin.** The plan notes early stopping
on `val_i` *inflates* R[i][i]. There is also **val-tail adjacency**: `val_i` is the temporal
tail of segment i (`val_tail_split`), so it sits next to segment i+1 and hands `ft_{i+1}` an
unearned advantage. Tested where both neighbours are equidistant, the successor beats the
predecessor on **3 of 3** datasets (SWaT 0.638 vs 0.698, PSM 0.665 vs 0.845, ETTh1 0.583 vs
0.908). This *deflates* the diagonal relative to the off-diagonal, so `specialisation`
**understates** the truth.

Not fixable by random splitting: with `stride=1, window_len=120` adjacent windows share 119 of
120 timesteps, so a random split would leak almost completely. **The tail split is the correct
design**; the bias is a limitation to document, not a bug to fix.

**6.6 The plan never asks what the test set contains.** See §5.1.

---

## 7. Operational knowledge — do not rediscover this

**Cluster:**

- **`sbatch --export=VAR=value` gets the job `CANCELLED by 0`** about two seconds in, with
  **no log file written at all**. Measured: no `--export` ✅, `--export=ALL` ✅,
  `--export=ALL,FOO=bar` ❌, `--export=FOO=bar` ❌, `--export=NONE` ❌. **Pass variables as an
  env prefix** and let the default `--export=ALL` carry them:
  ```bash
  SOURCE_RUN=$W/runs/exp/59077 STANDARD_RUN=$W/runs/std/59062 \
      sbatch scripts/sbatch_merge_diagnostics.sh
  ```
- **The session scratchpad under `/tmp` is node-local** and invisible to compute nodes. Never
  reference it from an sbatch script; put helper files on `$WORK`.
- **`geometry_report` OOMs on the login node at ~65 run dirs** in one invocation. Split per
  experiment.
- Heavy work goes through SLURM, never the login node.

**Code state (all committed to working tree, not pushed):**

- `MergeDiagnosticsPipeline` gained **`--pipeline_curve_include_val`** — off by default so
  existing behaviour and cost are unchanged. On, it traces every val column at every α.
- **`merge_scale_curve.csv` schema changed**: now
  `merge_scale, column, block, metric, value, n_windows, eval_seed` (was `split` instead of
  `column, block`). Readers must be updated.
- Summary keys: test metrics stay unprefixed for backward compatibility; val columns are
  namespaced `{column}/{metric}/scale_at_min`.

**Verified invariants (re-check after any merging change):**

- All **60** `merged/checkpoints/best.pt` under `$WORK/runs` recompute **bitwise** from their
  baseline + fine-tunes.
- Curve at α = the run's own scale reproduces the matrix `merged` test row exactly; curve at
  α = 0 reproduces `base` exactly; hand-computed GRR matches `result.json` to 1e-9.

---

## 8. Priority order

**Done 2026-08-05:** 4.1 continual fine-tuning arm ✅ · 5.3 noise floor ✅ · 5.4 ETTh1
`val_fraction` repair ✅ (2 runs; §9.4 of EXPERIMENTS.md)

1. **5.1 all-regime test evaluation** — cheap, and §5.1's three-scenario framing needs the
   measurement to back it
2. **5.2 held-out future regime** — the deployment question, and now the sharpest open one:
   does the merged model beat the newest specialist on a regime nobody trained on?
3. **2.1 random vs chronological** — the control the premise rests on
4. ~~**Seed the CL arm**~~ ✅ done — 3 seeds everywhere; merge cost at α\* is
   1.079 ±0.002 / 1.008 ±0.020 / 1.007 ±0.026, and α\* is 0.250 ±0.000 / 0.500 ±0.000 /
   0.367 ±0.050
5. **L2-SP sweep on the CL arm** — the stability/plasticity trade-off is now measured, so the
   question "does any λ match the merged model on the base regime?" is well-posed
6. **2.3 n_finetune_segments sweep, diagnosed** — α\* vs n
7. **4.3 BECAME** — derives α, side-stepping the AD tuning problem (fact 3)
8. **4.2 OPCM** — scope as a *streaming/memory* argument, not accuracy
9. **2.2 SWaT saturation figure** — evidence exists; this is presentation
10. **Remove `merge_scale` from `TRAIN_INCREMENTAL_SWEEP`** — verified no-op in training,
    triples compute for identical checkpoints (EXPERIMENTS.md §2)
11. **4.4, 4.5, 3.1** — after the above

## 9. Reference

**Headline runs and their diagnostics:**

| dataset | source run | standard ref | diagnostics (α=1.0) | diagnostics (+ val curve) |
|---|---|---|---|---|
| SWaT | `slurm_grid_swat_train_incremental/58941` | `..._train_standard/58930` | 78070 | **79148** |
| PSM | `slurm_grid_psm_train_incremental/59101` | `..._train_standard/59090` | 78071 | **79147** |
| ETTh1 (vf 0.10, dead ft_0) | `slurm_grid_etth_forecast_train_incremental/59071` | `..._train_standard/59060` | 78072 | 79146 |
| ETTh1 (vf 0.15, healthy) | `slurm_grid_etth_forecast_train_incremental/59077` | `..._train_standard/59062` | — | **79153** |

Use the **bold** ones. ETTh1 should be quoted from **59077**: 59071's diagonal contains a
fine-tune that never trained, which flatters its merge cost.

**Commands:**

```bash
# geometry — CPU, seconds, no dataset. Split per experiment or the login node OOMs.
python -m incremental_ad.analysis.geometry_report $RUNS_ROOT/<experiment>/* \
    --out $RUNS_ROOT/analysis/geometry_<name>

# full diagnostics incl. the val-vs-alpha curve — GPU, via SLURM, env prefix not --export
SOURCE_RUN=$W/runs/<exp>/<id> STANDARD_RUN=$W/runs/<std>/<id> \
    MERGE_SCALES="0.0 0.25 0.5 0.75 1.0 1.25 1.5" \
    EXTRA_ARGS="--pipeline_curve_include_val" \
    sbatch scripts/sbatch_merge_diagnostics.sh
```

Walltimes with the val curve on: ETTh1 ~35 s, PSM ~22 min, SWaT ~1 h 31 m.

**Where results land:**

```
$RUNS_ROOT/<source experiment>_diagnostics/<run_id>/merge_diagnostics/
  transfer_matrix.csv    model, column, block, metric, value, ratio_to_base, n_windows, eval_seed
  merge_scale_curve.csv  merge_scale, column, block, metric, value, n_windows, eval_seed
  result.json            summary scalars
  source.json            which run was analysed, its merge_scale, columns, window counts
```

**Interactive results page:** https://claude.ai/code/artifact/b9478404-0c3d-4629-a43b-0aedb1ee24bf

---

## 10. Open questions

- Does merging beat sequential fine-tuning? *(4.1 — unanswered, and it is the question)*
- Does the merged model generalise to a regime nobody fine-tuned on? *(5.2)*
- Is the 1–8% residual interference real or inside the noise? *(needs 5.3)*
- Does α\* decrease as the number of segments grows? *(2.3)*
- Is the drift-vs-partition-size confound resolved? *(2.1)*
- Can a cheap geometric signal decide *merge vs materialise*? Current evidence is weak — two
  usable GRR points, correctly ordered, is not evidence. ρ and λ\* are the candidates.
