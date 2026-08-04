# Task Arithmetic for Incremental Learning — Concepts, Theory, and What We Measured

Study notes. Self-contained: readable without any prior conversation, and designed to be
pasted into an LLM as context for follow-up questions.

**Scope.** The concepts behind merging fine-tuned models by adding their weight deltas,
the measurements that tell you whether it worked *and why*, and what those measurements
actually said on SWaT, PSM and ETTh1 in this repo.

**Relationship to the other docs.** [EXPERIMENTS.md](EXPERIMENTS.md) is the source of truth for numbers —
if this file and that one disagree, that one wins. [PHASE1_RUNBOOK.md](PHASE1_RUNBOOK.md) is the operational
procedure. [CLAUDE.md](CLAUDE.md) is the code map. This file is the *why*, and it is the only one of
the four that explains the reasoning rather than recording results.

Every number below is from run 58941 (SWaT), 59101 (PSM), 59071 (ETTh1) and their
diagnostics runs 78070 / 78071 / 78072, all single evaluation seed.

---

## Contents

1. [The setup](#1-the-setup)
2. [Three ways it can fail, all looking identical](#2-three-ways-it-can-fail-all-looking-identical)
3. [The transfer matrix](#3-the-transfer-matrix)
4. [Geometry](#4-geometry)
5. [Interference](#5-interference)
6. [Entanglement and disentanglement](#6-entanglement-and-disentanglement)
7. [Metrics that cannot see what you are measuring](#7-metrics-that-cannot-see-what-you-are-measuring)
8. [What the three datasets said](#8-what-the-three-datasets-said)
9. [Conclusions: supported and not supported](#9-conclusions-supported-and-not-supported)
10. [Open questions and the next measurements](#10-open-questions-and-the-next-measurements)
11. [Quick reference](#11-quick-reference)

---

## 1. The setup

### 1.1 A model is a point in a very large space

A trained network with *D* parameters is one point θ ∈ ℝ^D. Here D is 1,917,695 (SWaT),
649,341 (PSM), 714,780 (ETTh1). Distances and angles in that space are the objects of study.

### 1.2 Task vectors

- Train a **base model** θ₀ on the first half of the training data — the "early regime."
- Cut the remaining data into *n* segments (shards) in time order.
- Fine-tune **from θ₀** on each shard independently → θ₁ … θₙ. Never shard-to-shard, always
  from the same starting point. This matters: it makes the deltas comparable.
- The **task vector** is the difference:

  > **τᵢ = θᵢ − θ₀**

  One arrow in weight space encoding "everything fine-tuning on shard *i* changed."

These arrows are small. On SWaT ‖θ₀‖ = 110.46 while the τ's are 0.487 / 0.719 / 0.817 —
about **0.4–0.7% of the model's length**. Fine-tuning nudges; it does not relocate.

### 1.3 Task arithmetic

The bet is that arrows **compose**:

> **θ_merged = θ₀ + α · (τ₀ + τ₁ + … + τₙ)**

α is the *merge scale*. α = 1 sums them; α = 1/n averages them.

### 1.4 Why anyone cares

If this works you get incremental learning with no replay buffer: you never revisit old
data, you just keep adding arrows. Storage is one vector per regime; adding a regime never
touches the others. That is the whole promise, and everything below is about whether the
promise holds and how to tell.

---

## 2. Three ways it can fail, all looking identical

A normal training run produces two numbers you care about: how θ₀ does, and how θ_merged
does. Suppose merging doesn't help. **Three unrelated worlds produce that same signature:**

| | what's happening | what it implies |
|---|---|---|
| **Redundancy** | the τ's carry nothing the base didn't have | the premise is wrong; nothing to gain |
| **Interference** | each τ helps alone, the sum destroys the gains | the method needs fixing, not the premise |
| **Degenerate fine-tuning** | the fine-tunes never learned anything | you have a training bug |

Only **interference** says anything about non-orthogonality — the argument this line of
research rests on. Redundancy says the setup is pointless. Degeneracy says go fix training.

Nothing recorded by a normal run distinguishes them. That gap is what the transfer matrix
exists to close.

---

## 3. The transfer matrix

### 3.1 The missing measurement

> **θ₀ + τᵢ, evaluated on shard j, for j ≠ i.**

Why it is load-bearing. Say τ₁ improves shard 1. Two readings:

- τ₁ learned something *specific to shard 1* → a genuine specialist, worth composing.
- τ₁ just made the model better at *everything* → not a specialist, and n of them are n
  copies of one thing.

You cannot separate these without testing τ₁ on shards it never saw. A training run
structurally cannot produce that number: it only ever scores each specialist on its own
shard. That off-diagonal is the entire point.

### 3.2 Construction

Training-free — it reloads checkpoints that already exist and re-scores them.

- **Rows** (models): `base`, `ft_0 … ft_{n-1}` (meaning θ₀+τᵢ), `merged`, and optionally
  `standard` (a jointly-trained model, for GRR).
- **Columns** (data): each shard's held-out val slice (`val_0 …`), the baseline's own val
  slice (`val_base`), and the full test set.
- **Cells**: that model's error on that data ÷ *the base model's* error on the same data.

Checkpoints used are `best.pt` — the early-stopping best-val-loss checkpoint of each stage,
not the last epoch. `merged` is **recomputed** from base + fine-tunes rather than loaded, so
the row exists even when the merged checkpoint wasn't kept (it is bit-identical to what the
run wrote — verified on all 60 runs on disk).

**Val cells are loss-shaped, not detection-shaped.** For AD they are reconstruction error
(training data carries no labels, so AUROC is undefined there); for forecasting, MSE. Lower
is better, and 1.00 means "indistinguishable from the base model."

### 3.3 How to read it — four questions in order

Using PSM as the worked example:

```
         val_base   val_0   val_1   val_2
base       1.000    1.000   1.000   1.000
ft_0       1.039    0.774   0.845   0.909
ft_1       1.125    0.763   0.769   0.875
ft_2       1.090    0.950   0.665   0.714
merged     1.650    0.859   0.734   1.095
```

Read **down a column** = "on this fixed data, which model wins?" Read **across a row** =
"this fixed model, where does it help?"

**(a) Is the diagonal the column's minimum?** `ft_i` on `val_i` is a specialist at home. If
it doesn't win its own column, it didn't specialise. Here shard 0's best model is `ft_1`,
not `ft_0`; only shard 2 is won by its own specialist.

**(b) Is the diagonal better than the off-diagonal on average?** That difference is
`specialisation`. Positive means home beats away — the learning really was shard-specific.

**(c) Is `merged` worse than each specialist on that specialist's shard, while still beating
base?** That combination — **and only that one** — is the interference signature. Here
merged gets 0.859 where the best specialist got 0.763, and on shard 2 it is 1.095, *worse
than doing nothing*.

**(d) What happened to `val_base`?** Nobody fine-tuned on the early regime. Merged = 1.650.
That is forgetting, now a number instead of a worry.

### 3.4 The ideal matrix

> **Q: Is the ideal that each specialist is good on its own shard and *bad* everywhere else?**
>
> No. Bad elsewhere is *damage* — it means the arrow actively degrades other regimes, so
> adding several makes each wreck the others' turf. The ideal is **good at home, neutral
> away**.

```
         val_0   val_1   val_2
base     1.00    1.00    1.00
ft_0     0.60    1.00    1.00
ft_1     1.00    0.60    1.00
ft_2     1.00    1.00    0.60
merged   0.60    0.60    0.60     ← the target: merged == diagonal
```

Two separate properties:

- **Specialist rows are local** — each arrow changes its shard and leaves the rest alone.
  That is what makes them safe to add.
- **The merged row equals the diagonal.** This is the real goal: combining cost you nothing.
  One model performs on every shard as well as that shard's dedicated specialist.

The implied geometry of the ideal is cosine ≈ 0 and effective rank = n. See §4.2 for why
"cosine ≈ 0" is a weaker achievement than it sounds.

> **Q: What if the specialists are better *everywhere*, like PSM?**
>
> Genuinely ambiguous, and the matrix alone cannot resolve it. Two causes:
>
> **(a) Generic improvement** — the base was undertrained, so any further training helps
> everything. Then your n arrows are n copies of one "train more" direction, and broad
> usefulness means *redundancy*.
>
> **(b) Real shared structure** — the shards genuinely overlap and each arrow captures some
> of it plus its own local part. Then it is real value.
>
> You separate them with the geometry: **are the arrows independent?**
>
> | | mean cosine | eff. rank (of 3) | all rows good? | merge outcome |
> |---|---|---|---|---|
> | SWaT | 0.737 | 1.63 | yes | 2.43 — destroyed |
> | PSM | 0.399 | 2.51 | yes | 0.896 — works |
>
> Same matrix pattern, opposite outcomes. **"Better everywhere" is good news when the
> arrows are independent and a redundancy warning when they are aligned.**

### 3.5 The summary scalars, defined exactly

Computed in `_summarise` ([framework/pipelines/merge_diagnostics_pipeline.py](src/incremental_ad/framework/pipelines/merge_diagnostics_pipeline.py)). Every cell
is first converted to `ratio = value / base_value_on_the_same_column`. Then, **averaged over
the shard columns only — `val_base` is excluded**:

| scalar | definition | reads as |
|---|---|---|
| `diag_ratio_mean` | mean of the n cells where specialist meets its own shard (`ft_i` on `val_i`) | how much a specialist helps at home |
| `offdiag_ratio_mean` | mean of the n(n−1) cells where they don't match | how much it helps away |
| `specialisation` | `offdiag_ratio_mean − diag_ratio_mean` | **> 0 means the learning was shard-specific.** A difference, not a ratio |
| `merged_ratio_mean` | the merged model averaged across the shard columns | net effect of merging |
| `base_slice_ratio_mean` | mean of the **specialists'** cells on `val_base` | what adapting to later shards costs on the original regime |

**Two traps.**

`base_slice_ratio_mean` averages the **specialist** rows, *not* the merged row. The merged
model's forgetting is the `merged`/`val_base` cell (5.121 on SWaT) and **appears in no
summary scalar** — you must read it off the matrix.

`specialisation` is a difference of two means-of-ratios. It is a useful ordering statistic,
not a physical quantity.

### 3.6 Merge cost

> **Q: What is "merge cost" and what does it represent?**

The price of having **one model instead of n**: for each shard, the merged model's error
there ÷ the error of that shard's own specialist.

| | val_0 | val_1 | val_2 | mean |
|---|---|---|---|---|
| SWaT | 4.30× | 4.12× | 3.38× | **3.93×** |
| PSM | 1.11× | 0.95× | 1.53× | 1.20× |
| ETTh1 | 1.18× | 0.98× | 1.81× | 1.32× |

1.00× would mean merging is free. SWaT pays ~4× the error of keeping specialists separate;
PSM and ETTh1 pay 20–30%.

Note the cells **below 1.00** (PSM `val_1` at 0.95, ETTh1 `val_1` at 0.98): there the merged
model *beats* that shard's own specialist — the middle shard benefits from its neighbours.
So it is not uniformly a cost. That is constructive transfer, and it is what you want more of.

### 3.7 Reference decision table

| diagonal | off-diagonal | ‖τ‖ | scale curve | reading |
|---|---|---|---|---|
| ≈ 1 | ≈ 1 | small but nonzero | flat | **redundancy** |
| ≪ 1 | > diagonal | moderate | peaks, then falls | **interference** |
| ≈ 1 | ≈ 1 | ≈ 0 | flat everywhere | **degenerate fine-tuning** |

---

## 4. Geometry

> **Q: Geometry of what — direction, magnitude, or both?**

Three separate measurements, and keeping them separate matters.

### 4.1 The three quantities

**Direction only — cosine similarity.** The angle between τᵢ and τⱼ with magnitude divided
out. Purely "do these point the same way," range −1 to 1.

**Magnitude only — norms.** ‖τᵢ‖, and more usefully ‖τᵢ‖/‖θ₀‖: how far fine-tuning moved
the model relative to the model's own size.

**Both together — effective rank.** Stack the τ's as rows of an [n, D] matrix, take its
singular values σ₁ ≥ … ≥ σₙ, normalise pᵢ = σᵢ²/Σσ², and take **exp of the entropy of p**
([framework/merging/geometry.py](src/incremental_ad/framework/merging/geometry.py)). Intuition: "how many independent directions do these
arrows really occupy?" n = fully independent and equally weighted; 1 = one direction.

Plus one derived quantity: **cosine vs. temporal distance**, the mean cosine between shards
*k* apart. If it decays, time is what differentiates the vectors — the check that could
invalidate the whole framing.

### 4.2 The high-dimensional trap

> **Q: Is high effective rank good and low bad?**

Mostly, but not straightforwardly, for two reasons.

**Reason one: near-orthogonality is the default, not an achievement.** For two *random*
vectors in D dimensions the cosine is ~0 with standard deviation 1/√D. So:

| | D | mean cosine | 1/√D | distance from chance |
|---|---|---|---|---|
| SWaT | 1,917,695 | 0.737 | 0.00072 | **~1020 σ** |
| PSM | 649,341 | 0.399 | 0.00124 | ~321 σ |
| ETTh1 | 714,780 | 0.095 | 0.00118 | ~80 σ |

Even ETTh1 — which *looks* nearly orthogonal — is 80 standard deviations from random.
**None of these vector sets is meaningfully orthogonal.** They all encode strongly
overlapping information. A high rank could also mean the fine-tunes learned nothing and
merely drifted: random noise in high dimensions is automatically near-orthogonal and
full-rank. Always read rank alongside the norms and the diagonal.

**Reason two: rank conflates direction with magnitude.**

- **SWaT**: cosine 0.737, rank 1.63 → low rank *because the arrows are parallel*.
- **ETTh1**: cosine 0.095, rank 1.88 → nearly orthogonal, yet still low rank. Its norms are
  0.119 / 1.911 / 2.738, so the third singular value carries 0.1% of the energy. The rank is
  low **because one arrow is dead**, not because two are parallel.

Same number, opposite diagnosis.

### 4.3 What geometry is for

> **Q: If interference is what actually matters, why analyse geometry at all — optimising it
> doesn't guarantee improvement?**

Correct, it doesn't, and that is not its job. Four reasons it earns its place:

1. **It tells you which fix to reach for.** Merging failed — now what? Cosine 0.74 says
   *scale down / average / strip the shared component*. Norms differing 23× say *normalise
   the vectors*. Norms near zero say *your fine-tuning is broken; stop thinking about
   merging*. Without it you are guessing between a dozen interventions.
2. **It rules things out for free.** ETTh1's dead `finetune_0` was found in seconds, on CPU,
   with no dataset loaded — τ 23× smaller than its siblings, `best_epoch=1`. That saved
   interpreting a row that could never have meant anything.
3. **It is the only one you can optimise directly.** You can add an orthogonality penalty to
   a training loss. You cannot differentiate through "merge, then evaluate on n held-out
   slices." Geometry is the handle; the matrix is the scoreboard.
4. **Whether it predicts outcome is itself the research question.** If cheap geometry
   predicted expensive outcome you would have a GPU-free screening tool. Current evidence
   says it doesn't reliably — and *that is a finding*, being the concrete evidence for the
   caveat that parameter-space cosine is not weight disentanglement (§6.3).

---

## 5. Interference

### 5.1 Constructive, destructive, orthogonal

**Orthogonal** is the dream: each arrow occupies its own direction, they don't see each
other, ‖Στ‖ = √(Σ‖τᵢ‖²). Adding costs nothing; each keeps its own effect.

**Aligned arrows stack.** As cosine → 1, ‖Στ‖ → Σ‖τᵢ‖. Three arrows pointing the same way
take you three times as far.

**Anti-aligned arrows cancel.** You lose both.

Where the real vectors sit, comparing the actual sum against both theoretical bounds:

| | ‖Στ‖ | if orthogonal | if fully aligned | actual, as % of aligned bound |
|---|---|---|---|---|
| SWaT | 1.854 | 1.192 | 2.023 | **92%** |
| PSM | 1.191 | 0.889 | 1.535 | 78% |
| ETTh1 | 3.596 | 3.341 | 4.768 | 75% |

SWaT is at 92% of the worst case: almost nothing cancels.

### 5.2 Overshoot — the SWaT failure mode

> **Q: Why does *constructive* interference cause problems? And what does overshoot imply?**

Note that in the merging literature "interference" usually connotes *conflict*. SWaT's
failure is the opposite and is worth naming separately:

- **Conflict**: the arrows pull apart; the sum lands somewhere useless.
- **Overshoot**: the arrows pull *together*; the sum goes past the target.

Moving along τ₀ helps. Moving 3× as far in the same direction does not — you sail past the
point where it helped and out the other side. SWaT's base error on the early regime went
from 0.041 to 0.210 not because the direction was wrong but because the distance was.

**What overshoot implies is the optimistic part: the failure is in the *magnitude*, not the
direction.** Magnitude is a scalar you control. Conflict is not fixable by scaling; overshoot
is.

### 5.3 The decomposition that hasn't been measured yet

> total merge cost = **overshoot** (curable by scaling α) + **irreducible interference**
> (not curable by scaling)

These are currently **fused**, because every measurement was taken at α = 1.0. Turning α
down shrinks the useful contributions along with the harmful stacking. The question is
whether the merged curve **reaches the diagonal** at its best α or **plateaus above it**.
The gap remaining at the optimum is the irreducible part; everything above it was overshoot.

This is the single most informative missing measurement (§10).

### 5.4 Remedies

> **Q: Can normalisation avoid overshooting?**

Yes, in several forms addressing different pathologies:

| technique | what it fixes | relevant here? |
|---|---|---|
| **Global α** — `θ₀ + α·Στ` | pure magnitude | yes, and proven: PSM optimum 0.75, ETTh1 0.60, both below the 1.0 used |
| **α = 1/n** (average, not sum) | magnitude, the natural default when aligned | the obvious first try |
| **Per-vector normalisation** — rescale each τᵢ to unit length first | *imbalance* between vectors | ETTh1 (23× spread), not SWaT (norms already comparable) |
| **Norm-matching the sum** — pick α so merged sits a single specialist's distance from θ₀ | magnitude, analytically | gives 0.44 / 0.47 / 0.76 — **does not match measured optima** (see below) |
| **Strip the shared component** — decompose Στ into "what they all agree on" + residuals, apply the shared part *once* | alignment specifically | the fix aimed at SWaT's actual pathology |
| **TIES / DARE** — prune small entries, resolve sign disagreements before summing | conflict, redundancy | targets the conflict flavour more than overshoot |

**Important caveat on norm-matching.** The geometric estimate says α ≈ 0.47 for PSM where
the measured optimum is 0.75, and 0.76 for ETTh1 where it is 0.60. It gets the *direction*
right and the *number* wrong. Sweep α and measure; do not trust the analytic guess.

> **Q: If the vectors constructively interfere, should we add them at all, or ignore them?**
>
> Don't ignore them. The diagonal proves they carry real value — SWaT's 0.608 is a 39% error
> reduction on the shards. Discarding that is strictly worse.
>
> High alignment means the arrows are **redundant with each other**, not worthless. The
> right response to redundancy is **averaging, not discarding**: you get the consensus
> direction at a sane magnitude, plus some noise cancellation for free.
>
> The deeper reading: if n time periods produce nearly the same arrow, the *data* did not
> contain n distinct regimes. That is a statement about the dataset, not about task
> arithmetic.

### 5.5 Making specialists more ideal

> **Q: Is there a way to get each specialist to the ideal — good at home, neutral away?
> Maybe better splits?**

Better splits is probably the largest lever available here.

- **Splits at change-points, not equal time slices.** Currently shards are equal-length time
  chunks. On near-stationary data, equal chunks are near-identical distributions, so n
  fine-tunes learn n copies of the same thing. Cutting where the distribution actually
  changes is what would make the arrows differ. **This attacks the root cause.**
- **Constrain the vectors during training** — penalise alignment with previously-learned τ's
  (the orthogonal-gradient idea from continual learning), pushing each new arrow into
  unused directions.
- **Restrict where each arrow can live** — per-segment adapters/LoRA, or freeze shared
  layers. Structurally near-orthogonal because they occupy different parameters.
- **Sparsify the deltas** — keep only the largest changes; sparse high-dimensional vectors
  overlap far less. (This is what TIES/DARE do post hoc.)
- **Equalise the training budget** so magnitudes are comparable.

**L2-SP was already tried and is not this.** It shrinks all arrows toward zero rather than
making them *different* — empirically harmful on ETTh1, neutral on SWaT/PSM.

### 5.6 How to measure interference

> **Q: How do I measure interference?**

Two clean ways, neither yet done:

1. **Merged vs diagonal at the best α**, not at α = 1. Isolates the irreducible part.
   Requires tracing the *val block* against merge scale (the current curve is test-only).
2. **Pairwise merges.** Compare θ₀+τ₀+τ₁ against θ₀+τ₀ and θ₀+τ₁ separately. If pairs
   compose cleanly but triples don't, the damage is higher-order; if pairs already break,
   it's pairwise. Direct, needs no retraining, cheap.

---

## 6. Entanglement and disentanglement

### 6.1 The definition

A model is **weight-disentangled** with respect to a set of task vectors when, on inputs
belonging to task *i*, the model with *all* vectors applied behaves the same as the model
with *only* τᵢ applied. Adding the others doesn't disturb what τᵢ does on its own turf.

### 6.2 The matrix measures it

In matrix terms that statement is exactly:

> **merged row == diagonal**

So the merged-vs-diagonal gap **is** an empirical entanglement measure — the merge cost of
§3.6 under another name: 3.93× on SWaT (heavily entangled), 1.20× PSM, 1.32× ETTh1.

> **Q: Is the transfer matrix measuring geometry, entanglement, or just interactions?**
>
> Functional **interactions** — what the models *do*. Geometry is the separate, cheap
> measurement on the weights. But different comparisons inside the matrix measure different
> things:
>
> | comparison | measures |
> |---|---|
> | off-diagonal cell vs 1.00 | **transfer** — does τᵢ affect shard *j* at all? |
> | off-diagonal vs diagonal | **specialisation** — how shard-specific was the learning? |
> | **merged vs diagonal** | **weight disentanglement** |
> | merged vs 1.00 | net practical benefit |
> | cosine / rank / norms (separate) | weight-space **geometry** — a proxy for all of the above |
>
> Two honest caveats: the formal definition is pointwise over each task's input support,
> whereas we aggregate error over a time-sliced validation set — a legitimate proxy, not the
> exact quantity, and consecutive time slices of one process certainly overlap in
> distribution. And it is measured at α = 1.0 only; a proper disentanglement measure should
> be traced across α.

### 6.3 Why cosine is not disentanglement

Two arrows can be perfectly orthogonal in parameter space and still fight in function space,
because what matters is whether they change the model's behaviour **on the same inputs**.
Parameter orthogonality says nothing directly about that.

Hence: geometry is a *proxy*, the matrix is *ground truth*, and whether the proxy predicts
the truth is an open empirical question — currently answered "weakly at best" (§9).

---

## 7. Metrics that cannot see what you are measuring

This section exists because it produced the most confusing result in the whole exercise.

### 7.1 The apparent contradiction

On SWaT the merged model's reconstruction error is **2.4× worse on the shards and 5.1×
worse on the base slice** — yet every detection metric moves by less than 1%
(AUROC 0.8005 → 0.8060, window F1 0.7508 → 0.7513).

### 7.2 The resolution

**Every test metric in these runs is invariant to a monotone rescaling of the scores.**

- AUROC and AUPRC are **rank-based**: only the ordering of windows matters.
- The runs used `threshold_strategy=oracle`, which passes `None` as the threshold so **each
  metric sweeps its own optimum** ([framework/evaluators/ad_test_evaluator.py](src/incremental_ad/framework/evaluators/ad_test_evaluator.py)). A shifted
  scale is absorbed automatically.

And merging did something very close to a monotone rescale. SWaT merged ÷ base on `val_base`:

| statistic | ratio |
|---|---|
| median (p50) | **26.8×** |
| mean | 5.12× |
| p95 | 1.80× |
| p99 | 1.48× |
| **std** | **1.21×** |

The merge raises the *floor* — windows the base reconstructed almost perfectly get much
worse — while the hard tail barely moves and the spread is nearly unchanged. Precisely the
thing rank-based and oracle-thresholded metrics cannot see.

**So the two readings are not in conflict:** the val block measures the model as a
*reconstructor* in absolute terms; the test block measures it as a *ranker*. Merging wrecked
the first and left the second essentially untouched.

**The lesson, which generalises:** before concluding "X had no effect," check whether your
metric is *capable* of registering X. On SWaT a 5× change in the underlying model shows up
as +0.008 AUROC. That makes SWaT a **control**, not evidence.

### 7.3 GRR and the noise floor

> **GRR = (merged − base) / (standard − base)** on a test metric.

"What fraction of the gap between the frozen base and full joint training did merging
close?" ~1 means merging recovers joint training; 0 means inert; < 0 means destructive. The
formula needs no sign handling — for an error metric both differences are negative and the
ratio stays positive.

**GRR is only meaningful when the denominator clears the noise floor.** Where the base
already sits near the ceiling, GRR divides one noise term by another. The pipeline warns
below 2% of base (the size of the measured same-seed reproducibility gap).

| | gap | as % of base | GRR | usable? |
|---|---|---|---|---|
| SWaT | 0.0084 | 1.0% | 0.659 | **no** |
| PSM | 0.0262 | 3.4% | 0.958 | marginally |
| ETTh1 | −0.2221 | 36.2% | 0.652 | yes |

> **Q: How does effective rank relate to GRR?**
>
> Different universes. Effective rank is a property of the *weights*, free to compute. GRR is
> a *performance outcome* on the test metric, and expensive. The hypothesis is that the first
> predicts the second. Discarding SWaT (unusable), that leaves **two points**, ordered the
> right way. Consistent with the hypothesis; nowhere near evidence for it.

---

## 8. What the three datasets said

All at α = 1.0, one evaluation seed, val cells as ratio-to-base.

### 8.1 SWaT — arrows nearly parallel; metrics blind

```
         val_base   val_0   val_1   val_2      | test AUROC  window_f1
base       1.000    1.000   1.000   1.000      |    0.8005     0.7508
ft_0       1.026    0.740   0.698   0.750      |    0.7991     0.7508
ft_1       1.235    0.770   0.603   0.538      |    0.8016     0.7510
ft_2       1.413    0.861   0.638   0.481      |    0.8029     0.7511
merged     5.121    3.180   2.487   1.625      |    0.8060     0.7513
standard      —        —       —       —       |    0.8089     0.7514
```

specialisation **+0.101** (highest of the three) · diag 0.608 · offdiag 0.709 · merged 2.431
· base-slice 1.225 · cosine 0.737 · rank 1.63 · curve rises monotonically to 1.5, never peaks

- **The diagonal is the column minimum for all three shards** — the only dataset where every
  specialist wins at home. Specialisation is genuinely strongest here.
- Forgetting is clean and monotone in segment index: 1.03 → 1.24 → 1.41.
- And yet the merge is catastrophic on the val block, because the arrows are 92% aligned.
- **All test gaps ≤ 1% of base. Every SWaT GRR is noise ÷ noise and must not be quoted.**
- Read as a **control**: the reconstruction damage is real and the detector cannot see it.

### 8.2 PSM — the cleanest AD case

```
         val_base   val_0   val_1   val_2      | test AUROC  window_f1   pa_f1  event_f1
base       1.000    1.000   1.000   1.000      |    0.7740     0.6361  0.8067    0.2294
ft_0       1.039    0.774   0.845   0.909      |    0.7832     0.6517  0.7667    0.2957
ft_1       1.125    0.763   0.769   0.875      |    0.7936     0.6546  0.7675    0.1971
ft_2       1.090    0.950   0.665   0.714      |    0.7932     0.6557  0.7667    0.2145
merged     1.650    0.859   0.734   1.095      |    0.7991     0.6790  0.7716    0.2749
standard      —        —       —       —       |    0.8002     0.6918  0.7676    0.3960
```

specialisation **+0.082** · diag 0.753 · offdiag 0.835 · merged 0.896 · base-slice 1.085 ·
cosine 0.399 · rank **2.51** (most independent) · curve peaks at **α = 0.75**

- Every specialist beats base on every later shard — all nine cells below 1.00.
- Diagonal is column-minimum **only for shard 2**.
- Real locality: `ft_2` is best on shards 1 and 2 but *worst* on shard 0 (0.950), matching
  the cosine decay.
- Merged helps on shards 0 and 1, **hurts on shard 2** (1.095), forgets the early regime
  (1.650), and loses to the best specialist in every column.
- **Metrics disagree**, as they have before on PSM: `pa_f1` has a *negative* gap (standard
  0.7676 < base 0.8067), so its GRR describes recovery toward a worse model; on `event_f1`
  standard is far ahead and the merge recovers only ~27%. Ranking PSM by one metric is unsafe.

### 8.3 ETTh1 — interference plus a dead segment

```
         val_base   val_0   val_1   val_2      | test MSE
base       1.000    1.000   1.000   1.000      |  0.6132
ft_0       0.966    0.958   0.908   0.959      |  0.6334
ft_1       0.940    0.541   0.757   1.063      |  0.4721
ft_2       1.222    0.791   0.583   0.481      |  0.4575
merged     1.207    1.127   0.738   0.871      |  0.4684
standard      —        —       —       —       |  0.3911
```

specialisation **+0.076** · diag 0.732 · offdiag 0.808 · merged 0.912 · base-slice 1.043 ·
cosine 0.095 · rank 1.88 · curve is a clean U bottoming at **α = 0.60** (MSE 0.4228)

- **`ft_0` is degenerate**: ‖τ‖/‖θ₀‖ = 0.0012 against 0.0198 and 0.0283, `best_epoch=1`. Its
  row is inert by construction and the 3-vector merge is effectively a 2-vector merge. This
  affects **11 of 15** ETTh1 runs — a property of the fine-tuning schedule, not one bad job.
- `ft_1` helps shard 0 (0.541) *more than its own* shard 1 (0.757), and hurts shard 2.
- Merged is worse than base on two of four columns and loses to the best specialist everywhere.
- Cleanest gap of the three (36% of base), so its GRR of 0.652 is trustworthy.

### 8.4 Cross-dataset

| | SWaT | PSM | ETTh1 |
|---|---|---|---|
| mean off-diagonal cosine | 0.737 | 0.399 | 0.095 |
| cosine at temporal distance 1 → 2 | 0.771 → 0.667 | 0.466 → 0.265 | 0.121 → 0.042 |
| effective rank (of 3) | 1.63 | 2.51 | 1.88 |
| ‖τ‖/‖θ₀‖ per segment | .0044/.0065/.0074 | .0052/.0064/.0059 | **.0012**/.0198/.0283 |
| `specialisation` | +0.101 | +0.082 | +0.076 |
| `diag` / `offdiag` | 0.608 / 0.709 | 0.753 / 0.835 | 0.732 / 0.808 |
| `merged_ratio_mean` | 2.431 | 0.896 | 0.912 |
| merge cost (merged ÷ diagonal) | 3.93× | 1.20× | 1.32× |
| curve optimum α | none in range | 0.75 | 0.60 |
| gap vs base | 1.0% (below floor) | 3.4% | 36.2% |
| GRR | unusable | 0.958 | 0.652 |
| fine-tune epochs (seg 0/1/2) | 16/25/32 | 28/39/43 | 1/8/19 |

**Cosine decays with temporal distance on all three** — so temporal shift really is what
differentiates the vectors, and the framing survives the check that could have killed it.

**Merge damage tracks collinearity**: cosine 0.737 → 2.431, 0.399 → 0.896, 0.095 → 0.912.
Three points, monotone. Suggestive, not established.

**A confound to keep in view.** Fine-tuning epochs increase monotonically with segment index
on *all three* datasets. Later shards sit further from θ₀, so fine-tuning runs longer and the
vectors get bigger — and a bigger vector helps everywhere, not just at home. So part of
"later specialists win more columns" may be "later specialists are more trained." This
inflates the off-diagonal and therefore *deflates* `specialisation`. Unresolved.

---

## 9. Conclusions: supported and not supported

### Supported

- **Fine-tuning learns something real and shard-specific.** Diagonal ratios of 0.61–0.75 and
  positive specialisation on all three datasets rule out *redundancy* and *degenerate
  fine-tuning* as the global explanation (exceptions: ETTh1 `ft_0`, PSM run 59109).
- **The vectors are not orthogonal.** Substantially aligned — 80σ to 1020σ above chance —
  with alignment decaying as temporal distance grows.
- **Interference is the mechanism**, on all three: the merged model is worse than each
  specialist on that specialist's own shard.
- **α = 1.0 is the wrong scale.** Every reported result used it; where the curve is readable
  the optimum is below it (0.75 PSM, 0.60 ETTh1).
- **SWaT's detection metrics are saturated** — a 5× change in the model registers as +0.008
  AUROC.
- **The merge is bitwise reproducible** — all 60 merged checkpoints on disk recompute exactly
  from their baseline + fine-tunes.

### Not supported — do not claim these

- **That task arithmetic fails here.** It was never run at a sensible α. PSM at its optimum
  already recovers ~96% of the gap to joint training.
- **Anything about weight disentanglement from the cosine numbers.** Different quantity (§6.3).
- **Anything from SWaT's test column.** Every gap is inside the noise floor.
- **That geometry predicts outcome.** Two usable points, correctly ordered. Not evidence.
- **That the specialisation ordering is real** — the training-budget confound is unresolved.

### The one-sentence version

> The fine-tunes each learn something genuine and locally useful, but on these datasets they
> largely learn the *same* thing, so summing them overshoots rather than accumulates — and
> the standard detection metrics are too blunt to notice either way.

### The one practical takeaway

The highest-value finding is not geometric. It is that **every result reported so far used
α = 1.0, and the optimum is below it on every dataset where it can be measured.** That is a
one-line change worth more than anything the cosine numbers have said so far.

---

## 10. Open questions and the next measurements

Ordered by value per unit of compute.

1. **Pairwise merges.** Merge τ₀+τ₁ alone, τ₀+τ₂, τ₁+τ₂. Directly decomposes pairwise from
   higher-order interaction. Training-free, cheap, and its absence is the biggest hole.
2. **Trace the val block against merge scale.** The current curve is test-only — blind to
   exactly the effect under study. This is what separates *overshoot* from *irreducible
   interference* (§5.3). Small pipeline change plus a rerun.
3. **Equalise fine-tuning budgets** to remove the epochs confound (§8.4).
4. **Training seeds** (not just evaluation seeds) — do the findings replicate on a fresh θ₀?
5. **Change-point splits** instead of equal time slices, so shards are genuinely different
   regimes (§5.5).
6. **A dataset with real drift.** SWaT and PSM are near-stationary; that is *why* SWaT's
   metrics can't see a 5× model change.
7. **More segments** — with n = 3 the cosine-decay claim rests on two points per dataset.
8. **Evaluation seeds** on AD — lowest priority, since the claims rest on the val block
   (already averaged over 30 masks and thousands of windows) rather than on event metrics.

**On seeds, note the distinction.** *Evaluation* seed = same checkpoints, different random
masks during scoring; measures evaluation noise only; zero value on ETTh1 (deterministic).
*Training* seed = a different run entirely, different θ₀ and τ's; measures whether the
findings replicate. The second is the scientifically important one and the expensive one.

---

## 11. Quick reference

### Symbols

| symbol | meaning |
|---|---|
| θ₀ | base model, trained on the early regime |
| θᵢ | model fine-tuned on shard *i*, starting from θ₀ |
| τᵢ = θᵢ − θ₀ | task vector for shard *i* |
| α | merge scale in θ₀ + α·Στᵢ |
| `ft_i` | the model θ₀ + τᵢ (a matrix row) |
| `val_i` | shard *i*'s held-out validation slice (a matrix column) |
| `val_base` | the baseline's own validation slice — the early regime |

### Reading a cell

`value / base_value_on_the_same_column`. **Lower is better. 1.00 = indistinguishable from
base.** Val cells are loss-shaped (reconstruction error for AD, MSE for forecasting).

### Reading the matrix

| look at | tells you |
|---|---|
| diagonal vs 1.00 | did the specialist learn anything at home? |
| off-diagonal vs 1.00 | does it transfer, or damage, elsewhere? |
| off-diagonal vs diagonal | was the learning shard-specific? (`specialisation`) |
| merged vs diagonal | **weight disentanglement / merge cost** |
| merged vs 1.00 | net practical benefit |
| specialists on `val_base` | forgetting (`base_slice_ratio_mean`) |
| merged on `val_base` | the merge's own forgetting — **not in any summary scalar** |

### Commands

```bash
# geometry only — CPU, no dataset, seconds
python -m incremental_ad.analysis.geometry_report $RUNS_ROOT/<experiment>/* \
    --out $RUNS_ROOT/analysis/geometry
# NOTE: split by experiment — ~65 run dirs in one invocation OOMs the login node.

# full diagnostics — GPU, via SLURM
SOURCE_RUN=$WORK/runs/<exp>/<run_id> STANDARD_RUN=$WORK/runs/<std_exp>/<run_id> \
    sbatch scripts/sbatch_merge_diagnostics.sh
# NOTE: env prefix, NOT --export=VAR=value — an explicit --export gets the job
# "CANCELLED by 0" ~2s in with no log written at all.
```

### Outputs

```
$RUNS_ROOT/<source experiment>_diagnostics/<run_id>/merge_diagnostics/
  transfer_matrix.csv    model, column, block, metric, value, ratio_to_base, n_windows, eval_seed
  merge_scale_curve.csv  merge_scale, split, metric, value, n_windows, eval_seed
  result.json            the summary scalars of §3.5
  source.json            which run was analysed, its merge_scale, columns, window counts
```

### Code map

| what | where |
|---|---|
| `task_vector`, `apply_task_vectors`, `merge_task_arithmetic` | [framework/merging/task_vectors.py](src/incremental_ad/framework/merging/task_vectors.py) |
| cosine, norms, effective rank, cosine-vs-distance | [framework/merging/geometry.py](src/incremental_ad/framework/merging/geometry.py) |
| the transfer matrix and summary scalars | [framework/pipelines/merge_diagnostics_pipeline.py](src/incremental_ad/framework/pipelines/merge_diagnostics_pipeline.py) |
| what a val cell contains | [framework/evaluators/ad_val_evaluator.py](src/incremental_ad/framework/evaluators/ad_val_evaluator.py) |
| oracle vs percentile thresholding | [framework/evaluators/ad_test_evaluator.py](src/incremental_ad/framework/evaluators/ad_test_evaluator.py) |
| CLI entry points | [analysis/geometry_report.py](src/incremental_ad/analysis/geometry_report.py), [analysis/diagnose.py](src/incremental_ad/analysis/diagnose.py) |
