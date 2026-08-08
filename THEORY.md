# Theory and concepts — incremental learning by model merging

Written to be *studied*, not skimmed. It assumes no prior familiarity with task arithmetic,
model merging or continual-learning evaluation, and builds each idea from the setup upward.

Self-contained, and safe to paste into an LLM as background for follow-up questions.

**Where things live.** This file is the *why*: concepts, derivations and reasoning.
[EXPERIMENTS.md](EXPERIMENTS.md) holds the detailed results and observations — if a number
here disagrees with one there, that file wins. [EXECUTION_PLAN.md](EXECUTION_PLAN.md) tracks
what has been run and what to run next. [CLAUDE.md](CLAUDE.md) is the code map.

Every figure here is the current state at **three training seeds per dataset** (2026-08-05).
`±` is the half-range over seeds; a difference is only claimed where two intervals do not
overlap.

---

## Contents

1. [The setup](#1-the-setup)
2. [Why time series are not image classification](#2-why-time-series-are-not-image-classification)
3. [Three ways it can fail, and why they look identical](#3-three-ways-it-can-fail-and-why-they-look-identical)
4. [The transfer matrix](#4-the-transfer-matrix)
5. [Geometry](#5-geometry)
6. [Overshoot and interference](#6-overshoot-and-interference)
7. [Entanglement](#7-entanglement)
8. [Metrics that cannot see what you are measuring](#8-metrics-that-cannot-see-what-you-are-measuring)
9. [Versus sequential fine-tuning](#9-versus-sequential-fine-tuning)
10. [What the datasets said](#10-what-the-datasets-said)
11. [A decision rule — merge, route, or keep fine-tuning?](#11-a-decision-rule--merge-route-or-keep-fine-tuning)
12. [What is established and what is not](#12-what-is-established-and-what-is-not)
13. [Open questions](#13-open-questions)
14. [Quick reference](#14-quick-reference)
15. [Related work](#15-related-work)

---

## 1. The setup

### 1.1 A model is a point in a very large space

A trained network with *D* parameters is one point θ ∈ ℝ^D — 1,917,695 dimensions for SWaT,
649,341 for PSM, 714,780 for ETTh1. Distances and angles in that space are the objects of study.

### 1.2 Task vectors

- Train a **base model** θ₀ on the first half of the training data — the "early regime".
- Cut the remaining data into *n* segments in time order.
- Fine-tune **from θ₀** on each segment independently. Never segment-to-segment: always from
  the same starting point, which is what makes the deltas comparable and summable.
- The **task vector** is the difference:

  > **τᵢ = θᵢ − θ₀**

  One arrow in weight space encoding everything fine-tuning on segment *i* changed.

These arrows are small. On SWaT ‖θ₀‖ = 110.46 while the τ's are 0.487 / 0.719 / 0.817 — about
**0.4–0.7% of the model's length**. Fine-tuning nudges; it does not relocate.

### 1.3 Task arithmetic

The bet is that arrows **compose**:

> **θ_merged = θ₀ + α · (τ₀ + τ₁ + … + τₙ)**

α is the *merge scale*. α = 1 sums them; α = 1/n averages them. **α is not a detail** — §5 is
about how much it matters.

### 1.4 Why anyone cares

If it works you get incremental learning with no replay buffer: never revisit old data, just
add arrows. Storage is one vector per regime, and adding a regime never touches the others.

---

## 2. Why time series are not image classification

Task arithmetic was developed on image classification, where a task vector means *"the model
learned a new class"*. Two such vectors point at largely unrelated things, so in a
high-dimensional space they end up **near-orthogonal**, and adding them barely interferes.
That near-orthogonality is doing quiet work in every result the method is famous for.

Time series break the assumption in **both** directions, and which direction you get is a
property of the data, not of the method:

- **Stationary data** — consecutive periods are near-identical distributions, so *n* fine-tunes
  learn one thing *n* times. The vectors are highly aligned, and adding them at full strength
  travels *n* times too far in a direction only one step was needed in.
- **Drifting data** — each period genuinely differs, so the vectors carry distinct information.
  But they are still not orthogonal, because consecutive periods share most of their structure;
  only *distant* ones diverge.

Both are measured here. Subspace overlap ρ — the fraction of a new task vector already lying in
the span of its predecessors — ranges from **0.607 on SWaT** (highly redundant updates) to
**0.076 on ETTh1** (nearly new every time). And cosine between task vectors **decays with
temporal distance**: 0.737 between adjacent periods on SWaT against 0.095 on ETTh1. Time really
is the axis that separates them, which is the check that could have invalidated the whole
framing and did not.

### What follows from it

**Non-orthogonality does not break merging. It sets the scale.**

The merge is θ₀ + α·Στᵢ. If the vectors were orthogonal, their sum would be short relative to
Σ‖τᵢ‖, so a coefficient that decays only slowly with the number of tasks still works. When the
vectors are aligned, the sum is nearly Σ‖τᵢ‖ long, and the coefficient must fall like 1/n — at
α = 1 the merge overshoots by roughly the number of vectors.

That is broadly what the measurements show: **α\*·n lands at order 1 on five of the six datasets measured** (exchange_rate ≈1.5 is the exception)
(§6.6 — an empirical regularity, not a constant, and the count cannot be isolated from shard
size). Averaging rather than adding is the whole correction. It holds across shard counts partly
*because* alignment itself
falls as shards get finer — on ETTh1, 0.824 at n = 2 down to 0.633 at n = 5 — while the vectors
shrink in step, so ‖Στᵢ‖ stays nearly constant.

Everything previously written up as *interference* and *forgetting* on these datasets was this
scale error. The residual interference at α\* is small.

### The practical consequence

For time series you cannot inherit the *scaling behaviour* of the image-classification setting,
and you cannot assume orthogonality buys you free composition. What you can do is simpler:
**average the task vectors**, and expect a dataset-level coefficient of order 1 in front of the
mean (≈1.0 on SWaT, PSM, ETTh1, ETTh2 and ETTm2; ≈1.5 on exchange_rate).

> ⚠️ **Not "they use α = 1".** An earlier version of this section said time series "cannot
> inherit α = 1 from image classification", which misstates the literature: task-arithmetic
> papers **tune** the coefficient on held-out data rather than fixing it at 1 (α ≈ 1 is a
> default only for adding a *single* task vector, and model soups' uniform averaging is
> literally α = 1/n). The real contrast is the **rate**: there the tuned coefficient decays with
> the number of tasks but more slowly than 1/n, so α·n *grows*; here it tracks 1/n, so α·n stays
> order 1. That is the framing §6.6a uses, and this section now matches it. **The
> image-classification side remains unverified against sources** (§15) — check it before the
> comparison goes in the thesis.

---

## 3. Three ways it can fail, and why they look identical

A training run produces two numbers you care about: how θ₀ does, and how θ_merged does.
Suppose merging doesn't help. **Three unrelated worlds produce that same signature:**

| | what's happening | what it implies |
|---|---|---|
| **Redundancy** | the τ's carry nothing the base didn't have | the premise is wrong; nothing to gain |
| **Interference** | each τ helps alone, the sum destroys the gains | the method needs fixing, not the premise |
| **Degenerate fine-tuning** | the fine-tunes never learned anything | you have a training bug |

Only **interference** says anything about non-orthogonality. Redundancy says the setup is
pointless; degeneracy says go fix training. Nothing a normal run records distinguishes them —
that gap is what the transfer matrix exists to close.

*(A fourth possibility, discovered later and not on this list originally: the merge scale is
simply wrong. On these datasets that turned out to be the dominant one — see §6.)*

---

## 4. The transfer matrix

### 4.1 The missing measurement

> **θ₀ + τᵢ, evaluated on segment j, for j ≠ i.**

Say τ₁ improves segment 1. Two readings: τ₁ learned something *specific to segment 1*, or τ₁
just made the model better at everything. You cannot separate them without testing τ₁ on
segments it never saw — and a training run structurally cannot produce that number, because it
only ever scores each specialist on its own segment. **That off-diagonal is the entire point.**

### 4.2 Construction

Training-free: reload checkpoints that already exist and re-score them.

- **Rows**: `base`, `ft_0 … ft_{n-1}` (meaning θ₀+τᵢ), `merged`, and optionally a
  jointly-trained `standard`.
- **Columns**: each segment's held-out validation slice, the baseline's own slice
  (`val_base`), and the full test set.
- **Cells**: that model's error on that data ÷ the base model's error on the same data.

Checkpoints are `best.pt` — the early-stopping best-validation checkpoint. `merged` is
**recomputed** from base + fine-tunes rather than loaded; it is bit-identical to what the run
wrote, verified across all 87 runs on disk.

**Validation cells are loss-shaped, not detection-shaped** — reconstruction error for AD
(training data carries no labels, so AUROC is undefined there), MSE for forecasting. Lower is
better; 1.00 means indistinguishable from base.

### 4.3 How to read it

Using PSM, at the validation-selected scale:

```
model            val_base   val_0   val_1   val_2
base               1.000    1.000   1.000   1.000
θ₀+τ₀              1.039    0.774   0.845   0.909
θ₀+τ₁              1.125    0.763   0.769   0.875
θ₀+τ₂              1.090    0.950   0.665   0.714
merged @ α*=0.5    1.122    0.778   0.705   0.796
```

Read **down a column** = on this fixed data, which model wins? Read **across a row** = this
fixed model, where does it help?

**(a) Is the diagonal the column minimum?** `ft_i` on `val_i` is a specialist at home. If it
doesn't win its own column, it didn't specialise — *subject to the caveat in §4.5*.

**(b) Is the diagonal better than the off-diagonal on average?** That difference is
`specialisation`; positive means home beats away.

**(c) Is `merged` worse than each specialist on that specialist's segment, while still beating
base?** That combination is the interference signature — **but only when measured at α\***,
never at α = 1 (§5).

**(d) What happened to `val_base`?** Nobody fine-tuned on the early regime, so this is the
forgetting column.

### 4.4 The ideal matrix

The ideal is **good at home, neutral away** — *not* good at home and bad elsewhere. Bad
elsewhere is damage: it means adding several vectors makes each wreck the others' turf.

```
         val_0   val_1   val_2
base     1.00    1.00    1.00
ft_0     0.60    1.00    1.00
ft_1     1.00    0.60    1.00
ft_2     1.00    1.00    0.60
merged   0.60    0.60    0.60     <- the target: merged == diagonal
```

Two separate properties: the specialist rows are **local** (safe to add), and the merged row
**equals the diagonal** (combining cost nothing).

### 4.5 Two biases in the matrix, pointing opposite ways

**Selection bias inflates the diagonal.** Each fine-tune is early-stopped on its own
segment's validation slice, so `ft_i` was *chosen* using `val_i`.

**Tail adjacency deflates it.** The validation slice is the temporal *tail* of each segment
(`val_tail_split`), so `val_i` sits next to segment *i+1* and hands `ft_{i+1}` an unearned
advantage. Tested where both neighbours are equidistant, the successor beats the predecessor
on **3 of 3** datasets.

The second is not fixable by random splitting: with `stride=1, window_len=120` adjacent
windows share 119 of 120 timesteps, so a random split would leak almost entirely. **The tail
split is the correct design**; the bias is a limitation to document. Net effect: the
off-diagonal is inflated, so `specialisation` **understates** the truth.

### 4.5b The arithmetic, written out

Everything in this chapter — and most numbers in EXPERIMENTS.md — is built from one table and
five formulas. Notation:

| symbol | meaning |
|---|---|
| θ₀ | the base model, trained on the first half of the series |
| τᵢ = θᵢ − θ₀ | the task vector of segment *i*'s fine-tune, i = 0 … n−1 |
| θ_M(α) = θ₀ + α·Στᵢ | the merged model at scale α |
| θ_J | the jointly-trained reference (`StandardPipeline`, all data, from scratch) |
| D₀ … D_{n−1} | each segment's held-out validation slice; **D_b** the baseline's own slice; **D_T** the test set |
| ℓ(θ, D) | the metric of model θ on data D (below) |
| \|D\| | the number of sliding windows in D — the weight used when pooling |

**The metric ℓ.** Forecasting: `forecast/mse` = mean squared error over the horizon,
`F.mse_loss(preds, targets)` across every window. Anomaly detection uses two different
quantities depending on whether labels exist:

- *On validation* (no labels): `reconstruction/score_mean`, the mean anomaly score. For a
  random-mask model that score is, per window, the mean per-patch reconstruction error over
  **all** patches — `error_sum / error_counts.clamp(min=1)` accumulated over `n_eval_passes`
  masks, then `.mean(-1)`. Patches never masked in any pass contribute 0/1 = 0, which dilutes
  the mean; that is intentional and the answer is to use enough passes (30 in these runs).
- *On test* (labels exist): `window_auroc`. Each sliding window gets one score; its label is
  `labels.max()` over the window — anomalous if **any** timestep in it is. AUROC is then
  threshold-free over those (score, label) pairs. The `point_*` family instead labels a window
  by its **last** timestep; `pa_*` applies point adjustment; `event_*` scores contiguous
  anomaly events. §1.5 of EXPERIMENTS.md explains why AUROC is primary and `pa_f1` is not.

**1 — the transfer matrix.** One cell per (model, column), always as a *ratio to base* so that
runs, seeds and datasets are commensurable:

```
    M[m, c]  =  ℓ(θ_m, D_c)  /  ℓ(θ₀, D_c)
```

so `M[base, c] = 1` by construction. Rows are `base`, `ft_i` (= θ₀+τᵢ), `merged` (= θ_M at the
committed α) and optionally `standard` (= θ_J, test column only). For an error metric lower is
better; for AUROC the ratio inverts and the direction is read from the metric name, never
assumed.

**2 — specialisation** = how much better a specialist is at home than away:

```
    diag     =  (1/n) · Σᵢ M[ftᵢ, i]
    offdiag  =  mean over i ≠ j of M[ftᵢ, j]
    spec     =  offdiag − diag                     (> 0 ⇒ genuinely period-specific)
```

**3 — merge cost** = the price of one model instead of *n*, per period, on that period's own
slice — which is why it needs the matrix and cannot be got from test-set numbers:

```
    cost  =  (1/n) · Σᵢ  ℓ(θ_M, Dᵢ) / ℓ(ftᵢ, Dᵢ)
          =  (1/n) · Σᵢ  M[merged, i] / M[ftᵢ, i]
```

The base term cancels in the ratio of ratios, so this is exactly the raw-error ratio. **1.00
means merging costs nothing** against keeping one model per period. Note which ℓ this uses: the
columns are *validation* slices, so on AD it is `reconstruction/score_mean`, **not**
`window_auroc` — AUROC does not exist on the validation columns at all, because those slices
carry no labels. Verified: evaluated on the α = 1.0 diagnostics this formula returns 3.545
(SWaT), 1.187 (PSM) and 1.828 (ETTh1), each inside §4.6's published α = 1.0 interval.

**4 — GRR**, the share of the base-to-joint gap the merge closes, on the test column:

```
    GRR  =  ( ℓ(θ₀, D_T) − ℓ(θ_M, D_T) )  /  ( ℓ(θ₀, D_T) − ℓ(θ_J, D_T) )
```

Sign handling is automatic: for an error metric both differences are negative and the ratio
stays positive. Headroom is the denominator expressed relatively,
`(ℓ(θ₀,D_T) − ℓ(θ_J,D_T)) / ℓ(θ₀,D_T)`, and GRR is meaningless when that gap is inside the
noise floor — which is why the gap travels next to every ratio.

**5 — routing headroom**, the ceiling on what any router could recover. The column optimum over
specialists *is* the best possible router, because no real router beats choosing with hindsight:

```
    oracle_c  =  minᵢ M[ftᵢ, c]                    (max for a higher-is-better metric)
    headroom  =  ( mean_c M[merged, c] − mean_c oracle_c ) / mean_c oracle_c
```

Seeds are averaged **before** the optimum is taken. Taking the optimum per seed and averaging
afterwards lets a different seed win each column and biases the oracle low — an advantage no
deployed router has.

**6 — the merge scale α\***, the one quantity that is *chosen* rather than measured:

```
    α*  =  mean over seeds of   argmin over the α grid   Σ_c |D_c|·ℓ(θ_M(α), D_c) / Σ_c |D_c|
                                                          c ∈ {D_b, D₀ … D_{n−1}}
```

Three details carry weight. The pool is **window-weighted**, not a mean of per-column means.
It **includes the baseline's own slice D_b**, which at a 50% baseline is 50–68% of the weight —
so α\* is a *retention-weighted* optimum by construction, not a neutral one (§6.5, §6.6).
And the argmin is taken per seed and then averaged, which is why published values like 0.53 and
0.375 sit off the α grid.

**Aggregation, fixed once.** Every quantity above is computed from the per-seed *mean* of ℓ,
and ratios are ratios of means — never means of per-seed ratios, except α\* where the argmin
must be per seed. Mixing the two is what made an earlier version of the routing table
irreproducible. The reproducibility floor is `sd(ℓ) / mean(ℓ)` over the seeds of **one**
experiment, and is only comparable within that experiment.

The operational side of all this — which script computes what, which CSV it lands in, and what
was verified against what — is [EXPERIMENTS.md §0.6](EXPERIMENTS.md).

### 4.6 Merge cost

The price of one model instead of *n*: the merged model's error on a segment ÷ the error of
that segment's own specialist. **1.00× means merging is free.**

| | @ α = 1.0 | **@ α\*** |
|---|---|---|
| SWaT | 3.79 ±0.29× | **1.079 ±0.002×** |
| PSM | 1.18 ±0.11× | **1.008 ±0.020×** |
| ETTh1 | 1.69 ±0.32× | **1.007 ±0.026×** |

On PSM and ETTh1 the interval contains 1.00 — **merging is free**.

---

## 5. Geometry

Measured from the weights alone: no GPU, no dataset, seconds.

### 5.1 Three separate quantities

**Direction — cosine similarity.** The angle between τᵢ and τⱼ, magnitude divided out.

**Magnitude — norms.** ‖τᵢ‖/‖θ₀‖: how far fine-tuning moved the model relative to its own size.

**Direction, subspace form — principal angles.** For two subspaces U and V (say, the span of the
first k task vectors and the span of the rest), take orthonormal bases and the singular values
of Uᵀ V. Those singular values are cos θᵢ for the **principal angles** θ₁ ≤ … ≤ θₖ: θ₁ = 0 means
the subspaces share a direction exactly, all θᵢ = 90° means they are completely independent.
This generalises "the angle between two vectors" to "the angles between two *sets* of
directions", and is what ρ summarises in a single number.

**Both — effective rank.** Stack the τ's as rows, take singular values σᵢ, normalise
pᵢ = σᵢ²/Σσ², and take exp of the entropy of p. "How many independent directions do these
arrows really occupy?"

Plus **cosine vs temporal distance**: if similarity decays as segments get further apart, time
is what differentiates the vectors — the check that could have invalidated the whole framing.

### 5.2 Pairwise cosine is the wrong statistic; subspace overlap is the right one

In very high dimensions two *random* vectors are almost always near-orthogonal, which invites
"if orthogonality is free, why discuss it?" The objection is right about the statistic and
wrong about the concept: random vectors are not the relevant null, since task vectors are
gradients of the same loss on similar data from the same starting point.

**The question that matters is whether a new arrow carries information the earlier ones
didn't** — which is `sequential_overlap` (ρ): the fraction of τ_k lying inside the span of
τ₀…τ_{k−1}. **0 = all new, 1 = nothing new.**

| | ρ (subspace overlap) | mean cosine | effective rank (of 3) | mean ‖τ‖/‖θ₀‖ |
|---|---|---|---|---|
| SWaT | **0.607** | 0.737 | 1.63 | 0.0061 |
| PSM | 0.226 | 0.399 | 2.51 | 0.0059 |
| ETTh1 | 0.076 | 0.240 | 2.35 | 0.0185 |

On SWaT each new task vector sits **61% inside the span of its predecessors** — genuine
redundancy, not an artefact of dimensionality. Fine-tuning on segment 2 largely re-learned
what segments 0 and 1 already taught.

**Cosine decays with temporal distance on all three** — SWaT 0.771 → 0.667,
PSM 0.466 → 0.265, ETTh1 0.253 → 0.213. The framing survives.

**Effective rank conflates direction and magnitude** — a low value can mean "two vectors are
parallel" *or* "one vector is dead". Always read it with the norms.

### 5.3 What geometry is for

It does **not** predict outcome reliably, and that is not its job. Four reasons it earns its place:

1. **It tells you which fix to reach for.** High ρ → scale down or strip the shared component.
   Norms differing 20× → normalise. Norms near zero → your fine-tuning is broken; stop thinking
   about merging.
2. **It rules things out for free.** A dead fine-tune (τ 23× smaller than its siblings,
   `best_epoch=1`) was found in seconds on CPU with no dataset loaded.
3. **It is the only one you can optimise directly.** You can add an orthogonality penalty to a
   training loss; you cannot differentiate through "merge, then evaluate on n held-out slices".
4. **Whether it predicts outcome is itself the research question** — and the current answer,
   "not reliably", is the concrete evidence for the caveat that parameter-space cosine is not
   weight disentanglement (§7.3).

---

### 5.4 The regime indicator — measuring novelty exactly

### What it is

After fine-tuning segment *k* you hold τ_k and the accumulated history τ₀…τ_{k−1}. The useful
question before folding τ_k in is: **how much of it is genuinely new?**

Decompose τ_k against the subspace spanned by its predecessors. Write **P** for the
orthogonal projection onto that span:

> τ_k = **P**τ_k  +  (τ_k − **P**τ_k)
>
> ρ_k = ‖**P**τ_k‖² / ‖τ_k‖²   — the fraction of the update's *energy* already covered

By Pythagoras the orthogonal remainder has norm ‖τ_k‖·√(1 − ρ_k). Dividing by ‖θ₀‖ makes it
comparable across datasets and steps:

> **new_k = ‖τ_k‖ · √(1 − ρ_k) / ‖θ₀‖**

This is an **exact decomposition**, not a composite score: no weighting, no free parameters.
ρ ∈ [0,1], where 0 means the update is entirely orthogonal to everything learned so far and 1
means it adds no direction that wasn't already there.

> **Yes — this is the same ρ as everywhere else in this document.** §2 introduces it as
> "subspace overlap", §5.2 tabulates it per dataset, and this is its definition; in the code it
> is `sequential_overlap`. The only thing that changes is the *level of aggregation*:
>
> - **ρ_k** (here) is per step — one number per fold-in, k = 1 … n−1. Step 0 has no predecessors,
>   so ρ₀ is undefined and `new₀ = ‖τ₀‖/‖θ₀‖`.
> - **ρ** (§2, §5.2, §6.3) is the **mean of the ρ_k**, which is what `geometry_report` reports as
>   `mean_sequential_overlap`. SWaT's dataset-level 0.607 is just (0.483 + 0.733)/2; ETTh1's
>   0.076 is (0.075 + 0.065)/2.
>
> So a single per-dataset ρ hides the *trajectory*, and on these datasets the trajectory is
> where the interesting behaviour is — see the walkthrough below.

In practice the span is truncated to the leading singular directions covering ~90% of the
accumulated update's energy, which keeps ρ from being inflated by numerical noise in the tail.
That truncation rank is recorded alongside ρ.

### Why it should matter

The mechanism from §5: aligned vectors stack, so summing *n* near-parallel updates travels
roughly *n* times too far. A vector that is mostly *inside* the existing span adds little new
information but contributes its full magnitude to that stacking. So:

- **ρ → 1**: new segments are re-learning what is already there. The merge has little unique
  content to dilute, and scaling down costs almost nothing. **Accumulating is safe.**
- **ρ → 0 with large new_k**: each segment contributes a genuinely new direction, which a
  scaled sum averages away. **A separate model might preserve more.**

**Why those directions, spelled out.** The two methods fail differently, and ρ changes which
failure bites:

|  | merging (θ₀ + α·Στᵢ, α ≈ 1/n) | sequential (θ_k = θ_{k−1} + update) |
|---|---|---|
| what it does to a direction | keeps **every** τᵢ, each at ≈1/n strength | keeps the **last** at full strength, overwrites earlier ones |
| its failure mode | dilution | forgetting |

At **high ρ** every τᵢ points nearly the same way, so the average of *n* of them is close to any
one of them: merging's 1/n dilution costs almost nothing, and it buys stability for free. At
**low ρ** the vectors point different ways, so the average is a compromise serving each regime
at reduced strength — while sequential arrives fully adapted to the most recent segment. Hence
low ρ ⇒ sequential, high ρ ⇒ merge.

> ⚠️ **But that argument silently assumes what you evaluate on.** It compares the two on **new**
> segments, which is what the table below scores. Evaluate instead across *all* regimes and the
> reasoning reverses at low ρ: merging's 1/n-of-everything is exactly what a balanced objective
> wants, while sequential has overwritten the early regimes. §1.16 and §1.20 show that in
> practice — merging beats always-using-the-newest-specialist on **every** dataset. So ρ does
> not predict "which method is better" at all; at best it predicts "which is better *on the
> newest data*", and the sign of the rule flips with the evaluation target. A rule whose
> direction depends on an unstated choice is already in trouble before any data arrives.

That is the reasoning behind an *accumulate versus materialise* rule.

**The rule, stated explicitly — and where the threshold comes from.** The prose above gives a
mechanism but not a decision procedure, and the "ρ predicts" column below needs one:

> **predict *merge* if ρ > θ, *sequential* otherwise.**

⚠️ **There is no principled value for θ. It is fitted, and the conclusion depends on it.** The
original 4-dataset table only had *one* dataset on the merge side (SWaT, ρ = 0.607) against
0.226 / 0.117 / 0.076, so **any** cut in (0.226, 0.607) — a window 0.33 wide — reproduces the
same 3-of-4. The data pinned θ down to nothing sharper than "SWaT differs from the rest".

Fitting θ on the full 18-configuration sweep (below) puts it near **0.10**, far *lower*
than that window. This matters more than it sounds: an arbitrary θ = 0.35 scores 13/18, while
the fitted θ = 0.128 scores 16/18. **The same statistic looks useless or strong depending on a
number nobody derived.** Any use of ρ as an indicator has to report θ and how it was chosen.
All of these figures come from `analysis/novelty_report.py indicator`, which emits the fixed
and fitted thresholds, the permutation P and the leave-one-dataset-out score together, so the
three cannot be quoted apart.

### What it actually does — 3 of 4, and then not at all

> **Read this subsection as history.** ρ was tested on four datasets at one segment count
> each, where it scored 3 of 4. The segment sweep later raised the sample to **nine decisive
> configurations** and found that **no cheap signal separates the outcomes beyond chance** —
> and, more fundamentally, that the outcome is not a property of a dataset at all: it flips
> with the segment count on exchange_rate. The measurements below are correct and still worth
> understanding, because the *reasoning* about the two routes survives; the 3-of-4 hit rate
> does not. EXPERIMENTS.md §1.14.

| dataset | per-step ρ and new component | ρ predicts | actual winner on new segments |
|---|---|---|---|
| SWaT | step 0: ρ=—, new=0.00441 · step 1: ρ=0.478, new=0.00471 · step 2: ρ=0.737, new=0.00379 | merge | **merge** ✓ |
| PSM | step 0: ρ=—, new=0.00523 · step 1: ρ=0.303, new=0.00537 · step 2: ρ=0.149, new=0.00546 | sequential | **sequential** ✓ |
| ETTh1 | step 0: ρ=—, new=0.01517 · step 1: ρ=0.067, new=0.02594 · step 2: ρ=0.085, new=0.01281 | sequential | **sequential** ✓ |
| exchange_rate | step 0: ρ=—, new=0.01163 · step 1: ρ=0.208, new=0.01012 · step 2: ρ=0.026, new=0.03098 | sequential | **merge** ✗ |

**It fails on exchange_rate**, which has the lowest ρ of the four (0.026) and yet the
most decisive merge win (0.331 ±0.034 against sequential's
0.474 ±0.061).

### Step by step, on each dataset

> **Provenance.** Per-step ρ and new_k from `analysis/novelty_report.py steps` over
> `geometry_report` output; the indicator test from `novelty_report.py indicator` on a table
> built by `novelty_report.py outcomes` from `analysis_specs/rho_indicator_spec.csv`.
> Definitions in [EXPERIMENTS.md §0.6](EXPERIMENTS.md).


The table above collapses each dataset to one verdict. The per-step trajectory is more
informative, and it is what the rule is actually reading. `ρ_k` is the share of step *k*'s
update already inside the span of its predecessors; `new_k = ‖τ_k‖·√(1−ρ_k)/‖θ₀‖` is the part
that is genuinely new, as a fraction of the base model's norm. Mean over the seeds available;
n = 3 throughout, so there are two fold-in steps.

| dataset | ‖τ_k‖/‖θ₀‖ (k=0,1,2) | ρ₁ , ρ₂ | new_k (k=0,1,2) | trajectory |
|---|---|---|---|---|
| **SWaT** | 0.0058 / 0.0052 / 0.0076 | 0.483 , **0.733** | 0.0058 / 0.0037 / 0.0039 | overlap **climbing** |
| **PSM** | 0.0047 / 0.0067 / 0.0060 | 0.268 , 0.152 | 0.0047 / 0.0057 / 0.0055 | overlap falling, novelty flat |
| **ETTh1** | 0.0174 / 0.0235 / 0.0076 | 0.075 , 0.065 | 0.0174 / **0.0226** / 0.0074 | novelty spikes then collapses |
| **exchange_rate** | 0.0124 / 0.0114 / **0.0305** | 0.227 , **0.030** | 0.0124 / 0.0100 / **0.0300** | novelty **growing** |
| **ETTh2** | 0.0390 / 0.0316 / 0.0250 | 0.051 , 0.092 | 0.0390 / 0.0308 / 0.0239 | large, decaying |
| **ETTm2** | **0.0741** / 0.0504 / 0.0428 | **0.026** , 0.073 | 0.0741 / 0.0498 / 0.0412 | largest updates of all |

**SWaT — the only dataset that behaves the way the theory wants.** ρ climbs 0.483 → 0.733:
each period increasingly re-learns what is already there, and the genuinely new content shrinks
to a third of the first segment's. That is the accumulate-safely regime, and merging does win.

**PSM — the mirror image.** ρ *falls* (0.268 → 0.152) while the new component stays flat at
≈0.0055. Every period keeps contributing a fresh direction that a scaled sum would average
away, so the rule says keep them separate — and sequential wins.

**ETTh1 — novelty concentrated in one step.** The second segment is the big one (new = 0.0226,
the largest of the three), then the third contributes a third as much. ρ is near zero
throughout (0.075, 0.065): almost nothing repeats. Sequential wins, as the rule says.

**exchange_rate — where the rule breaks, and it breaks hard.** ρ collapses to 0.030 while the
new component *triples* by the last step (0.0100 → 0.0300) — the most novel final update of any
dataset here. The rule reads that as "each period is distinct, keep separate models" and
predicts sequential. Merging wins decisively instead. So the failure is not a near-miss on a
threshold; the signal points the wrong way at its most extreme value.

**ETTh2 and ETTm2 — added later, and they extend the negative result.** Both have low ρ
(0.05–0.09), so the rule predicts sequential for both. On **ETTh2 it is right** — sequential
beats merging at all three segment counts. On **ETTm2 it is right at n = 2 and 3 and wrong at
n = 5**, where the merge overtakes. One dataset, one rule, two answers depending on how the
same data is chunked: that is the sharpest form of the §1.14 finding that the outcome **is not
a property of the dataset**.

**One thing the trajectories do explain — but it is not the merge/sequential question.** The
relative update size ‖τ‖/‖θ₀‖ orders the datasets almost exactly as headroom does: ETTm2
(0.074, 86.6% headroom) > ETTh2 (0.039, 83.8%) > ETTh1 (0.017, 38–43%) > exchange_rate (0.012,
43.8%) > the saturated AD pair (0.005, ~1–3%). When the base model is far from what the data
supports, fine-tuning moves it a long way; when it is already near the ceiling, the task vectors
are tiny. That is a sanity check on the whole setup rather than a finding — but it does mean a
small ‖τ‖ is a reliable sign that **no** update strategy has much to win, which is exactly the
situation on SWaT and PSM.

### Is ρ a good indicator? The answer depends on choices nobody derived — and it ends at "no"

ρ is computed on the actual segmentation, so it varies with n and can be tested on **every**
configuration measured: six datasets × n ∈ {2, 3, 5} = 18. Winner = whichever of merged /
continual is better on the test metric.

| threshold θ | accuracy | note |
|---|---|---|
| 0.35 (arbitrary, "SWaT vs rest") | 13/18 = 72% | vs 56% majority baseline; binomial P = 0.124, **not significant** |
| **0.096 (fitted on these 18)** | **16/18 = 89%** | permutation test, θ re-fitted per shuffle: **P = 0.0037, significant** |

So the first verdict — "ρ has no skill" — was an artefact of a threshold chosen by hand and set
three times too high. With θ fitted, ρ genuinely separates the outcomes, and the separation
survives a permutation test that re-fits θ on every shuffle. Held out properly
(leave-one-*dataset*-out, θ fitted on five datasets and tested on the sixth, θ landing in
0.084–0.127 across folds) it scores **14/18 = 78%** against a 56% baseline.

**Then it collapses.** Restrict to the configurations where the merge/continual difference
actually clears that dataset's reproducibility floor — 16 of 18 are decisive — and it still
holds: 14/16 fitted (P = 0.0103), **12/16 = 75% out-of-sample** against 56%. But **remove the two saturated
AD datasets** and:

> **decisive, forecasting only (11 configurations): leave-one-dataset-out = 8/11 = 73%, and the
> majority baseline is also 8/11 = 73%. Zero out-of-sample skill.**

All of ρ's apparent predictive power comes from SWaT and PSM. And those "decisive" AD outcomes
are decisive only in the narrowest technical sense: margins of **0.11%–0.45%**, clearing floors
of 0.07%–0.09%, on datasets where the base model is already within 1.1% and 3.4% of joint
training. They clear the noise bar while being practically meaningless — which is exactly the
situation §0.1b warns should be read as *"nothing separates"* rather than as evidence.

**Verdict.** On the datasets where merge-versus-sequential is a real question with real margins,
ρ predicts nothing a coin biased to "sequential" would not. EXPERIMENTS.md §1.14's conclusion
stands, and this is the sharper statement of *why*: not that ρ is uninformative about geometry —
it is exactly informative about geometry — but that **the geometry of the updates does not
determine which training strategy wins**, and the appearance that it does was carried by two
saturated datasets and a hand-picked threshold.

**Worth recording as a methodological point.** This subsection reached three different verdicts
in a row — "no skill" (arbitrary θ), "significant" (fitted θ), "no skill on the data that
matters" (fitted θ, saturated datasets excluded). All three are correctly computed from the same
numbers. The lesson is that a threshold rule reported without its threshold, its fitting
procedure, and its held-out score is not a result at all.

### What each route actually requires, operationally

Two questions the mechanism above glosses over: merging needs the task vectors — must you keep
all *n* fine-tuned models? And what exactly is "sequential" fine-tuning fine-tuning *from*?

**Merging does not require storing n models.** The merge is θ₀ + α·Στᵢ, and Στᵢ is a running
sum: after period *k* you update **S ← S + (θ_k − θ₀)** and discard θ_k. Serving needs θ₀, S and
a counter *k*; the served weights are θ₀ + (k⁻¹)·S. **Storage is two model-sized objects
regardless of how many periods have passed** — which is the whole operational case for merging,
and why it belongs in the "stores no raw data, stores one model" column of §11.3. The *n*
specialists exist only inside the experiments, because measuring the transfer matrix and merge
cost requires scoring each one; a deployment never materialises them together.

**Why a uniform average, and not a weighted sum?** α·Στᵢ with α = 1/n *is* the unweighted mean,
and nothing here tested alternatives. Three obvious weightings are untested:

- **By shard size.** In this project every segment is the same size by construction, so uniform
  weighting and size weighting **coincide** — the experiments cannot distinguish them. Under
  unequal periods (the realistic case) they diverge, and nothing here says which is right.
- **By recency.** Weakly contraindicated: §1.20 finds the newest specialist is the best model
  for a regime in 1 case out of 16, so up-weighting recent vectors optimises for the wrong
  thing. Untested directly.
- **By ‖τᵢ‖ or by novelty √(1−ρᵢ).** The natural geometric candidates, given that α\* tracks
  1/alignment (§1.18). Also untested, and the most interesting of the three.

**So "average the task vectors" is a finding about the *scale*, not about the *weights*.** What
was measured is that the coefficient in front of the sum must be ≈1/n rather than 1; whether the
*n* terms should carry equal weight is a separate question this project never asked.

**Sequential fine-tuning: from what, on what.** The chain is

> θ₀ → (train on segment 0) → θ₁ → (train on segment 1) → θ₂ → … → θ_n

Each step starts from the **immediately preceding model**, not from θ₀, and trains on **only the
new segment's training slice** — no replay, no retained history. Early stopping and checkpoint
selection use **that segment's own validation slice**. This is the one-line difference from the
merging pipeline, where every segment instead starts from the **frozen** θ₀ — which is precisely
what makes the τᵢ = θᵢ − θ₀ comparable to one another and therefore summable. The two are not
variants of one method: sequential models share no common base, so **task arithmetic does not
apply to them at all**.

One option exists and is off by default: `reg_lambda > 0` adds an L2-SP penalty pulling each
step back toward an anchor — θ₀ by default, bounding cumulative drift, or the previous model.
At `reg_lambda = 0` it is inert, and every number in this document is at that setting.

**The update loop, period by period.** Two common misreadings are worth heading off — one about
what merging stores, one about what sequential fine-tunes *from*:

| | **merging** | **sequential** |
|---|---|---|
| stored between periods | **θ₀ (frozen, forever) + running sum S** | the current model only |
| each period fine-tunes from | **frozen θ₀** — reloaded every time | **the previous model** θ_{k−1} |
| trained on | that period's data only | that period's data only |
| update | τ_k = θ_k − θ₀ ; **S ← S + τ_k** ; discard θ_k | θ_k *becomes* the model |
| served | θ₀ + (1/k)·S | θ_k |
| storage | 2 models, **independent of k** | 1 model |

**You cannot keep only the merged model.** Storing θ₀ + S is equivalent to storing θ₀ and the
current merge (since S = k·(merged − θ₀)), but **θ₀ itself can never be discarded**: the next
task vector is defined against it, and re-scaling the average from 1/k to 1/(k+1) needs it too.
Drop θ₀ and the scheme cannot continue. That is the real storage cost of merging — two models,
not one, and not *n*.

**And sequential does *not* fine-tune the merged model.** It fine-tunes the previous *sequential*
model. Fine-tuning the merge would be a **third method**, and one this project never ran:

> **merge-then-continue** — serve the merge, then fine-tune *it* on the next period.

It is a tempting hybrid and it breaks task arithmetic. Once you fine-tune the merged model the
base has moved, so the next τ is measured against a drifting reference and is no longer
comparable to its predecessors — the property that makes the sum meaningful (§4.2) is exactly
what is lost. You would have to either re-anchor to the new model each period, which *is*
sequential fine-tuning with extra steps, or keep θ₀ as the anchor while serving something else,
in which case the merge and the fine-tune pull against each other. **Untested here, and flagged
as future work rather than recommended.**

**Which is why the two fail differently.** Merging keeps a fixed-size summary of *all* periods
and dilutes each by 1/n. Sequential keeps full strength on the newest period and lets earlier
ones decay through overwriting — with nothing anchoring them unless L2-SP is switched on. That
asymmetry, not ρ, is what §11.3's decision rule is built on.

### Why the failure is informative

exchange_rate is also the dataset with the **largest headroom** (43.8% base-to-joint) and
the **strongest input drift**, and it is the only one where the merge beats *joint training*
outright (GRR 1.207 at n = 3 with α chosen on validation — 1.164 ±0.080 as a per-seed mean,
EXPERIMENTS.md §1.10 — and above 1.0 at every
segment count). That points at a second, independent route by
which merging can win:

- **Route 1 — redundancy.** Updates repeat each other, sequential training drifts cumulatively
  and forgets, merging caps the drift. ρ sees this. *(SWaT)*
- **Route 2 — ensembling across regimes.** Under strong progressive drift, joint training
  weights every regime equally and sequential training over-commits to the newest, while
  base + scaled task vectors sits between them and generalises forward better than either.
  **ρ cannot see this**, because it is a statement about the *data's* structure, not the task
  vectors' geometry. *(exchange_rate)*

So ρ is a **sufficient** condition for one route and blind to the other. A usable controller
would need at least a headroom term alongside it — and headroom is not cheaply observable
online, since measuring it is what joint training does. The larger sweep tested exactly that:
headroom, α\*·n, merge cost and n itself were each given a threshold and asked to separate the
nine decisive configurations. **None does.**

### The practical alternative

Because merging is training-free, you rarely need to *predict* this. After each segment,
build both candidates and score them on the held-out slices you have already accumulated:
that measures the decision criterion directly rather than inferring it from a proxy validated
on four points. See §9.

The exception is unsupervised anomaly detection, where those accumulated slices carry no
labels, so you can only measure reconstruction — and §8.3 shows reconstruction and detection
come apart. There, a cheap proxy is all you have, and no proxy has been found. That is the
strongest argument for a merging method that needs no coefficient at all
([EXECUTION_PLAN.md §3.7](EXECUTION_PLAN.md), the AD-dataset scope question).

---

## 6. Overshoot and interference

### 6.1 Aligned, orthogonal, anti-aligned

**Orthogonal** is the dream: each arrow occupies its own direction, ‖Στ‖ = √(Σ‖τᵢ‖²), and
adding costs nothing. **Aligned arrows stack**: as cosine → 1, ‖Στ‖ → Σ‖τᵢ‖. **Anti-aligned
arrows cancel** and you lose both.

Where the real vectors sit on SWaT: ‖Στ‖ = 1.854 against 1.192 if orthogonal and 2.023 if
perfectly aligned — **92% of the fully-aligned bound**. Almost nothing cancels.

### 6.2 The distinction that matters

> total merge cost = **overshoot** (curable by scaling α) + **irreducible interference**
> (what remains at α\*)

The usual story about interference is *conflict* — vectors pulling apart, the sum landing
somewhere useless. What happens here is the opposite: the vectors **agree**, so summing them
**overshoots**. Like three people each saying "turn it up a bit" and turning it up by the sum.

Two different failures of the same assumption, with very different implications: **overshoot
is a magnitude error and a scalar fixes it; conflict is a direction error and no scaling
saves you.**

### 6.3 Measured: it is almost all overshoot

Tracing the **validation block** (actual model quality) against α separates them. If the
merged curve descends to the diagonal, the cost was overshoot; if it plateaus above, the
residual is real interference.

| | merge cost @ α=1.0 | merge cost @ α\* | α\* | old regime @ α=1.0 | old regime @ α\* |
|---|---|---|---|---|---|
| SWaT | 3.79 ±0.29× | **1.079 ±0.002×** | 0.250 ±0.000 | 4.931 ±0.287 | **1.013 ±0.012** |
| PSM | 1.18 ±0.11× | **1.008 ±0.020×** | 0.500 ±0.000 | 1.664 ±0.158 | **1.139 ±0.053** |
| ETTh1 | 1.69 ±0.32× | **1.007 ±0.026×** | 0.367 ±0.050 | 1.897 ±0.379 | **0.967 ±0.029** |

**Merging is essentially free on all three once the scale is right.** SWaT's collapse at
α = 1.0 — nearly 4× merge cost and ~5× damage to the base regime — is *entirely* overshoot.
The model was never destroyed; it was pushed about four times too far.

**Forgetting goes with it.** At α\* it is 1.013 on SWaT and **0.967** on ETTh1 — the
merged model is *better* than the base model on the base model's own regime.

> ⚠️ **This claim is partly circular, and the size of the circularity is measurable.** α\* is
> chosen on the pooled validation union, which *includes the baseline's own validation slice*
> (`val_base`) weighted by window count — and at `baseline_fraction=0.5` that slice is **50–68%
> of the selection signal** (SWaT 50%, exchange_rate 52%, ETTh2 57%, ETTh1 at n=5 **68%**). So α
> is selected with roughly half its weight on staying close to θ₀, and we then report that at
> that α the base regime is undamaged. The finding is not empty — nothing forced the base-regime
> ratio below 1.0, and the *magnitude* of the overshoot at α = 1 is unaffected — but it should
> be read as **"at an α chosen with ~50% weight on base retention, base-regime damage
> vanishes"**, not as an unconditional property of merging. Excluding `val_base` moves α\*
> materially (ETTh1 n=3: 0.30 → 0.40; PSM n=2: 0.50 → 0.75); the full sensitivity is in
> [EXPERIMENTS.md §1.18](EXPERIMENTS.md).

**Non-orthogonality still matters, but its consequence changes.** It is precisely *because*
the vectors agree that summing overshoots. Non-orthogonality does not make merging fail — it
dictates a smaller α.

> **Withdrawn:** an earlier version of this section claimed the ordering holds — *"the most
> collinear dataset needs the smallest scale, the least collinear the largest"* — and named PSM
> as the least collinear. It is not; ETTh1 is (ρ = 0.076 against PSM's 0.226), and ETTh1's α\*
> is *smaller* than PSM's, not larger. Across four datasets the ranking by ρ
> (SWaT, PSM, exchange_rate, ETTh1) does not match the ranking by α\*
> (SWaT, ETTh1, exchange_rate, PSM). **ρ sets the direction — more agreement means a smaller
> scale is needed — but it does not order the datasets.** What does hold is the weaker
> within-dataset regularity α\*·n ≈ order 1 (§6.6) — near-flat rather than constant, and
> conditional on how α\* is selected.

### 6.4 Remedies

| technique | fixes | relevant here? |
|---|---|---|
| **Global α** | pure magnitude | yes, and proven — this is the whole finding |
| **α = 1/n** (average, not sum) | magnitude, natural default when aligned | **approximately right for a fixed base** — §6.6 |
| **Per-vector normalisation** | *imbalance* between vectors | when norms differ a lot |
| **Norm-matching the sum** analytically | magnitude | gives roughly the right range but **does not** predict the measured optimum; sweep and measure |
| **Strip the shared component** — apply it once, keep residuals at full strength | alignment specifically | aimed at SWaT's actual pathology |
| **TIES / DARE** | conflict, redundancy | targets conflict more than overshoot |

### 6.5 Making specialists more ideal

- **Split at change-points, not equal time slices.** Equal chunks of near-stationary data are
  near-identical distributions, so *n* fine-tunes learn one thing *n* times. This attacks the
  root cause.
- **Constrain the vectors during training** — penalise alignment with previously-learned τ's.
- **Restrict where each arrow lives** — per-segment adapters, or freeze shared layers.
- **Sparsify the deltas** — sparse high-dimensional vectors overlap far less.
- **Equalise the training budget** so magnitudes are comparable.

L2-SP is *not* this: it shrinks all arrows toward zero rather than making them different.

**And it was measured.** Anchoring each fine-tune to the base with an L2-SP penalty is the
obvious way to keep the task vectors in the linear regime, and it is the first thing anyone
asks about — so it was tested cleanly on ETTh1, three seeds per λ, α selected per variant
(EXPERIMENTS.md §3.1). **No measurable effect at λ ∈ {1e-3, 1e-2}**: the difference is ~3%
against a ~5% within-variant spread, and its *sign flips* depending on whether you read absolute
error, ratio-to-own-baseline, or hardware-matched pairs. That instability is the point — the
effect is smaller than the noise. Constraining magnitude is not what merging needs here,
which is consistent with §6.3: the problem was never that the vectors were too long, it was
that they were being *summed* rather than averaged.

---

### 6.6 The optimal merge is the *mean* task vector

The remedy table lists α = 1/n as a sane prior. It is better than that: it is what the data
says, and it is the cleanest quantitative result in the project.

Sweeping the number of segments n ∈ {2, 3, 5} on four datasets and selecting α on validation
at each, the product **α\*·n stays at order 1 within a dataset on five of six** — near-flat there,
though not literally constant (ETTh2, added later, drifts 0.97 → 0.80 → 0.75; see
[EXPERIMENTS.md §1.18](EXPERIMENTS.md)), and *not* attributable to the count alone for the
identifiability reason at the end of this subsection:

| dataset | α\* at n=2 | n=3 | n=5 | α\*·n |
|---|---|---|---|---|
| SWaT | 0.40 | 0.23 | 0.20 | 0.80 / 0.70 / 1.00 |
| PSM | 0.53 | 0.43 | 0.30 | 1.07 / 1.30 / 1.50 |
| ETTh1 | 0.50 | 0.30 | 0.20 | 1.00 / 0.90 / 1.00 |
| exchange_rate | 0.70 | 0.53 | 0.30 | 1.40 / 1.60 / 1.50 |
| ETTh2 | 0.48 | 0.27 | 0.15 | 0.97 / 0.80 / 0.75 |
| ETTm2 | 0.38 | 0.27 | 0.15 | 0.77 / 0.80 / 0.75 |

> ⚠️ **Correction withdrawn (2026-08-07).** This row was briefly changed to 0.75 / 0.50 / 0.50
> → 1.50 / 1.50 / 2.50 on the grounds that it reproduced under no rule. That was wrong: it
> reproduces **exactly** (0.500 / 0.375 / 0.250) once the baseline's validation slice
> (`val_base`) is included in the pooled selection signal, which is what the forecasting rule
> does and what §0.6 now specifies for both. The failed reconstruction used a rule that excluded
> it — an under-specification in §0.6, not an error in the data. The original values stand.
> SWaT is insensitive to this choice, which is why it reproduced either way and masked the
> ambiguity.

Why that is a statement about the *mean*: the merge is θ₀ + α·Στᵢ. Write α = k/n and it
becomes θ₀ + k·(Στᵢ)/n = θ₀ + k·**mean**(τ). A constant α·n = k says the best merge always
steps a fixed multiple k of the *average* task vector — k ≈ 1 on five datasets, ≈1.5 on
exchange_rate — no matter how many vectors are being averaged.

**Why it is not just "the vectors got smaller".** Individual task vectors *do* shrink as
segments shrink (ETTh1 mean ‖τᵢ‖: 1.85 → 1.53 → 1.10). If α\* were compensating for
magnitude it would have to **grow** as ‖τ‖ falls. It falls instead, as 1/n. Meanwhile ‖Στᵢ‖
stays nearly constant (3.05 → 3.39 → 3.49), because the vectors get smaller *and* less
mutually aligned (0.824 → 0.737 → 0.633) and the two effects cancel. So α\* is not merely
compensating for vector **size** — which rules out one explanation but does not establish the
count as the cause, for the reason immediately below.

**The caveat, and why it cannot be removed.** The sweep varied n with the baseline fixed, so
shard size fell as 1/n. Two further designs tried to separate them:

- *Fixed shard size*, letting the baseline shrink: ETTh1 stays flat (1.00 / 1.10 / 1.08),
  **exchange_rate grows** (0.97 / 1.25 / 1.58).
- *Prefix merges* from one run — base **and** shard size fixed, only the count varying:
  **both grow** (ETTh1 0.48 → 0.83, exchange_rate 1.20 → 1.50).

And there is no fourth design. On a fixed series `baseline + n × shard = total`, so fixing any
two forces the third to move: **the count can never be varied alone.** Every design confounds it
with shard size, baseline size, or total coverage. *"Is α\* a function of the count?"* is not an
identifiable question here, and more datasets reproduce the same constraint rather than escaping
it.

So the honest status is: **an empirical regularity in one parameterisation, not a law.** In the
setting that matters for deployment — a fixed base model with shards tiling what arrived after
it — starting from the mean of the task vectors is a good default and much better than α = 1.
That is what to carry forward.

Two measurement cautions attach to every number here. The apparent *exact* agreement across
seeds was an artefact of the 0.1 grid (at 0.05 the seeds differ by one step), and quantisation
alone contributes ±(grid × n) — ±0.50 at n = 5. The minimum itself is sharp enough to test
(15–51% penalty between α·n = 1.0 and 1.5, against floors of 8.2% and 5.3%), so resolution and
identifiability were the binding limits, never the data.
[EXPERIMENTS.md §1.18](EXPERIMENTS.md).

### 6.6a Where α = 1/n comes from — a derivation

The measurement says α\*·n ≈ 1. Here is a model under which that is *exactly* right, which is
worth having because it also says what the model cannot explain.

**Setup.** Suppose each period's task vector decomposes into a shared drift direction plus
period-specific noise:

> **τᵢ = d + εᵢ**,  with E[εᵢ] = 0, Cov(εᵢ) = σ²I, and the εᵢ independent.

That is the "the periods are teaching the same thing, measured noisily" hypothesis. Then

> **Στᵢ = n·d + Σεᵢ**

and the merged model is θ₀ + α·(n·d + Σεᵢ).

**Signal term.** If the correct update is to move *once* along d, you need α·n·d ≈ d, hence

> **α\* = 1/n**,  i.e.  **α\*·n = 1** ∎

**Noise term.** The independent noise adds in quadrature: E‖Σεᵢ‖² = n·σ²·D for dimension D, so
‖Σεᵢ‖ ≈ σ√(nD). Scaled by α = 1/n the residual noise is

> **α·‖Σεᵢ‖ ≈ σ√(D/n)** — it *shrinks* as 1/√n.

So under this model **merging is denoising**: averaging n noisy estimates of one direction beats
any single one, and more periods help. That is the formal version of the reading in [EXPERIMENTS.md §1.19](EXPERIMENTS.md) —
merging may be denoising a single update rather than composing distinct knowledge.

**The spectrum.** Two extremes bracket the behaviour:

| if the periods… | then Στ ≈ | optimal α | α·n |
|---|---|---|---|
| teach the **same** thing (τᵢ = d + noise) | n·d | 1/n | **1** |
| teach **disjoint** things (τᵢ all needed) | Σdᵢ | 1 | **n** |

**α·n is therefore a measurement of how much the periods repeat each other** — 1 means "one
thing, measured n times", n means "n genuinely different things". Measured values of 1.0–1.5
say the periods mostly repeat, slightly less so on exchange_rate.

**Where the model fails, and this matters.** It predicts α·n should rise as the updates become
*less* redundant — i.e. α·n should order inversely with ρ. **It does not.** SWaT has the highest
ρ (0.607, most repetitive) yet not the lowest α·n; ETTh1 has the lowest ρ (0.076) and α·n ≈ 1.
Across four datasets the ρ ordering and the α·n ordering disagree (§6.3). So the
shared-drift-plus-isotropic-noise picture gets the *magnitude* right and the *cross-dataset
ordering* wrong — the εᵢ are evidently neither isotropic nor independent of d. Treat the
derivation as an explanation of why α·n is order 1, not as a predictive model.

### Why this is not just "average the task vectors"

Simple averaging is a known baseline — model soups, `mergekit`'s normalisation — so *"use the
mean"* on its own would be a rediscovery. The contribution is the **contrast between regimes**.

In image classification, where tasks are near-orthogonal, the published task-arithmetic scaling
coefficient *falls* with the number of tasks but far more slowly than 1/n, so the product α·n
**grows** as tasks accumulate. Here, with aligned segments, it is **constant**.

| | orthogonal tasks (image classification) | aligned segments (this work) |
|---|---|---|
| α·n as n grows | grows | constant (≈1, or ≈1.5 on exchange_rate) |
| averaging vs summing | averaging reported to lose ground | averaging is optimal |

**The claim worth making:** in the aligned regime the optimal coefficient collapses to exactly
1/n, which does not happen in the orthogonal regime — so α·n is itself a **measurable signature
of which regime a task set is in**, reachable without computing any geometry.

> ⚠️ **The signature is convention-dependent, and the claim must say which α it uses.** α\* is
> selected on the validation union *including the baseline's own slice*, which carries 50–68% of
> the signal ([EXPERIMENTS.md §0.6](EXPERIMENTS.md)). Drop that slice — asking instead "what α
> best incorporates the *new* information?" — and the products **grow** on ETTh1 (1.13 → 1.33)
> and PSM (1.50 → 2.50): the orthogonal signature, from vectors whose measured pairwise cosine
> is 0.20–0.74. The contrast with image classification is still real, because the underlying
> geometry differs, but *"α·n is constant here"* holds for the **deployment** objective and not
> for the new-information one. Stated without that qualifier the signature is not falsifiable.
>
> The geometry in fact predicts the *rising* version: alignment ‖Στ‖ ÷ Σ‖τ‖ falls with n
> (ETTh1 0.808 → 0.598), which should push α\* above 1/n. So the flat product is the composite
> of a rising geometric term and a retention term that grows with n — not a bare property of
> the task vectors. Confirmed **within** each dataset — as n grows and alignment falls, α\*·n
> rises, with a positive per-dataset correlation in 5 of 6 (median r = +0.87) — but **absent
> between** datasets (pooled r = −0.03 over all 18 points): ETTm2 is less aligned than SWaT at
> every n yet has a much lower α\*·n. Alignment predicts the *trend* in n, not the *level*, and
> the ratio α\*·n ÷ (1/alignment) spans 0.54–1.71, so proportionality is *not* supported. Quoting
> the pooled figure alone is what produced three different readings (+0.44, +0.31, −0.03) of the
> same relationship as datasets were added; see [EXPERIMENTS.md §1.18](EXPERIMENTS.md). That is a
statement about time series specifically (§2), and it is a contribution; "average the vectors"
is not.

> **Unverified.** The image-classification side of that table is recorded from prior knowledge,
> not from fetched sources. Published coefficients around ~0.4 at 2 tasks falling to ~0.15 at
> 20 would give α·n ≈ 0.8 → 3.0, and TIES-Merging reports simple averaging losing ground to
> task arithmetic on image pairs. **Check these against the papers before the comparison goes
> into the thesis** — the argument stands on the shape of the trend, but the numbers must be
> cited from source.

**What this buys.** If it holds under that control, the merge scale stops being a
hyperparameter: use the mean, scaled by one dataset-level constant. On AD, where §8.4 shows α
cannot be tuned honestly at all, that would be the difference between a method you can deploy
and one you cannot.

## 7. Entanglement

### 7.1 The definition

A model is **weight-disentangled** with respect to a set of task vectors when, on inputs
belonging to task *i*, the model with *all* vectors applied behaves the same as the model with
*only* τᵢ applied. Adding the others doesn't disturb what τᵢ does on its own turf.

### 7.2 It is a comparison, not a cell value

**The trap:** a small cell means the model is *good in absolute terms* on that column.
Entanglement is a *comparison*. The measure is

> **entanglement on segment i = (merged on segment i) ÷ (θ₀ + τᵢ on segment i)**, both at α\*
>
> 1.00 = perfectly disentangled · above 1 = the other vectors got in the way ·
> **below 1 = positive transfer**, the merge beats the lone specialist

Measured at α\*:

| | segment 0 | segment 1 | segment 2 |
|---|---|---|---|
| SWaT | 0.98× | 1.04× | **1.28×** |
| PSM | 1.01× | 0.92× | **1.11×** |
| ETTh1 | 0.76× | 0.97× | **1.53×** |

Two readings. **Positive transfer is real** — values below 1.00 mean the merged model *beats*
the specialist that owns that segment, because the other vectors carried information that
helped there. And **entanglement concentrates on the last segment**, ≈1.0 elsewhere. That
follows mechanically from α\* < 1: scaling down shrinks *every* vector, and the last segment
sits furthest from θ₀, so it is the one that most needed its own vector at full strength.
**The newest regime pays for the merge.**

### 7.3 Why cosine is not disentanglement

Two arrows can be orthogonal in parameter space and still fight in function space, because
what matters is whether they change the model's behaviour **on the same inputs**. Parameter
orthogonality says nothing directly about that. Geometry is a *proxy*; the matrix is *ground
truth*; whether the proxy predicts the truth is an open question currently answered "weakly".

---

## 8. Metrics that cannot see what you are measuring

### 8.1 The observation

On SWaT at α = 1.0 the merged model's reconstruction error is nearly **5× worse** on the base
regime — and its AUROC goes **up** by 0.09%.

### 8.2 The resolution

**Every test metric in these runs is invariant to a monotone rescaling of the scores.** AUROC
and AUPRC are rank-based; the runs use `threshold_strategy=oracle`, which sweeps each metric's
own best threshold, so a shifted scale is absorbed. And merging did something very close to a
monotone rescale — it raises the *floor* (windows the base reconstructed almost perfectly get
much worse) while the hard tail barely moves.

**So the two readings are not in conflict:** the validation block measures the model as a
*reconstructor*; the test block measures it as a *ranker*. Merging wrecked the first and left
the second alone.

**The lesson generalises:** before concluding "X had no effect", check whether the metric is
*capable* of registering X.

### 8.3 Why improving reconstruction does not improve AUROC

Careful: AUROC is invariant to *rescaling*, not to model improvement in general. The deeper
reason is different:

> **The validation block measures reconstruction error on *normal* data only. Detection
> depends on the *contrast* between normal and anomalous.**

A better autoencoder reconstructs **everything** better, anomalies included, so the gap can
stay flat while every raw number improves. Like making a smoke alarm's sensor more sensitive:
useless if it becomes equally more sensitive to burnt toast and to real fires.

### 8.4 The consequence: α cannot be tuned honestly in unsupervised AD

To pick a merge scale defensibly you need a signal that tracks the metric you will report and
is not the test set. For **forecasting** that exists — validation and test are both MSE.

For **anomaly detection it does not**: validation measures reconstruction, the test metric
measures detection, §8.3 is exactly the statement that these come apart, and you cannot select
on test because that is selecting on the number you report. The reason there is no third
option is structural:

> **AD training data carries no labels by construction.** That is the premise of the whole
> setup. So there is no held-out set on which detection can be measured.

**Practical rule: select α on validation for forecasting; use a fixed, pre-declared α for AD.**
The stability helps — **α\* does not move across seeds** on either AD dataset — but it does not
rescue the situation, and the segment sweep measured how badly.

The two optima do not merely differ in principle; they point to different values:

| dataset | α minimising validation reconstruction | α maximising test AUROC |
|---|---|---|
| SWaT | 0.50 | ≥1.50 — AUROC rises monotonically past the grid edge |
| PSM | 0.50 | 0.75 |

Choosing α on validation therefore costs **almost everything achievable on SWaT**
(96–98% of the achievable GRR) **and costs 25–44% on PSM**, against
**1–8% on forecasting** (EXPERIMENTS.md §1.12). And AUROC is not insensitive to α — it moves
6× the noise floor on SWaT and 85× on PSM. It moves in a direction validation cannot see.

Note what the SWaT row means concretely: at α = 1.5 the merged model reconstructs 2.5× worse
than at its validation optimum, and detects *better*. **A model that reconstructs badly can
separate anomalies well.** Reconstruction level is simply not the quantity detection depends
on — the score *distribution* is.

That last observation is the one opening left. Mean reconstruction error throws away exactly
the information detection uses. A spread statistic on the validation scores — p99/p50, std,
kurtosis — might have its optimum where AUROC does, and those columns are already recorded in
every curve, so it is analysis rather than new training ([EXECUTION_PLAN.md §3.2](EXECUTION_PLAN.md)). If none of
them works, the honest conclusion is that **unsupervised AD cannot tune the merge scale**, and
needs a coefficient-free merging method instead.

---

## 9. Versus sequential fine-tuning

### 9.1 The comparison that matters

Everything above compares merging against the *frozen base* and against *joint training*.
Neither is what a practitioner would do. The obvious alternative is **sequential fine-tuning**:

    θ₀ --seg 0--> θ₁ --seg 1--> θ₂ --seg 2--> θ₃

A genuinely different method, not a variant — sequential models share no common base, so there
are no comparable task vectors and task arithmetic does not apply to them at all.

| | ρ | sequential — new | merged — new | verdict | sequential — old | merged — old | verdict |
|---|---|---|---|---|---|---|---|
| SWaT | 0.607 | 0.697 ±0.011 | **0.656** | **merge** | 1.635 ±0.049 | **1.013** | **merge** |
| PSM | 0.226 | **0.702 ±0.013** | 0.759 | **sequential** | 1.159 ±0.024 | 1.139 | *tie* |
| ETTh1 | 0.076 | **0.560 ±0.038** | 0.705 | **sequential** | 1.110 ±0.143 | 0.967 | *tie* |

**It is a stability/plasticity trade, and which side wins depends on the segment count, not
only on the dataset** (EXPERIMENTS.md §1.13). Sequential
fine-tuning is more *plastic* — it adapts harder to the new segments on PSM and ETTh1. Merging
is more *stable*, but demonstrably so only on SWaT; elsewhere the old-regime intervals overlap.

### 9.2 Forgetting tracks how much the updates repeat each other

Base-regime ratio after each sequential step (>1 = worse than the model you started with):

| | after seg 0 | after seg 1 | after seg 2 | ρ | headroom |
|---|---|---|---|---|---|
| SWaT | 1.115 | 1.209 | **1.586** | 0.607 | 1% |
| PSM | 1.046 | 1.217 | 1.158 | 0.226 | 3% |
| ETTh1 | 1.037 | 0.936 | **0.972** | 0.076 | 39% |

**SWaT gives the textbook forgetting curve — monotone, ending ~59% worse. ETTh1 shows none.**

> **Merging's advantage over sequential fine-tuning appears exactly where the task vectors
> most repeat each other** — the opposite of the intuition that redundant updates make merging
> pointless.

### 9.3 Two explanations, not separable here

**Redundancy.** SWaT's updates repeat each other (ρ = 0.607), so sequential training drifts
cumulatively in one consistent direction. Merging applies that same drift scaled by α\*, which
is why its base slice stays near 1.00.

**Headroom.** SWaT's base is already within 1% of joint training, so extra training
cannot improve it — only move it. ETTh1's base is 39% below its ceiling, so more
training genuinely helps everything. **ETTh1's lack of forgetting is therefore not immunity —
it is an undertrained base.**

Both point the same way on all three datasets. Separating them needs a dataset with *headroom*
and *redundant updates* simultaneously, which none of these provides.

---

## 10. What the datasets said

| | SWaT | PSM | ETTh1 |
|---|---|---|---|
| ρ (update overlap) | **0.607** | 0.226 | 0.076 |
| mean cosine | 0.737 | 0.399 | 0.240 |
| α\* | 0.250 ±0.000 | 0.500 ±0.000 | 0.367 ±0.050 |
| merge cost @ α\* | 1.079 ±0.002× | **1.008 ±0.020×** | **1.007 ±0.026×** |
| old regime @ α\* | 1.013 ±0.012 | 1.139 ±0.053 | 0.967 ±0.029 |
| base → joint gap | 1.0% | 3.4% | 39.2% |
| GRR @ α\* | *inside noise* | **0.857 ±0.031** | 0.767 ±0.141 |
| seed noise floor | 0.09% | 0.07% | 8.76% |
| vs sequential | **merge wins both** | sequential on new, tie on old | sequential on new, tie on old |

**SWaT is a control, not evidence.** A frozen base on half the data is already within
1.0% of training on everything, and 14 of its 19 test metrics sit inside their own
noise floor. It is the one dataset where merging beats sequential fine-tuning outright, and
also the one where nothing can be demonstrated on the test set.

**PSM is the clean success.** Merging matches joint training, its noise floor is 0.07%, and
merge cost is indistinguishable from free.

**ETTh1 is the only dataset that genuinely drifts** — a 39% gap between base and joint
training — but it is also the noisiest (8.76% on absolute values), so it must be reported
as ratios.

---

## 11. A decision rule — merge, route, or keep fine-tuning?

### 11.0 First: can you keep the data?

Before choosing among update strategies, settle the branch that decides whether you need one.

**If history can be retained and periodic retraining is affordable — retrain.** A single
fine-tune on the unsplit post-baseline data beats every merge on ETTh1 by 9.3%, ties on the AD
pair, and loses only on exchange_rate ([EXPERIMENTS.md §1.17](EXPERIMENTS.md)). This is also
what most deployed systems do. **Merging is not more accurate; it is available under constraints
where retraining is not.**

Those constraints are real — retention limits, data volume, per-update compute, updates arriving
from separate teams that cannot pool raw data. And *"cannot keep all history"* is not *"cannot
keep any"*, so the honest baseline is a **sliding window**: retrain θ₀ on the last W periods.
That has now been measured, and it gives the sharpest statement in the project:

> **Merging is worth two to four periods of retained history, and the number tracks how much
> data each period carries.** It beats small retention budgets on every dataset tested and loses
> to larger ones. The crossover is **W = 3** on the small-shard datasets (ETTh1, exchange_rate)
> and **W = 5** on the data-rich ones (ETTh2, ETTm2).
> ([EXPERIMENTS.md §1.21](EXPERIMENTS.md), [§1.23](EXPERIMENTS.md), [§1.24](EXPERIMENTS.md))

The rule is therefore a **retention-budget** question, not a methods question:

> **Keep ≤ 2 periods → merge. Keep ≥ 5 → retrain on the window. In between, measure.**

What merging buys is *a few periods' worth of accuracy while storing no data at all* — only
weight deltas. That is a genuine and roughly-sized benefit under retention limits, and no
benefit without them.

**Whether to keep *everything* depends on how much data you have per period.** A 3-period
window beats using all history on exchange_rate (+26%) — but exchange_rate has 607-row shards.
On **ETTh2 and ETTm2**, which share its drift with 2× and 9× the data, the opposite holds:
more data is monotonically better and joint training is essentially unbeatable *on this dataset* (ETTm2: using
everything is **34% better** than a 3-period window). So *"retrain on everything is the wrong
default"* is **withdrawn as a general claim** — it holds in the small-shard regime and reverses
outside it ([EXPERIMENTS.md §1.24](EXPERIMENTS.md)). **Drift does not explain it, and neither does drift *shape*.** exchange_rate and ETTh2 have
nearly the same segment-shift statistic and behave oppositely; a trend-aware refinement was
tested and fails too — each period's distance to the test period gives near-identical profiles
on all four datasets, with the test period nearest a *middle* period every time
([EXPERIMENTS.md §1.23](EXPERIMENTS.md)).

**What does explain it is shard size.** ETTm2 holds drift fixed at exchange_rate's level and
multiplies the data ninefold, and three of exchange_rate's four distinctive behaviours vanish.
With 607-row shards each task vector is estimated from very little data — the regime where
averaging helps most and where an old period is most likely to be out-of-regime. So the rule of
thumb is about **data per period**, not about how fast the series moves.

Practically: **measure it on your own stream.** Retrain at two window lengths and compare — two
jobs, and it answers the question directly for your data, which is more than any statistic here
can do.

### 11.0a When to materialise, and how to know

Accumulation does not decay on its own — an accumulated merge still beats the base on shards no
task vector has seen, out to k = 4 ([EXPERIMENTS.md §1.19](EXPERIMENTS.md)). What changes with
drift is whether it beats the *alternative*:

| on the next unseen period | ETTh1 (mild drift) | exchange_rate (strong drift) |
|---|---|---|
| accumulate wins | k = 4, ties elsewhere | k = 3 |
| materialise wins | never | k = 1, 2, 4 |

**Materialise when drift is strong enough that recency beats accumulation on the next period.**
The measurement is free: you fine-tune a fresh specialist every period anyway, so compare it
against the accumulated merge on the newest held-out slice, and branch when the fresh model
starts winning. That is the same performance-triggered rule the dynamic-weighted-ensemble
literature converges on (§15).

### 11.0a-bis Branching the chain — what is tested, and what is not

The two routes are usually presented as a choice, but nothing forces a single flat chain. The
natural family in between is **re-anchoring**: accumulate for a while, then declare the current
model a new base and start a fresh accumulation from it.

**What this project tested: the trigger, not the structure.** §11.0a answers *when* to branch —
materialise when a freshly fine-tuned specialist starts beating the accumulated merge on the
newest held-out slice, which is free to measure. What happens *after* the branch was never run:
every experiment here is one flat chain of one type.

**Re-anchoring is well-defined, and worth writing down.** Anchor at period *m*, with
θ_A = θ₀ + (1/m)·S_m the merge of the first block. Later periods define their vectors against
the new base, τ′ᵢ = θᵢ − θ_A, and the served model is

> θ_A + α′·Σᵢ τ′ᵢ  =  θ₀ + (1/m)·S_m + α′·Σᵢ τ′ᵢ

so the scheme telescopes into a **two-level tree**: a within-block average at scale 1/m, and a
between-block term at its own scale. Nothing breaks, because every vector is still measured
against a base its siblings share — the property §4.2 needs. Storage is one extra model per
live anchor.

**Why it is more than bookkeeping.** It attacks the failure mode merging actually has. Flat
merging dilutes every contribution by 1/n, so after twenty periods each is at 1/20 — the
"shard starvation" mode of §11.3. **Re-anchoring bounds that dilution**: with blocks of five,
each vector sits at 1/5 within its block regardless of how long the stream runs. The two
routes' failure modes are dilution (merging) and forgetting (sequential); a re-anchored chain
trades between them by choosing the block length, with flat merging (m = n) and sequential
(m = 1, roughly) as the two endpoints.

**What it would take to test.** Nothing new in the framework: the pipeline already fine-tunes
from a frozen base and already accumulates, so re-anchoring is a base swap on a schedule. The
open questions are empirical — where the block length should sit, whether the trigger in
§11.0a picks it well, and whether α within a block still tracks 1/m as §6.6 predicts, which is
the sharpest test since it would extend the scale rule to a regime it was not fitted on.

**Related work makes this the live direction.** **OPCM** is continual merging — folding in one
model at a time while keeping only the orthogonal component — which is this territory
approached from the geometry side rather than the schedule side. Its prediction here is
pessimistic: the orthogonal component is what re-anchoring preserves, and §5.4 measures ρ at
0.02–0.13 on the forecasting datasets, meaning the vectors are *already* nearly orthogonal
there and projection has little to remove. **Where it should bite is SWaT** (ρ ≈ 0.61), the one
dataset where updates genuinely repeat. That is a falsifiable prediction, and the datasets to
test it on are already built.

### 11.0b Recency is not relevance

A tempting simplification is that the newest model supersedes the older ones — serve the latest,
discard the rest. The transfer matrix says otherwise: across 20 regime-columns the **matching**
specialist wins 10, and excluding the trivial last regime the **newest** specialist wins
**1 of 16** ([EXPERIMENTS.md §1.20](EXPERIMENTS.md)).

Which regime the data belongs to determines which model is best — not how recent that model is.
That is why old and new regimes are separate targets, why routing has headroom at all, and why
"just keep fine-tuning on the newest data" is not automatically the right default.

### 11.0c The four strategies — and why two of them are easy to confuse

| strategy | each period it… | stores | starts from |
|---|---|---|---|
| **Model merging** | fine-tunes a copy of θ₀ on the new period, keeps τ = θ − θ₀, serves θ₀ + α·Στ | **deltas, no data** | θ₀ |
| **Window retrain** | fine-tunes a **fresh copy of θ₀** on the last W periods pooled | **W periods of data** | **θ₀ every time** |
| **Continual fine-tuning** | keeps training the *current* model on the new period | one model, no data | **the previous model** |
| **Materialise + route** | keeps several models and picks one per input | several models | varies |

**Window retrain is a third method, not a variation of the other two**, and the distinction is
the one most easily lost:

- *Not continual fine-tuning*, because it restarts from θ₀ every time. Nothing compounds across
  updates, so there is no forgetting chain — which is exactly what makes continual fine-tuning
  degrade as steps accumulate.
- *Not merging*, because it trains on **retained raw data**. Merging never stores data, only
  weight deltas. That single difference is the whole practical trade in §11.0.

A **specialist** is a window retrain with W = 1 — the same object under a different name.

### 11.1 Why routing is usually not worth it

The transfer matrix answers this directly, and the argument is short.

Score every model on every period. For each period, the **lowest error in that column is the
best any router could ever do** — it is the router that magically picks correctly every single
time. Call it the ceiling. A real router, choosing from limited recent data and sometimes
wrong, can only fall below it.

So: how far is the merged model from that ceiling?

| dataset | drift (5-way) | merged vs the ceiling | newest specialist vs the ceiling |
|---|---|---|---|
| exchange_rate | 0.833 | **+106.9%** | +474.7% |
| ETTh2 | 0.753 | **+81.0%** | +54.8% |
| ETTm2 | 0.752 | **+66.0%** | +67.4% |
| ETTh1 | 0.412 | **+6.1%** | +36.5% |
| SWaT / PSM | — | **not measurable** | not measurable |

> ⚠️ Recomputed 2026-08-07 and **these figures replace** SWaT +6.2%, PSM +7.8%, ETTh1 +7.0%,
> exchange_rate +102.3%/+318.0%. The AD rows are withdrawn, not restated: on AD the per-regime
> columns carry only reconstruction statistics, so a per-regime oracle cannot be formed at all
> (see [EXPERIMENTS.md §1.16](EXPERIMENTS.md)).

**The prize is small only when drift is small.** On ETTh1 a router that never makes a mistake
would beat the merged model by 6%, which does not repay storing *n* models and building
selection logic that can itself pick wrong. But at drift ≈0.75 and above the same perfect router
recovers **66–107%**, and that does repay it.

The mechanism is the one you would guess: where regimes genuinely differ, a specialist trained
on period *k* is far better on period *k* than any average of specialists, so averaging costs
real performance; where periods resemble each other, the average is nearly as good as the best
individual and there is nothing left to win. **This is the one behaviour in the project that
tracks drift rather than shard size** — ETTm2 has nine times exchange_rate's data and still
shows 66%, and ETTh1 and ETTh2 differ thirteenfold at identical shard size.

One more result to carry: **merging beat "always use the newest model" on all four datasets.**
If you are keeping a single model, the merge is the right single model — better than the most
recent specialist, which is the obvious naive choice.

### 11.2 And merging is not an accuracy win over not splitting at all

Against one fine-tune on the same data *unsplit* ([EXPERIMENTS.md §1.17](EXPERIMENTS.md)): ETTh1 **loses by
9.3%**, SWaT and PSM win by 0.24% and 1.01% — clearing their very tight floors but practically
negligible — and only exchange_rate wins by a real margin (+8.4%).

So merging does not buy accuracy. **It buys the ability to operate under the streaming
constraint**, where the unsplit fine-tune is simply not available because the data cannot be
retained. That is a sound justification; it is just a different one from "merging is better".

### 11.2b The decision, as a tree

```
Can you retain >= 3 periods of raw data?
|
+-- YES -> WINDOW RETRAIN on ~3 periods.
|          Fine-tune a fresh copy of theta_0 on them each update.
|          Beats every merge (13-20%) and beats continual fine-tuning.
|          Do NOT use all history: a 3-period window beats using
|          everything by 26% on strongly drifting data.
|
+-- NO (0-2 periods) -> you need a method that stores no data:
    |
    +-- can you select alpha on validation?
        |
        +-- YES (forecasting) -> MODEL MERGING
        |     ~a two-period window's accuracy for zero stored data;
        |     resists forgetting; generalises forward; one model served.
        |
        +-- NO (unsupervised AD) -> CONTINUAL FINE-TUNING
              needs no coefficient at all, which is the blocker
              merging cannot get past there.

Routing between specialists: only when drift is strong AND you can
measure which model is best. Otherwise serve the merge (6% headroom at
low drift; 66-107% at high drift; not measurable at all on AD).

Materialise a new specialist when drift is strong enough that a fresh
model beats the accumulated merge on the newest period -- measurable
free each period, since you fine-tune one anyway.
```

**The thing to carry away: merging's competitor is not continual fine-tuning, it is your
retention policy.** Merging buys roughly a two-period window's accuracy for zero stored data.
If policy already permits three periods, none of this machinery earns its place.

### 11.3 The verdict

| | window retrain | merge, keep 1 | n specialists + router | continual fine-tuning |
|---|---|---|---|---|
| **raw data retained** | **W periods** (+0.15W val) | **none** with α = 1/n fixed; **0.9–2.5 periods** of validation if α is selected — see below | same as merging | none |
| models stored | 1 | 1 | n | 1 |
| starts each update from | θ₀ | θ₀ | θ₀ | previous model |
| needs a hyperparameter? | W | **yes — α** | yes, a selection rule | **no** |
| works on unlabelled AD? | yes | only with a pre-declared α | **no** | **yes** |
| accuracy | **best, at W ≥ 3** | ≈ a 2-period window | +6% (low drift) to +107% (high drift) over merge | trades with merge, 5/4 |
| degrades as periods accumulate by | — (restarts each time) | shard starvation | storage | forgetting |

> ⚠️ **"Stores no data" is only true if you do not tune α.** Selecting α needs the pooled
> validation union, and that is *raw data*, retained — `val_base` forever, plus every segment's
> slice, because each selection scores the whole union. Counted in periods (one period = one
> incremental segment), total retention including training data:
>
> | | merging + α selection | W = 2 window | W = 3 window | merging, α = 1/n fixed |
> |---|---|---|---|---|
> | ETTh1, n = 3 | **0.90** | 2.30 | 3.45 | **0.00** |
> | ETTh1, n = 5 | **1.50** | 2.30 | 3.45 | **0.00** |
> | exchange_rate, n = 5 | **2.50** | 2.50 | 3.75 | **0.00** |
>
> Merging still retains **less in total** than the window it is compared against — the
> comparison it wins is unaffected. But *"none"* was wrong: it is 0.9–2.5 periods of validation
> data, and it **grows with n**, since `val_base` is permanent and every new period adds a
> slice. On exchange_rate at n = 5 it has already drawn level with a two-period window.
>
> **Which is what makes α\*·n ≈ 1 operationally load-bearing** rather than merely interesting.
> Fixing α = 1/n in advance needs no validation data at all, and the measured cost against the
> *test-optimal* α — the harshest possible comparator — is:
>
> | | n = 2 | n = 3 | n = 5 | dataset floor |
> |---|---|---|---|---|
> | ETTh1 | 0.6% | 2.7% | 4.9% | 8.76% — **all inside the floor** |
> | exchange_rate | 11.2% | 9.0% | 12.0% | 5.73% — real, but ~10% |
>
> On ETTh1 pre-declaring 1/n is free; on exchange_rate it costs about 10%. **That, not the
> selected-α version, is the variant whose storage claim is genuinely zero** — and every merging
> number elsewhere in this document is at a *selected* α. Running the fixed-α variant properly
> is EXECUTION_PLAN.md §3.9.

**If you can retain ≥ 3 periods: window retrain.** It beat every merge by 13–20% on both
datasets with headroom, needs no coefficient, and cannot accumulate forgetting because it
restarts from θ₀. Its only cost is the retained data. Everything below assumes you cannot pay
that cost.

**Otherwise, default: merge and keep one model.** It is within ~7% of an unreachable ceiling, beats the
newest specialist everywhere, costs ~1.05× a dedicated specialist, and stores one model. Its
one requirement is α — and α is not a free parameter you must search blindly, because α\*·n
lands at order 1 on five of six datasets (§6.6). Start from the mean of the task vectors.
Note this is an empirical regularity across six datasets, **not** a constant of the dataset:
§6.6 shows the product drifts within a dataset as n varies, and no design can isolate the
count, so treat α\*·n ≈ 1 as a starting point rather than a formula to trust.

**Route only under strong drift**, and only if you can measure which model is best. Routing is
the one behaviour in this project that **does** track drift rather than shard size, measured on
all four forecasting datasets ([EXPERIMENTS.md §1.16](EXPERIMENTS.md)):

| dataset | drift (5-way) | shard @ n=5 | a perfect router would recover |
|---|---|---|---|
| exchange_rate | 0.833 | 607 | **+106.9%** |
| ETTh2 | 0.753 | 1,393 | **+81.0%** |
| ETTm2 | 0.752 | 5,574 | **+66.0%** |
| ETTh1 | 0.412 | 1,393 | **+6.1%** |

The ordering is drift's, not size's — ETTm2 has nine times exchange_rate's data and still shows
66%, while ETTh1, at the *same* shard size as ETTh2, shows thirteen times less headroom. The
mechanism is the obvious one: when periods genuinely differ, the right specialist for a period
is far better than any average of all of them; when they resemble each other, the average is
nearly as good as the best individual. The signal that routing has something to offer is the
same one that makes merging beat joint training — regimes that genuinely differ.

⚠️ An earlier version of this section attributed the headroom to thin shards, generalising the
scarcity result of §11.0. ETTm2 shows that was wrong: **scarcity explains retention behaviour,
drift explains routing.** Two behaviours, two drivers.

**Continual fine-tuning is the right default when you cannot choose α** — most importantly
unsupervised anomaly detection, where §8.4 shows no unlabelled signal tracks detection quality.
It needs no coefficient at all. Its failure mode is forgetting, and that failure grows with the
number of steps: on exchange_rate its error goes 0.220 → 0.362 → 0.531 as periods accumulate,
which is why merging overtakes it at five periods there and not at two.

**Between merge and continual, there is no universal winner.** Nine decisive configurations
split 5 / 4, and the outcome *flips with the number of periods on the same dataset* — so it is
not a property of the data. Choose by which failure mode you can tolerate: merging starves as
shards shrink, continual forgets as steps accumulate.

### 11.4 What changes if AD has a labelled calibration set

Labels dissolve the blocker: with them you can measure detection directly, so α becomes
selectable and a router becomes buildable. ⚠️ **An earlier version of this section claimed the
routing headroom "does not change", citing SWaT and PSM at 6.2% and 7.8% from the oracle
router. Those two figures are withdrawn** — §1.16 shows a per-regime oracle cannot be formed on
AD at all, because the per-regime columns carry only reconstruction statistics and no detection
metric. **With labels the headroom becomes measurable for the first time**, and its size is
genuinely unknown: on forecasting it ranges from 6% at low drift to 107% at high drift, and
both AD datasets are saturated (base within 1.1% and 3.4% of joint training), which argues for
little but does not measure it.

So a labelled calibration set is worth having for **α selection**, which is worth 96–98% on
SWaT and 25–44% on PSM of the
achievable GRR on AD (§8.4) — a much larger prize than routing. Whether routing pays off on AD
is untested on drifting AD data, and both current AD datasets are the wrong place to look. The
prediction, from §11.1, is that AD routing pays off exactly when the regimes differ enough for
specialists to separate — which is what a drifting AD benchmark would be for.

---

## 12. What is established and what is not

### Established

- **Fine-tuning learns something real and segment-specific.** Diagonal ratios of 0.61–0.75 and
  positive `specialisation` on all three rule out redundancy and degenerate fine-tuning as the
  global explanation.
- **The vectors are substantially aligned**, and alignment decays with temporal distance.
- **Merging is essentially free at the right scale** — merge cost ~1.0–1.1× on SWaT, PSM and
  ETTh1, and **flat as the segment count grows** (n = 2, 3, 5).
- **What looked like interference at α = 1.0 was overshoot.** Residual interference is small
  and concentrated on the newest segment.
- ~~α\* is stable across seeds, exactly~~ — **withdrawn**, that was 0.1-grid quantisation; at
  0.05 resolution the seeds differ by one step.
- **α\*·n is near-constant**, so the optimal merge is roughly a fixed multiple of the *mean*
  task vector
  (§6.6) — but see §6.6 itself: the count can never be isolated on a fixed series, so this is
  an empirical regularity in one parameterisation, not a law.
- **Validation cannot select α on AD.** The val and test optima point to different values, and
  choosing on validation costs 96–98% on SWaT / 25–44% on PSM against 1–8% on forecasting
  (§8.4).
- **The merge is bitwise reproducible** — all 87 merged checkpoints recompute exactly.

### Not established — do not claim these

- **That merging protects the base regime on all datasets.** With error bars it does so
  only on SWaT; elsewhere the intervals overlap.
- **That geometry — or anything else — predicts the outcome.** Tested at nine decisive
  configurations: no signal separates them beyond chance (EXPERIMENTS.md §1.14).
- **That merge-vs-sequential is a property of a dataset.** It flips with the segment count.
- **That GRR degrades as segments accumulate.** An intermediate reading said so; under uniform
  α it holds only on PSM. Withdrawn.
- **That redundancy or headroom is *the* driver of forgetting.** Both explain all three
  datasets equally well.
- **Anything from SWaT's test column.** Most of its metrics are inside their own noise floor,
  and its base-to-joint gap of 0.0095 AUROC makes its GRR ill-conditioned.
- **Anything about weight disentanglement from cosine numbers** — different quantity.

### The one-sentence version

> The fine-tunes each learn something genuine and locally useful; because they largely learn
> the *same* thing, summing them at full strength overshoots — but that is a scale error, and
> once corrected, one merged model comes within a few percent of keeping a separate specialist
> per regime. Whether that beats simply continuing to train depends on how much the updates
> repeat each other.

---

## 13. Open questions

1. **Does the merged model handle a regime nobody fine-tuned on**, better than the newest
   specialist? This is the case merging is actually for, and it is unmeasured.
2. **Is the residual interference real or inside the noise?** Needs per-metric floors applied
   to the val block.
3. **Does α\* decrease as the number of segments grows?** Direct evidence on accumulating
   interference.
4. **Is the drift-vs-partition-size confound resolved?** Requires the random-split control.
5. **Can a cheap geometric signal decide merge-vs-materialise?** ρ and BECAME's λ\* are the
   candidates.
6. **Can redundancy and headroom be separated?** Needs a dataset with both.

---

## 14. Quick reference

### Formulas in one place

| quantity | formula |
|---|---|
| task vector | τᵢ = θᵢ − θ₀ |
| merge | θ_merged = θ₀ + α·Στᵢ = θ₀ + (α·n)·mean(τ) |
| subspace overlap | ρ_k = ‖**P**τ_k‖² / ‖τ_k‖², **P** = projection onto span(τ₀…τ_{k−1}) |
| new component | new_k = ‖τ_k‖·√(1 − ρ_k) / ‖θ₀‖ |
| alignment | ‖Στᵢ‖ / Σ‖τᵢ‖ — equals 1 when parallel, √(Σ‖τᵢ‖²)/Σ‖τᵢ‖ when orthogonal |
| effective rank | exp(H(p)), pᵢ = σᵢ²/Σσⱼ² over singular values of the stacked τ's |
| principal angles | cos θᵢ = singular values of Uᵀ V for orthonormal bases U, V |
| ratio to base | error(model, slice) / error(θ₀, slice) |
| merge cost | error(merged, val_i) / error(θ₀+τᵢ, val_i), averaged over i |
| GRR | (merged − base) / (joint − base) on the test metric |
| specialisation | mean off-diagonal − mean diagonal of the transfer matrix |
| optimal scale (model) | α\* = 1/n under τᵢ = d + εᵢ; see §6.6a |


### Symbols

| symbol | meaning |
|---|---|
| θ₀ | base model, trained on the early regime |
| τᵢ = θᵢ − θ₀ | task vector for segment *i* |
| α, α\* | merge scale; α\* is the validation-selected one |
| ρ | subspace overlap — does a new vector add anything? |
| `val_i` / `val_base` | segment *i*'s held-out slice / the early regime's |

### Reading a cell

`value / base_value_on_the_same_column`. **Lower is better; 1.00 = indistinguishable from base.**

### Reading the matrix

| look at | tells you |
|---|---|
| diagonal vs 1.00 | did the specialist learn anything at home? |
| off-diagonal vs 1.00 | does it transfer, or damage, elsewhere? |
| off-diagonal vs diagonal | was the learning segment-specific? (`specialisation`) |
| **merged vs diagonal, at α\*** | **entanglement / merge cost** |
| merged vs 1.00 | net practical benefit |
| merged on `val_base` | the merge's own forgetting |

### Commands

```bash
# geometry — CPU, seconds. Split per experiment or the login node OOMs.
python -m incremental_ad.analysis.geometry_report $RUNS_ROOT/<experiment>/* \
    --out $RUNS_ROOT/analysis/geometry_<name>

# diagnostics incl. the val-vs-alpha curve — GPU, via SLURM.
# env prefix, NOT --export: an explicit --export gets the job CANCELLED by root.
SOURCE_RUN=$W/runs/<exp>/<id> STANDARD_RUN=$W/runs/<std>/<id> \
    MERGE_SCALES="0.0 0.25 0.5 0.75 1.0 1.25 1.5" \
    EXTRA_ARGS="--pipeline_curve_include_val" \
    sbatch scripts/sbatch_merge_diagnostics.sh

# the sequential baseline
python -m incremental_ad.main --pipeline ContinualFineTuningPipeline ...
```

### Code map

| what | where |
|---|---|
| `task_vector`, `apply_task_vectors`, `merge_task_arithmetic` | [framework/merging/task_vectors.py](src/incremental_ad/framework/merging/task_vectors.py) |
| cosine, norms, effective rank, ρ, cosine-vs-distance | [framework/merging/geometry.py](src/incremental_ad/framework/merging/geometry.py) |
| the transfer matrix, α curve, summary scalars | [framework/pipelines/merge_diagnostics_pipeline.py](src/incremental_ad/framework/pipelines/merge_diagnostics_pipeline.py) |
| the sequential baseline, ACC/BWT | [framework/pipelines/continual_finetuning_pipeline.py](src/incremental_ad/framework/pipelines/continual_finetuning_pipeline.py) |
| what a val cell contains | [framework/evaluators/ad_val_evaluator.py](src/incremental_ad/framework/evaluators/ad_val_evaluator.py) |
| oracle vs percentile thresholding | [framework/evaluators/ad_test_evaluator.py](src/incremental_ad/framework/evaluators/ad_test_evaluator.py) |
| CLI entry points | [analysis/geometry_report.py](src/incremental_ad/analysis/geometry_report.py), [analysis/diagnose.py](src/incremental_ad/analysis/diagnose.py) |

---

## 15. Related work

> **Verified against sources 2026-08-08.** Every author, title, venue and year below was checked;
> full titles are given so a reader can find each paper. **One error was found and fixed:** L2-SP
> was cited as "Xuhong et al." — *Xuhong* is the author's given name, and the correct short form
> is **Li et al.** Three references that were named without a citation (DARE, OPCM, BECAME) now
> carry one. Page-Hinkley, AUE and VUS-ROC/PR are mentioned by name only, without author or year,
> and so were not verifiable as citations.

**Model merging / task arithmetic.** *Editing Models with Task Arithmetic* (Ilharco, Ribeiro,
Wortsman, Gururangan, Schmidt, Hajishirzi & Farhadi, **ICLR 2023**, arXiv:2212.04089) introduces
τ = θ_ft − θ_base and the scaled sum this project uses. *Model soups: averaging weights of
multiple fine-tuned models improves accuracy without increasing inference time* (Wortsman et al.,
**ICML 2022**, PMLR 162:23965–23998) averages fine-tuned checkpoints. *TIES-Merging: Resolving
Interference When Merging Models* (Yadav, Tam, Choshen, Raffel & Bansal, **NeurIPS 2023**,
arXiv:2306.01708) resolves sign conflicts and parameter interference between task vectors;
*DARE*, from *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a
Free Lunch* (Yu, Yu, Yu, Huang & Li, **ICML 2024**, arXiv:2311.03099), randomly **d**rops delta
parameters at rate p **a**nd **re**scales the rest by 1/(1−p). *Merging Models with
Fisher-Weighted Averaging* (Matena & Raffel, **NeurIPS 2022**) weights by parameter importance.
**OPCM** — *Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable
Continual Model Merging* (Tang et al., **2025**, arXiv:2501.09522) — targets *continual* merging,
folding in one model at a time at constant memory by projecting each update onto the orthogonal
complement of the span of previous ones. **BECAME** — *BayEsian Continual Learning with Adaptive
Model MErging* (Li et al., **ICML 2025**, arXiv:2504.02666) — derives a **closed-form optimal
merging coefficient** rather than tuning it, which is the property that matters most for the
unsupervised-AD case (§8.4).

*What is different here:* that literature merges **near-orthogonal** tasks (different image
classes). Time-series shards are aligned, which is what changes the scale rule (§2, §6.6).

**Concept drift and streaming adaptation.** The *"when do I rebuild"* question is classically a
drift-detection problem: ADWIN (*Learning from Time-Changing Data with Adaptive Windowing*,
Bifet & Gavaldà, **SDM 2007**, pp. 443–448), DDM (*Learning with Drift Detection*, Gama, Medas,
Castillo & Rodrigues, **SBIA 2004**, LNCS 3171), Page-Hinkley. Gama, Žliobaitė, Bifet,
Pechenizkiy & Bouchachia's *A survey on concept drift adaptation* (**ACM Computing Surveys**
46(4), 2014) is the standard entry point. The **keep-n-experts-and-route** design is a dynamic
weighted ensemble: *Dynamic Weighted Majority: An Ensemble Method for Drifting Concepts* (Kolter
& Maloof, **JMLR** 8:2755–2790, 2007), Learn++.NSE (*Incremental Learning of Concept Drift in
Nonstationary Environments*, Elwell & Polikar, **IEEE Trans. Neural Networks** 22(10):1517–1531,
2011), AUE.
Those methods add and prune experts on measured performance — which is the same trigger this
project arrives at for materialisation (§11).

**Continual learning.** EWC (*Overcoming catastrophic forgetting in neural networks*,
Kirkpatrick et al., **PNAS** 2017) and L2-SP (*Explicit Inductive Bias for Transfer Learning with
Convolutional Networks*, **Li**, Grandvalet & Davoine, **ICML 2018**) constrain drift from a
reference; L2-SP is the one tested here and found neutral (§6.5).
Rehearsal/replay methods assume retained data, which is exactly the constraint that motivates
merging.

**Anomaly-detection evaluation.** *Towards a Rigorous Evaluation of Time-series Anomaly
Detection* (**S. Kim**, Choi, Choi, Lee & Yoon, **AAAI 2022**, arXiv:2109.05257) shows
point-adjusted F1 is so permissive that a random score reaches state of the art; replacements
include PA%K (same paper), range-based precision/recall (*Precision and Recall for Time Series*,
Tatbul, Lee, Zdonik, Alam & Gottschlich, **NeurIPS 2018**, arXiv:1803.03639), affiliation metrics
(*Local Evaluation of Time Series Anomaly Detection Algorithms*, Huet, Navarro & Rossi,
**KDD 2022**, arXiv:2206.13167) and VUS-ROC/PR. This
project follows their recommendation — threshold-free metrics primary, PA-F1 for legacy
comparability only (EXPERIMENTS.md §1.5).
