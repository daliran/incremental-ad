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
Σ‖τᵢ‖ and α ≈ 1 would be sensible — which is why the image-classification literature can often
get away with it. When the vectors are aligned, the sum is nearly Σ‖τᵢ‖ long, and α = 1
overshoots by roughly the number of vectors.

That is exactly what the measurements show: **α\*·n is constant** (§6.6). Averaging rather than
adding is the whole correction. And it holds at every shard count *because* alignment itself
falls as shards get finer — on ETTh1, 0.824 at n = 2 down to 0.633 at n = 5 — while the vectors
shrink in step, so ‖Στᵢ‖ stays nearly constant.

Everything previously written up as *interference* and *forgetting* on these datasets was this
scale error. The residual interference at α\* is small.

### The practical consequence

For time series you cannot inherit α = 1 from the image-classification setting, and you cannot
assume orthogonality buys you free composition. What you can do is simpler: **average the task
vectors**, and expect a dataset-level constant of order 1 in front of the mean (1.0 on SWaT, PSM
and ETTh1; ≈1.5 on exchange_rate, the most strongly drifting).

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

That is the reasoning behind an *accumulate versus materialise* rule.

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

### Why the failure is informative

exchange_rate is also the dataset with the **largest headroom** (43.8% base-to-joint) and
the **strongest input drift**, and it is the only one where the merge beats *joint training*
outright (GRR 1.164 ±0.099 at n = 3 with α chosen on validation, and above 1.0 at every
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
(EXECUTION_PLAN.md §4.5).

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

**Non-orthogonality still matters, but its consequence changes.** It is precisely *because*
the vectors agree that summing overshoots. Non-orthogonality does not make merging fail — it
dictates a smaller α.

> **Withdrawn:** an earlier version of this section claimed the ordering holds — *"the most
> collinear dataset needs the smallest scale, the least collinear the largest"* — and named PSM
> as the least collinear. It is not; ETTh1 is (ρ = 0.076 against PSM's 0.226), and ETTh1's α\*
> is *smaller* than PSM's, not larger. Across four datasets the ranking by ρ
> (SWaT, PSM, exchange_rate, ETTh1) does not match the ranking by α\*
> (SWaT, ETTh1, exchange_rate, PSM). **ρ sets the direction — more agreement means a smaller
> scale is needed — but it does not order the datasets.** What does hold is the within-dataset
> law α\*·n = constant (§6.6).

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
asks about — so it was tested cleanly on ETTh1, three seeds per λ, α selected per arm
(EXPERIMENTS.md §3.0b). **No measurable effect at λ ∈ {1e-3, 1e-2}**: the difference is ~3%
against a ~5% within-arm spread, and its *sign flips* depending on whether you read absolute
error, ratio-to-own-baseline, or hardware-matched pairs. That instability is the point — the
effect is smaller than the noise. Constraining magnitude is not what merging needs here,
which is consistent with §6.3: the problem was never that the vectors were too long, it was
that they were being *summed* rather than averaged.

---

### 6.6 The optimal merge is the *mean* task vector

The remedy table lists α = 1/n as a sane prior. It is better than that: it is what the data
says, and it is the cleanest quantitative result in the project.

Sweeping the number of segments n ∈ {2, 3, 5} on four datasets and selecting α on validation
at each, the product **α\*·n is constant within a dataset**:

| dataset | α\* at n=2 | n=3 | n=5 | α\*·n |
|---|---|---|---|---|
| SWaT | 0.50 | 0.25 | 0.25 | 1.00 / 0.75 / 1.25 |
| PSM | 0.50 | 0.38 | 0.25 | 1.00 / 1.12 / 1.25 |
| ETTh1 | 0.50 | 0.30 | 0.20 | 1.00 / 0.90 / 1.00 |
| exchange_rate | 0.70 | 0.53 | 0.30 | 1.40 / 1.60 / 1.50 |

Why that is a statement about the *mean*: the merge is θ₀ + α·Στᵢ. Write α = k/n and it
becomes θ₀ + k·(Στᵢ)/n = θ₀ + k·**mean**(τ). A constant α·n = k says the best merge always
steps a fixed multiple k of the *average* task vector — k ≈ 1 on three datasets, ≈1.5 on
exchange_rate — no matter how many vectors are being averaged.

**Why it is not just "the vectors got smaller".** Individual task vectors *do* shrink as
segments shrink (ETTh1 mean ‖τᵢ‖: 1.85 → 1.53 → 1.10). If α\* were compensating for
magnitude it would have to **grow** as ‖τ‖ falls. It falls instead, as 1/n. Meanwhile ‖Στᵢ‖
stays nearly constant (3.05 → 3.39 → 3.49), because the vectors get smaller *and* less
mutually aligned (0.824 → 0.737 → 0.633) and the two effects cancel. So α\* tracks the
**count**, not the size.

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
of which regime a task set is in**, reachable without computing any geometry. That is a
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

Choosing α on validation therefore costs **22–99% of the achievable GRR on AD**, against
**1–8% on forecasting** (EXPERIMENTS.md §1.12). And AUROC is not insensitive to α — it moves
6× the noise floor on SWaT and 85× on PSM. It moves in a direction validation cannot see.

Note what the SWaT row means concretely: at α = 1.5 the merged model reconstructs 2.5× worse
than at its validation optimum, and detects *better*. **A model that reconstructs badly can
separate anomalies well.** Reconstruction level is simply not the quantity detection depends
on — the score *distribution* is.

That last observation is the one opening left. Mean reconstruction error throws away exactly
the information detection uses. A spread statistic on the validation scores — p99/p50, std,
kurtosis — might have its optimum where AUROC does, and those columns are already recorded in
every curve, so it is analysis rather than new training (EXECUTION_PLAN.md §4.2). If none of
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
| seed noise floor | 0.13% | 0.07% | 8.20% |
| vs sequential | **merge wins both** | sequential on new, tie on old | sequential on new, tie on old |

**SWaT is a control, not evidence.** A frozen base on half the data is already within
1.0% of training on everything, and 14 of its 19 test metrics sit inside their own
noise floor. It is the one dataset where merging beats sequential fine-tuning outright, and
also the one where nothing can be demonstrated on the test set.

**PSM is the clean success.** Merging matches joint training, its noise floor is 0.07%, and
merge cost is indistinguishable from free.

**ETTh1 is the only dataset that genuinely drifts** — a 39% gap between base and joint
training — but it is also the noisiest (8.20% on absolute values), so it must be reported
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
from separate teams that cannot pool raw data. But note the middle ground: *"cannot keep all
history"* is not *"cannot keep any"*. The honest baseline is a **sliding window** — retrain on
the last W periods — and how merging compares against a realistic W is, as of this writing,
**not yet measured** (EXECUTION_PLAN.md §3.1a). Treat the strategies below as the answer for
the no-retention case, and as provisional for the partial-retention case.

### 11.0b Recency is not relevance

A tempting simplification is that the newest model supersedes the older ones — serve the latest,
discard the rest. The transfer matrix says otherwise: across 20 regime-columns the **matching**
specialist wins 10, and excluding the trivial last regime the **newest** specialist wins
**1 of 16** ([EXPERIMENTS.md §1.20](EXPERIMENTS.md)).

Which regime the data belongs to determines which model is best — not how recent that model is.
That is why old and new regimes are separate targets, why routing has headroom at all, and why
"just keep fine-tuning on the newest data" is not automatically the right default.

### 11.0c The three strategies

Three strategies are on the table once data arrives over time.

- **Merge and keep one model.** Fine-tune a copy per period, keep only what changed, add the
  deltas back onto the base. One model on disk.
- **Keep n specialists and route.** Store every per-period model and pick the right one for
  each incoming batch.
- **Continual fine-tuning.** Never branch: keep training the same model as data arrives. One
  model on disk.

### 11.1 Why routing is usually not worth it

The transfer matrix answers this directly, and the argument is short.

Score every model on every period. For each period, the **lowest error in that column is the
best any router could ever do** — it is the router that magically picks correctly every single
time. Call it the ceiling. A real router, choosing from limited recent data and sometimes
wrong, can only fall below it.

So: how far is the merged model from that ceiling?

| dataset | merged vs the ceiling | newest specialist vs the ceiling |
|---|---|---|
| SWaT | **+6.2%** | +11.9% |
| PSM | **+7.8%** | +15.7% |
| ETTh1 | **+7.0%** | +37.0% |
| exchange_rate | **+102.3%** | +318.0% |

On three of four datasets, **a router that never makes a mistake would beat the merged model by
about 7%.** That is the entire prize. Against it you must store *n* models instead of one, and
build selection logic that can itself pick wrong — spending the 7% you were trying to win. The
answer there is: **keep the merge.**

exchange_rate is the exception, and the reason is instructive. Its periods genuinely differ, so
a specialist trained on period *k* is roughly twice as good on period *k* as any average of
specialists. Where regimes are distinct, averaging them costs real performance and routing has
something to recover. Where periods resemble each other, the average is nearly as good as the
best individual and there is nothing left to win.

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

### 11.3 The verdict

| | merge, keep 1 | n specialists + router | continual fine-tuning |
|---|---|---|---|
| models stored | 1 | n | 1 |
| cost vs a per-period specialist | ~1.05× | 1.00× by definition | — |
| gap to the best possible router | +6–8%, or +102% under strong drift | 0 (unreachable) | worse than merge on 3 of 4 |
| needs a hyperparameter? | **yes — α** | yes, a selection rule | **no** |
| works on unlabelled AD? | only with a pre-declared α | **no** | **yes** |
| degrades as periods accumulate by | shard starvation | storage | forgetting |

**Default: merge and keep one model.** It is within ~7% of an unreachable ceiling, beats the
newest specialist everywhere, costs ~1.05× a dedicated specialist, and stores one model. Its
one requirement is α — and α is not a free parameter you must tune, because α\*·n ≈ a
dataset constant of order 1 (§6.6). Start from the mean of the task vectors.

**Route only under strong drift**, and only if you can measure which model is best. The signal
that routing has something to offer is the same one that makes merging beat joint training:
regimes that genuinely differ. On exchange_rate that is +102%, which is worth the storage.

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
selectable and a router becomes buildable. But **the headroom does not change** — SWaT and PSM
sit 6.2% and 7.8% from the ceiling, so routing would still be recovering very little on *these*
datasets, and both are saturated to begin with (base within 1.1% and 3.4% of joint training).

So a labelled calibration set is worth having for **α selection**, which is worth 22–99% of the
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
- **α\*·n is constant**, so the optimal merge is a fixed multiple of the *mean* task vector
  (§6.6). Subject to the count-vs-size control in EXECUTION_PLAN.md §4.1.
- **Validation cannot select α on AD.** The val and test optima point to different values, and
  choosing on validation costs 22–99% of achievable GRR there against 1–8% on forecasting
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

> **Recorded from prior knowledge, not fetched — verify every one against the source before it
> goes into the thesis.** Titles, venues and years are the parts most likely to be wrong; the
> ideas attributed to them are what matters for positioning this work.

**Model merging / task arithmetic.** *Editing Models with Task Arithmetic* (Ilharco et al.,
ICLR 2023) introduces τ = θ_ft − θ_base and the scaled sum this project uses. *Model Soups*
(Wortsman et al., ICML 2022) averages fine-tuned checkpoints. *TIES-Merging* (Yadav et al.,
NeurIPS 2023) resolves sign conflicts and parameter interference between task vectors; *DARE*
(Yu et al., 2024) drops and rescales delta parameters. *Fisher-Weighted Averaging* (Matena &
Raffel, NeurIPS 2022) weights by parameter importance. **OPCM** targets *continual* merging —
folding in one model at a time without storing every vector — and **BECAME** derives the
merging coefficient rather than tuning it, which is the property that matters most for the
unsupervised-AD case (§8.4).

*What is different here:* that literature merges **near-orthogonal** tasks (different image
classes). Time-series shards are aligned, which is what changes the scale rule (§2, §6.6).

**Concept drift and streaming adaptation.** The *"when do I rebuild"* question is classically a
drift-detection problem: ADWIN (Bifet & Gavaldà, 2007), DDM (Gama et al., 2004), Page-Hinkley.
Gama et al.'s *survey on concept drift adaptation* (ACM Computing Surveys, 2014) is the standard
entry point. The **keep-n-experts-and-route** design is a dynamic weighted ensemble: Dynamic
Weighted Majority (Kolter & Maloof, JMLR 2007), Learn++.NSE (Elwell & Polikar, 2011), AUE.
Those methods add and prune experts on measured performance — which is the same trigger this
project arrives at for materialisation (§11).

**Continual learning.** EWC (Kirkpatrick et al., PNAS 2017) and L2-SP (Xuhong et al., ICML 2018)
constrain drift from a reference; L2-SP is the one tested here and found neutral (§6.5).
Rehearsal/replay methods assume retained data, which is exactly the constraint that motivates
merging.

**Anomaly-detection evaluation.** *Towards a Rigorous Evaluation of Time-series Anomaly
Detection* (Kim et al., AAAI 2022) shows point-adjusted F1 is so permissive that a random score
reaches state of the art; replacements include PA%K (same paper), range-based precision/recall
(Tatbul et al., NeurIPS 2018), affiliation metrics (Huet et al., KDD 2022) and VUS-ROC/PR. This
project follows their recommendation — threshold-free metrics primary, PA-F1 for legacy
comparability only (EXPERIMENTS.md §1.5).
