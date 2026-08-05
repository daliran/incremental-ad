# Task Arithmetic for Incremental Learning — theory, reasoning, and findings

The *why* behind this project. Self-contained: readable with no prior context, and written to
be pasted into an LLM as background for follow-up questions.

- [EXPERIMENTS.md](EXPERIMENTS.md) — the numbers. If this file disagrees, that one wins.
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) — what is done and what is next.
- [PHASE1_RUNBOOK.md](PHASE1_RUNBOOK.md) — how to run the diagnostics.
- [CLAUDE.md](CLAUDE.md) — code map and repo invariants.

Every figure here is the current state at **three training seeds per dataset** (2026-08-05).
`±` is the half-range over seeds; a difference is only claimed where two intervals do not
overlap.

---

## Contents

1. [The setup](#1-the-setup)
2. [Three ways it can fail, and why they look identical](#2-three-ways-it-can-fail-and-why-they-look-identical)
3. [The transfer matrix](#3-the-transfer-matrix)
4. [Geometry](#4-geometry)
5. [Overshoot and interference](#5-overshoot-and-interference)
6. [Entanglement](#6-entanglement)
7. [Metrics that cannot see what you are measuring](#7-metrics-that-cannot-see-what-you-are-measuring)
8. [Versus sequential fine-tuning](#8-versus-sequential-fine-tuning)
9. [What the three datasets said](#9-what-the-three-datasets-said)
10. [A decision rule](#10-a-decision-rule)
11. [What is established and what is not](#11-what-is-established-and-what-is-not)
12. [Open questions](#12-open-questions)
13. [Quick reference](#13-quick-reference)

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

## 2. Three ways it can fail, and why they look identical

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
simply wrong. On these datasets that turned out to be the dominant one — see §5.)*

---

## 3. The transfer matrix

### 3.1 The missing measurement

> **θ₀ + τᵢ, evaluated on segment j, for j ≠ i.**

Say τ₁ improves segment 1. Two readings: τ₁ learned something *specific to segment 1*, or τ₁
just made the model better at everything. You cannot separate them without testing τ₁ on
segments it never saw — and a training run structurally cannot produce that number, because it
only ever scores each specialist on its own segment. **That off-diagonal is the entire point.**

### 3.2 Construction

Training-free: reload checkpoints that already exist and re-score them.

- **Rows**: `base`, `ft_0 … ft_{n-1}` (meaning θ₀+τᵢ), `merged`, and optionally a
  jointly-trained `standard`.
- **Columns**: each segment's held-out validation slice, the baseline's own slice
  (`val_base`), and the full test set.
- **Cells**: that model's error on that data ÷ the base model's error on the same data.

Checkpoints are `best.pt` — the early-stopping best-validation checkpoint. `merged` is
**recomputed** from base + fine-tunes rather than loaded; it is bit-identical to what the run
wrote, verified across all 60 runs on disk.

**Validation cells are loss-shaped, not detection-shaped** — reconstruction error for AD
(training data carries no labels, so AUROC is undefined there), MSE for forecasting. Lower is
better; 1.00 means indistinguishable from base.

### 3.3 How to read it

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
doesn't win its own column, it didn't specialise — *subject to the caveat in §3.5*.

**(b) Is the diagonal better than the off-diagonal on average?** That difference is
`specialisation`; positive means home beats away.

**(c) Is `merged` worse than each specialist on that specialist's segment, while still beating
base?** That combination is the interference signature — **but only when measured at α\***,
never at α = 1 (§5).

**(d) What happened to `val_base`?** Nobody fine-tuned on the early regime, so this is the
forgetting column.

### 3.4 The ideal matrix

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

### 3.5 Two biases in the matrix, pointing opposite ways

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

### 3.6 Merge cost

The price of one model instead of *n*: the merged model's error on a segment ÷ the error of
that segment's own specialist. **1.00× means merging is free.**

| | @ α = 1.0 | **@ α\*** |
|---|---|---|
| SWaT | 3.79 ±0.29× | **1.079 ±0.002×** |
| PSM | 1.18 ±0.11× | **1.008 ±0.020×** |
| ETTh1 | 1.69 ±0.32× | **1.007 ±0.026×** |

On PSM and ETTh1 the interval contains 1.00 — **merging is free**.

---

## 4. Geometry

Measured from the weights alone: no GPU, no dataset, seconds.

### 4.1 Three separate quantities

**Direction — cosine similarity.** The angle between τᵢ and τⱼ, magnitude divided out.

**Magnitude — norms.** ‖τᵢ‖/‖θ₀‖: how far fine-tuning moved the model relative to its own size.

**Both — effective rank.** Stack the τ's as rows, take singular values σᵢ, normalise
pᵢ = σᵢ²/Σσ², and take exp of the entropy of p. "How many independent directions do these
arrows really occupy?"

Plus **cosine vs temporal distance**: if similarity decays as segments get further apart, time
is what differentiates the vectors — the check that could have invalidated the whole framing.

### 4.2 Pairwise cosine is the wrong statistic; subspace overlap is the right one

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

### 4.3 What geometry is for

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
   weight disentanglement (§6.3).

---

## 5. Overshoot and interference

### 5.1 Aligned, orthogonal, anti-aligned

**Orthogonal** is the dream: each arrow occupies its own direction, ‖Στ‖ = √(Σ‖τᵢ‖²), and
adding costs nothing. **Aligned arrows stack**: as cosine → 1, ‖Στ‖ → Σ‖τᵢ‖. **Anti-aligned
arrows cancel** and you lose both.

Where the real vectors sit on SWaT: ‖Στ‖ = 1.854 against 1.192 if orthogonal and 2.023 if
perfectly aligned — **92% of the fully-aligned bound**. Almost nothing cancels.

### 5.2 The distinction that matters

> total merge cost = **overshoot** (curable by scaling α) + **irreducible interference**
> (what remains at α\*)

The usual story about interference is *conflict* — vectors pulling apart, the sum landing
somewhere useless. What happens here is the opposite: the vectors **agree**, so summing them
**overshoots**. Like three people each saying "turn it up a bit" and turning it up by the sum.

Two different failures of the same assumption, with very different implications: **overshoot
is a magnitude error and a scalar fixes it; conflict is a direction error and no scaling
saves you.**

### 5.3 Measured: it is almost all overshoot

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
dictates a smaller α. The ordering holds: the most collinear dataset needs the smallest scale
(SWaT 0.25), the least collinear the largest (PSM 0.5).

### 5.4 Remedies

| technique | fixes | relevant here? |
|---|---|---|
| **Global α** | pure magnitude | yes, and proven — this is the whole finding |
| **α = 1/n** (average, not sum) | magnitude, natural default when aligned | a sane prior |
| **Per-vector normalisation** | *imbalance* between vectors | when norms differ a lot |
| **Norm-matching the sum** analytically | magnitude | gives roughly the right range but **does not** predict the measured optimum; sweep and measure |
| **Strip the shared component** — apply it once, keep residuals at full strength | alignment specifically | aimed at SWaT's actual pathology |
| **TIES / DARE** | conflict, redundancy | targets conflict more than overshoot |

### 5.5 Making specialists more ideal

- **Split at change-points, not equal time slices.** Equal chunks of near-stationary data are
  near-identical distributions, so *n* fine-tunes learn one thing *n* times. This attacks the
  root cause.
- **Constrain the vectors during training** — penalise alignment with previously-learned τ's.
- **Restrict where each arrow lives** — per-segment adapters, or freeze shared layers.
- **Sparsify the deltas** — sparse high-dimensional vectors overlap far less.
- **Equalise the training budget** so magnitudes are comparable.

L2-SP is *not* this: it shrinks all arrows toward zero rather than making them different.

---

## 6. Entanglement

### 6.1 The definition

A model is **weight-disentangled** with respect to a set of task vectors when, on inputs
belonging to task *i*, the model with *all* vectors applied behaves the same as the model with
*only* τᵢ applied. Adding the others doesn't disturb what τᵢ does on its own turf.

### 6.2 It is a comparison, not a cell value

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

### 6.3 Why cosine is not disentanglement

Two arrows can be orthogonal in parameter space and still fight in function space, because
what matters is whether they change the model's behaviour **on the same inputs**. Parameter
orthogonality says nothing directly about that. Geometry is a *proxy*; the matrix is *ground
truth*; whether the proxy predicts the truth is an open question currently answered "weakly".

---

## 7. Metrics that cannot see what you are measuring

### 7.1 The observation

On SWaT at α = 1.0 the merged model's reconstruction error is nearly **5× worse** on the base
regime — and its AUROC goes **up** by 0.09%.

### 7.2 The resolution

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

### 7.3 Why improving reconstruction does not improve AUROC

Careful: AUROC is invariant to *rescaling*, not to model improvement in general. The deeper
reason is different:

> **The validation block measures reconstruction error on *normal* data only. Detection
> depends on the *contrast* between normal and anomalous.**

A better autoencoder reconstructs **everything** better, anomalies included, so the gap can
stay flat while every raw number improves. Like making a smoke alarm's sensor more sensitive:
useless if it becomes equally more sensitive to burnt toast and to real fires.

### 7.4 The consequence: α cannot be tuned honestly in unsupervised AD

To pick a merge scale defensibly you need a signal that tracks the metric you will report and
is not the test set. For **forecasting** that exists — validation and test are both MSE.

For **anomaly detection it does not**: validation measures reconstruction, the test metric
measures detection, §7.3 is exactly the statement that these come apart, and you cannot select
on test because that is selecting on the number you report. The reason there is no third
option is structural:

> **AD training data carries no labels by construction.** That is the premise of the whole
> setup. So there is no held-out set on which detection can be measured.

**Practical rule: select α on validation for forecasting; use a fixed, pre-declared α for AD.**
This is more defensible than it sounds, because **α\* does not move across seeds** on either AD
dataset (0.250 ±0.000 and 0.500 ±0.000) — the optimum is a stable property, not a
per-run accident.

---

## 8. Versus sequential fine-tuning

### 8.1 The comparison that matters

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

**It is a stability/plasticity trade, and merging wins outright only on SWaT.** Sequential
fine-tuning is more *plastic* — it adapts harder to the new segments on PSM and ETTh1. Merging
is more *stable*, but demonstrably so only on SWaT; elsewhere the old-regime intervals overlap.

### 8.2 Forgetting tracks how much the updates repeat each other

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

### 8.3 Two explanations, not separable here

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

## 9. What the three datasets said

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

## 10. A decision rule

Derived from the results above, at three seeds per dataset. **Treat it as a hypothesis**:
three datasets, two of them saturated, so roughly one informative case per branch.

```
1. Is there a real gap between the frozen base and joint training?
   NO  (SWaT, ~1% of base)  -> neither method can help. The benchmark cannot show a
                               difference; don't do continual learning here.
   YES -> continue.

2. Do you actually need the old regimes?
   NO  (only ever serve the newest data) -> keep the newest specialist. Merging only
                                            dilutes it.
   YES -> continue.

3. Are the updates repetitive (high rho), or is the base near its ceiling?
   YES -> MERGE. Sequential training can only drift, and it forgets badly.
   NO  (real headroom, novel updates) -> SEQUENTIAL FINE-TUNING. It adapts better to the
                                          new periods and there is little to forget.

4. Can you not tell which regime the incoming data belongs to?
   -> MERGE. It is routing-free insurance. UNTESTED -- and the case merging is really for.
```

**Merging's advantage is retention, not adaptation.** Step 4 is the case it exists for, and
nothing has measured it yet.

---

## 11. What is established and what is not

### Established

- **Fine-tuning learns something real and segment-specific.** Diagonal ratios of 0.61–0.75 and
  positive `specialisation` on all three rule out redundancy and degenerate fine-tuning as the
  global explanation.
- **The vectors are substantially aligned**, and alignment decays with temporal distance.
- **Merging is essentially free at the right scale** — merge cost intervals containing 1.00 on
  PSM and ETTh1.
- **What looked like interference at α = 1.0 was overshoot.** Residual interference is small
  and concentrated on the newest segment.
- **α\* is stable across seeds** on both AD datasets, which is what makes a pre-declared α
  defensible there.
- **Merging beats sequential fine-tuning outright only on SWaT**, the dataset whose updates
  most repeat each other.
- **The merge is bitwise reproducible** — all 60 merged checkpoints recompute exactly.

### Not established — do not claim these

- **That merging protects the base regime on all three datasets.** With error bars it does so
  only on SWaT; elsewhere the intervals overlap.
- **That geometry predicts outcome.** Two or three usable points, correctly ordered, is not
  evidence.
- **That redundancy or headroom is *the* driver of forgetting.** Both explain all three
  datasets equally well.
- **Anything from SWaT's test column.** Most of its metrics are inside their own noise floor.
- **Anything about weight disentanglement from cosine numbers** — different quantity.

### The one-sentence version

> The fine-tunes each learn something genuine and locally useful; because they largely learn
> the *same* thing, summing them at full strength overshoots — but that is a scale error, and
> once corrected, one merged model comes within a few percent of keeping a separate specialist
> per regime. Whether that beats simply continuing to train depends on how much the updates
> repeat each other.

---

## 12. Open questions

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

## 13. Quick reference

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
