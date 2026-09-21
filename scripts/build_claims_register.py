"""Build `results_archive/audit/claims_register.csv` — every claim this project makes, and its status.

    python scripts/build_claims_register.py --out results_archive/audit

The register exists because the failure mode this project keeps hitting is not a wrong number —
the checker catches those — but a **correct number carrying a claim wider than it supports**.
§1.23 and §1.24 both began as "under drift, X"; both turned out to be "on small shards, X" once a
second dataset with the same drift was run. §1.26's "merging never wins a decisive forecasting
configuration" was true of one table and false of the wider one. Nothing in the pipeline notices
that, because every cell involved reproduces.

So each claim is recorded with the evidence it actually rests on, and the status is **derived**,
not asserted:

    kind == "mechanism" and (n_datasets < 3 or not falsification_tested)  ->  hypothesis

A mechanism claim is one about *why* something happens — those are the ones that generalise
beyond their evidence if nobody stops them. A measurement claim ("on ETTh2, sequential beats
merging at all three n") is bounded by its own wording and does not need the rule. A scope claim
states a limit and is self-bounding.

`status_declared` is what the prose says; `status` is what the rule allows. Where they differ the
prose is wrong and the `prose_action` column says what has to change. Running this script is how
that list is produced, so the register cannot quietly drift from its own rule.

**The freeze rule (CLAUDE.md):** no merging experiment is added unless it maps to a row here whose
status is not `supported` — i.e. unless there is a stated open question it answers.
"""

import argparse
import csv
import logging
import re
from pathlib import Path

log = logging.getLogger("claims_register")

FIELDS = ["id", "claim", "section", "datasets", "n_datasets", "seeds", "floor_cleared",
          "kind", "falsification_tested", "status_declared", "status", "prose_action", "evidence"]

# Each entry: (id, claim, sections, datasets, seeds, floor_cleared, kind, falsification_tested,
#              status_declared, evidence)
#
# floor_cleared: "yes" the reported effect exceeds the dataset's own floor; "no" it does not;
#   "mixed" some cells do and some do not; "n/a" the claim is not a magnitude comparison.
# falsification_tested: "yes" only if an experiment was run that *could* have overturned it and
#   did not. Re-measuring the same quantity on more seeds is not a falsification test.
CLAIMS: list[tuple] = [
    # ---- headline measurements -------------------------------------------------------------
    ("C01", "Merging costs ~1.0-1.1x a per-regime specialist",
     "TL;DR, 1.11", "SWaT,PSM,ETTh1", 3, "yes", "measurement", "yes", "supported",
     "§1.35 P3 REFUTED it as a general claim: ETTh2 1.454->2.064 and exchange 1.626->2.042, both "
     "growing with n. Holds on the three datasets named and is now scoped to them."),
    ("C02", "alpha*.n stays order 1 in the deployment parameterisation",
     "TL;DR, 1.18", "SWaT,PSM,ETTh1,ETTh2,ETTm2,exchange_rate", 6, "yes", "measurement", "yes",
     "supported",
     "§1.18 states it as an empirical regularity and records that it fails a fixed-shard control "
     "on exchange_rate and a prefix-merge design on both - the falsification test was run and "
     "bounded the claim rather than breaking it."),
    ("C03", "alpha*.n rises with n on PSM because the task vectors de-align",
     "1.18", "PSM", 1, "yes", "mechanism", "no", "hypothesis",
     "Already reported as mechanistically uncorroborated in §1.18: ETTh1 de-aligns fastest and "
     "has the flattest product, so the geometry points the other way. One dataset, no "
     "falsification test. "
     "DETERMINATION 2026-09-21, after C31 showed a hypothesis can be settled from the archive "
     "with zero runs: C03 CANNOT be. The archive holds alpha*.n for PSM at n=2,3,5 and the "
     "geometry that contradicts it, and both are already reported - there is no third quantity "
     "sitting in finished runs that separates 'the task vectors de-align' from 'something else "
     "rises with n'. Settling it needs PSM at more segment counts with per-shard geometry, i.e. "
     "NEW RUNS, and the merging chapter is frozen. DECLARED OUT OF SCOPE for the thesis: the "
     "measurement stands (alpha*.n rises, 2.4x the quantisation bound), the mechanism does not, "
     "and §1.18 already says so. An open row here is a deliberate boundary, not unfinished "
     "work."),
    ("C04", "Merging is worth two to four periods of retained history",
     "TL;DR, 1.21, 1.23", "ETTh1,ETTh2,exchange_rate", 3, "yes", "measurement", "yes",
     "supported",
     "Crossover W=3 on ETTh1/exchange_rate, W=5 on ETTh2. §1.26b re-selects the budget on "
     "validation and reports what that costs, which is the honest-selection control."),
    ("C05", "Whether old history hurts is dataset-specific and drift does not predict it",
     "TL;DR, 1.21, 1.23", "ETTh2,ETTm2,exchange_rate", 3, "yes", "measurement", "yes",
     "supported",
     "exchange_rate +26% for a 3-period window; ETTh2 monotonically better with more data at "
     "near-identical drift. The ETTh2 run is itself the falsification test of the drift reading."),
    ("C06", "Shard size, not drift, drives most cross-dataset differences",
     "TL;DR, 1.23, 1.24", "ETTh2,ETTm2,exchange_rate", 3, "yes", "mechanism", "yes", "supported",
     "§1.24 was run specifically to break the drift explanation and did: three of exchange_rate's "
     "four distinctive behaviours fail to reproduce on datasets sharing its drift. §1.16's "
     "routing-headroom exception is recorded rather than smoothed over."),
    ("C07", "Continual fine-tuning degrades as steps chain",
     "TL;DR, 1.24", "ETTm2,exchange_rate", 2, "yes", "measurement", "yes", "supported",
     "Stated over the two datasets where it is measured, and §1.23's ETTh2 counterweight is "
     "published beside it rather than omitted - the claim is already scoped in the prose."),
    ("C08", "There is no universal winner between merging and sequential fine-tuning",
     "TL;DR, 1.13, 1.23", "SWaT,PSM,ETTh1,ETTh2,ETTm2,exchange_rate", 6, "mixed", "measurement",
     "yes", "supported",
     "A negative claim, and the strongest kind here: it survives every dataset added. ETTh2 "
     "(sequential wins at all three n) and exchange_rate (flips with n) are both counterexamples "
     "to any ranking."),
    ("C09", "Recency is not relevance - the newest specialist is rarely the best model",
     "TL;DR, 1.20", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "n/a", "measurement", "no", "supported",
     "1 of 16 excluding the trivial last regime. A counting result over four datasets; no "
     "falsification test exists because there is no mechanism claimed."),
    ("C10", "Accumulated merges generalise forward to unseen regimes",
     "TL;DR, 1.19", "ETTh1,exchange_rate", 2, "mixed", "measurement", "no",
     "supported",
     "Beats base in 7 of 8 cases with no decay as vectors accumulate - and the 8 are TWO "
     "datasets x four prefix points, not four datasets: `prefix_merges.csv` is written only by "
     "the prefix_etth1/prefix_exchange runs. 1 of 8 is a counterexample (exchange_rate at k=2, "
     "1.050) and is reported. (The datasets field read four until 2026-09-19.)"),
    ("C11", "Validation cannot select alpha for unsupervised anomaly detection",
     "TL;DR, 1.12", "SWaT,PSM", 2, "yes", "measurement", "yes", "supported",
     "Costs 25-98% of achievable GRR on AD against 1-8% on forecasting. §1.12's own early-stopping "
     "probe came back under 1%, which narrowed the claim to alpha specifically rather than "
     "indicting validation selection generally - a falsification test that bounded it."),
    ("C12", "The reproducibility floor is dataset-specific, not a universal 2%",
     "TL;DR, 1.9, 1.30", "SWaT,PSM,ETTh1,ETTh2,ETTm2,exchange_rate,PSM-forecast,SWaT-forecast", 8,
     "n/a", "measurement", "yes", "supported",
     "Floors span 0.07% to 69.75%. §1.9a tests the variance convention itself and reports which "
     "verdicts depend on it."),
    ("C13", "What used to look like interference was a scale error, not forgetting",
     "TL;DR, 1.3", "SWaT,PSM,ETTh1,exchange_rate", 4, "yes", "mechanism", "yes", "supported",
     "The alpha sweep is the falsification test: at alpha* the damage disappears on every dataset "
     "where the alpha=1.0 'forgetting' had been reported."),
    ("C14", "No cheap regime predictor separates merge-favouring from sequential-favouring configs",
     "1.13, 1.14", "SWaT,PSM,ETTh1,exchange_rate", 4, "n/a", "measurement", "yes", "supported",
     "Nine decisive configurations, 5/4 split, no signal beyond chance. Reported with the "
     "diagnosis that the question was mis-posed without segment count as an input."),
    ("C15", "Task vectors shrink and de-align as segments multiply",
     "TL;DR, 1.15", "ETTh1", 1, "n/a", "measurement", "no", "supported",
     "A direct geometric measurement on ETTh1, stated as such. Not a mechanism claim in itself; "
     "C03 is the mechanism built on top of it and is downgraded."),
    ("C16", "Routing's advantage over merging is real and universal",
     "1.16b", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "yes", "scope", "no", "supported",
     "§1.16b already labels it 'the weakest possible sense' - a per-window oracle router is an "
     "upper bound, not a buildable system. Self-bounding."),
    ("C17", "A router cannot be evaluated on AD from these runs",
     "1.16", "SWaT,PSM", 2, "n/a", "scope", "n/a", "supported",
     "AD transfer matrices carry no detection metric per regime; the refusal is intended and the "
     "previously published +6.2%/+7.8% is withdrawn as underivable."),
    ("C18", "On exchange_rate merging beats joint training on all the data at once",
     "TL;DR, 1.11", "exchange_rate", 1, "yes", "measurement", "no", "supported",
     "GRR 1.399/1.207/1.224 at n=2/3/5, alpha chosen on validation. Single dataset and the claim "
     "names it; §1.24 shows it does not transfer to ETTh2/ETTm2."),
    ("C19", "Merging never wins a decisive forecasting configuration",
     "1.26", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "yes", "measurement", "yes", "refuted",
     "Withdrawn in-place at §1.26: true of that table's 18 cells, false across §1.5b's 48. The "
     "warning block stays because the sentence is quoted elsewhere."),
    # ---- advanced merging ------------------------------------------------------------------
    ("C20", "OPCM's cost grows with the accumulated-subspace overlap rho",
     "1.35", "SWaT,PSM,ETTh1,ETTh2,ETTm2,exchange_rate,PSM-forecast,SWaT-forecast", 8, "mixed",
     "mechanism", "yes", "refuted",
     "Registered as P1 before the sweep and refuted by it: the cost does not order by rho. The "
     "prediction and its refutation are both published. REPLACED, not merely refuted: C31 gives "
     "the predictor that does work at threshold 0.3 - base-to-joint headroom, i.e. how much "
     "there was to lose rather than how much was removed (§1.36)."),
    ("C21", "In the MERGING frame, BECAME's coefficient cannot reach the strength AD needs "
              "because its convex fold pins alpha.n = 1.0",
     "1.31, 1.35", "SWaT,PSM,PSM-forecast", 3, "yes", "mechanism", "yes", "supported",
     "Structural, and verified per run rather than argued: implied_alpha_times_n is emitted by "
     "remerge.py and equals 1.0 on every BECAME row. §1.36 P2 is the falsification test - "
     "rescaling the weights to the committed alpha.n separates weighting from magnitude. SCOPE (2026-09-20): this concerns BECAME's COEFFICIENT applied inside this project's merging frame, where every shard is fine-tuned from the frozen theta_0. It is not a verdict on BECAME, which is a continual-learning method whose merge interpolates two endpoints of ONE training trajectory. The method is tested in its own frame as strategy 6, adaptive-lambda sequential fine-tuning (§1.39); this row is the evidence for why that was worth doing, not a result about it."),
    ("C22", "In the MERGING frame, Fisher weighting contributes nothing beyond setting the "
              "merge magnitude",
     "1.36", "SWaT,PSM,PSM-forecast", 3, "yes", "mechanism", "yes", "supported",
     "§1.36's P2 ran the isolating test - BECAME's relative weights rescaled so their sum equals "
     "the source run's committed alpha*n, against uniform 1/n at that same alpha*n through the "
     "same evaluation path - and confirmed it: 5 of 7 cells are ties inside the floor, and "
     "NEITHER exception favours the weighting (PSM n=3 +0.11%, PSM n=5 +1.36% against a 0.07% "
     "floor, i.e. 19x the floor). implied_alpha_times_n equalled the target on every row, so the "
     "comparison really was at matched magnitude. Once magnitude is held fixed the Fisher "
     "weighting contributes nothing, and at larger n it costs. SCOPE (2026-09-20): this concerns BECAME's COEFFICIENT applied inside this project's merging frame, where every shard is fine-tuned from the frozen theta_0. It is not a verdict on BECAME, which is a continual-learning method whose merge interpolates two endpoints of ONE training trajectory. The method is tested in its own frame as strategy 6, adaptive-lambda sequential fine-tuning (§1.39); this row is the evidence for why that was worth doing, not a result about it."),
    ("C23", "OPCM helps on exchange_rate at n<=3 because it acts as a recency filter",
     "1.35, 1.36", "exchange_rate", 1, "yes", "mechanism", "pending", "supported",
     "§1.35 asserts it as a finding - 'OPCM is not a merge improvement; it is a recency filter, "
     "and it pays exactly where recency pays' - on ONE dataset with no test that could have "
     "broken it. §1.36's P3 ran that test and came back INCONCLUSIVE, not supporting: feeding "
     "the periods newest-first does destroy the exchange_rate win (+13.25% at n=2, +9.67% at "
     "n=3 against a 5.73% floor), but reversal is decisively worse on EVERY dataset - ETTm2 +38% "
     "to +54%, ETTh1 +11% to +14% - so exchange_rate's penalty is mid-range and the test cannot "
     "separate a recency effect from the general cost of reversing. The ETTh2 half of P3 was "
     "refuted outright (its loss grew rather than shrank). The direction does not fit the "
     "stale-directions story either: under reversal the OLDER shards are stripped, which should "
     "have helped on the one dataset where old data hurts, and did not. No further merging run "
     "can settle this; what stands is the measurement C32, not the mechanism. "
     "DETERMINATION 2026-09-21: INCONCLUSIVE IS THE FINAL STATE, and that is the honest answer "
     "rather than a placeholder. The archive cannot settle it - the falsification test has "
     "already been run (§1.36's P3) and its result is not ambiguous about the data, only about "
     "what the data can attribute: reversal is decisively worse everywhere, so exchange_rate's "
     "penalty being mid-range cannot separate a recency effect from the general cost of "
     "reversing. A different experiment could (feeding shards in random orders, or filtering by "
     "age without reordering), but that is NEW RUNS on a frozen chapter. DECLARED OUT OF SCOPE: "
     "the win is real and measured (C32); the recency explanation is recorded as untested and "
     "must not be quoted as the reason."),
    ("C43", "At the paper's threshold, OPCM recovers less of the base-to-joint gap than plain "
     "task arithmetic on every measurable configuration",
     "1.36", "ETTh1,ETTh2,ETTm2,exchange_rate,PSM,SWaT,PSM-forecast", 7, "yes", "measurement",
     "n/a", "supported",
     "Stated in GRR, the unit §1.11 already uses - (base - merged)/(base - joint), §0.6's "
     "definition - so it answers the question actually asked of this work: does sophisticated "
     "merging help where task arithmetic falls short of joint training? It does not. 21 of 21 "
     "cells at threshold 0.5, mean delta GRR = -0.311; on SWaT the OPCM GRR goes NEGATIVE at "
     "every n (-0.077, -0.141, -0.156), i.e. worse than the base model. The sd quoted is PAIRED "
     "(both arms from the same seed's checkpoints, so the per-seed difference cancels their "
     "shared variance) and 19 of 21 deltas exceed 1x it; the two that do not - ETTh2 n=2 and "
     "exchange n=3 - are named in the section rather than folded into the count. "
     "SCOPE: at threshold 0.3 it is 17 of 21, mean -0.218 - same direction, not the same "
     "unanimity - so the claim names the threshold. SWaT-forecast is excluded because its "
     "headroom is negative at n=3,5, so the gap GRR divides by has the wrong sign. "
     "CROSS-CHECKED: the baseline arm is each run's stored plain-sum merge, so its GRR must "
     "reproduce derived.csv's own grr, computed by different code from a different file - all "
     "241 rows agree TO 1e-4, which is the effective tolerance: derived.csv stores grr to 4 "
     "decimals, so the check cannot detect an error below that, and the observed worst gap is "
     "exactly 1.0e-4 on PSM n=3 (stored precision, not disagreement). The check applies to 222 "
     "of 241 rows; the other 19 carry a blank grr_baseline_derived because their baseline is "
     "not the stored merge - P3_order_reversal (12, forward-order OPCM) and P2_became_rescaled "
     "(7, uniform 1/n at the same alpha*n)."),
    ("C31", "At threshold 0.3, OPCM hurts most where there is most base-to-joint headroom",
     "1.35, 1.36", "ETTh1,ETTh2,ETTm2,exchange_rate,PSM-forecast", 5, "mixed", "mechanism", "yes",
     "supported",
     "SETTLED 2026-09-21 WITHOUT NEW RUNS, which is what the freeze requires of this row: "
     "§1.36's P1 deltas joined to derived.csv's headroom on (dataset, n_segments), 15 cells over "
     "5 forecasting datasets. At threshold 0.3 the per-cell correlation is r=+0.732, rho=+0.729; "
     "collapsed to datasets - where the dependence actually lives, since headroom is close to a "
     "dataset property and the 15 cells carry only 5 distinct x values - r=+0.954 with an EXACT "
     "permutation p=0.025 over all 120 permutations. It survives dropping any single dataset "
     "(r=+0.897 to +0.999). This is the predictor refuted C20 was looking for: the cost tracks "
     "the size of the PRIZE (how much a joint model improves on the base at all), not the size "
     "of the deletion. "
     "SCOPE, and it is a real limit: the claim holds AT THRESHOLD 0.3 ONLY. At the paper's own "
     "recommended 0.5 the dataset-level p is 0.125 and at 0.7 it is 0.667 - not distinguishable "
     "from chance with five datasets. The decay is CONSISTENT with aggressive deletion swamping "
     "the dependence (at 0.7 exchange_rate jumps to +53.4% and breaks the ordering), but five "
     "datasets cannot separate 'swamped' from 'underpowered' and no such claim is made. "
     "The permutation test's own floor is 2/120 = 0.017, so p=0.025 is real evidence and also "
     "near the resolution limit of five datasets. "
     "TWO EXCLUSIONS, both load-bearing: SWaT-forecast (headroom NEGATIVE at n=3,5, so 'how much "
     "there was to lose' is undefined) and PSM/SWaT on window_auroc (headroom 1-3%, at which "
     "every merge scores alike and |delta| is small BY CONSTRUCTION - including them raises the "
     "per-cell r for an arithmetic rather than a mechanistic reason). "
     "SENSITIVITY: headroom is averaged over every experiment carrying it; using the experiment "
     "of record instead gives r=+0.627 at threshold 0.3. Both positive, same story, both emitted "
     "to headroom_cost_fit.csv. NOTE the label: remerge_closeout.csv spells this dataset "
     "'exchange' while floors.csv and this register spell it 'exchange_rate'."),
    ("C32", "The measurement that survives P3 regardless: OPCM beats plain summation on "
     "exchange_rate at n=2 and n=3, at all three thresholds",
     "1.35", "exchange_rate", 1, "yes", "measurement", "n/a", "supported",
     "-10.88% and -14.18% against a 5.73% floor, three thresholds each. This is what remains if "
     "C23's mechanism is refuted."),
    ("C24", "In the MERGING frame, lambda* is unstable because the Fisher estimate is noisy",
     "1.32", "PSM-forecast", 1, "no", "mechanism", "yes", "refuted",
     "§1.32 ran the falsification test and the estimate saturates at the full pass - the "
     "instability is not sampling noise. STILL REFUTED, but the test was insensitive to the "
     "defect that does exist (2026-09-20): §1.39b shows diagonal_fisher squares the gradient of "
     "a batch-MEAN loss, so E[F_hat] = g^2 + sigma^2/B - a function of the BATCH SIZE B, not of "
     "the sample count K. §1.32 saturated K, which is exactly the variable the expectation does "
     "not contain, so it could not have detected this however it came out. The defect is a bias, "
     "not noise, so C24's wording is refuted on its own terms; what is withdrawn is the strength "
     "of the evidence, not the verdict. In the merging frame the bias also cancels: "
     "_shard_fishers evaluates every Fisher at its own specialist, all minima, so sigma^2/B is "
     "common to numerator and denominator - the same algebra as the t=1 control in §1.39b. "
     "SCOPE (2026-09-20): this concerns BECAME's COEFFICIENT applied inside this project's merging frame, where every shard is fine-tuned from the frozen theta_0. It is not a verdict on BECAME, which is a continual-learning method whose merge interpolates two endpoints of ONE training trajectory. The method is tested in its own frame as strategy 6, adaptive-lambda sequential fine-tuning (§1.39); this row is the evidence for why that was worth doing, not a result about it."),
    ("C25", "Attention-exclusive fine-tuning (the testable half of QOMM) helps",
     "1.33", "PSM-forecast", 1, "no", "measurement", "yes", "refuted",
     "Measured on PSM-forecast n=3, three seeds, and did not clear the floor - reported as a "
     "negative result. §1.33 also refutes QOMM's stated PREMISE: attention-only fine-tuning was "
     "predicted to make task vectors more orthogonal and does the opposite, raising the "
     "off-diagonal cosine ~35% and nearly doubling rho. (The datasets field read "
     "'ETTh1,exchange_rate' until 2026-09-19; §1.33 never used those.)"),
    ("C26", "The simplified OPCM operator (§1.31, §1.35) is the paper's OPCM",
     "1.31, 1.35, 1.36", "", 0, "n/a", "scope", "n/a", "refuted",
     "Never claimed and must never be: `opcm_residual` projects out the span of the flattened "
     "predecessors; the paper's operator (Tang et al., NeurIPS 2025, Algorithm 1) projects out "
     "the top-alpha singular subspace of the ACCUMULATED MERGED matrix, on both sides, drops the "
     "i==j diagonal, and carries a norm-stabilising lambda. Both are now implemented and reported "
     "side by side as separate rules (`merge_opcm_paper` in framework/merging/opcm.py)."),
    ("C33", "The paper's OPCM sets its own merge magnitude, which no validation can move",
     "1.36", "", 0, "n/a", "scope", "n/a", "supported",
     "Structural, from Algorithm 1 line 14 and Theorem 5.2: lambda^(T) pins "
     "||theta_merged - theta_0|| to the MEAN task-vector norm exactly. Verified per merge "
     "(`opcm_norm_ratio` must be 1.0, asserted in remerge.py and negative-tested in "
     "verify_merge_rules.py) on all 252 merges. Across the 72 published cells it lands at "
     "implied alpha*n 1.063-1.696, above 1.0 on every one, against the order-1 alpha*.n of "
     "§1.18. Same class of finding as C21: the method chooses a magnitude, and on these datasets "
     "it is not the one that wins. Whether that is WHY it loses is C34, and untested."),
    # ---- design and measurement scope ------------------------------------------------------
    ("C34", "The paper's OPCM loses because of its fixed magnitude, not its projection "
              "(AD: refuted; forecasting: refuted)",
     "1.36, 1.37, 1.38", "SWaT,PSM,ETTh1,ETTh2,ETTm2,exchange_rate,PSM-forecast,SWaT-forecast",
     8, "yes", "mechanism", "yes", "refuted",
     "§1.37's P4 ran the isolating test - the paper's projection held bit-for-bit fixed "
     "(collinear to 3.7e-15) and rescaled to each run's committed alpha - and refuted it: 0 "
     "better, 10 ties, 62 worse over 72 cells. The decisive cells are SWaT and PSM, where the "
     "committed alpha is already 1.0 so the correction is a near no-op (distance ratio "
     "0.96-1.15) and the loss is UNCHANGED to within 0.17pp while still 7-25x its floor. "
     "Magnitude is not the cause on AD; the projection is. FORECASTING, settled separately by "
     "§1.38's P5 after §1.37's coefficient-matched cells proved confounded (those merges "
     "travelled only 0.18-0.66x the intended distance): re-run distance-matched, every cell "
     "improved - median -16.1pp, up to -69.7pp - so the confound was real and material, yet the "
     "verdict is unchanged. 2 better, 10 ties, 42 worse over 54 cells; ETTh1, ETTm2 and "
     "PSM-forecast lose decisively at every threshold; 9 of the 10 ties are SWaT-forecast, whose "
     "84.23% floor ties everything. Refuted on BOTH task families: magnitude is part of the story "
     "on forecasting and not the cause of it. The only genuine wins in the sweep are exchange_rate "
     "at alpha=0.3, n=2 and n=3 - the one dataset where old data actively hurts (§1.24)."),
    ("C27", "The base model's 50% history fraction is a free parameter, not a tuned choice",
     "1.28", "ETTm2", 1, "yes", "scope", "no", "supported",
     "§1.28 varied it for the first time and found the training-size term alone is large. Stated "
     "as a limitation of every other section rather than as a result."),
    ("C28", "The winner survives moving the train/test cut (rolling origin)",
     "1.27, 1.27a", "exchange_rate", 1, "no", "measurement", "yes", "refuted",
     "It does not: the ranking is unstable across origins. §1.27b isolates the test block and "
     "§1.27a pools ranks rather than raw means because the MSE scale spans 0.20-2.05."),
    ("C29", "Merging sits closer to the routing ceiling than the newest specialist in 8 of 10 "
              "forecasting configurations",
     "1.16", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "mixed", "measurement", "yes", "supported",
     "Restated 2026-09-19 as a COUNT. It previously read 'on all four forecasting datasets', "
     "which routing_summary.csv contradicts: merging is closer to the ceiling in 8 of 10 groups, "
     "and the newest specialist is closer on ETTh1 n=2 (0.0% vs 5.6% - there the newest model IS "
     "the per-regime optimum) and ETTh2 n=5 (54.8% vs 81.0%). ETTh2 n=3 is a coin flip at 108.30 "
     "vs 108.60, so a conservative count reads 7 of 10. Per dataset: ETTm2 3/3, exchange_rate "
     "2/2, ETTh2 2/3, ETTh1 1/2 - so it does not hold on all four datasets under any reading. "
     "The section field also pointed at 1.16b while the sentence lives in 1.16."),
    ("C30", "Continual fine-tuning forgets, measured as BWT",
     "1.34", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "mixed", "measurement", "no", "supported",
     "ACC/BWT read off the sequential chains that already existed; no new training."),
    # ---- strategy 6: adaptive-lambda sequential fine-tuning ---------------------------------
    # NOT merging claims. The freeze tally quoted in CLAUDE.md is a statement about the merging
    # chapter and stays at 34; these are counted separately in §0.7.
    ("C35", "On ACC, adaptive-lambda sequential fine-tuning loses to the plain chain on "
     "forecasting",
     "1.39", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "yes", "measurement", "yes", "supported",
     "P1 registered before the runs and refuted by them: 16 worse, 0 ties, 0 better over 8 "
     "configurations x 2 estimator settings, each paired against the chain that supplied its own "
     "theta_0. The falsification test WAS run - the Fisher estimator was identified as defective "
     "and the whole sweep re-run at the corrected setting, which could have overturned the "
     "verdict and did not: every cell improved and none changed verdict. ETTh1 n=3 is the one "
     "borderline cell (1.43x its floor, inside the <1.5x band) and is flagged as such rather "
     "than pooled with cells at 10-30x. AD (PSM, SWaT) is NOT covered: its ACC has no floor for "
     "reconstruction/score_mean and it was not re-run at the corrected estimator. SCOPED TO ACC: "
     "ACC is the mean over regimes of the LOSS, so it charges the chain for forgetting old "
     "shards. On the task's own final test metric the picture differs on two cells - see C38 - "
     "and neither metric is the right one on its own."),
    ("C38", "On the task's own test metric adaptive-lambda still loses on forecasting, except "
     "exchange_rate n=3",
     "1.39", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "yes", "measurement", "yes", "supported",
     "The pipeline already evaluated the configurator's real test evaluator at every step, AFTER "
     "the pullback, so the model scored is theta*_t; no new compute was needed. Corrected "
     "estimator, final step: 6 worse, 1 tie (ETTm2 n=5, 0.86x its floor), 1 BETTER - "
     "exchange_rate n=3 at -9.79%, 1.71x its floor, with all three seeds improving (-9.08, "
     "-2.10, -17.18%) though the middle seed is itself inside the floor. At B=128 that same cell "
     "was a tie (+5.65%), so the estimator defect was HIDING the one win. Consistent with §1.24: "
     "exchange_rate is the dataset where old data actively hurts, so a rule that pulls the chain "
     "back toward theta_0 has the least to destroy. ACC and this metric DISAGREE on two cells "
     "and both are reported (C35)."),
    ("C40", "Adaptive-lambda's exchange_rate win is scarcity (braking as regularisation), not "
     "recency",
     "1.39, 1.39c", "exchange_rate,ETTm2", 2, "n/a", "mechanism", "yes", "refuted",
     "REFUTED by §1.39c, run the same day it was registered: fixed lambda = 1/t brakes on the "
     "same schedule with no Fisher at all and does NOT win - +6.86% against the adaptive rule's "
     "-9.79%, 16.7pp apart and on the wrong side of the floor. If the win were regularisation of "
     "an overfitted ~1,012-row fine-tune, the imposed schedule would have delivered it. What the "
     "coefficient actually does is the OPPOSITE of braking harder: on exchange_rate the derived "
     "lambda EXCEEDS 1/t at every period (1.25x, 2.18x, 1.30x), so the Fisher says take more of "
     "the new shard, and taking more is what wins - 1/t over-brakes here. That is a reading of "
     "§1.24 the withdrawn recency argument had backwards. No replacement mechanism is claimed: "
     "one cell, one dataset, three seeds, and the coefficient is only right under the corrected "
     "estimator (at B=128 the same rule ties). Original wording follows. "
     "The recency reading was STATED in an earlier draft and is withdrawn as backwards: the "
     "pullback interpolates toward theta*_(t-1), which RETAINS more of the earlier periods, so "
     "on a dataset where old data hurts (§1.24) braking should cost more, not less. Scarcity is "
     "the likelier reading - at baseline_fraction=0.5, n=3 each period is ~1,012 of "
     "exchange_rate's 6,071 rows, small enough that the fine-tune overfits and braking is simply "
     "regularisation. §1.28's discriminating test applies (ETTm2 has essentially the same drift, "
     "0.752 vs 0.833, with 9x the data) and ETTm2 n=5 at ~5,574 rows per period came out a TIE, "
     "not a win - one cell, weak, pointing at scarcity. NOT settled: the two readings are not "
     "separated on this dataset, which is why every exchange_rate result in this document "
     "carries the 'strong drift or small shards' caveat. The decisive test is registered and "
     "cheap: fixed lambda = 1/t on exchange_rate n=3 (three fine-tunes). If braking per se wins "
     "there, the adaptive coefficient is not doing the work."),
    ("C41", "Adaptive-lambda's premise - a useful point strictly between the accumulated model "
     "and the fresh fine-tune - holds on some configurations and fails on most; where it holds "
     "the derived coefficient locates it better than a fixed schedule",
     "1.39d", "exchange_rate,ETTh2,ETTm2", 3, "n/a", "mechanism", "yes", "refuted",
     "REFUTED by its own registered test (45 runs, 0 failures). The claim had two halves and "
     "only one survives. FIRST HALF WRONG: the premise does NOT fail on most configurations - a "
     "useful interior point exists on TWO of the three swept, including ETTh2 n=3, which was "
     "registered as the loser that should decrease monotonically toward lambda=1 and instead "
     "has an optimum at lambda=0.7 beating the plain chain by 9.11% (1.35x its floor, which is "
     "borderline). exchange_rate is interior at 3.85x its floor as predicted; ETTm2 is flat "
     "within the floor, demonstrating no interior point. SECOND HALF SURVIVES and is re-"
     "registered as C42. REGISTERED 2026-09-21 BEFORE the fixed-lambda grid, from evidence "
     "already in hand. §1.39c "
     "gives three coefficients that bracket one cell: lambda=1 IS the plain chain (gate 1 "
     "asserts it reproduces it bitwise) at 0.3586, adaptive at 0.3235 (-9.79%), fixed 1/t at "
     "0.3832 (+6.86%) - worse at the boundary, better in the middle, worse again when braking "
     "harder, so the optimum on that cell is INTERIOR and bounded on both sides. On the other "
     "seven configurations the plain chain wins outright, so the optimum is at the BOUNDARY, and "
     "Eq. 20 cannot reach it: lambda* = A/(A+B) with A,B>0 is strictly inside (0,1). The "
     "falsification test is registered with the claim: a fixed-lambda grid over {0.1,0.3,0.5,"
     "0.7,0.9} x 3 seeds must show exchange_rate n=3 with an interior minimum and ETTh2 n=3 / "
     "ETTm2 n=3 decreasing monotonically toward lambda=1. ETTm2 is chosen as the third dataset "
     "because §1.28 already names it exchange_rate's discriminating comparison (same drift, 9x "
     "the data), and because 3 datasets is what the register's rule needs before this can be "
     "anything but a hypothesis."),
    ("C42", "Where adaptive-lambda loses on forecasting, the coefficient over-brakes: a useful "
     "interior point exists and Eq. 20 sits well below it",
     "1.39d", "exchange_rate,ETTh2,ETTm2", 3, "mixed", "mechanism", "yes", "supported",
     "Measured by the fixed-lambda grid registered as C41's falsification test, so the test "
     "that could have overturned it was run before the claim was written. The curve's own "
     "optimum against the derived coefficient at t>=2: exchange_rate 0.5 vs 0.727/0.326 (0.9x - "
     "lands on it, and wins), ETTm2 0.5 vs 0.070/0.135 (4.9x), ETTh2 0.7 vs 0.081/0.087 (8.4x - "
     "the point exists and Eq. 20 sits 8.4x below it). The ordering matches C37's independently: "
     "the configurations where the derived lambda collapses are exactly those with a large "
     "Lambda/F excess over its floor of t (exchange 0.1-0.8x, ETTm2 2.0-8.6x, ETTh2 3.4-7.1x), "
     "which is the at-a-minimum asymmetry §1.39b measures. Two separately-built quantities agree "
     "about which configurations the coefficient will mis-set. SCOPE: three configurations, one "
     "dataset each, and ETTh2's interior gain is borderline (1.35x its floor). It does NOT show "
     "that a well-chosen fixed lambda beats the plain chain in general - ETTm2 says it does not "
     "- and it does not change any §1.39 verdict. It relocates the failure from 'the premise "
     "rarely applies' to 'the premise often applies and this coefficient mis-sets it'."),
    ("C39", "Adaptive-lambda does not improve anomaly detection",
     "1.39", "PSM,SWaT", 2, "mixed", "measurement", "yes", "supported",
     "Measured on window_auroc, which HAS a published floor on both datasets - not on "
     "reconstruction/score_mean, which §1.12 showed is blind to detection quality and has no "
     "floor. Adaptive lambda loses AUROC on both: PSM -0.79%, SWaT -0.44%, against floors of "
     "0.068% and 0.087%. window_auprc ties on both against its larger floor. CAVEAT, on the "
     "record: PSM's margin is 0.75x THESE RUNS' own seed spread (1.05%), so it is decisive by "
     "the published floor - measured on a dedicated, quieter base-model experiment - only. "
     "SWaT's is 2.7x its own spread (0.16%) and is decisive either way. This is §1.9's open "
     "question landing on a live cell; own_spread_pct is emitted for every row. NOT re-run at "
     "the corrected estimator: on AD the bias is expected to largely self-cancel (§1.39b) and "
     "that expectation is an inference, not a measurement."),
    ("C36", "diagonal_fisher's batch-mean gradient suppresses lambda* by 6-24x at every step "
     "past the first",
     "1.39b", "ETTm2", 1, "n/a", "mechanism", "yes", "supported",
     "The algebra is exact - E[F_hat] = g^2 + sigma^2/B, so a form at a minimum scales as B^-1 "
     "and one away from a minimum saturates - and the differential exponent is measured: "
     "numerator B^-0.90, Lambda B^-0.29, with t=1 as a built-in negative control where Lambda is "
     "Lambda_0 alone, both forms sit at minima, and lambda is B-invariant across a 128x range. "
     "The scaling share derived from those exponents (42.6% at t=2, 38.2% at t=3, falling with t "
     "in all three seeds) is a second prediction the fit was not tuned to produce. BUT the "
     "exponent fit is ONE dataset (ETTm2 n=3, 3 seeds): the B-sweep was not repeated elsewhere, "
     "so the rule downgrades this and it is right to. The CONSEQUENCE is broader - all eight "
     "forecasting configurations were re-run at B=1 and every one improved - but the mechanism "
     "itself rests on one dataset."),
    ("C37", "The residual suppression left after correcting the estimator is Eq. 20's "
     "at-a-minimum asymmetry",
     "1.39b", "ETTh1,ETTh2,ETTm2,exchange_rate", 4, "n/a", "mechanism", "no", "supported",
     "d^T F_t(theta_hat_t) d / d^T F_t(theta*_t) d is below 1 in all 78 cells - the Fisher is "
     "always larger at the merged point - and regressing log(Lambda/F / t) on log(1/asymmetry) "
     "over the 54 cells at t>1 gives slope +0.972, i.e. correctly SCALED. But r = 0.585, so "
     "r^2 = 0.342 and two thirds of the per-cell variation is unaccounted for: the asymmetry "
     "accounts for the residual in magnitude, not cell by cell. No test has been run that could "
     "have overturned it, which is why the rule holds it at hypothesis. A near miss is on the "
     "record: measured at the PUBLISHED batch size the asymmetry reads ~90x instead of ~5-8x, "
     "because F(theta_hat) is itself suppressed by sigma^2/B - quoting that would have confirmed "
     "the mechanism at fifteen times its size."),
]

# The professor's method list: what was asked for, and what actually exists.
METHODS: list[tuple] = [
    ("Task arithmetic (plain sum at alpha)", "full",
     "The project's baseline throughout; merge is bitwise reproducible from checkpoints (412/412).",
     "alpha* ~ 1/n in the deployment parameterisation; merging within 1.0-1.1x of specialists on "
     "SWaT/PSM/ETTh1 but 1.5-2.1x on ETTh2/exchange_rate (C01)."),
    ("OPCM - residual against flattened predecessors", "partial",
     "Implemented and run everywhere, but this is a SIMPLIFICATION of the published operator, "
     "labelled as such in every section that uses it (C26).",
     "Cost does not scale with rho (C20, refuted). Helps on exchange_rate at n<=3; the "
     "recency-filter mechanism was tested by §1.36's P3 and came back INCONCLUSIVE, so C23 "
     "stays a hypothesis and only the measurement (C32) stands."),
    ("OPCM - paper operator (Tang et al., NeurIPS 2025, Algorithm 1)", "full",
     "Implemented from the paper once it was supplied: full SVD of the accumulated merged task "
     "matrix, two-sided projection out of the top-alpha singular subspace, i==j dropped, "
     "norm-stabilising lambda, 1-D tensors passed through. Eq. 8 and Thm 5.2 are asserted as "
     "unit checks rather than assumed (`verify_merge_rules.py`).",
     "Takes no merge scale - it fixes its own magnitude at the mean task-vector norm (C33). On "
     "ETTh1 that lands at alpha*n = 1.60 against the 1.00 validation selected, and the merge is "
     "decisively worse than plain summation at every threshold, worsening monotonically as the "
     "threshold rises (§1.36 P1). §1.37 isolated the cause: with its norm rule replaced by the "
     "committed strength the loss is unchanged on the cells where magnitude needed no correction, "
     "so the PROJECTION is what fails on this backbone, not the norm rule (C34 refuted)."),
    ("BECAME (Li et al., ICML 2025)", "full",
     "Implemented as a CONTINUAL-LEARNING strategy in the paper's own frame, without gradient "
     "projection, per the paper's Appendix C.3 configuration - see strategy 6, adaptive-lambda "
     "sequential fine-tuning (§1.39). It was first tried as a coefficient inside the MERGING "
     "frame (C21, C22, C24), which measures a frame mismatch rather than the method: BECAME's "
     "merge interpolates two endpoints of one training trajectory, and the merging frame has no "
     "trajectory. That result is the reason it was then implemented properly, not a verdict.",
     "Its total strength is pinned at alpha.n = 1.0 by the convex fold, verified per run (C21). "
     "lambda* instability is not Fisher sampling noise (C24, refuted)."),
    ("BECAME's weighting at a chosen strength (rescaled)", "full",
     "Added in §1.36 to separate weighting from magnitude. NOT BECAME, and never labelled as it.",
     "Confirmed by §1.36's P2, which closed C22: at matched alpha*n the Fisher weighting is "
     "inert - 5 of 7 ties, and neither exception favours it (PSM n=5 +1.36% against a 0.07% "
     "floor). Magnitude was the whole story."),
    ("QOMM", "partial",
     "Only the attention-exclusive fine-tuning half is testable with this backbone; the "
     "quadratic-form outer-product machinery is not implemented.",
     "The testable half does not clear the floor (C25, refuted)."),
    ("AEFT (attention-exclusive fine-tuning)", "full",
     "Run on PSM-forecast n=3, three seeds, with its own geometry report (`geometry_aeft/`). "
     "NOT ETTh1/exchange_rate - that pairing was a bookkeeping error corrected 2026-09-19.",
     "No effect above the floor (C25)."),
    ("Fisher-weighted averaging (non-sequential)", "not run",
     "Subsumed by BECAME, which is the sequential Fisher method and was the one asked for.",
     "-"),
    ("Sequential / continual fine-tuning with L2-SP", "full",
     "Opt-in in StandardTrainer; the early-stopping confound was found and fixed 2026-08-06.",
     "The recorded rejection of L2-SP is WITHDRAWN, not confirmed - cross-lambda comparison was "
     "confounded and must be re-tested before any claim is made (§3.1)."),
    ("Window retraining (W periods of retained history)", "full",
     "W = 1/2/3 on four forecasting datasets, plus honest validation-based budget selection.",
     "Merging is worth 2-4 periods of history, dataset-dependent (C04)."),
    ("Routing / regime indicator", "partial",
     "A per-window ORACLE router only; no buildable router exists, and it cannot be computed on "
     "AD from these runs at all (C17).",
     "Routing's advantage is real in the weakest possible sense - as an upper bound (C16)."),

]


def derive_status(kind: str, n_datasets: int, falsification_tested: str,
                  declared: str) -> tuple[str, str]:
    """(status, prose_action). The downgrade rule is mechanical so it cannot be argued with."""
    if declared == "refuted":
        return "refuted", ""
    if kind == "mechanism" and (n_datasets < 3 or falsification_tested != "yes"):
        reason = ("rests on fewer than 3 datasets" if n_datasets < 3
                  else "never falsification-tested")
        if declared == "hypothesis":
            return "hypothesis", ""
        return "hypothesis", f"DOWNGRADE in prose: mechanism claim that {reason}"
    return declared, ""


# Words that make a sentence a universal claim. A sentence carrying one of these has to name the
# datasets, the segment counts and the floor it was judged against, or it asserts more than any
# table in this project can support - "merging never wins a decisive forecasting configuration"
# was true of 18 cells and false across 48.
QUANTIFIERS = ("never", "always", "every dataset", "no dataset", "universal", "any dataset",
               "all datasets", "in all cases")

# Sentences that legitimately carry a quantifier: method/provenance statements about how the code
# behaves, and sentences that are themselves the scoping. Keyed by a distinctive fragment.
QUANTIFIER_ALLOWED = (
    "never shuffled", "never used for", "never across", "always means", "always p =",
    "never assumed per dataset", "never be varied alone", "never a significance test",
    "never mix them", "never recorded", "never tabulated", "never been", "never entered",
    "never touched", "never writes into", "never combines them", "never contradicted",
    "never from experiment-name patterns", "never reproducible", "never made", "never said so",
    "never identifiable", "universal 2%", "no universal winner", "never varied",
    "never called the paper", "never gets to express it", "cannot be chosen honestly",
    "never labelled as it", "never quietly", "never quoted",
)


def scan_quantifiers(text: str) -> list[tuple[int, str]]:
    """Lines in the documents that assert a universal without visible scope."""
    flagged = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith((">", "|", "```", "#")) or stripped.startswith("- **Provenance"):
            continue
        lowered = line.lower()
        if not any(q in lowered for q in QUANTIFIERS):
            continue
        if any(a.lower() in lowered for a in QUANTIFIER_ALLOWED):
            continue
        flagged.append((number, stripped))
    return flagged


def self_test() -> None:
    """Prove the downgrade rule can fire. A rule that never fires is decoration, and this one
    passed vacuously on its first run until C23's declaration was corrected to match the prose."""
    cases = [
        (("mechanism", 1, "yes", "supported"), "hypothesis", "1 dataset"),
        (("mechanism", 6, "no", "supported"), "hypothesis", "never falsification-tested"),
        (("mechanism", 6, "yes", "supported"), "supported", "well-evidenced mechanism survives"),
        (("measurement", 1, "no", "supported"), "supported", "measurement claims are exempt"),
        (("mechanism", 1, "no", "refuted"), "refuted", "refuted stays refuted"),
    ]
    failures = 0
    for arguments, expected, label in cases:
        status, _ = derive_status(*arguments)
        ok = status == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {arguments} -> {status} "
              f"(wanted {expected})")
    raise SystemExit(1 if failures else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true",
                        help="prove the downgrade rule can fire, then exit")
    parser.add_argument("--scan_docs", type=Path, nargs="*",
                        default=[Path("EXPERIMENTS.md")],
                        help="documents to scan for unscoped universal claims")
    parser.add_argument("--out", type=Path, default=Path("results_archive/audit"))
    parser.add_argument("--experiments", type=Path, default=Path("EXPERIMENTS.md"))
    parser.add_argument("--floors", type=Path,
                        default=Path("results_archive/audit/floors.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        print("CLAIMS REGISTER SELF-TEST")
        self_test()

    # Validate the register against the documents it indexes, so a renamed section or a
    # mistyped dataset is a failure here rather than a dangling reference in a published table.
    text = args.experiments.read_text()
    # Both levels: §1.27a/b/c are `####` subsections, and a register that only knew about
    # `###` would reject a live reference as dangling.
    sections = set(re.findall(r"^#{3,4} (\d+\.\d+[a-z]?)", text, re.M))
    known = {row["dataset"] for row in csv.DictReader(args.floors.open())}
    problems = []
    for claim in CLAIMS:
        for ref in claim[2].split(","):
            ref = ref.strip()
            if ref and ref != "TL;DR" and ref not in sections:
                problems.append(f"{claim[0]}: section {ref} not in {args.experiments}")
        for dataset in filter(None, (d.strip() for d in claim[3].split(","))):
            if dataset not in known:
                problems.append(f"{claim[0]}: dataset {dataset} not in floors.csv")
        if len(list(filter(None, (d.strip() for d in claim[3].split(","))))) != claim[4]:
            problems.append(f"{claim[0]}: n_datasets={claim[4]} but {claim[3]!r} lists "
                            f"{len(list(filter(None, claim[3].split(','))))}")
    if problems:
        for problem in problems:
            log.error("  %s", problem)
        raise SystemExit(f"{len(problems)} register problem(s)")

    rows = []
    for (cid, claim, section, datasets, n_datasets, floor, kind, tested,
         declared, evidence) in CLAIMS:
        status, action = derive_status(kind, n_datasets, tested, declared)
        rows.append({"id": cid, "claim": claim, "section": section, "datasets": datasets,
                     "n_datasets": n_datasets, "seeds": 3, "floor_cleared": floor,
                     "kind": kind, "falsification_tested": tested,
                     "status_declared": declared, "status": status,
                     "prose_action": action, "evidence": evidence})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "claims_register.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "methods_register.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "implemented", "why", "what_was_found"])
        writer.writerows(METHODS)

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    log.info("%d claims: %s", len(rows),
             "  ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    actions = [r for r in rows if r["prose_action"]]
    if actions:
        log.info("\n%d claim(s) the rule downgrades below what the prose declares:", len(actions))
        for row in actions:
            log.info("  %s  %-62s  %s -> %s", row["id"], row["claim"][:62],
                     row["status_declared"], row["status"])
    log.info("\n%d methods: %s", len(METHODS),
             "  ".join(f"{k} {sum(1 for m in METHODS if m[1] == k)}"
                       for k in ("full", "partial", "not run")))
    flagged = []
    for document in args.scan_docs:
        if document.is_file():
            flagged += [(document, n, line) for n, line in scan_quantifiers(document.read_text())]
    with (args.out / "unscoped_universals.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["document", "line", "text"])
        writer.writerows([(str(d), n, line) for d, n, line in flagged])
    log.info("\n%d line(s) assert a universal without visible scope -> %s",
             len(flagged), args.out / "unscoped_universals.csv")
    for document, number, line in flagged:
        log.info("  %s:%d  %s", document, number, line[:110])

    log.info("wrote %s and %s", args.out / "claims_register.csv",
             args.out / "methods_register.csv")


if __name__ == "__main__":
    main()
