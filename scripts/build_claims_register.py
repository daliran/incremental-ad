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
     "falsification test."),
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
     "prediction and its refutation are both published."),
    ("C21", "BECAME cannot reach the strength AD needs because its convex fold pins alpha.n = 1.0",
     "1.31, 1.35", "SWaT,PSM,PSM-forecast", 3, "yes", "mechanism", "yes", "supported",
     "Structural, and verified per run rather than argued: implied_alpha_times_n is emitted by "
     "remerge.py and equals 1.0 on every BECAME row. §1.36 P2 is the falsification test - "
     "rescaling the weights to the committed alpha.n separates weighting from magnitude."),
    ("C22", "Fisher weighting itself contributes nothing beyond setting the merge magnitude",
     "1.36", "SWaT,PSM,PSM-forecast", 3, "yes", "mechanism", "yes", "supported",
     "§1.36's P2 ran the isolating test - BECAME's relative weights rescaled so their sum equals "
     "the source run's committed alpha*n, against uniform 1/n at that same alpha*n through the "
     "same evaluation path - and confirmed it: 5 of 7 cells are ties inside the floor, and "
     "NEITHER exception favours the weighting (PSM n=3 +0.11%, PSM n=5 +1.36% against a 0.07% "
     "floor, i.e. 19x the floor). implied_alpha_times_n equalled the target on every row, so the "
     "comparison really was at matched magnitude. Once magnitude is held fixed the Fisher "
     "weighting contributes nothing, and at larger n it costs."),
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
     "can settle this; what stands is the measurement C32, not the mechanism."),
    ("C31", "OPCM hurts most where there is most base-to-joint headroom",
     "1.35", "ETTh2,ETTm2", 2, "mixed", "mechanism", "no", "hypothesis",
     "§1.35 already labels it 'a hypothesis from six points, not a finding'. Recorded so it is "
     "not later quoted as one."),
    ("C32", "The measurement that survives P3 regardless: OPCM beats plain summation on "
     "exchange_rate at n=2 and n=3, at all three thresholds",
     "1.35", "exchange_rate", 1, "yes", "measurement", "n/a", "supported",
     "-10.88% and -14.18% against a 5.73% floor, three thresholds each. This is what remains if "
     "C23's mechanism is refuted."),
    ("C24", "lambda* is unstable because the Fisher estimate is noisy",
     "1.32", "PSM-forecast", 1, "no", "mechanism", "yes", "refuted",
     "§1.32 ran the falsification test and the estimate saturates at the full pass - the "
     "instability is not sampling noise."),
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
    ("BECAME - Fisher-weighted convex fold", "full",
     "Implemented with diagonal Fishers per shard and lambda* solved per step.",
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
