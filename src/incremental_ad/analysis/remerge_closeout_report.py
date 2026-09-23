"""Collect §1.36's three falsification tests into one table each.

    python -m incremental_ad.analysis.remerge_closeout_report --runs_root $RUNS_ROOT \\
        --remerge_dir $WORK/remerge_closeout --floors results_archive/audit/floors.csv \\
        --out results_archive/audit/remerge_closeout

`remerge_report.py` compares each re-merge against the run's own stored plain merge, which is the
right baseline for "different rule, same strength". Two of the closeout tests need a different
comparison and get their own script rather than a flag, because the *verdict* differs:

- **P2** compares `became_rescaled` against `control_uniform` — 1/n weights at the **same** α·n,
  evaluated through the same code path. The question is whether Fisher *weighting* does anything
  once magnitude is held fixed, so the baseline has to be the same magnitude, not the same rule.
  `implied_alpha_times_n` must equal `committed_alpha * n` on every row; where it does not, the
  rescale did not do what it says and the row is void, not merely surprising.
- **P3** compares `opcm_*_rev` against the **forward** OPCM of the same run from the §1.35 sweep.
  A reversal that changed anything else would show up first in `sum_rev_nullcheck`: plain
  summation is order-inert, so those rows must match the stored merge **exactly**. That check is
  reported before the P3 table and gates it — an inexact null check makes every P3 row
  uninterpretable, so it is a failure rather than a footnote.

Differences smaller than the dataset-and-metric floor are ties, and the raw delta is kept beside
the verdict so a reader sees what was set aside.
"""

import argparse
import csv
import json
import logging
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from incremental_ad.analysis.remerge_provenance import (
    load_result, report_exclusions,
)

log = logging.getLogger("remerge_closeout")

# Tolerance for the order-reversal null check, derived rather than fitted.
#
# Plain summation is order-inert in exact arithmetic; in float32 it is not, because reversing the
# list reorders the additions and `apply_task_vectors` sums the task vectors before scaling. The
# relative error of summing n float32 terms is bounded by about n*eps, eps ~ 1.2e-7, so a few
# parts in 1e-7 is the arithmetic, not the code. 1e-6 sits an order above that and:
#   * four orders BELOW the smallest floor in this project (PSM, 0.07% = 7e-4 relative), so it
#     cannot mask a difference any table would call real; and
#   * five orders below the order effect OPCM actually produces (1.2e-3 absolute in
#     `verify_merge_rules.py`'s fixture), so a genuine reversal bug still fails loudly.
# The worst observed deviation is printed beside the verdict so the margin is visible rather than
# asserted -- if it ever creeps toward the tolerance, that is the signal, not the pass/fail.
NULL_CHECK_RTOL = 1e-6

FIELDS = ["test", "dataset", "n_segments", "metric", "n_seeds", "floor_pct",
          "baseline_label", "baseline", "baseline_sd",
          "variant_label", "variant", "variant_sd", "delta_pct", "verdict",
          "committed_alpha", "target_alpha_times_n", "implied_alpha_times_n",
          "threshold", "opcm_norm_ratio", "distance_ratio", "confounded", "alpha_n_ok",
          "n_seeds_expected", "complete",
          "source_experiment",
          # GRR: the unit §1.11 already uses, so the re-merges can be read against the committed
          # merges without a hand-join. Appended, never inserted — 60+ checks read this file by
          # column name and a shifted column would break every regex at once.
          "base", "joint", "joint_from",
          "grr_baseline", "grr_baseline_sd", "grr_baseline_derived",
          "grr_variant", "grr_variant_sd", "grr_delta", "grr_delta_paired_sd",
          # Matched-seed correction for AD (see `attach_matched_seed`). Appended, like GRR.
          "baseline_matched_seed", "delta_pct_matched_seed", "verdict_matched_seed",
          "matched_seed_n", "verdict_changed"]


# Mirrors scripts/generate_remerge_closeout.py, so "how many cells should exist" is derived from
# the same scope that decided which commands to emit.
BECAME_SCOPE = {("SWaT", None), ("PSM", None), ("PSM-forecast", "3")}
REVERSAL_DATASETS = {"exchange", "ETTh2", "ETTh1", "ETTm2"}


def in_p2_scope(entry: dict) -> bool:
    return ((entry["dataset"], None) in BECAME_SCOPE
            or (entry["dataset"], entry["n"]) in BECAME_SCOPE)


# P4 (coefficient-matched) is interpretable only where the rescale barely moved the merge. The
# projection SHRINKS the task vectors, so matching per-vector coefficients to alpha*n does NOT
# match the distance travelled, and a row whose distance moved a lot is comparing direction AND
# magnitude at once (EXPERIMENTS.md §1.37, and §1.38's P5 is the repair).
#
# The flag is DERIVED from the measurement, not from a dataset list: `opcm_norm_ratio` is the
# merge's distance relative to the paper's own norm-stabilised one, which is the exact quantity
# §1.37 quotes as 0.96-1.15 on the clean cells and 0.18-0.66 on the confounded ones. A row inside
# the band below is a near no-op and is interpretable; outside it, the row is flagged.
#
# In the event the two populations separate on their own -- seed-averaged ratios run 0.195-0.819
# on forecasting and 0.935-1.469 on the AD pair -- so the band is not doing delicate work and the
# flag reproduces §1.37's reading exactly. It also flags two AD rows at threshold 0.3, correctly:
# their rescale moved the merge ~47%, which is not the no-op the other AD cells are. The ratio is
# carried in the same row so a reader can check every call.
NO_OP_BAND = (0.90, 1.20)


def confounded_flag(test: str, norm_ratio) -> str:
    """"coefficient_not_distance_matched" for an uninterpretable P4 row, else "".

    Empty for every other test: P5 is distance-matched by construction, P1 uses the paper's own
    norm rule (so there is no rescale to confound), and P2/P3 apply no transform at all.
    """
    if test != "P4_opcm_committed" or norm_ratio in ("", None):
        return ""
    low, high = NO_OP_BAND
    return "" if low <= float(norm_ratio) <= high else "coefficient_not_distance_matched"


def threshold_of(tag: str) -> float | str:
    """The projection threshold encoded in a tag like ``opcm_paper_t050`` -> 0.5.

    Tags carry it as three digits of hundredths. Returning "" for a tag that has none keeps the
    column blank for the rules that have no threshold (BECAME) rather than inventing a value.
    """
    match = re.search(r"_t(\d{3})$", tag)
    return round(int(match.group(1)) / 100, 2) if match else ""


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower()
               for k in ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))


def load_floors(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("role", "floor") == "floor":
                try:
                    out[(row["dataset"], row["metric"])] = float(row["floor_pct"])
                except (TypeError, ValueError):
                    continue
    return out


EXCLUSIONS: dict[str, int] = defaultdict(int)
LEGACY: dict[str, int] = defaultdict(int)
PENDING: dict[str, int] = defaultdict(int)
SCANNED = [0]


def read_tag(remerge_dir: Path, experiment: str, run: str, tag: str, metric: str,
             require_schema: bool = True):
    """(metric value, payload) for one re-merge, or (None, None) — with the reason recorded.

    A missing directory means the job was never run and is not an exclusion; anything else is,
    and is tallied so `report_exclusions` can print it. Dropping a result without saying so is
    how a table quietly loses a seed, which is the failure this whole module guards against.
    """
    path = remerge_dir / f"{experiment}__{run}" / tag / "result.json"
    if not path.parent.is_dir():
        return None, None
    SCANNED[0] += 1
    payload, reason = load_result(path, require_metric=f"test/{metric}",
                                  require_schema=require_schema)
    if payload is None:
        (PENDING if reason == "absent" else EXCLUSIONS)[f"{reason}  [{tag}]"] += 1
        return None, None
    if payload.get("schema") is None:
        LEGACY[f"unversioned, accepted from the archived §1.35 sweep  [{tag}]"] += 1
    return payload["metrics"][f"test/{metric}"], payload


def stored_merge(run_dir: Path, metric: str):
    path = run_dir / "merged" / "test" / "result.json"
    if not path.is_file():
        return None
    try:
        return (json.loads(path.read_text()) or {}).get("metrics", {}).get(metric)
    except (json.JSONDecodeError, OSError):
        return None


def summarise(test: str, entry: dict, floor, pairs: list[tuple],
              baseline_label: str, variant_label: str, extra: list[dict]) -> dict | None:
    """One row: mean over seeds of baseline and variant, signed so positive = variant is worse."""
    if not pairs:
        return None
    baselines = [p[0] for p in pairs]
    variants = [p[1] for p in pairs]
    base, new = st.mean(baselines), st.mean(variants)
    delta = (100.0 * (base - new) / base if higher_is_better(entry["metric"])
             else 100.0 * (new - base) / base)
    verdict = ("tie (inside floor)" if floor is not None and abs(delta) < floor
               else ("worse" if delta > 0 else "better"))
    row = {
        "test": test, "dataset": entry["dataset"], "n_segments": entry["n"],
        "metric": entry["metric"], "n_seeds": len(pairs),
        "floor_pct": round(floor, 3) if floor is not None else "",
        "baseline_label": baseline_label, "baseline": round(base, 6),
        "baseline_sd": round(st.stdev(baselines), 6) if len(baselines) > 1 else 0.0,
        "variant_label": variant_label, "variant": round(new, 6),
        "variant_sd": round(st.stdev(variants), 6) if len(variants) > 1 else 0.0,
        "delta_pct": round(delta, 3), "verdict": verdict,
        # The per-seed baseline-minus-variant differences, kept because they are PAIRED: both
        # arms come from the same seed's checkpoints, so the difference cancels the run-to-run
        # variance the two share. Its sd is the honest spread of the comparison, and it is
        # smaller than either arm's own sd whenever the two move together. Converted into GRR
        # units by `attach_grr`, which is where base and joint are known.
        "paired_diff_mean": round(st.mean([b - v for b, v in pairs]), 9),
        "paired_diff_sd": (round(st.stdev([b - v for b, v in pairs]), 9)
                           if len(pairs) > 1 else 0.0),
        "committed_alpha": "", "target_alpha_times_n": "", "implied_alpha_times_n": "",
        "threshold": "", "opcm_norm_ratio": "", "distance_ratio": "", "confounded": "",
        "alpha_n_ok": "",
        # Carried IN THE FILE, not only in stdout. A CSV is read long after the run that made it,
        # and "3 rows built from fewer seeds" printed to a terminal does not survive into the
        # archive. Without this, a row averaged over 1 seed is indistinguishable from one averaged
        # over 3, and it is compared against a floor defined on 3 (§1.9).
        "n_seeds_expected": "", "complete": "",
        "source_experiment": entry["merge_experiment"],
    }
    # Per-seed alpha bookkeeping, averaged across seeds — except `alpha_n_ok`, which takes the
    # MINIMUM: one seed that missed its target invalidates the row, and a mean would dilute it to
    # 0.67 and read as "mostly fine".
    collected: dict[str, list] = defaultdict(list)
    for item in extra:
        for key, value in item.items():
            if value is not None:
                collected[key].append(value)
    for key, values in collected.items():
        if not values:
            continue
        row[key] = min(values) if key == "alpha_n_ok" else round(st.mean(values), 6)
    return row


# The one baseline label that means "the run's own stored plain-sum merge". Rows carrying any
# other label are comparing against something else and must not be cross-checked against
# derived.csv's `grr`.
STORED_MERGE_LABEL = "plain sum at committed alpha"

# Covers two things, and the second is NOT rounding. derived.csv stores `grr` to 4 decimals, so
# 5e-5 of any gap is storage rounding. The worst observed gap is 6.9e-5, so roughly 2e-5 is
# something else — most likely float ordering in how the two files pool seeds. Deliberately not
# investigated: it is four orders below the smallest floor in this project and changes no
# verdict. The tolerance covers both, and this comment exists so nobody later reads the whole
# gap as rounding.
GRR_CROSSCHECK_TOL = 1e-4


def attach_grr(rows: list[dict], derived_path: Path | None) -> int:
    """Add GRR columns to every row whose source experiment has a base and a joint.

    **GRR = (base − merged) / (base − joint)** — §0.6's definition, not a new one. The point of
    putting it here is that `remerge_closeout.csv` otherwise carries only raw metric values, so
    "does the sophisticated merge recover more of the base-to-joint gap than plain task
    arithmetic?" could not be answered from the repo without joining two CSVs by hand.

    Paired on **`source_experiment`**, never on (dataset, n_segments): several experiments can
    share a dataset and segment count — `selalpha_etth1_n3` and `segsweep_etth1_merge_n3` both
    exist — and pairing on the coarser key would silently average across them.

    Three sd's, and they are not interchangeable:

    - `grr_baseline_sd` / `grr_variant_sd` are **exact**, not approximations: with base and joint
      held at their means, GRR is affine in the merged value, so sd divides straight through by
      |base − joint|.
    - `grr_delta_paired_sd` comes from the per-seed *differences*, so it cancels the variance the
      two arms share. It is the one to quote for a comparison.

    `grr_baseline_derived` is `derived.csv`'s own `grr` for the same experiment, carried as a
    **cross-check**: the baseline arm IS the stored merge, so the two must agree. They are
    computed by different code from different files, and a disagreement means one of them is
    wrong rather than that the number is uncertain.
    """
    if derived_path is None or not derived_path.is_file():
        return 0
    lookup = {}
    with derived_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("base") and row.get("joint"):
                lookup[(row["experiment"], row["metric"])] = row
    attached = mismatched = 0
    for row in rows:
        source = lookup.get((row["source_experiment"], row["metric"]))
        if source is None:
            continue
        base, joint = float(source["base"]), float(source["joint"])
        gap = base - joint
        if gap == 0:
            continue
        row["base"], row["joint"] = base, joint
        row["joint_from"] = source.get("joint_from", "")
        row["grr_baseline"] = round((base - float(row["baseline"])) / gap, 6)
        row["grr_variant"] = round((base - float(row["variant"])) / gap, 6)
        row["grr_delta"] = round(row["grr_variant"] - row["grr_baseline"], 6)
        row["grr_baseline_sd"] = round(float(row["baseline_sd"]) / abs(gap), 6)
        row["grr_variant_sd"] = round(float(row["variant_sd"]) / abs(gap), 6)
        row["grr_delta_paired_sd"] = round(float(row["paired_diff_sd"]) / abs(gap), 6)
        # ⚠️ The cross-check applies ONLY where the baseline arm really is the stored plain-sum
        # merge. `P3_order_reversal` baselines against the FORWARD-order OPCM instead, so its
        # `grr_baseline` legitimately differs from derived.csv's `grr` — comparing them fired on
        # all 12 forecasting experiments and the data was right every time. `grr_baseline_derived`
        # is left blank on those rows rather than filled with a number that means something else.
        stored = row["baseline_label"] == STORED_MERGE_LABEL
        row["grr_baseline_derived"] = source.get("grr", "") if stored else ""
        if stored and source.get("grr"):
            # ⚠️ The EFFECTIVE tolerance of this cross-check is set by `derived.csv`, which
            # stores `grr` to 4 decimals: two values agreeing here can still differ by up to
            # 5e-5 before rounding, so the check cannot detect an error below ~1e-4. This file
            # now stores 6 decimals so it is not itself the limit. "All rows agree" therefore
            # means "agree to 1e-4", and the section that quotes it says so.
            if abs(float(source["grr"]) - row["grr_baseline"]) > GRR_CROSSCHECK_TOL:
                mismatched += 1
                log.warning("[grr] %s %s: baseline arm gives %.4f but derived.csv says %s — "
                            "one of the two is wrong", row["source_experiment"], row["metric"],
                            row["grr_baseline"], source["grr"])
        attached += 1
    if mismatched:
        log.warning("[grr] %d row(s) disagree with derived.csv", mismatched)
    else:
        log.info("[grr] %d row(s) carry GRR; every stored-merge baseline reproduces "
                 "derived.csv's grr", attached)
    return attached


# Tests whose baseline arm is the run's STORED merge metric rather than a re-scored one.
STORED_BASELINE_TESTS = {"P1_paper_opcm", "P4_opcm_committed", "P5_opcm_distance"}
AD_METRICS = {"window_auroc", "window_auprc"}


def attach_matched_seed(rows: list[dict], rescored_dir: Path | None, runs_root: Path) -> int:
    """Re-read AD comparisons with BOTH arms scored under the same evaluation seed.

    ⚠️ **The defect this corrects.** The pipeline scores the stored merge with
    ``eval_seed = seed + 1`` (`framework/experiment.py`); `remerge.py` scored every variant with
    ``seed``. AD scoring averages ``n_eval_passes`` RANDOM masks, so for every AD row whose
    baseline is the stored merge (P1, P4) the two arms were scored under different mask draws.
    Forecasting scoring is deterministic and is unaffected, which is also why the P3 null check
    — run on forecasting datasets only — could never have seen it.

    **The correction costs no run.** §1.40's sweep re-scores task arithmetic at the committed
    alpha through `remerge.py`, i.e. the stored merge's exact model (bitwise, by the self-check;
    every AD run committed alpha = 1.0) under ``seed`` — the variant's seed. So its ``ta_a1.00``
    result IS the matched-seed baseline. The published columns are left as they were and the
    corrected ones are added beside them, with ``verdict_changed`` so a flip cannot hide.
    """
    if rescored_dir is None or not rescored_dir.is_dir():
        return 0
    corrected = 0
    for row in rows:
        if row["test"] not in STORED_BASELINE_TESTS or row["metric"] not in AD_METRICS:
            continue
        group = runs_root / row["source_experiment"]
        values = []
        for run in sorted(p for p in group.iterdir() if p.is_dir()) if group.is_dir() else []:
            payload, _why = load_result(
                rescored_dir / f"{row['source_experiment']}__{run.name}" / "ta_a1.00"
                / "result.json", require_metric=f"test/{row['metric']}")
            if payload is not None and abs(float(payload.get("committed_alpha", 1.0)) - 1.0) < 1e-9:
                values.append(payload["metrics"][f"test/{row['metric']}"])
        if len(values) != int(row["n_seeds"]):
            continue
        base = st.mean(values)
        variant = float(row["variant"])
        # Same sign convention as `summarise`: positive = the variant is worse.
        delta = 100.0 * (base - variant) / base
        floor = float(row["floor_pct"]) if row["floor_pct"] != "" else None
        verdict = ("tie (inside floor)" if floor is not None and abs(delta) < floor
                   else ("worse" if delta > 0 else "better"))
        row.update({"baseline_matched_seed": round(base, 6),
                    "delta_pct_matched_seed": round(delta, 3),
                    "verdict_matched_seed": verdict, "matched_seed_n": len(values),
                    "verdict_changed": verdict != row["verdict"]})
        corrected += 1
    flips = [r for r in rows if r.get("verdict_changed") is True]
    log.info("[matched-seed] %d AD row(s) re-read with both arms at one eval seed; %d verdict(s) "
             "changed%s", corrected, len(flips),
             "" if not flips else ": " + ", ".join(
                 f"{r['test']} {r['dataset']} n={r['n_segments']} {r['variant_label']}"
                 for r in flips[:8]))
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--rescored_dir", type=Path,
                        help="§1.40's per-run grid (remerge.py --baseline_rule): its ta_a1.00 "
                             "results are the stored merge re-scored at the variant's eval seed")
    parser.add_argument("--derived", type=Path,
                        help="derived.csv, for base/joint so GRR can be emitted per row")
    parser.add_argument("--per_seed", type=Path,
                        help="accepted for symmetry with the other reports; the paired spread "
                             "this file needs comes from the per-seed pairs already in hand")
    parser.add_argument("--remerge_dir", type=Path, required=True,
                        help="output of the closeout sweep")
    parser.add_argument("--forward_dir", type=Path,
                        help="output of the §1.35 sweep, for P3's forward-order OPCM baseline; "
                             "defaults to --remerge_dir")
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--floors", type=Path, required=True)
    parser.add_argument("--forward_tag", default="opcm_t050")
    parser.add_argument("--paper_thresholds", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    parser.add_argument("--reverse_tag", default="opcm_t050_rev")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    forward_dir = args.forward_dir or args.remerge_dir

    def source_for(tag: str) -> Path:
        """The §1.35 sweep wrote the simplified OPCM; the closeout sweep wrote everything else."""
        return forward_dir if tag == args.forward_tag else args.remerge_dir

    floors = load_floors(args.floors)
    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    rows: list[dict] = []
    null_ok, null_bad, null_absent, null_worst = 0, [], 0, [0.0]
    expected_seeds: dict[str, int] = {}
    # Rows ATTEMPTED per test. The per-row `complete` flag says whether a row that exists used all
    # its seeds; it cannot say a row is MISSING altogether, because there is no row to carry the
    # flag. Without this, a P1 table holding 72 of its 96 (dataset, n, variant) cells reports
    # "72/72 at full seed count" and reads as finished.
    attempted: dict[str, int] = defaultdict(int)

    for entry in spec:
        experiment = entry["merge_experiment"]
        group = args.runs_root / experiment
        if not group.is_dir():
            continue
        floor = floors.get((entry["dataset"], entry["metric"]))
        if floor is None:
            try:
                floor = float(entry["floor_pct"])
            except (TypeError, ValueError, KeyError):
                floor = None
        runs = sorted(p for p in group.iterdir() if p.is_dir())
        expected_seeds[experiment] = sum(
            1 for r in runs if stored_merge(r, entry["metric"]) is not None
        )

        # --- the null check first: it gates P3 -------------------------------------------
        for run in runs:
            value, _ = read_tag(args.remerge_dir, experiment, run.name,
                               "sum_rev_nullcheck", entry["metric"])
            if value is None:
                continue
            published = stored_merge(run, entry["metric"])
            if published is None:
                null_absent += 1
            elif abs(published - value) <= NULL_CHECK_RTOL * abs(published):
                null_ok += 1
                null_worst[0] = max(null_worst[0], abs(published - value) / abs(published))
            else:
                null_bad.append((entry["dataset"], entry["n"], run.name, published, value))

        # --- P1: the paper's OPCM, at each threshold, against plain summation ------------
        #
        # The baseline is the run's stored plain-sum merge at the strength it committed to. The
        # two are NOT at the same magnitude and cannot be made so: OPCM's lambda pins it at the
        # mean task-vector norm (Thm 5.2), which is why `implied_alpha_times_n` is carried into
        # every row -- the reader has to be able to see that a loss may be a magnitude effect
        # rather than a projection effect. The simplified operator's row for the same cell is
        # emitted under the same `test` name with its own variant label, so the two sit side by
        # side without ever being summed or averaged together.
        for tag, label in ([(f"opcm_paper_t{int(x * 100):03d}", f"paper OPCM alpha={x}")
                            for x in args.paper_thresholds]
                           + [(args.forward_tag, "simplified OPCM (1.35)")]):
            pairs, extra = [], []
            for run in runs:
                variant, payload = read_tag(source_for(tag), experiment, run.name, tag,
                                            entry["metric"],
                                            require_schema=(tag != args.forward_tag))
                plain = stored_merge(run, entry["metric"])
                if variant is None or plain is None:
                    continue
                pairs.append((plain, variant))
                extra.append({
                    "committed_alpha": payload.get("alpha"),
                    "implied_alpha_times_n": payload.get("implied_alpha_times_n"),
                    "opcm_norm_ratio": payload.get("opcm_norm_ratio"),
                })
            attempted["P1_paper_opcm"] += 1
            row = summarise("P1_paper_opcm", entry, floor, pairs,
                            "plain sum at committed alpha", label, extra)
            if row:
                row["threshold"] = "" if tag == args.forward_tag else threshold_of(tag)
                # Thm 5.2 must hold on every real merge, not only on the fixture. A row where it
                # does not is not the paper's operator and is void rather than surprising.
                ratio = row.get("opcm_norm_ratio", "")
                row["alpha_n_ok"] = ("" if ratio in ("", None)
                                     else int(abs(float(ratio) - 1.0) < 1e-6))
                rows.append(row)

        # --- P4 (§1.37): the paper's PROJECTION at the run's committed magnitude ------------
        #
        # Baseline is the stored plain-sum merge at that same committed alpha, so the two differ
        # only by the projection -- which is the whole point: P1 could not separate the projection
        # from the Thm-5.2 norm rule, and this can.
        for threshold in args.paper_thresholds:
            tag = f"opcm_committed_t{int(threshold * 100):03d}"
            pairs, extra = [], []
            for run in runs:
                variant, payload = read_tag(args.remerge_dir, experiment, run.name, tag,
                                            entry["metric"])
                plain = stored_merge(run, entry["metric"])
                if variant is None or plain is None:
                    continue
                pairs.append((plain, variant))
                alpha = payload.get("alpha")
                extra.append({
                    "committed_alpha": alpha,
                    "target_alpha_times_n": (alpha * payload.get("n_shards", 0)
                                             if alpha is not None else None),
                    "implied_alpha_times_n": payload.get("implied_alpha_times_n"),
                    # Needed to derive `confounded`; see NO_OP_BAND.
                    "opcm_norm_ratio": payload.get("opcm_norm_ratio"),
                })
            attempted["P4_opcm_committed"] += 1
            row_threshold = threshold
            row = summarise("P4_opcm_committed", entry, floor, pairs,
                            "plain sum at committed alpha",
                            f"paper projection at committed alpha, thr={threshold}", extra)
            if row:
                row["threshold"] = row_threshold
                target, implied = row["target_alpha_times_n"], row["implied_alpha_times_n"]
                row["alpha_n_ok"] = ("" if "" in (target, implied)
                                     else int(abs(float(implied) - float(target)) < 1e-6))
                rows.append(row)

        # --- P5 (§1.38): the paper's projection at matched DISTANCE, forecasting only -------
        #
        # Same baseline as P4 -- the stored plain-sum merge at the committed alpha -- but now the
        # variant travels exactly as far from theta_0, so the two differ only in DIRECTION. P4's
        # coefficient rescale left these merges at 0.18-0.66x that distance, which is the confound
        # this replaces. AD is absent by design: C34 is already settled there.
        for threshold in args.paper_thresholds:
            tag = f"opcm_distance_t{int(threshold * 100):03d}"
            pairs, extra = [], []
            for run in runs:
                variant, payload = read_tag(args.remerge_dir, experiment, run.name, tag,
                                            entry["metric"])
                plain = stored_merge(run, entry["metric"])
                if variant is None or plain is None:
                    continue
                pairs.append((plain, variant))
                alpha = payload.get("alpha")
                extra.append({
                    "committed_alpha": alpha,
                    "target_alpha_times_n": (alpha * payload.get("n_shards", 0)
                                             if alpha is not None else None),
                    "implied_alpha_times_n": payload.get("implied_alpha_times_n"),
                    "distance_ratio": payload.get("opcm_distance_ratio"),
                })
            if pairs:
                attempted["P5_opcm_distance"] += 1
            row_threshold = threshold
            row = summarise("P5_opcm_distance", entry, floor, pairs,
                            "plain sum at committed alpha",
                            f"paper projection at matched distance, thr={threshold}", extra)
            if row:
                row["threshold"] = row_threshold
                ratio = row.get("distance_ratio", "")
                # The identity is the control. A row that missed it is not evidence.
                row["alpha_n_ok"] = ("" if ratio in ("", None)
                                     else int(abs(float(ratio) - 1.0) < 1e-6))
                rows.append(row)

        # --- P2: BECAME's weighting at a chosen strength, against 1/n at the same strength --
        pairs, extra = [], []
        for run in runs:
            variant, payload = read_tag(args.remerge_dir, experiment, run.name,
                                        "became_rescaled", entry["metric"])
            control, _ = read_tag(args.remerge_dir, experiment, run.name,
                                  "control_uniform", entry["metric"])
            if variant is None or control is None:
                continue
            pairs.append((control, variant))
            alpha = payload.get("alpha")
            target = alpha * payload.get("n_shards", 0) if alpha is not None else None
            implied = payload.get("implied_alpha_times_n")
            extra.append({"committed_alpha": alpha, "target_alpha_times_n": target,
                          "implied_alpha_times_n": implied,
                          "alpha_n_ok": int(implied is not None and target is not None
                                            and abs(implied - target) < 1e-6)})
        if in_p2_scope(entry):
            attempted["P2_became_rescaled"] += 1
        row = summarise("P2_became_rescaled", entry, floor, pairs,
                        "uniform 1/n at same alpha*n", "BECAME weighting at same alpha*n", extra)
        if row:
            rows.append(row)

        # --- P3: reversed order against forward order, same rule and strength ---------------
        pairs = []
        for run in runs:
            reverse, _ = read_tag(args.remerge_dir, experiment, run.name,
                                  args.reverse_tag, entry["metric"])
            forward, _ = read_tag(forward_dir, experiment, run.name,
                                  args.forward_tag, entry["metric"], require_schema=False)
            if reverse is None or forward is None:
                continue
            pairs.append((forward, reverse))
        if entry["dataset"] in REVERSAL_DATASETS:
            attempted["P3_order_reversal"] += 1
        row = summarise("P3_order_reversal", entry, floor, pairs,
                        f"OPCM forward ({args.forward_tag})",
                        f"OPCM reversed ({args.reverse_tag})", [])
        if row:
            rows.append(row)

    # Report exclusions BEFORE giving up on an empty table. "no outputs found — check
    # --remerge_dir" is a lie when the directory is full and every file was rejected, and it
    # sends the reader to the wrong problem.
    waiting = sum(PENDING.values())
    if waiting:
        log.info("provenance: %d re-merge(s) not yet on disk — the sweep is still running",
                 waiting)
    excluded = report_exclusions(log, EXCLUSIONS, SCANNED[0] - waiting)
    for reason, count in sorted(LEGACY.items()):
        log.info("provenance: %d result(s) %s", count, reason)
    if not rows:
        raise SystemExit(
            f"no usable closeout outputs: {SCANNED[0]} result(s) scanned, {excluded} excluded "
            f"for the reasons above" if SCANNED[0]
            else "no closeout outputs found — check --remerge_dir"
        )

    # A cell built from fewer seeds than its source group has runs is not wrong, but it is not
    # what it looks like either. §1.9's floors are defined on three seeds, so a two-seed cell is
    # compared against a threshold derived from a different n and nothing in the row says so.
    for row in rows:
        row["confounded"] = confounded_flag(row["test"], row.get("opcm_norm_ratio", ""))
        expected = expected_seeds.get(row["source_experiment"], 0)
        row["n_seeds_expected"] = expected
        row["complete"] = int(expected > 0 and row["n_seeds"] >= expected)
    short = [r for r in rows if r["n_seeds"] < expected_seeds.get(r["source_experiment"], 0)]
    if short:
        log.warning("⚠️  %d row(s) built from fewer seeds than the source group has runs:", len(short))
        for row in short:
            log.warning("      %-15s n=%-2s %-34s %d of %d seeds", row["dataset"],
                        row["n_segments"], row["variant_label"], row["n_seeds"],
                        expected_seeds[row["source_experiment"]])

    log.info("\nNULL CHECK — plain summation under --reverse_order must equal the stored merge")
    log.info("  %d within %.0e relative (worst %.2e), %d mismatched, %d with no stored merge",
             null_ok, NULL_CHECK_RTOL, null_worst[0], len(null_bad), null_absent)
    for dataset, n, run, published, value in null_bad:
        log.error("  MISMATCH %s n=%s %s: stored %.9f vs reversed %.9f",
                  dataset, n, run, published, value)
    if null_bad:
        log.error("  ⚠️  the reversal changes something other than the order — every P3 row below "
                  "is uninterpretable until this is fixed")

    # Per-test completeness, so a partially-collected sweep cannot be read as a finished one.
    log.info("")
    for test in ("P1_paper_opcm", "P4_opcm_committed", "P5_opcm_distance",
                 "P2_became_rescaled", "P3_order_reversal"):
        subset = [r for r in rows if r["test"] == test]
        want = attempted.get(test, 0)
        if subset or want:
            done = sum(r["complete"] for r in subset)
            finished = done == len(subset) == want
            log.info("%-20s %2d/%2d cell(s) present, %d at full seed count%s",
                     test, len(subset), want, done,
                     "" if finished else "  <-- INCOMPLETE, do not publish as final")

    for test in ("P1_paper_opcm", "P4_opcm_committed", "P5_opcm_distance",
                 "P2_became_rescaled", "P3_order_reversal"):
        subset = [r for r in rows if r["test"] == test]
        if not subset:
            continue
        log.info("\n%s", test)
        log.info("  %-15s %-3s %10s %10s %9s  %-18s %s", "dataset", "n", "baseline", "variant",
                 "delta", "verdict", "variant")
        for row in subset:
            log.info("  %-15s %-3s %10.4f %10.4f %+8.2f%%  %-18s %s",
                     row["dataset"], row["n_segments"], row["baseline"], row["variant"],
                     row["delta_pct"], row["verdict"], row["variant_label"])
        ties = sum(1 for r in subset if r["verdict"].startswith("tie"))
        log.info("  %d comparisons, %d ties", len(subset), ties)

    bad_alpha = [r for r in rows if r["alpha_n_ok"] not in ("", 1, 1.0)]
    if bad_alpha:
        log.error("\n⚠️  %d rescaled row(s) did NOT land on the requested alpha*n — the rescale is "
                  "wrong and those rows are void: %s", len(bad_alpha),
                  [(r["dataset"], r["n_segments"], r["implied_alpha_times_n"],
                    r["target_alpha_times_n"]) for r in bad_alpha[:5]])
    elif any(r["alpha_n_ok"] != "" for r in rows):
        log.info("\nRescaled rows: implied alpha*n equals the committed alpha x n on every one")

    attach_grr(rows, args.derived)
    attach_matched_seed(rows, args.rescored_dir, args.runs_root)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "remerge_closeout.csv"
        with path.open("w", newline="") as fh:
            # `extrasaction="ignore"` drops the per-seed difference columns, which are working
            # state for `attach_grr` rather than published quantities: what they support is
            # `grr_delta_paired_sd`, and emitting both would invite reading the raw difference
            # as if it were in GRR units.
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with (args.out / "reversal_nullcheck.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dataset", "n", "run", "stored_merge", "reversed_sum", "status"])
            for dataset, n, run, published, value in null_bad:
                writer.writerow([dataset, n, run, published, value, "MISMATCH"])
            writer.writerow(["", "", "", "", "", f"{null_ok} exact"])
        log.info("wrote %s", path)
    raise SystemExit(1 if (null_bad or bad_alpha or excluded) else 0)


if __name__ == "__main__":
    main()
