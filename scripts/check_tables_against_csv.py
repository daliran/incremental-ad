"""Assert that numbers written into EXPERIMENTS.md still match the generated CSVs.

    python scripts/check_tables_against_csv.py --audit_dir $WORK/audit_2026_08_07
    python scripts/check_tables_against_csv.py --audit_dir ... --strict   # exit 1 on drift

**Why this exists.** The CSVs under `audit_*/` are generated; the markdown is hand-transcribed.
Every documentation error found in the August 2026 audit lived in that gap: a value was correct
when written, the underlying computation later changed or was re-derived differently, and
nothing connected the two. The only defence was somebody recomputing by hand and noticing —
which is how one error survived long enough to spread into three documents.

This is deliberately a *spot* checker, not a parser of every table. Parsing arbitrary markdown
tables and guessing which CSV column each belongs to is guesswork that fails silently; instead
each check below names a document claim, the CSV cell that backs it, and a tolerance. Adding a
claim is three lines, and an unbacked claim is visible by its absence.

Run it after any re-audit, and before committing a change to a published number.
"""

import argparse
import copy
import csv
import re
import sys
from pathlib import Path

def _to_float(text: str) -> float:
    """Parse a number as written in the document.

    Prose uses U+2212 MINUS SIGN, not ASCII hyphen, so a negative value like the pooled
    correlation "−0.03" fails float() outright. Normalise before parsing, and keep the sign
    inside the captured group so that a sign flip registers as drift rather than a match.
    """
    return float(text.replace("\u2212", "-").replace("\u2013", "-"))


# (label, regex capturing one number from EXPERIMENTS.md, csv file, row filter, column, tol)
# The regex must capture exactly one group: the number as written in the document.
CHECKS = [
    ("§0.1b ETTh2 headroom", r"\*\*ETTh2\*\* \| forecasting \|(?:[^|]*\|){7} ([\d.]+)%",
     "derived.csv", {"experiment": "etth2_gate_base"}, "headroom_pct", 0.15),
    ("§0.1b ETTm2 headroom", r"\*\*ETTm2\*\* \| forecasting \|(?:[^|]*\|){7} ([\d.]+)%",
     "derived.csv", {"experiment": "ettm2_gate_base"}, "headroom_pct", 0.15),
    ("§1.9 ETTh2 floor", r"\*\*ETTh2\*\* \(forecast/mse\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "etth2_gate_base"}, "floor_pct", 0.05),
    ("§1.9 ETTm2 floor", r"\*\*ETTm2\*\* \(forecast/mse\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "ettm2_gate_base"}, "floor_pct", 0.05),
    # The remaining four §1.9 floors, each bound to the experiment named in
    # analysis_specs/floor_spec.csv. exchange_rate's published 5.29% derived from no experiment
    # at all and survived for months precisely because nothing checked it.
    ("§1.9 SWaT floor", r"\*\*SWaT\*\* \(window_auroc\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "noisefloor_swat"}, "floor_pct", 0.005),
    ("§1.9 PSM floor", r"\*\*PSM\*\* \(window_auroc\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "noisefloor_psm"}, "floor_pct", 0.005),
    ("§1.9 ETTh1 floor", r"\*\*ETTh1\*\* \(forecast/mse\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "noisefloor_etth"}, "floor_pct", 0.05),
    ("§1.9 Exchange floor", r"\*\*Exchange\*\* \(forecast/mse\) \| 3 \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)%\*\*",
     "derived.csv", {"experiment": "n1_exchange"}, "floor_pct", 0.05),
    # §0.1b repeats the same six floors in its overview row. They drifted apart before, so pin
    # both columns to the one source rather than to each other.
    ("§0.1b SWaT floor", r"\*\*SWaT\*\* \| anomaly detection \|(?:[^|]*\|){6} ([\d.]+)%",
     "derived.csv", {"experiment": "noisefloor_swat"}, "floor_pct", 0.005),
    ("§0.1b PSM floor", r"\*\*PSM\*\* \| anomaly detection \|(?:[^|]*\|){6} ([\d.]+)%",
     "derived.csv", {"experiment": "noisefloor_psm"}, "floor_pct", 0.005),
    ("§0.1b ETTh1 floor", r"\*\*ETTh1\*\* \| forecasting \|(?:[^|]*\|){6} ([\d.]+)%",
     "derived.csv", {"experiment": "noisefloor_etth"}, "floor_pct", 0.05),
    ("§0.1b exchange floor", r"\*\*exchange_rate\*\* \| forecasting \|(?:[^|]*\|){6} ([\d.]+)%",
     "derived.csv", {"experiment": "n1_exchange"}, "floor_pct", 0.05),
    ("§0.1b SWaT headroom", r"\*\*SWaT\*\* \| anomaly detection \|(?:[^|]*\|){7} ([\d.]+)%",
     "derived.csv", {"experiment": "segsweep_swat_merge_n5"}, "headroom_pct", 0.05),
    # §1.18's α*·n(out) column. ETTh1 n=3 was documented as 1.20 for months while
    # `noisefloor_etth_diagnostics` — the group α* itself is read from — gives 1.10; the error
    # inverted that dataset's shape from "monotone rising" to a dip. Pin the whole column.
    ("§1.18 ETTh1 α*·n out n=3", r"\| ETTh1 \| 3 \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
     "scale_forecast/scale_summary.csv", {"group": "noisefloor_etth_diagnostics"},
     "alpha_excl_n", 0.02),
    ("§1.18 ETTh1 α*·n out n=5", r"\| ETTh1 \| 5 \| [\d.]+ \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "scale_forecast/scale_summary.csv", {"group": "segsweep_etth1_merge_n5_diagnostics"},
     "alpha_excl_n", 0.02),
    ("§1.18 PSM α*·n out n=5", r"\| PSM \| 5 \| [\d.]+ \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "scale_ad/scale_summary.csv", {"group": "segsweep_psm_merge_n5_diagnostics"},
     "alpha_excl_n", 0.02),
    ("§1.18 SWaT α*·n out n=3", r"\| SWaT \| 3 \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
     "scale_ad/scale_summary.csv", {"group": "noisefloor_swat_diagnostics"},
     "alpha_excl_n", 0.02),
    ("§1.18 ETTm2 α*·n out n=5", r"\| ETTm2 \| 5 \| [\d.]+ \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "scale_forecast/scale_summary.csv", {"group": "ettm2_merge_n5_diagnostics"},
     "alpha_excl_n", 0.02),
    # The within/between correlation split, which replaced a pooled r that moved from +0.44 to
    # +0.31 to -0.03 as datasets were added. Sign matters more than magnitude here.
    ("§1.18 within-dataset r", r"\| dataset-centred \(within, 18 points\) \| \*\*\+([\d.]+)\*\*",
     "alignment/alignment_correlation.csv", {"statistic": "within_r"}, "value", 0.02),
    ("§1.18 pooled r", r"\| pooled across all 18 points \| \*\*(−?[\d.]+)\*\*",
     "alignment/alignment_correlation.csv", {"statistic": "pooled_r"}, "value", 0.02),
    # The n=2 cell read "—" until those diagnostics landed; accept either form so filling the
    # table in does not silently disable the n=5 check (it did, and only --strict caught it).
    ("§1.16 ETTh2 routing n=5",
     r"\*\*ETTh2\*\* \| 0\.753 \| 1,393 \| (?:—|\*\*\+[\d.]+%\*\*) \| \*\*\+[\d.]+%\*\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "etth2_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTm2 routing n=5",
     r"\*\*ETTm2\*\* \| 0\.752 \| 5,574 \| (?:—|\*\*\+[\d.]+%\*\*) \| \*\*\+[\d.]+%\*\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "ettm2_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTh2 routing n=2", r"\*\*ETTh2\*\* \| 0\.753 \| 1,393 \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "etth2_merge_n2_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTm2 routing n=2", r"\*\*ETTm2\*\* \| 0\.752 \| 5,574 \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "ettm2_merge_n2_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTh1 routing n=5", r"ETTh1 \| 0\.412 \| 1,393 \| \+[\d.]+% \| \*\(α=1 only\)\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "segsweep_etth1_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
]

# Whole-row expansion --------------------------------------------------------------------
#
# Hand-writing one regex per cell is where this file's own bugs came from: a pattern that
# counted columns wrong silently grabbed the neighbouring value and still "passed". For tables
# that are a plain grid of numbers, describe the row once and generate a check per cell.

_CELL = r"\*{0,2}[\d.]+%?\*{0,2}"
_CAP = r"\*{0,2}([\d.]+)%?\*{0,2}"


def row_checks(section, row_label, picks, rel, column, tol, cell=_CELL, cap=_CAP):
    """One check per selected cell of a markdown row.

    `row_label` is the literal leading cell (regex-escaped by the caller if needed); `picks`
    maps a 0-based numeric-cell index to the CSV row filter for that cell. Cells may be bold
    and may carry a trailing %, both of which are tolerated. `cell`/`cap` override the cell
    shape for tables whose cells are not bare numbers (§1.13's are `winner (±x.xx%)`).
    """
    out = []
    for index, (suffix, where) in sorted(picks.items()):
        pattern = row_label + r" \| " + "".join(f"{cell} \\| " for _ in range(index)) + cap
        out.append((f"{section} {suffix}", pattern, rel, where, column, tol))
    return out


# §1.11's α* and α*·n grid: `| **SWaT** | α*(2) | α*(3) | α*(5) | α*n(2) | α*n(3) | α*n(5) |`.
# Both halves come from the same scale_summary rows, so a column-offset error would show up as
# α* being compared against α*·n — which is exactly what the cell indices below pin down.
_SCALE_GROUPS = {
    "SWaT": ("scale_ad/scale_summary.csv", ["segsweep_swat_merge_n2_diagnostics",
                                            "noisefloor_swat_diagnostics",
                                            "segsweep_swat_merge_n5_diagnostics"]),
    "PSM": ("scale_ad/scale_summary.csv", ["segsweep_psm_merge_n2_diagnostics",
                                           "noisefloor_psm_diagnostics",
                                           "segsweep_psm_merge_n5_diagnostics"]),
    "ETTh1": ("scale_forecast/scale_summary.csv", ["segsweep_etth1_merge_n2_diagnostics",
                                                   "noisefloor_etth_diagnostics",
                                                   "segsweep_etth1_merge_n5_diagnostics"]),
    "exchange": ("scale_forecast/scale_summary.csv", ["segsweep_exchange_merge_n2_diagnostics",
                                                      "exch_incremental_diagnostics",
                                                      "segsweep_exchange_merge_n5_diagnostics"]),
}
for _ds, (_rel, _groups) in _SCALE_GROUPS.items():
    for _pos, _n in enumerate((2, 3, 5)):
        _where = {"group": _groups[_pos]}
        CHECKS += row_checks("§1.11", rf"\| \*\*{_ds}\*\*",
                             {_pos: (f"{_ds} α* n={_n}", _where)}, _rel, "alpha_star", 0.02)
        CHECKS += row_checks("§1.11", rf"\| \*\*{_ds}\*\*",
                             {_pos + 3: (f"{_ds} α*·n n={_n}", _where)}, _rel,
                             "alpha_star_n", 0.02)

# §1.12's honest-α cost: `| **SWaT** | 96% | 97% | 98% | seeds | verdict |`. The reconciliation
# block below cross-checks these against GRR; this pins them to the CSV they are read from.
for _ds, (_rel, _groups) in _SCALE_GROUPS.items():
    for _pos, _n in enumerate((2, 3, 5)):
        CHECKS += row_checks("§1.12", rf"\| \*\*{_ds}\*\*",
                             {_pos: (f"{_ds} honest-α cost n={_n}", {"group": _groups[_pos]})},
                             _rel, "honest_alpha_cost_pct", 1.0)



# §1.13 `| **ETTh1** | sequential (−12.73%) | ... |` — the magnitude is outcomes.csv:margin_pct;
# the sign carries the winner, checked separately below.
_M_CELL = r"\w+ \([+−-][\d.]+%\) ?†?"   # a cell may carry a trailing footnote dagger
_M_CAP = r"\w+ \([+−-]([\d.]+)%\)"
for _ds in ("SWaT", "PSM", "ETTh1", "ETTh2", "ETTm2", "exchange"):
    for _pos, _n in enumerate((2, 3, 5)):
        CHECKS += row_checks("§1.13", rf"\| \*\*{_ds}\*\*",
                             {_pos: (f"{_ds} margin n={_n}", {"dataset": _ds, "n": str(_n)})},
                             "outcomes.csv", "margin_pct", 0.02, cell=_M_CELL, cap=_M_CAP)

# §1.26 `| ETTh1 | 2 | joint | merge | sequential | window | best | margin |`. Every numeric
# column, because this table is the one that ranks the five strategies against each other.
for _ds in ("ETTh1", "ETTh2", "ETTm2", "exchange", "SWaT", "PSM"):
    for _n in (2, 3, 5):
        _where = {"dataset": _ds, "n": str(_n)}
        for _pos, _col in enumerate(("joint", "merge", "sequential", "window_best")):
            CHECKS += row_checks("§1.26", rf"\| {_ds} \| {_n}",
                                 {_pos: (f"{_ds} n={_n} {_col}", _where)},
                                 "methods/method_comparison.csv", _col, 0.001)


# §1.21 `| SWaT | base | W=1 | W=2 | W=3 | W=5 | best merge |`. The base and W=1..3 columns are
# `window_<ds>_W<k>` blocks in run_metrics.csv. The W=5 and "best merge" columns are NOT pinned:
# W=5 does not map to a single experiment (n1_* matches ETTh1 but not SWaT/PSM/exchange) and
# "best merge" is a per-dataset argmin over experiments. Both are still unverified.
_WINDOW = {"SWaT": ("window_swat", "window_auroc"), "PSM": ("window_psm", "window_auroc"),
           "ETTh1": ("window_etth1", "forecast/mse"),
           "exchange_rate": ("window_exchange", "forecast/mse")}
# The `base` column is bound for the forecasting datasets only. On SWaT and PSM the documented
# value matches neither `window_*_W1/baseline/test` (0.8001 / 0.7756) nor any single consistent
# experiment — SWaT's 0.8013 matches `slurm_grid_swat_train_incremental`, PSM's 0.7746 matches
# `noisefloor_psm`. Binding them would mean picking whichever run reproduces the printed number,
# which is backwards. Recorded in EXECUTION_PLAN.md §3.13 instead.
for _ds, (_stem, _metric) in _WINDOW.items():
    if _ds in ("ETTh1", "exchange_rate"):
        CHECKS += row_checks("§1.21", rf"\| {_ds}",
                             {0: (f"{_ds} base", {"experiment": f"{_stem}_W1",
                                                  "block": "baseline/test", "metric": _metric})},
                             "run_metrics.csv", "mean", 0.001)
    # Only W=3 can be bound: `method_comparison.csv` carries the winning window as
    # `window_best`, and W=3 wins on all six datasets. W=1 and W=2 are read from
    # `finetune_0/test`, a block `results_audit` does not emit (EXECUTION_PLAN.md §3.13), so
    # they stay unverified rather than bound to the wrong block — `merged/test` looks right on
    # ETTh1 only because at W=1 the merged model *is* finetune_0.
    # PSM is excluded: §1.21 prints 0.7952 where §1.26 prints 0.7987 for the same quantity —
    # the best window model on PSM — a 0.0035 gap against a 0.0005 floor. See §3.13.
    if _ds != "PSM":
        CHECKS += row_checks("§1.21", rf"\| {_ds}",
                             {3: (f"{_ds} W=3", {"dataset": "exchange" if _ds == "exchange_rate"
                                                 else _ds, "n": "2"})},
                             "methods/method_comparison.csv", "window_best", 0.001)

# §1.15's ETTh1 geometry: mean ||tau_i|| and the alignment recomputed from the measured cosine.
for _n, _exp in ((2, "segsweep_etth1_merge_n2"), (5, "segsweep_etth1_merge_n5")):
    CHECKS += row_checks("§1.15", rf"\| {_n}",
                         {0: (f"ETTh1 n={_n} mean tau norm",
                              {"experiment_name": _exp, "n_segments": str(_n)})},
                         "geometry/geometry_summary.csv", "mean_tau_norm", 0.01)


def section_slice(text: str, section: str) -> str:
    """The document text belonging to `section` (e.g. "§1.11"), else the whole document.

    Row labels repeat across tables — `| **SWaT** |` appears in a dozen — so an unscoped
    search binds to the first table in the file rather than the intended one. Falling back to
    the whole document keeps checks whose label is not a section reference working.
    """
    number = section.lstrip("§")
    start = re.search(rf"^#{{2,5}} {re.escape(number)}[ \\]", text, re.M)
    if start is None:
        return text
    nxt = re.search(r"^#{1,3} ", text[start.end():], re.M)
    return text[start.start(): start.end() + nxt.start()] if nxt else text[start.start():]


# Cross-section consistency ---------------------------------------------------------------

# §1.11 publishes GRR at the validation-selected alpha and at the oracle alpha; §1.12 publishes
# the cost of choosing on validation. They are the same measurement twice:
#
#     cost = 1 - GRR(alpha_val) / GRR(alpha_oracle)
#
# Nothing enforced that, and twice they drifted apart — once silently (the AD rows in 1.11 were
# a seed behind 1.12) and once in *sign* (1.11 said +0.116 where 1.12 said -0.026, which is the
# difference between "merging still helps" and "merging hurts"). Four lines of arithmetic catch
# both, so they belong here rather than in a reviewer's head.

GRR_ROW = re.compile(
    r"^\| \*\*(?P<ds>\w+)\*\* \| (?P<v>[^|]+)\|(?P<v2>[^|]+)\|(?P<v3>[^|]+)\| \|"
    r"(?P<o>[^|]+)\|(?P<o2>[^|]+)\|(?P<o3>[^|]+)\|", re.M)
COST_ROW = re.compile(
    r"^\| \*\*(?P<ds>\w+)\*\* \| (?P<c1>[^|]+)\|(?P<c2>[^|]+)\|(?P<c3>[^|]+)\|", re.M)


def _num(cell: str) -> float | None:
    """First number in a table cell, ignoring bold markers, footnote stars and ± spreads."""
    m = re.search(r"-?\d+\.?\d*", cell.replace("−", "-"))
    return float(m.group()) if m else None


def check_reconciliation(text: str, tolerance: float = 0.03) -> int:
    """Assert §1.12's honest-alpha cost follows from §1.11's GRR block. Returns failure count."""
    gstart = text.find("#### GRR — share of the base-to-joint gap")
    gend = text.find("### 1.12 ", gstart + 1) if gstart >= 0 else -1
    gblock = text[gstart:gend] if gstart >= 0 and gend > gstart else ""
    grr = {m.group("ds"): [_num(m.group(k)) for k in ("v", "v2", "v3")]
                          + [_num(m.group(k)) for k in ("o", "o2", "o3")]
           for m in GRR_ROW.finditer(gblock)}
    # Scope the search to §1.12 — several sections have three percentage columns, and matching
    # on shape alone silently picks up §1.13's merge-vs-continual table instead.
    start = text.find("### 1.12 ")
    end = text.find("### 1.13 ", start + 1) if start >= 0 else -1
    section = text[start:end] if start >= 0 and end > start else ""
    costs = {}
    for m in COST_ROW.finditer(section):
        cells = [m.group("c1"), m.group("c2"), m.group("c3")]
        if all("%" in c for c in cells):
            costs[m.group("ds")] = [_num(c) for c in cells]

    failures = 0
    print("\nRECONCILIATION — §1.12 cost must equal 1 - GRR(a_val)/GRR(a_oracle) from §1.11:")
    for ds, cost in sorted(costs.items()):
        block = grr.get(ds)
        if not block or any(v is None for v in block):
            print(f"  SKIP  {ds}: no usable GRR row in §1.11")
            failures += 1
            continue
        vals, oracles = block[:3], block[3:]
        for i, (v, o, c) in enumerate(zip(vals, oracles, cost)):
            if v is None or o is None or c is None or not o:
                continue
            derived = 100.0 * (1.0 - v / o)
            ok = abs(derived - c) <= max(1.0, tolerance * max(abs(c), 1.0))
            n = (2, 3, 5)[i]
            print(f"  {'ok  ' if ok else 'FAIL'}  {ds} n={n}: 1-{v}/{o} = {derived:5.1f}%  "
                  f"documented {c:5.1f}%")
            if not ok:
                failures += 1
    return failures


def load(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise SystemExit(f"missing generated file: {csv_path}\nrun analysis/results_audit.py first")
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, default=Path("EXPERIMENTS.md"))
    parser.add_argument("--strict", action="store_true", help="exit non-zero on any drift")
    parser.add_argument("--self-test", action="store_true",
                        help="corrupt each backing cell and assert the check catches it")
    args = parser.parse_args()

    text = args.doc.read_text(encoding="utf-8")
    cache: dict[str, list[dict]] = {}
    drift, missing = 0, 0

    for label, pattern, rel, where, column, tol in CHECKS:
        # Search only inside the section the label names. Matching the whole document lets a
        # row label like `| **SWaT** |` bind to whichever table happens to appear first, which
        # silently compared §1.11's α* against an unrelated table's first column.
        match = re.search(pattern, section_slice(text, label.split()[0]))
        if match is None:
            print(f"  NOT FOUND  {label}: the documented pattern no longer appears — "
                  f"the table was edited, so this check needs updating")
            missing += 1
            continue
        documented = _to_float(match.group(1))
        rows = cache.setdefault(rel, load(args.audit_dir / rel))
        hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        if not hits:
            print(f"  NO ROW     {label}: no CSV row matching {where}")
            missing += 1
            continue
        # Average when the filter matches several rows. Per-run CSVs (geometry_summary) carry
        # one row per seed while the document quotes the seed mean, and taking hits[0] compared
        # a single seed against a 3-seed average — which reads as drift in a correct document.
        values = [float(r[column]) for r in hits if r.get(column, "") != ""]
        if not values:
            print(f"  EMPTY      {label}: {rel}:{column} is blank in {len(hits)} matching row(s)")
            missing += 1
            continue
        actual = sum(values) / len(values)
        over = f" (mean of {len(values)} rows)" if len(values) > 1 else ""
        if abs(actual - documented) > tol:
            print(f"  DRIFT      {label}: document says {documented}, CSV says {actual}{over} "
                  f"(tolerance {tol})")
            drift += 1
        else:
            print(f"  ok         {label}: {documented} ≈ {round(actual, 6)}{over}")

    # Coverage: how many numeric tables exist in the document versus how many are checked.
    tables = len(re.findall(r"^\|[^\n]*\|\n\|[-: |]+\|$", text, re.M))
    checked_sections = sorted({c[0].split()[0] for c in CHECKS})
    print(f"\n{len(CHECKS)} checks over {len(checked_sections)} sections — "
          f"{drift} drifted, {missing} unresolvable")
    print(f"COVERAGE: {len(CHECKS)} cells checked against {tables} numeric tables in "
          f"{args.doc.name}. Checked sections: {', '.join(checked_sections)}.")
    # Derive the unchecked list from the document rather than hardcoding it: the hardcoded
    # version went stale the moment sections were added, and still named §1.11/§1.12/§1.13 as
    # unchecked after they had been covered — a coverage report that overstates the gap is only
    # marginally better than one that understates it.
    documented = re.findall(r"^#{2,5} (\d+(?:\.\d+)*[a-z]*)", text, re.M)
    with_tables = [s for s in documented if re.search(r"^\|", section_slice(text, s), re.M)]
    unchecked = [s for s in with_tables if f"§{s}" not in checked_sections]
    print(f"UNCHECKED sections containing tables ({len(unchecked)} of {len(with_tables)}): "
          f"{', '.join('§' + s for s in unchecked)}.")
    print("Add checks before trusting cells in those sections.")

    recon = check_reconciliation(text)
    if recon:
        print(f"  -> {recon} reconciliation failure(s): §1.11 and §1.12 disagree")
        drift += recon

    if args.self_test:
        print("\nSELF-TEST — corrupting each backing cell; every check must then fail:")
        failures = 0
        for label, pattern, rel, where, column, tol in CHECKS:
            rows = copy.deepcopy(cache.get(rel) or load(args.audit_dir / rel))
            hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
            match = re.search(pattern, text)
            if not hits or match is None:
                print(f"  SKIP  {label}: not resolvable"); failures += 1; continue
            # Perturb far beyond tolerance and confirm the comparison would reject it.
            corrupted = float(hits[0][column]) + 10 * max(tol, 1.0)
            if abs(corrupted - _to_float(match.group(1))) > tol:
                print(f"  ok    {label}: corruption detected")
            else:
                print(f"  LEAK  {label}: corrupted value still passes — check is vacuous")
                failures += 1
        print(f"self-test: {failures} problem(s)")
        if args.strict and failures:
            sys.exit(1)

    if args.strict and (drift or missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
