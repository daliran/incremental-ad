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
    ("§1.16 ETTh2 routing n=5", r"\*\*ETTh2\*\* \| 0\.753 \| 1,393 \| — \| \*\*\+[\d.]+%\*\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "etth2_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTm2 routing n=5", r"\*\*ETTm2\*\* \| 0\.752 \| 5,574 \| — \| \*\*\+[\d.]+%\*\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "ettm2_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
    ("§1.16 ETTh1 routing n=5", r"ETTh1 \| 0\.412 \| 1,393 \| \+[\d.]+% \| \*\(α=1 only\)\* \| \*\*\+([\d.]+)%\*\*",
     "routing_forecast/routing_summary.csv", {"group": "segsweep_etth1_merge_n5_diagnostics"},
     "merged_vs_oracle_pct", 0.15),
]



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
        match = re.search(pattern, text)
        if match is None:
            print(f"  NOT FOUND  {label}: the documented pattern no longer appears — "
                  f"the table was edited, so this check needs updating")
            missing += 1
            continue
        documented = float(match.group(1))
        rows = cache.setdefault(rel, load(args.audit_dir / rel))
        hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        if not hits:
            print(f"  NO ROW     {label}: no CSV row matching {where}")
            missing += 1
            continue
        raw = hits[0].get(column, "")
        if raw == "":
            print(f"  EMPTY      {label}: {rel}:{column} is blank")
            missing += 1
            continue
        actual = float(raw)
        if abs(actual - documented) > tol:
            print(f"  DRIFT      {label}: document says {documented}, CSV says {actual} "
                  f"(tolerance {tol})")
            drift += 1
        else:
            print(f"  ok         {label}: {documented} ≈ {actual}")

    # Coverage: how many numeric tables exist in the document versus how many are checked.
    tables = len(re.findall(r"^\|[^\n]*\|\n\|[-: |]+\|$", text, re.M))
    checked_sections = sorted({c[0].split()[0] for c in CHECKS})
    print(f"\n{len(CHECKS)} checks over {len(checked_sections)} sections — "
          f"{drift} drifted, {missing} unresolvable")
    print(f"COVERAGE: {len(CHECKS)} cells checked against {tables} numeric tables in "
          f"{args.doc.name}. Checked sections: {', '.join(checked_sections)}.")
    print("Everything else is UNCHECKED — notably §1.3 merge-scale curves, §1.4/§1.23/§1.24 "
          "transfer matrices, §1.5 metric reports, §1.8/§1.15 geometry, §1.11 α*·n and merge "
          "cost, §1.12 honest-α, §1.13 merge-vs-continual, §1.17 n=1, §1.19 forward transfer, "
          "§1.20 recency, §1.21/§1.22 retention. Add checks before trusting those cells.")

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
            if abs(corrupted - float(match.group(1))) > tol:
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
