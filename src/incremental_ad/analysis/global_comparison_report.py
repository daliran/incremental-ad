"""One table with every update strategy and every merge rule, per (dataset, n) cell.

    python -m incremental_ad.analysis.global_comparison_report \\
        --method_comparison results_archive/audit/methods_windowval/method_comparison.csv \\
        --merge_baselines results_archive/audit/merge_baselines/merge_baselines.csv \\
        --adaptive results_archive/audit/adaptive_lambda/adaptive_lambda_test.csv \\
        --out $OUT/global_comparison

Pure aggregation over three CSVs that are each already the script of record for their own
section — `method_comparison` with validation-selected windows (§1.26b), `merge_baselines_report` (§1.40) and
`adaptive_lambda_report` (§1.39). Nothing is recomputed; every value here is copied from one of
them, and `source` names which. It exists because those three were never shown side by side, and
it does NOT change `method_comparison.csv`'s schema, which other checks bind to.

What is joined, and what each value is:

- **References, never ranked:** `base` (θ₀, no update), `joint` (all data at once — the upper
  reference GRR divides by), `specialists` (mean of per-segment models, not one model),
  `window_oracle` (best W picked on **test**, §1.21), and `merge_ta_published` — task
  arithmetic as §1.26 publishes it.
- **Candidates, ranked within the cell:** `ta` (task arithmetic on §1.40's grid), `sequential`,
  `window_val` (W picked on validation) when present, the §1.40 rules `dare` / `ties` / `iso_c` /
  `tsv`, and `adaptive_lambda` (strategy 6) where it was run.
- **Why TA is ranked from §1.40, not §1.26.** Both are validation-selected, but on different α
  grids (§1.26's runs selected from their own `extra_merge_scales`; §1.40 uses 0.1–3.0 for every
  rule). Ranking the four rules against a TA chosen from another grid would let a grid difference
  pose as a rule difference — on ETTh2 n = 3 DARE beats the published TA (0.1961 vs 0.2153) but
  not TA on its own grid (0.1929). The published value is kept beside it so the two can be
  compared; on AD they are the same model.

⚠️ Not everything is on one protocol, and the table says so rather than hiding it:
  * On AD the §1.40 rules are read at a **distance-matched** α (§1.40), not a selected one.
  * `adaptive_lambda` on AD uses B = 64 — the estimator §1.39b shows is defective; AD was not
    re-run at B = 1 (§1.39). Forecasting uses the corrected B = 1.
  * `adaptive_lambda` is paired in §1.39 against **its own** plain chain; `adaptive_paired_plain`
    carries that chain's value, which is the fair comparison. Its rank here is against the
    published `sequential`, a different set of runs.
  * Ranks are *orderings of means*, not verdicts — the per-section decision rules (floor-scaled
    margins) are what each section's claims rest on. `gap_over_floor` is emitted so a reader can
    see which orderings are inside the noise.

The summary excludes SWaT-forecast (floor 84%: no ordering there is meaningful), as §1.40 does.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("global_comparison_report")

ALIAS = {"exchange_rate": "exchange"}
EXCLUDED_FROM_SUMMARY = {"SWaT-forecast"}
REFERENCES = ("base", "joint", "specialists", "window_oracle", "merge_ta_published")
RULES = ("dare", "ties", "iso_c", "tsv")
# Strategy 6 estimator of record: corrected per-sample Fisher on forecasting; AD has only B = 64.
ADAPTIVE_BATCH = {"forecast": "1", "ad": "64"}

FIELDS = ["dataset", "n", "metric", "higher_is_better", "floor_pct", "method", "role",
          "selection", "value", "sd", "n_seeds", "rank", "n_ranked", "gap_to_best_pct",
          "gap_over_floor", "adaptive_paired_plain", "source"]
SUMMARY_FIELDS = ["method", "role", "n_cells", "n_best", "n_better_than_best", "mean_rank",
                  "mean_normalised_rank", "mean_gap_to_best_pct", "cells"]


def higher_is_better(metric: str) -> bool:
    return "auroc" in metric or "auprc" in metric or "f1" in metric


def _f(value):
    return float(value) if value not in ("", None) else None


def read(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build(method_rows, baseline_rows, adaptive_rows) -> list[dict]:
    cells: dict[tuple, dict] = {}

    def cell(dataset, n, metric, floor=None):
        key = (ALIAS.get(dataset, dataset), str(n), metric)
        entry = cells.setdefault(key, {"floor": None, "entries": []})
        if floor is not None and entry["floor"] is None:
            entry["floor"] = floor
        return entry

    for row in method_rows:
        c = cell(row["dataset"], row["n"], row["metric"], _f(row["floor_pct"]))
        for column, method, role, selection in (
                ("base", "base", "reference", "none"),
                ("joint", "joint", "reference", "shared configuration"),
                ("specialists", "specialists", "reference", "mean of per-segment models"),
                ("merge", "merge_ta_published", "reference", "val-selected alpha (§1.26 grid)"),
                ("sequential", "sequential", "candidate", "final step"),
                ("window_val", "window_val", "candidate", "W selected on validation"),
                ("window_best", "window_oracle", "reference", "W selected on TEST")):
            value = _f(row.get(column))
            if value is not None:
                c["entries"].append({"method": method, "role": role, "selection": selection,
                                     "value": value, "source": "method_comparison"})

    for row in baseline_rows:
        c = cell(row["dataset"], row["n_segments"], row["metric"], _f(row["floor_pct"]))
        rule = row["rule"]
        common = {"selection": row["protocol"], "sd": row["value_sd"],
                  "n_seeds": row["n_seeds"], "source": "merge_baselines"}
        if rule == "ta":
            c["entries"].append({"method": "ta", "role": "candidate",
                                 "value": float(row["value"]), **common})
        elif rule in RULES:
            c["entries"].append({"method": rule, "role": "candidate",
                                 "value": float(row["value"]), **common})

    for row in adaptive_rows:
        family = "forecast" if row["metric"].startswith("forecast/") else "ad"
        if row["lambda_source"] != "became" or row["fisher_batch_size"] != ADAPTIVE_BATCH[family]:
            continue
        c = cell(row["dataset"], row["n_segments"], row["metric"], _f(row["floor_pct"]))
        estimator = "B = 1 (corrected)" if family == "forecast" else "B = 64 (defect; AD not re-run)"
        c["entries"].append({"method": "adaptive_lambda", "role": "candidate",
                             "selection": f"final step, {estimator}",
                             "value": float(row["adaptive"]), "sd": row["adaptive_sd"],
                             "n_seeds": row["n_seeds"], "source": "adaptive_lambda",
                             "adaptive_paired_plain": row["plain"]})

    out = []
    for (dataset, n, metric), c in sorted(cells.items()):
        up = higher_is_better(metric)
        ranked = sorted((e for e in c["entries"] if e["role"] == "candidate"),
                        key=lambda e: e["value"], reverse=up)
        # Only the primary-metric cells the published comparison defines. Strategy 6 also
        # reports AUPRC, where it would otherwise be ranked 1st against nobody.
        if not any(e["source"] == "method_comparison" for e in c["entries"]):
            continue
        best = ranked[0]["value"]
        ranks = {id(e): i for i, e in enumerate(ranked, 1)}
        for e in c["entries"]:
            gap = 100.0 * ((best - e["value"]) if up else (e["value"] - best)) / best
            out.append({
                "dataset": dataset, "n": n, "metric": metric, "higher_is_better": up,
                "floor_pct": c["floor"], "method": e["method"], "role": e["role"],
                "selection": e["selection"], "value": e["value"], "sd": e.get("sd", ""),
                "n_seeds": e.get("n_seeds", ""), "rank": ranks.get(id(e), ""),
                "n_ranked": len(ranked), "gap_to_best_pct": round(gap, 4),
                "gap_over_floor": round(gap / c["floor"], 4) if c["floor"] else "",
                "adaptive_paired_plain": e.get("adaptive_paired_plain", ""),
                "source": e["source"]})
    return out


def summarise(rows: list[dict]) -> list[dict]:
    by_method = defaultdict(list)
    for row in rows:
        if row["rank"] == "" or row["dataset"] in EXCLUDED_FROM_SUMMARY:
            continue
        by_method[row["method"]].append(row)
    summary = []
    for method, items in by_method.items():
        k = len(items)
        summary.append({
            "method": method, "role": "candidate", "n_cells": k, "n_better_than_best": "",
            "n_best": sum(1 for r in items if r["rank"] == 1),
            "mean_rank": round(sum(r["rank"] for r in items) / k, 4),
            "mean_normalised_rank": round(sum((r["rank"] - 1) / max(r["n_ranked"] - 1, 1)
                                              for r in items) / k, 4),
            "mean_gap_to_best_pct": round(sum(r["gap_to_best_pct"] for r in items) / k, 4),
            "cells": " ".join(f"{r['dataset']}/{r['n']}" for r in items)})
    summary.sort(key=lambda r: r["mean_normalised_rank"])
    # Joint training is a REFERENCE, never ranked: it retains the full history, which is the
    # thing every candidate is trying to avoid. Its row is emitted so the scope of the ranking is
    # visible in the table itself — how often it beats the best candidate is what "task
    # arithmetic has the best mean rank" leaves out.
    joint = [r for r in rows if r["method"] == "joint" and r["dataset"] not in EXCLUDED_FROM_SUMMARY]
    if joint:
        summary.append({
            "method": "joint", "role": "reference, not ranked", "n_cells": len(joint),
            "n_best": "", "n_better_than_best": sum(r["gap_to_best_pct"] < 0 for r in joint),
            "mean_rank": "", "mean_normalised_rank": "",
            "mean_gap_to_best_pct": round(sum(r["gap_to_best_pct"] for r in joint) / len(joint), 4),
            "cells": " ".join(f"{r['dataset']}/{r['n']}" for r in joint)})
    return summary


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d row(s))", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method_comparison", type=Path, required=True)
    parser.add_argument("--merge_baselines", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = build(read(args.method_comparison), read(args.merge_baselines), read(args.adaptive))
    if not rows:
        raise SystemExit("no cells built — refusing to write an empty table")
    summary = summarise(rows)
    for row in summary:
        if row["role"] != "candidate":
            log.info("  %-16s cells %2d  (%s) better than the best candidate on %d, mean gap "
                     "%.2f%%", row["method"], row["n_cells"], row["role"],
                     row["n_better_than_best"], row["mean_gap_to_best_pct"])
            continue
        log.info("  %-16s cells %2d  best %2d  mean normalised rank %.3f  mean gap %.2f%%",
                 row["method"], row["n_cells"], row["n_best"], row["mean_normalised_rank"],
                 row["mean_gap_to_best_pct"])
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "global_comparison.csv", FIELDS, rows)
    write(args.out / "global_comparison_summary.csv", SUMMARY_FIELDS, summary)


if __name__ == "__main__":
    main()
