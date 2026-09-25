"""Seed-level summary of `oracle_router.py`'s per-run CSVs (§1.16b) — pure aggregation.

    python -m incremental_ad.analysis.oracle_router_report \\
        --per_run_dir $OUT/oracle_router --out $OUT/oracle_router/oracle_router_summary.csv

Until 2026-09-25 `oracle_router_summary.csv` had no generator in the repository: the per-run files
did (`oracle_router.py`, a checkpoint reader), the summary the document quotes did not. This is
that missing step, written to reproduce the archived summary byte for byte.

Per (dataset, n, metric): mean and sd over seeds of the per-window oracle router, mean of the
best single model, and `oracle_vs_best_single_pct` = 100 · (best_single − oracle) / best_single
on those means.
"""

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

LABEL = {"EtthForecastDataset": "ETTh1", "Etth2ForecastDataset": "ETTh2",
         "Ettm2ForecastDataset": "ETTm2", "ExchangeRateForecastDataset": "exchange"}
FIELDS = ["dataset", "n", "metric", "n_seeds", "oracle_router", "oracle_sd", "best_single",
          "oracle_vs_best_single_pct"]
ORDER = ["ETTh1", "ETTh2", "ETTm2", "exchange"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per_run_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.per_run_dir.glob("oracle_router_*.csv")):
        if path.name == "oracle_router_summary.csv":
            continue
        for row in csv.DictReader(path.open(encoding="utf-8")):
            groups[(LABEL[row["dataset"]], int(row["n_segments"]), row["metric"])].append(row)
    if not groups:
        raise SystemExit(f"no per-run oracle_router CSVs under {args.per_run_dir}")
    rows = []
    for (dataset, n, metric), items in sorted(groups.items(),
                                              key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1],
                                                              kv[0][2])):
        oracle = [float(r["oracle_router"]) for r in items]
        best = st.fmean(float(r["best_single"]) for r in items)
        mean = st.fmean(oracle)
        rows.append({"dataset": dataset, "n": n, "metric": metric, "n_seeds": len(items),
                     "oracle_router": round(mean, 6),
                     "oracle_sd": round(st.stdev(oracle), 6) if len(oracle) > 1 else 0.0,
                     "best_single": round(best, 6),
                     "oracle_vs_best_single_pct": round(100.0 * (best - mean) / best, 2)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
