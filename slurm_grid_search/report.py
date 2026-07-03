"""Compare Standard vs Incremental (baseline/merged) performance for a dataset, once
both training-parameter sweeps have been collected.

Usage:
    python report.py --dataset swat --metric pa_f1
    python report.py --dataset etth_forecast --metric "forecast/mse" --lower-is-better

Answers: what's the best Standard pipeline performance, what's the best Incremental
performance (baseline vs merged), does Incremental reach Standard's level, and does
fine-tuning actually help the merge beat just keeping the baseline frozen (found on
SWaT/PSM's first grid search that this isn't always true depending on the metric).

Run collect.py for train_standard and train_incremental first -- this only reads their
already-collected CSVs, it doesn't scan run directories itself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from harness import output_root


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"(not found: {path} -- run collect.py for this stage first)")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best(rows: list[dict[str, str]], column: str, lower_is_better: bool) -> dict[str, str] | None:
    usable = [r for r in rows if r.get(column) not in (None, "")]
    if not usable:
        return None
    return min(usable, key=lambda r: float(r[column])) if lower_is_better else \
        max(usable, key=lambda r: float(r[column]))


def _print_row(label: str, row: dict[str, str] | None, column: str) -> None:
    if row is None:
        print(f"  {label}: (no rows with a value for {column})")
        return
    value = row[column]
    extra = {k: v for k, v in row.items() if k not in ("run_id", "status") and v not in (None, "")}
    print(f"  {label}: {column}={value}")
    print(f"    run_id={row.get('run_id')}  params={extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["swat", "psm", "etth_forecast"])
    parser.add_argument(
        "--metric", required=True,
        help='Bare metric name as it appears inside a result.json\'s "metrics" dict, '
             'e.g. "pa_f1" or "forecast/mse" -- report.py adds the train_test_/'
             "baseline_test_/merged_test_ prefixes itself.",
    )
    parser.add_argument(
        "--lower-is-better", action="store_true",
        help="Set for metrics like MSE/MAE where lower is better (default: higher is better, e.g. F1/AUROC).",
    )
    args = parser.parse_args()

    lower = args.lower_is_better
    root = output_root()

    standard_rows = _read_csv(root / f"{args.dataset}_train_standard_results.csv")
    incremental_rows = _read_csv(root / f"{args.dataset}_train_incremental_results.csv")

    std_col = f"train_test_{args.metric}"
    base_col = f"baseline_test_{args.metric}"
    merged_col = f"merged_test_{args.metric}"

    print(f"=== {args.dataset} -- Standard vs Incremental, metric={args.metric} "
          f"({'lower' if lower else 'higher'} is better) ===\n")

    best_standard = _best(standard_rows, std_col, lower)
    _print_row("Best StandardPipeline", best_standard, std_col)

    best_merged = _best(incremental_rows, merged_col, lower)
    _print_row("Best Incremental (merged)", best_merged, merged_col)

    best_baseline = _best(incremental_rows, base_col, lower)
    _print_row("Best Incremental (baseline alone)", best_baseline, base_col)

    print()
    if best_standard and best_merged:
        std_v, merged_v = float(best_standard[std_col]), float(best_merged[merged_col])
        reaches = merged_v <= std_v if lower else merged_v >= std_v
        print(f"Does Incremental reach Standard's level? {'YES' if reaches else 'no'} "
              f"(merged={merged_v} vs standard={std_v})")

    if best_merged:
        run_id = best_merged["run_id"]
        same_run = next((r for r in incremental_rows if r["run_id"] == run_id), None)
        if same_run and same_run.get(base_col):
            base_v, merged_v = float(same_run[base_col]), float(same_run[merged_col])
            helped = merged_v <= base_v if lower else merged_v >= base_v
            print(f"For the best merged run specifically, did fine-tuning+merge beat "
                  f"its own frozen baseline? {'YES' if helped else 'no'} "
                  f"(baseline={base_v} vs merged={merged_v})")


if __name__ == "__main__":
    main()
