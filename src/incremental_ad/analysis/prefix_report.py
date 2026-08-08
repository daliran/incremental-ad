"""Forward transfer and the accumulate-vs-materialise comparison, from prefix merges.

    python -m incremental_ad.analysis.prefix_report $RUNS_ROOT/prefix_etth1/* \\
        --label ETTh1 --out /tmp/prefix

**Why this file exists.** EXPERIMENTS.md 1.19 and 1.22 were the last published tables backed by
no script. `prefix_merges.csv` stores raw `value` per (prefix_k, merge_scale, column) with no
ratio column and no record of which alpha produced the published numbers -- and none could be
recovered: scanning every grid point reproduced 3 of 8 cells, at inconsistent alphas. So the
numbers were restated under a convention that is now fixed here.

**The convention, chosen and fixed:** alpha = the grid point nearest **1/k** for a prefix of k
task vectors. That is the project's own *pre-declared* rule (the `alpha = 1/n fixed` variant of
1.25), it needs no validation pass, and it deliberately gives merging no oracle advantage on a
table whose claim is that merging beats the base model. The alpha actually used is emitted per
row, because 1/k is not always on the grid (1/3 = 0.333 lands on 0.35).

**accumulate** = merge(tau_0 ... tau_{k-1}) scored on `val_k`, the first shard it has not seen.
**materialise** = `ft_{k-1}`, a specialist fine-tuned on the most recent shard alone, scored on
that same `val_k`. Both as a ratio to the base model on that shard, so 1.0 means "no better than
the frozen base" and lower is better for an error metric.

Both come from the same run, so the comparison is within-seed; ratios are formed per seed and
then averaged, which is not the same as the ratio of averages and is the form that carries a
spread.
"""

import argparse
import csv
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("prefix_report")

FIELDS = ["label", "k", "column", "n_seeds", "alpha_used", "accumulate", "accumulate_sd",
          "materialise", "materialise_sd", "winner", "margin_pct"]


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in
               ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))


def _prefix_values(run: Path, metric: str) -> dict:
    """(k, alpha, column) -> value, for one run."""
    path = run / "merge_diagnostics" / "prefix_merges.csv"
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == metric:
                out[(int(row["prefix_k"]), float(row["merge_scale"]), row["column"])] = \
                    float(row["value"])
    return out


def _specialist_ratios(run: Path, metric: str) -> dict:
    """(model, column) -> ratio_to_base, for one run's transfer matrix."""
    path = run / "merge_diagnostics" / "transfer_matrix.csv"
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == metric:
                out[(row["model"], row["column"])] = float(row["ratio_to_base"])
    return out


def analyse(runs: list[Path], label: str, metric: str) -> list[dict]:
    """One row per prefix length k: accumulate vs materialise on the first unseen shard."""
    per_k_acc: dict[int, list[float]] = defaultdict(list)
    per_k_mat: dict[int, list[float]] = defaultdict(list)
    alpha_used: dict[int, float] = {}

    for run in runs:
        values = _prefix_values(run, metric)
        specialists = _specialist_ratios(run, metric)
        if not values:
            continue
        grid = sorted({a for _, a, _ in values})
        for k in sorted({kk for kk, _, _ in values}):
            column = f"val_{k}"
            base = values.get((k, 0.0, column))
            if not base:
                continue
            # Pre-declared alpha: the grid point nearest 1/k. Emitted, because it is not
            # always exactly 1/k.
            alpha = min(grid, key=lambda a: abs(a - 1.0 / k)) if grid else None
            merged = values.get((k, alpha, column))
            if merged is None:
                continue
            alpha_used[k] = alpha
            per_k_acc[k].append(merged / base)
            # The freshly materialised model is the specialist on the most recent shard.
            fresh = specialists.get((f"ft_{k - 1}", column))
            if fresh is not None:
                per_k_mat[k].append(fresh)

    rows = []
    for k in sorted(per_k_acc):
        acc, mat = per_k_acc[k], per_k_mat.get(k, [])
        better = max if higher_is_better(metric) else min
        row = {
            "label": label, "k": k, "column": f"val_{k}", "n_seeds": len(acc),
            "alpha_used": round(alpha_used[k], 4),
            "accumulate": round(st.mean(acc), 4),
            "accumulate_sd": round(st.stdev(acc), 4) if len(acc) > 1 else 0.0,
            "materialise": round(st.mean(mat), 4) if mat else "",
            "materialise_sd": round(st.stdev(mat), 4) if len(mat) > 1 else 0.0,
        }
        if mat:
            a, m = st.mean(acc), st.mean(mat)
            row["winner"] = "accumulate" if better(a, m) == a else "materialise"
            row["margin_pct"] = round(100.0 * abs(a - m) / m, 2)
        else:
            row["winner"], row["margin_pct"] = "", ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--label", required=True, help="dataset name for the output rows")
    parser.add_argument("--metric", default="forecast/mse")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = analyse([p for p in args.run_dirs if p.is_dir()], args.label, args.metric)
    if not rows:
        raise SystemExit("no prefix_merges.csv found under the given run directories")

    log.info(f"{'k':>3}{'alpha':>8}{'seeds':>7}{'accumulate':>13}{'materialise':>13}"
             f"{'winner':>14}{'margin':>9}")
    for row in rows:
        mat = f"{row['materialise']:.4f}" if row["materialise"] != "" else "—"
        margin = f"{row['margin_pct']:.1f}%" if row["margin_pct"] != "" else "—"
        log.info(f"{row['k']:>3}{row['alpha_used']:>8.2f}{row['n_seeds']:>7}"
                 f"{row['accumulate']:>13.4f}{mat:>13}{row['winner']:>14}{margin:>9}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "prefix_forward_transfer.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
