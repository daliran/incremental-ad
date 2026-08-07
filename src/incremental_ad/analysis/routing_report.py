"""Bound what a router could ever gain over the merged model.

Reads only `transfer_matrix.csv` files already written by `MergeDiagnosticsPipeline` — no
dataset, no model, no GPU.

    python -m incremental_ad.analysis.routing_report $RUNS_ROOT/etth2_merge_n5_diagnostics/*
    python -m incremental_ad.analysis.routing_report $RUNS_ROOT/*_merge_n5_diagnostics/* --out /tmp/routing

**Why this file exists.** The routing-headroom table in EXPERIMENTS.md §1.16 was originally
computed ad hoc, with no script of record, and could not be re-derived afterwards — recomputing
it reproduced ETTh1 to within a point but missed exchange_rate by 5 points and its
newest-specialist column by 157. Numbers that cannot be recomputed cannot be defended, so the
definition is pinned here and every published figure is regenerated from it.

**The measurement.** The transfer matrix's column optimum *is* the best possible router: for
each period, whichever specialist turns out best on that period's own held-out slice. No real
router beats it, because a real router must choose without seeing the answer. So the distance
from the merged model to that optimum upper-bounds what routing could ever recover.

Three models are compared per column, all as ratio-to-base so runs and seeds are commensurable:
`oracle` (column optimum over specialists), `merged` (at its validation-selected scale), and
`newest` (the last specialist — the always-use-the-latest-model policy).

Two further transfer-matrix quantities are reported here because they need the same matrix:
**merge cost** = mean over periods of merged(val_i) ÷ ft_i(val_i), where 1.00 means merging
costs nothing against keeping one model per period; and **specialisation** = mean off-diagonal
− mean diagonal, positive when specialists really are period-specific. Both are ratio-to-base
throughout, so the base row cancels and seeds are commensurable.

**Aggregation order matters and is fixed here:** average each cell over seeds first, then take
the column optimum, then average over columns. Choosing the optimum per seed and averaging
afterwards biases the oracle low, because it lets a different seed win in each column — an
advantage no deployed router has.

Direction is read from the metric, not assumed: `window_auroc` and other `auroc`/`auprc`/`f1`
metrics are higher-better, so the oracle is a column *maximum* and the gap is
(oracle − model) / oracle; error metrics invert both. Interpretation is left out — this writes
measurements, not conclusions.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("routing_report")

# Substrings marking a metric where larger is better. Everything else is treated as an error.
HIGHER_IS_BETTER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")

SUMMARY_FIELDS = [
    "group",
    "dataset",
    "metric",
    "merge_scale",
    "n_segments",
    "n_seeds",
    "oracle",
    "merged",
    "newest",
    "merged_vs_oracle_pct",
    "newest_vs_oracle_pct",
    "merged_vs_oracle_pct_excl_last",
    "merge_cost",
    "specialisation",
]


def _higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER_IS_BETTER)


def _gap_pct(model: float, oracle: float, higher_better: bool) -> float:
    """How far `model` sits from `oracle`, as a percentage of the oracle.

    Always non-negative when the oracle really is optimal, and always expressed as a
    fraction of the oracle so the two directions are comparable.
    """
    return 100.0 * (oracle - model) / oracle if higher_better else 100.0 * (model - oracle) / oracle


def _read_matrices(run_dirs: list[Path], metric: str) -> tuple[dict, dict]:
    """Collect ratio-to-base cells keyed (model, column) -> [value per seed], plus run info."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    info: dict[str, set] = {"dataset": set(), "n_segments": set(), "seeds": set(),
                            "merge_scale": set()}
    for run in run_dirs:
        matrix = run / "merge_diagnostics" / "transfer_matrix.csv"
        source = run / "merge_diagnostics" / "source.json"
        if not matrix.exists():
            log.warning("no transfer_matrix.csv under %s — skipped", run)
            continue
        if source.exists():
            src = json.loads(source.read_text())
            for key, field in (("dataset", "dataset"), ("n_segments", "n_segments"),
                               ("merge_scale", "merge_scale")):
                if src.get(field) is not None:
                    info[key].add(src[field])
        cfg = run / "config.json"
        if cfg.exists():
            info["seeds"].add(json.loads(cfg.read_text()).get("seed"))
        with matrix.open() as fh:
            for row in csv.DictReader(fh):
                if row["metric"] != metric:
                    continue
                cells[(row["model"], row["column"])].append(float(row["ratio_to_base"]))
    return cells, info


def analyse(run_dirs: list[Path], metric: str) -> dict | None:
    """Return the routing summary for one group of runs (same dataset, same n, several seeds)."""
    cells, info = _read_matrices(run_dirs, metric)
    if not cells:
        return None
    # Average over seeds FIRST — see the module docstring on aggregation order.
    mean_cell = {key: st.mean(vals) for key, vals in cells.items()}

    # NB: dedup with a set — mean_cell is keyed (model, column), so iterating it without
    # deduplicating yields each column once per model. That silently mispairs ft_i with the
    # wrong column in the merge-cost and specialisation loops below.
    columns = sorted({c for _, c in mean_cell if c.startswith("val_") and c != "val_base"},
                     key=lambda c: int(c.split("_")[1]))
    specialists = sorted(
        {m for m, _ in mean_cell if m.startswith("ft_")}, key=lambda m: int(m.split("_")[1])
    )
    if not columns or not specialists:
        return None

    higher = _higher_is_better(metric)
    pick = max if higher else min

    oracle, merged, newest = [], [], []
    for col in columns:
        scores = [mean_cell[(m, col)] for m in specialists if (m, col) in mean_cell]
        if not scores or ("merged", col) not in mean_cell:
            continue
        oracle.append(pick(scores))
        merged.append(mean_cell[("merged", col)])
        newest.append(mean_cell[(specialists[-1], col)])
    if not oracle:
        return None

    o, m, n = st.mean(oracle), st.mean(merged), st.mean(newest)

    # Merge cost: the merged model against each period's OWN specialist, on that period.
    own = [(mean_cell[("merged", c)], mean_cell[(f"ft_{i}", c)])
           for i, c in enumerate(columns)
           if ("merged", c) in mean_cell and (f"ft_{i}", c) in mean_cell]
    merge_cost = st.mean([a / b for a, b in own if b]) if own else None

    # Specialisation: how much better a specialist is on its own period than on the others.
    diag = [mean_cell[(f"ft_{i}", c)] for i, c in enumerate(columns) if (f"ft_{i}", c) in mean_cell]
    off = [mean_cell[(mm, c)] for mm in specialists for i, c in enumerate(columns)
           if mm != f"ft_{i}" and (mm, c) in mean_cell]
    specialisation = (st.mean(off) - st.mean(diag)) if diag and off else None
    # The final regime is trivial for the newest specialist — it trained on it. Reporting the
    # gap with it excluded keeps the newest-vs-oracle column from flattering that policy.
    excl = (
        _gap_pct(st.mean(merged[:-1]), st.mean(oracle[:-1]), higher) if len(oracle) > 1 else None
    )
    return {
        "dataset": "/".join(sorted(str(d) for d in info["dataset"])) or "?",
        "metric": metric,
        # The scale the merged model was actually built at. A run merged at alpha=1 is
        # measuring overshoot, not routing headroom -- those rows are not comparable with
        # runs whose scale was selected on validation, and must not be pooled with them.
        "merge_scale": "/".join(f"{s:g}" for s in sorted(info["merge_scale"])) or "?",
        "n_segments": "/".join(sorted(str(s) for s in info["n_segments"])) or str(len(columns)),
        "n_seeds": len({s for s in info["seeds"] if s is not None}),
        "oracle": round(o, 4),
        "merged": round(m, 4),
        "newest": round(n, 4),
        "merged_vs_oracle_pct": round(_gap_pct(m, o, higher), 1),
        "newest_vs_oracle_pct": round(_gap_pct(n, o, higher), 1),
        "merged_vs_oracle_pct_excl_last": None if excl is None else round(excl, 1),
        "merge_cost": None if merge_cost is None else round(merge_cost, 4),
        "specialisation": None if specialisation is None else round(specialisation, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="diagnostics run directories")
    parser.add_argument(
        "--metric",
        default="forecast/mse",
        help="metric column of transfer_matrix.csv (e.g. window_auroc for AD)",
    )
    parser.add_argument(
        "--group-by-parent",
        action="store_true",
        default=True,
        help="group runs by their parent experiment directory (seeds of one config)",
    )
    parser.add_argument("--out", type=Path, help="write routing_summary.csv here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    groups: dict[str, list[Path]] = defaultdict(list)
    for run in args.run_dirs:
        groups[run.parent.name if args.group_by_parent else run.name].append(run)

    rows = []
    for name, runs in sorted(groups.items()):
        result = analyse(runs, args.metric)
        if result is None:
            log.warning("%s: no usable cells for metric %s", name, args.metric)
            continue
        rows.append({"group": name, **result})

    if not rows:
        raise SystemExit("no routing results — check --metric against transfer_matrix.csv")

    width = max(len(r["group"]) for r in rows)
    log.info(
        "\n%-*s %10s %10s %10s   %s",
        width, "group", "oracle", "merged", "newest", "merged vs oracle / newest vs oracle",
    )
    for r in rows:
        log.info(
            "%-*s %10.4f %10.4f %10.4f   %+7.1f%% / %+7.1f%%",
            width, r["group"], r["oracle"], r["merged"], r["newest"],
            r["merged_vs_oracle_pct"], r["newest_vs_oracle_pct"],
        )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "routing_summary.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
