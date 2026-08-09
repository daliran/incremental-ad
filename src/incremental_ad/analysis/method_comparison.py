"""All five update strategies on one footing, per (dataset, segment count).

    python -m incremental_ad.analysis.method_comparison --runs_root $RUNS_ROOT \\
        --spec analysis_specs/method_comparison_spec.csv --out /tmp/cmp \\
        --routing_dir $OUT/routing_forecast

**Why this file exists.** The method comparisons in EXPERIMENTS.md live in separate tables
against separate baselines — §1.13 merge vs sequential, §1.21 merge vs windowed, §1.11's GRR
merge vs joint — so no single table puts them on the same rows, and a reader cannot see which
method wins overall without doing the joins by hand. Doing those joins by hand is how three
irreproducible figures got into the documents.

The five strategies, all scored on the **same test set** with the same seeds:

| column | what it is |
|---|---|
| `joint` | trained from scratch on everything — a reference, **not** a ceiling (merging beats it on exchange_rate) |
| `merge` | θ₀ + α·Στ at the committed α |
| `sequential` | the continual chain's final model |
| `window_W` | a fresh fine-tune of θ₀ on the last W periods (best W reported) |
| `oracle_router` | the transfer matrix's column optimum — the best any router could do. **Forecasting only**: on AD the per-regime columns carry no detection metric (§1.16) |

**Decisiveness, not winners.** Every pairwise gap is compared against that dataset's
reproducibility floor and marked `~` when it falls inside — a difference smaller than run-to-run
variation is not a result. The `best` column names a method only when its margin over the
runner-up clears the floor; otherwise it reads `tie`.

Floors come from the spec rather than being recomputed, so this stays a pure join over
`results_audit` output and cannot silently disagree with §1.9.
"""

import argparse
import csv
import json
import logging
import statistics as st
from pathlib import Path

log = logging.getLogger("method_comparison")

FIELDS = ["dataset", "n", "metric", "floor_pct", "base", "specialists", "joint", "merge",
          "sequential", "window_W1", "window_W2", "window_W3", "window_best", "window_W",
          "oracle_router", "best", "margin_pct", "decisive"]

HIGHER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER)


def mean_metric(runs_root: Path, experiment: str, block: str, metric: str) -> float | None:
    """Mean over seeds of one metric in one block of one experiment."""
    values = []
    for result in sorted((runs_root / experiment).glob(f"*/{block}/result.json")):
        try:
            metrics = json.loads(result.read_text()).get("metrics") or {}
        except (OSError, ValueError):
            continue
        if metric in metrics:
            values.append(metrics[metric])
    return st.mean(values) if values else None


def _specialists_mean(runs_root: Path, experiment: str, n: int, metric: str) -> float | None:
    """Mean over the per-shard specialists on the global test set.

    This is the *average* specialist, not the best one — a router would pick per period, so this
    understates what routing could reach. It is reported anyway because the average is what you
    get without a router, and on AD no router can be built at all (1.16).
    """
    values = [mean_metric(runs_root, experiment, f"finetune_{i}/test", metric) for i in range(n)]
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def compare(runs_root: Path, spec_row: dict, routing: dict, metric: str | None = None
            ) -> dict | None:
    metric = metric or spec_row["metric"]
    n = int(spec_row["n"])
    floor = float(spec_row["floor_pct"])
    better = max if higher_is_better(metric) else min

    entries: dict[str, float] = {}
    for name, experiment, block in (
        ("joint", spec_row.get("joint_experiment"), "train/test"),
        ("merge", spec_row.get("merge_experiment"), "merged/test"),
        ("sequential", spec_row.get("seq_experiment"), f"continual_{n - 1}/test"),
        # The base model is the same frozen theta_0 every method starts from; read it off the
        # merge run so it is the base those numbers were actually produced against.
        ("base", spec_row.get("merge_experiment"), "baseline/test"),
    ):
        if not experiment:
            continue
        value = mean_metric(runs_root, experiment, block, metric)
        if value is not None:
            entries[name] = value
    specialists = _specialists_mean(runs_root, spec_row.get("merge_experiment", ""), n, metric)

    # The window axis is independent of n, so report the best W and say which it was.
    windows = {}
    for w in (1, 2, 3):
        experiment = spec_row.get(f"window_W{w}_experiment")
        if not experiment:
            continue
        value = mean_metric(runs_root, experiment, "finetune_0/test", metric)
        if value is not None:
            windows[w] = value
    window_w = better(windows, key=windows.get) if windows else None
    if window_w is not None:
        entries["window_best"] = windows[window_w]

    # Oracle router is a ratio-to-base, so convert it onto the same scale as the rest:
    # merged / (1 + headroom_over_merged) is not recoverable, so report it only as a ratio.
    router = routing.get(spec_row.get("diagnostics_group", ""))

    if len(entries) < 2:
        return None
    # `base` is the starting point every method improves on, not a competitor for "best".
    ranked = sorted(((k, v) for k, v in entries.items() if k != "base"),
                    key=lambda kv: kv[1], reverse=higher_is_better(metric))
    top, second = ranked[0], ranked[1]
    margin = 100.0 * abs(top[1] - second[1]) / abs(second[1]) if second[1] else 0.0
    decisive = margin > floor

    row = {"dataset": spec_row["dataset"], "n": n, "metric": metric, "floor_pct": floor,
           "window_W": window_w, "oracle_router": router,
           "specialists": round(specialists, 6) if specialists is not None else "",
           "best": top[0] if decisive else "tie",
           "margin_pct": round(margin, 2), "decisive": decisive}
    for name in ("base", "joint", "merge", "sequential", "window_best"):
        row[name] = round(entries[name], 6) if name in entries else ""
    for w in (1, 2, 3):
        row[f"window_W{w}"] = round(windows[w], 6) if w in windows else ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--routing_dir", type=Path,
                        help="directory holding routing_summary.csv, for the oracle column")
    parser.add_argument("--metrics", nargs="+",
                        help="override the spec's metric; repeat to emit one row per metric "
                             "(e.g. window_auroc window_auprc point_auroc point_auprc). "
                             "Secondary metrics were never tabulated, which is why the "
                             "windowed column read '—' for everything but the primary one.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    routing: dict[str, str] = {}
    if args.routing_dir and (args.routing_dir / "routing_summary.csv").is_file():
        with (args.routing_dir / "routing_summary.csv").open() as fh:
            for row in csv.DictReader(fh):
                routing[row["group"]] = row.get("merged_vs_oracle_pct", "")

    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    metrics = args.metrics or [None]
    rows = [r for s in spec for m in metrics
            if (r := compare(args.runs_root, s, routing, m)) is not None]
    if not rows:
        raise SystemExit("nothing comparable — check the spec's experiment names")

    log.info("%-10s %2s %10s %10s %11s %11s %9s  %-11s %s",
             "dataset", "n", "joint", "merge", "sequential", "window", "router", "best", "margin")
    for r in rows:
        f = lambda k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else "—")  # noqa: E731
        router = f"+{r['oracle_router']}%" if r.get("oracle_router") else "—"
        log.info("%-10s %2d %10s %10s %11s %11s %9s  %-11s %s",
                 r["dataset"], r["n"], f("joint"), f("merge"), f("sequential"),
                 f("window_best"), router, r["best"],
                 f"{r['margin_pct']:.1f}%{'' if r['decisive'] else ' (inside floor)'}")

    decisive = [r for r in rows if r["decisive"]]
    tally: dict[str, int] = {}
    for r in decisive:
        tally[r["best"]] = tally.get(r["best"], 0) + 1
    log.info("\ndecisive configurations: %d of %d", len(decisive), len(rows))
    for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        log.info("   %-12s wins %d", name, count)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "method_comparison.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
