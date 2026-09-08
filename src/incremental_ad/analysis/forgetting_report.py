"""ACC / BWT for the sequential chains — the direct evidence for "sequential forgets".

    python -m incremental_ad.analysis.forgetting_report --runs_root results_archive/runs \\
        --out $OUT/forgetting

`ContinualFineTuningPipeline` has always written ACC, BWT and the base-slice retention ratio to
`continual_summary/result.json`, and **no table in EXPERIMENTS.md reads them**. §1.13 says
"merging starves as shards shrink, continual forgets as steps accumulate" and supports the
second half with the *symptom* — test error rising from 0.220 to 0.531 on exchange_rate — rather
than with the measurement that names it. This is the measurement.

Definitions come from the pipeline, not from here, so the two cannot diverge:

- **ACC** — mean over regimes of the final model's loss on each regime. Loss-shaped, so lower is
  better, which is the opposite orientation from the classification literature's ACC. The
  pipeline's docstring is the authority.
- **BWT** — mean over earlier regimes of (final loss − loss right after that regime was trained).
  **Positive means later training hurt earlier regimes**, i.e. positive is forgetting. Again the
  opposite sign convention from the accuracy-shaped original, and the reason this is stated twice.
- **base-slice retention** — the final model's loss on the baseline's own slice, as a ratio to the
  baseline model's. > 1 means the chain drifted away from where it started.

The merged model has no chain, so it has no BWT in the same sense. Its comparable quantity is the
merge cost from the transfer matrix — merged loss on regime i ÷ that regime's own specialist —
which is bounded and does not accumulate over steps. Both are reported side by side because that
contrast *is* the §1.13 claim, but they are different quantities and are labelled as such.

Training-free: reads `result.json` and `transfer_matrix.csv` only.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("forgetting_report")

FIELDS = ["dataset", "experiment", "n_segments", "metric", "n_seeds",
          "acc", "acc_sd", "bwt", "bwt_sd", "acc_ratio_to_baseline",
          "base_slice_ratio_final", "merge_cost", "merge_cost_source"]


def summary_metrics(run: Path) -> dict[str, float]:
    path = run / "continual_summary" / "result.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text()) or {}).get("metrics") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def merge_costs(routing_dir: Path | None) -> dict[str, float]:
    """group -> merge cost, read from `routing_report`'s summary.

    Deliberately *not* recomputed here. A first version derived it from `transfer_matrix.csv`
    directly and produced 1.43-2.13x where §1.16 publishes ~1.0-1.1x — two definitions of one
    quantity, which is the failure mode CLAUDE.md names explicitly. `routing_report` owns this
    number; this reads it.
    """
    costs: dict[str, float] = {}
    if routing_dir is None:
        return costs
    for name in ("routing_forecast", "routing_ad", "routing_psm_forecast"):
        path = routing_dir.parent / name / "routing_summary.csv"
        if not path.is_file():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    costs[row["group"]] = float(row["merge_cost"])
                except (TypeError, ValueError, KeyError):
                    continue
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+",
                        default=["forecast/mse", "window_auroc"])
    parser.add_argument("--routing_dir", type=Path,
                        help="an audit routing_* directory; merge cost is read from "
                             "routing_report rather than recomputed")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    costs = merge_costs(args.routing_dir)
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    info: dict[tuple, dict] = {}
    for config_path in sorted(args.runs_root.glob("*/*/config.json")):
        run = config_path.parent
        metrics = summary_metrics(run)
        if not metrics:
            continue
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        run_args = config.get("args") or {}
        # `*_oldmask` are the pre-§1.30 runs, kept on disk as evidence for that correction
        # and explicitly superseded. They must not enter a published aggregate.
        if run.parent.name.endswith("_oldmask"):
            continue
        key = (run.parent.name, run_args.get("dataset_n_finetune_segments"))
        info[key] = {"dataset": config.get("dataset"),
                     "n_segments": run_args.get("dataset_n_finetune_segments")}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped[key][name].append(value)

    if not grouped:
        raise SystemExit("no continual_summary blocks found — is --runs_root correct?")

    rows: list[dict] = []
    for key, metrics in sorted(grouped.items()):
        experiment, _n = key
        for base_metric in args.metrics:
            acc = metrics.get(f"{base_metric}/ACC")
            bwt = metrics.get(f"{base_metric}/BWT")
            if not acc or not bwt:
                continue
            merge_experiment = experiment.replace("continual", "merge").replace("seq", "merge")
            rows.append({
                "dataset": info[key]["dataset"], "experiment": experiment,
                "n_segments": info[key]["n_segments"], "metric": base_metric,
                "n_seeds": len(acc),
                "acc": round(st.mean(acc), 6),
                "acc_sd": round(st.stdev(acc), 6) if len(acc) > 1 else 0.0,
                "bwt": round(st.mean(bwt), 6),
                "bwt_sd": round(st.stdev(bwt), 6) if len(bwt) > 1 else 0.0,
                "acc_ratio_to_baseline": round(
                    st.mean(metrics.get(f"{base_metric}/ACC_ratio_to_baseline", [float("nan")])), 6),
                "base_slice_ratio_final": round(
                    st.mean(metrics.get(f"{base_metric}/base_slice_ratio_final",
                                        [float("nan")])), 6),
                "merge_cost": "",
                "merge_cost_source": "",
            })
            group = f"{merge_experiment}_diagnostics"
            if group in costs:
                rows[-1]["merge_cost"] = round(costs[group], 4)
                rows[-1]["merge_cost_source"] = f"routing_report:{group}"

    log.info("%-30s %-4s %-14s %8s %8s %8s", "experiment", "n", "metric", "ACC", "BWT", "merge")
    for row in rows:
        log.info("%-30s %-4s %-14s %8.4f %8.4f %8s", row["experiment"], row["n_segments"],
                 row["metric"], row["acc"], row["bwt"], row["merge_cost"] or "—")
    positive = [r for r in rows if r["bwt"] > 0]
    log.info("\n%d of %d chains have positive BWT (later training hurt earlier regimes)",
             len(positive), len(rows))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "forgetting_summary.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
