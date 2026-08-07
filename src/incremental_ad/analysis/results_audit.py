"""Recompute every published scalar from the runs on disk — the numbers of record.

Reads only `config.json` and `result.json` files. No checkpoints, no model, no GPU, no
dataset: it is pure aggregation over finished runs, so it is cheap and can be re-run at any
time to check the documentation against the data.

    python -m incremental_ad.analysis.results_audit --runs_root $RUNS_ROOT --out /tmp/audit
    python -m incremental_ad.analysis.results_audit --runs_root $RUNS_ROOT --metric window_auroc

Writes two CSVs. `run_metrics.csv` is one row per (experiment, block, metric) with the
per-seed mean, sample standard deviation and seed list — the raw material. `derived.csv` is
one row per experiment with the quantities the write-ups actually quote: the reproducibility
floor, the base-to-joint headroom, the committed merge scale, GRR and merge cost.

**Why this file exists.** Three separate numbers published in EXPERIMENTS.md turned out to be
computed with a different denominator, a different aggregation order, or from an experiment
group other than the one named — each defensible in isolation, none reproducible afterwards
because no script of record existed. Everything below is therefore stated as an executable
definition rather than as prose.

Definitions, all fixed here and used nowhere else in the codebase:

**Aggregation.** Every quantity is computed from the per-seed *mean* of a metric. Where a
quantity is a ratio, the ratio is taken of the means — not the mean of per-seed ratios. The
two differ, and mixing them is what made the routing table irreproducible.

**Reproducibility floor** = sample standard deviation ÷ mean of the baseline-stage test
metric across the seeds of one experiment, as a percentage. It requires ≥2 seeds and is only
comparable *within* one experiment: several groups training the same configuration give
estimates that differ by up to 3.3×, so treat it as a rough threshold and never as a
significance test.

**Headroom** = (base − joint) ÷ base for error metrics, (joint − base) ÷ base for score
metrics. It is the share of the base model's error that training on all the data removes, and
it bounds what any update strategy can win.

**GRR** (gap recovery ratio) = (base − merged) ÷ (base − joint), the share of the
base-to-joint gap the merge closes. The joint model almost always lives in a *different*
experiment (a `StandardPipeline` run), so it is matched on dataset plus every `mae_tx_*` and
`dataset_*` argument except the partition arguments a joint run must differ on
(`baseline_fraction`, `n_finetune_segments`, `baseline_use_fraction`, `val_fraction`).
Matching by seed alone once paired four incompatible runs, so compatibility is checked first
and the seed is used only to choose among compatible candidates. GRR = 1.0 is parity with
joint training; read >1.0 as "the denominator was soft", not "better than possible", since
joint training is itself beatable by a window retrain.

**Merge cost** is deliberately *not* computed here. It is defined per period, on that
period's own held-out slice — merged(val_i) ÷ ft_i(val_i) — so it needs the transfer matrix,
not the test block. Computing it from test-set numbers gives a different quantity that happens
to look plausible (0.56–0.92 instead of ≈1.05); `routing_report.py` computes the real one.

**Percentage comparisons.** "A beats B by p%" is always p = (B − A) ÷ B for error metrics —
expressed as a fraction of **B, the alternative**. The reverse denominator inflates every
figure and is never used.

**Direction** is read from the metric name, never assumed: names containing auroc, auprc, f1,
precision, recall or accuracy are higher-better; everything else is treated as an error.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("results_audit")

HIGHER_IS_BETTER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")

# Blocks a pipeline may write, in the order they are reported.
BLOCKS = ("baseline/test", "train/test", "merged/test", "merged/val")

# Arguments a joint (StandardPipeline) run necessarily differs on — exempt from the
# compatibility check. Everything else must match exactly or the runs are not comparable.
PARTITION_ARGS = frozenset({
    "dataset_baseline_fraction", "dataset_n_finetune_segments",
    "dataset_baseline_use_fraction", "dataset_val_fraction",
})


def retention_periods(args: dict) -> dict[str, float] | None:
    """How much raw data each update strategy must retain, in units of one period.

    Merging stores no *training* data, but selecting alpha scores the pooled validation union
    (see EXPERIMENTS.md 0.6), which means keeping `val_base` permanently and every segment's
    validation slice as it arrives. A window retrain instead keeps W periods of training data
    plus a validation slice cut from them. Expressing both in periods makes them comparable to
    the retention crossover, which is measured in the same unit.

    Pure arithmetic on the run's own configuration — no measurement involved.
    """
    bf = args.get("dataset_baseline_fraction")
    n = args.get("dataset_n_finetune_segments")
    vf = args.get("dataset_val_fraction")
    if bf is None or vf is None or not n:
        return None
    period = (1.0 - bf) / n            # one segment, as a fraction of the series
    if period <= 0:
        return None
    merge_val = (bf * vf + period * vf * n) / period
    out = {"merge_alpha_selected": round(merge_val, 3), "merge_alpha_fixed": 0.0}
    for w in (1, 2, 3):
        out[f"window_W{w}"] = round((w * period + w * period * vf) / period, 3)
    return out


def comparable(a: dict, b: dict) -> bool:
    """True when two runs differ only in how the series was partitioned."""
    keys = {k for k in set(a) | set(b)
            if (k.startswith("dataset_") or k.startswith("mae_tx_")) and k not in PARTITION_ARGS}
    return all(a.get(k) == b.get(k) for k in keys)

RUN_FIELDS = ["experiment", "dataset", "pipeline", "n_segments", "block", "metric",
              "n_seeds", "seeds", "mean", "sd", "sd_pct"]
DERIVED_FIELDS = ["experiment", "dataset", "metric", "n_segments", "n_seeds", "base", "joint",
                  "joint_from", "merged", "floor_pct", "headroom_pct", "merge_scale_selected",
                  "grr", "grr_paired", "n_paired", "best_specialist",
                  "retention_merge_alpha_selected_periods", "retention_window_W2_periods",
                  "retention_window_W3_periods"]


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER_IS_BETTER)


def _metrics(path: Path) -> dict:
    f = path / "result.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text()).get("metrics") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def collect(runs_root: Path) -> dict:
    """experiment -> {"info": {...}, "blocks": {block: {metric: {seed: value}}}}."""
    out: dict[str, dict] = {}
    for cfg_path in sorted(runs_root.glob("*/*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        args = cfg.get("args") or {}
        seed = args.get("seed")
        if seed is None:
            continue
        run = cfg_path.parent
        experiment = run.parent.name
        entry = out.setdefault(experiment, {
            "info": {
                "dataset": cfg.get("dataset"),
                "pipeline": cfg.get("pipeline"),
                "n_segments": args.get("dataset_n_finetune_segments"),
                "args": args,
            },
            "blocks": defaultdict(lambda: defaultdict(dict)),
            "specialists": defaultdict(lambda: defaultdict(dict)),
        })
        # Named blocks, plus every per-segment specialist (finetune_i / continual_i).
        for block in BLOCKS:
            for metric, value in _metrics(run / block).items():
                entry["blocks"][block][metric][seed] = value
        for sub in sorted(run.glob("finetune_*/test")) + sorted(run.glob("continual_*/test")):
            name = sub.parent.name
            for metric, value in _metrics(sub).items():
                entry["specialists"][name][metric][seed] = value
    return out


def _agg(per_seed: dict) -> tuple[float, float, int]:
    values = list(per_seed.values())
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return st.mean(values), sd, len(values)


def audit(runs_root: Path, metric_filter: set[str] | None) -> tuple[list, list]:
    data = collect(runs_root)
    run_rows, derived_rows = [], []

    for experiment, entry in sorted(data.items()):
        info = entry["info"]
        for block, metrics in sorted(entry["blocks"].items()):
            for metric, per_seed in sorted(metrics.items()):
                if metric_filter and metric not in metric_filter:
                    continue
                mean, sd, n = _agg(per_seed)
                run_rows.append({
                    "experiment": experiment, "dataset": info["dataset"],
                    "pipeline": info["pipeline"], "n_segments": info["n_segments"],
                    "block": block, "metric": metric, "n_seeds": n,
                    "seeds": " ".join(str(s) for s in sorted(per_seed)),
                    "mean": round(mean, 6), "sd": round(sd, 6),
                    "sd_pct": round(100 * sd / mean, 3) if mean else "",
                })

        # Derived quantities need a primary metric; pick the first available per dataset.
        candidates = [m for m in ("forecast/mse", "window_auroc")
                      if m in entry["blocks"].get("baseline/test", {})
                      or m in entry["blocks"].get("train/test", {})]
        for metric in candidates:
            if metric_filter and metric not in metric_filter:
                continue
            base_seeds = entry["blocks"].get("baseline/test", {}).get(metric, {})
            merged_seeds = entry["blocks"].get("merged/test", {}).get(metric, {})
            scale_seeds = entry["blocks"].get("merged/val", {}).get("merge_scale/selected", {})

            base = _agg(base_seeds)[0] if base_seeds else None
            merged = _agg(merged_seeds)[0] if merged_seeds else None
            floor = (100 * _agg(base_seeds)[1] / base) if base_seeds and len(base_seeds) > 1 else None

            # The joint reference: this experiment's own train/test if it is a Standard run,
            # otherwise the best *compatible* joint experiment on the same dataset.
            joint_seeds = entry["blocks"].get("train/test", {}).get(metric, {})
            joint_from = experiment if joint_seeds else ""
            if not joint_seeds:
                for other, oe in sorted(data.items()):
                    if oe["info"]["dataset"] != info["dataset"]:
                        continue
                    cand = oe["blocks"].get("train/test", {}).get(metric, {})
                    if not cand or not comparable(info["args"], oe["info"]["args"]):
                        continue
                    # Prefer a candidate covering the same seeds.
                    if joint_seeds and not (set(cand) >= set(base_seeds or merged_seeds)):
                        continue
                    joint_seeds, joint_from = cand, other
                    if set(cand) >= set(base_seeds or merged_seeds):
                        break
            joint = _agg(joint_seeds)[0] if joint_seeds else None

            spec_means = {}
            for name, metrics_by_seed in entry["specialists"].items():
                if metric in metrics_by_seed:
                    spec_means[name] = _agg(metrics_by_seed[metric])[0]
            better = max if higher_is_better(metric) else min
            best_spec = better(spec_means.values()) if spec_means else None

            headroom = grr = None
            if base and joint:
                headroom = (100 * (joint - base) / base if higher_is_better(metric)
                            else 100 * (base - joint) / base)
            if base is not None and joint is not None and merged is not None and base != joint:
                grr = (base - merged) / (base - joint)

            # GRR is fragile: base and joint come from different experiments, so a mean-of-means
            # silently mixes different seed sets. Where the same seed exists in all three, pair
            # them and average the per-seed ratios instead; dividing by a run's own base cancels
            # the shared run-to-run variance. Both are reported so the gap is visible.
            grr_paired = None
            shared = set(base_seeds) & set(joint_seeds) & set(merged_seeds)
            if shared:
                ratios = [(base_seeds[s] - merged_seeds[s]) / (base_seeds[s] - joint_seeds[s])
                          for s in shared if base_seeds[s] != joint_seeds[s]]
                if ratios:
                    grr_paired = st.mean(ratios)

            if base is None and merged is None:
                continue
            derived_rows.append({
                "experiment": experiment, "dataset": info["dataset"], "metric": metric,
                "n_segments": info["n_segments"],
                "n_seeds": len(base_seeds or merged_seeds),
                "base": round(base, 6) if base is not None else "",
                "joint": round(joint, 6) if joint is not None else "",
                "joint_from": joint_from,
                "merged": round(merged, 6) if merged is not None else "",
                "floor_pct": round(floor, 3) if floor is not None else "",
                "headroom_pct": round(headroom, 2) if headroom is not None else "",
                "merge_scale_selected": (round(_agg(scale_seeds)[0], 4) if scale_seeds else ""),
                "grr": round(grr, 4) if grr is not None else "",
                "grr_paired": round(grr_paired, 4) if grr_paired is not None else "",
                "n_paired": len(shared) if joint_seeds else 0,
                "best_specialist": round(best_spec, 6) if best_spec is not None else "",
                **({"retention_merge_alpha_selected_periods": ret["merge_alpha_selected"],
                    "retention_window_W2_periods": ret["window_W2"],
                    "retention_window_W3_periods": ret["window_W3"]}
                   if (ret := retention_periods(info["args"])) else {}),
            })
    return run_rows, derived_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", action="append", help="restrict to these metrics")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_rows, derived_rows = audit(args.runs_root, set(args.metric) if args.metric else None)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (("run_metrics.csv", run_rows, RUN_FIELDS),
                               ("derived.csv", derived_rows, DERIVED_FIELDS)):
        path = args.out / name
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
