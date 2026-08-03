"""Measure the parameter-space geometry of the task vectors in completed runs.

Reads only the checkpoints of `IncrementalTaskArithmeticPipeline` runs — no dataset, no
model construction, no GPU — so it is cheap enough to run over every run on disk.

    python -m incremental_ad.analysis.geometry_report runs/ad_grid_psm/20260703_165802_5b43001d
    python -m incremental_ad.analysis.geometry_report runs/ad_grid_psm/* --out /tmp/geom

Run directories are named explicitly — use the shell to expand a group. Each run gets a
directory of detail CSVs; one `geometry_summary.csv` across the named runs collects the
per-run scalars, which is the table to correlate against merging outcome. Interpretation
is deliberately left out — this writes measurements, not conclusions.
"""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path

from incremental_ad.framework.core.checkpoints import load_model_state
from incremental_ad.framework.merging import geometry_report

log = logging.getLogger("geometry_report")

INCREMENTAL_PIPELINE = "IncrementalTaskArithmeticPipeline"

SUMMARY_FIELDS = [
    "experiment_name",
    "run_id",
    "dataset",
    "task",
    "n_segments",
    "merge_scale",
    "baseline_fraction",
    "n_parameters",
    "mean_offdiag_cosine",
    "min_offdiag_cosine",
    "max_offdiag_cosine",
    "cosine_at_distance_1",
    "effective_rank",
    "mean_tau_norm",
    "mean_tau_over_base",
    "min_tau_over_base",
    "max_tau_over_base",
    "run_dir",
]


def read_run_config(run_dir: Path) -> dict:
    """The run's config.json, rejecting anything this script cannot analyse."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{run_dir}: not a run directory (no config.json)")

    config = json.loads(config_path.read_text())
    pipeline = config.get("pipeline")
    if pipeline != INCREMENTAL_PIPELINE:
        raise ValueError(
            f"{run_dir}: pipeline is {pipeline!r}, expected {INCREMENTAL_PIPELINE!r} "
            "- there are no task vectors to measure"
        )
    return config


def load_checkpoints(run_dir: Path, name: str) -> tuple[dict, list[dict]] | None:
    """Baseline state plus each fine-tune state in segment order, or None if incomplete."""
    baseline = run_dir / "baseline" / "checkpoints" / f"{name}.pt"
    finetunes = sorted(
        (d for d in run_dir.iterdir() if d.is_dir() and re.fullmatch(r"finetune_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),  # finetune_10 must not sort before finetune_2
    )
    finetune_checkpoints = [d / "checkpoints" / f"{name}.pt" for d in finetunes]

    if not baseline.is_file() or not finetune_checkpoints:
        return None
    missing = [c for c in finetune_checkpoints if not c.is_file()]
    if missing:
        log.warning("%s: missing %d fine-tune checkpoint(s), skipping", run_dir, len(missing))
        return None

    return load_model_state(baseline), [load_model_state(c) for c in finetune_checkpoints]


def _offdiag(matrix: list[list[float]]) -> list[float]:
    return [
        matrix[i][j]
        for i in range(len(matrix))
        for j in range(len(matrix))
        if i != j and matrix[i][j] == matrix[i][j]  # drop nan
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_run_csvs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry.json").write_text(json.dumps(report, indent=2))

    cosine = report["cosine"]
    with (out_dir / "cosine_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "cosine", "temporal_distance"])
        for i, row in enumerate(cosine):
            for j, value in enumerate(row):
                writer.writerow([i, j, value, abs(i - j)])

    norms = report["norms"]
    with (out_dir / "norms.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "tau_norm", "tau_over_base", "base_norm"])
        for i, (tau, ratio) in enumerate(zip(norms["tau_norm"], norms["tau_over_base"])):
            writer.writerow([i, tau, ratio, norms["base_norm"]])

    rank = report["effective_rank"]
    with (out_dir / "effective_rank.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["singular_index", "singular_value", "p", "cumulative_energy"])
        cumulative = 0.0
        for i, (value, p) in enumerate(zip(rank["singular_values"], rank["p"])):
            cumulative += p
            writer.writerow([i, value, p, cumulative])

    with (out_dir / "cosine_vs_distance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["temporal_distance", "mean", "std", "n_pairs"])
        for distance, stats in sorted(report["cosine_vs_distance"].items()):
            writer.writerow([distance, stats["mean"], stats["std"], stats["n"]])

    with (out_dir / "per_tensor_cosine.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "i", "j", "cosine"])
        for parameter, matrix in report["per_tensor_cosine"].items():
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if i != j:
                        writer.writerow([parameter, i, j, value])


def summary_row(config: dict, report: dict, run_dir: Path) -> dict:
    args = config.get("args", {})
    offdiag = _offdiag(report["cosine"])
    distance_1 = report["cosine_vs_distance"].get(1, {}).get("mean", float("nan"))
    ratios = report["norms"]["tau_over_base"]

    return {
        "experiment_name": config.get("experiment_name"),
        "run_id": config.get("run_id"),
        "dataset": config.get("dataset"),
        "task": config.get("task"),
        "n_segments": report["n_segments"],
        "merge_scale": args.get("pipeline_merge_scale"),
        "baseline_fraction": args.get("dataset_baseline_fraction"),
        "n_parameters": report["n_parameters"],
        "mean_offdiag_cosine": _mean(offdiag),
        "min_offdiag_cosine": min(offdiag) if offdiag else float("nan"),
        "max_offdiag_cosine": max(offdiag) if offdiag else float("nan"),
        "cosine_at_distance_1": distance_1,
        "effective_rank": report["effective_rank"]["effective_rank"],
        "mean_tau_norm": _mean(report["norms"]["tau_norm"]),
        "mean_tau_over_base": _mean(ratios),
        "min_tau_over_base": min(ratios),
        "max_tau_over_base": max(ratios),
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="run directories to analyse (expand groups with a shell glob)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output root (default: $RUNS_ROOT/analysis/geometry)")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_root = args.out or Path(os.environ.get("RUNS_ROOT", "runs")) / "analysis" / "geometry"
    log.info("Analysing %d run(s); writing to %s", len(args.run_dirs), out_root)

    rows = []
    for run_dir in args.run_dirs:
        try:
            config = read_run_config(run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            log.error("%s", error)
            return 1

        states = load_checkpoints(run_dir, args.checkpoint)
        if states is None:
            continue
        base_state, ft_states = states

        report = geometry_report(base_state, ft_states)
        row = summary_row(config, report, run_dir)

        write_run_csvs(report, out_root / str(row["experiment_name"]) / str(row["run_id"]))
        rows.append(row)

        log.info(
            "  %-38s n=%d  mean_offdiag_cos=%+.4f  eff_rank=%.2f  mean_|tau|/|theta0|=%.4f",
            f"{row['experiment_name']}/{row['run_id']}",
            row["n_segments"], row["mean_offdiag_cosine"],
            row["effective_rank"], row["mean_tau_over_base"],
        )

    if not rows:
        log.error("No run had a complete set of checkpoints.")
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "geometry_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("\n%d run(s) summarised -> %s", len(rows), summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
