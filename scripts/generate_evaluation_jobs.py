"""Emit §1.42's calibration re-merges and §1.43's prequential scoring jobs.

    python scripts/generate_evaluation_jobs.py --runs_root $RUNS_ROOT \\
        --calibration_out $WORK/merge_calibration --prequential_out $WORK/prequential \\
        --out $WORK/sweeps

Both are training-free and registered in EXPERIMENTS.md before any run.

- **§1.42** — one job per (AD run, rule): §1.40's AD grid, plus α = 0 (the base) on TA only,
  plus the distance-matched point on the four other rules, with the test recording split at
  c ∈ {5, 10, 20, 30}%. Validation is NOT skipped: §1.12's reconstruction-selected α is read from
  the same job. Written to its own directory, so §1.40's tree is never touched.
- **§1.43** — one job per forecasting merge run, paired with the sequential run of the same seed.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RULES = ("ta", "dare", "ties", "iso_c", "tsv")
AD_GRID = (0.5, 1.0, 2.0, 3.0, 5.0)
FRACTIONS = (0.05, 0.10, 0.20, 0.30)


def runs_with(group: Path, checkpoint: str) -> list[Path]:
    if not group.is_dir():
        return []
    return [p for p in sorted(group.iterdir()) if (p / checkpoint).is_file()]


def seed_of(run: Path):
    return json.loads((run / "config.json").read_text())["args"].get("seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--calibration_out", type=Path, required=True)
    parser.add_argument("--prequential_out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    calibration, prequential, problems = [], [], []
    fractions = " ".join(f"{f:g}" for f in FRACTIONS)
    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))
    for row in spec:
        merges = runs_with(args.runs_root / row["merge_experiment"], "merged/checkpoints/best.pt")
        if row["metric"] == "window_auroc":
            for run in merges:
                for rule in RULES:
                    grid = ((0.0,) + AD_GRID) if rule == "ta" else AD_GRID
                    extra = "" if rule == "ta" else " --distance_match_alpha 1"
                    calibration.append(
                        f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                        f"--out {args.calibration_out} --baseline_rule {rule} "
                        f"--alpha_grid {' '.join(f'{a:g}' for a in grid)}{extra} "
                        f"--calibration_split {fractions}")
            continue
        chains = defaultdict(list)
        for chain in runs_with(args.runs_root / row["seq_experiment"],
                               "continual_0/checkpoints/best.pt"):
            chains[seed_of(chain)].append(chain)
        for run in merges:
            matched = chains.get(seed_of(run), [])
            if len(matched) != 1:
                problems.append(f"{row['dataset']} n={row['n']} {run.name}: "
                                f"{len(matched)} chain run(s) with seed {seed_of(run)}")
                continue
            prequential.append(
                f"python -m incremental_ad.analysis.prequential --run_dir {run} "
                f"--chain_dir {matched[0]} --out {args.prequential_out}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, commands in (("calibration.sh", calibration), ("prequential.sh", prequential)):
        (args.out / name).write_text("\n".join(commands) + "\n")
        print(f"  {name:18s} {len(commands):>3} job(s)")
    for item in problems:
        print(f"  ⚠️  {item}")


if __name__ == "__main__":
    main()
