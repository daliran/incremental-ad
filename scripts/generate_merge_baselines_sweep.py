"""Emit §1.40's sweep: every rule, every run of record, one alpha grid per task family.

    python scripts/generate_merge_baselines_sweep.py --runs_root $RUNS_ROOT \\
        --remerge_out $WORK/merge_baselines --out $WORK/sweeps

Which runs are authoritative comes from `analysis_specs/method_comparison_spec.csv`, never from
experiment-name patterns. Training-free: each command recombines checkpoints that exist.

**Two grids, and why.** Forecasting selects alpha on validation, so it needs a grid wide enough
to hold every rule's optimum — the rules' alphas are on different scales — and it is cheap: a
full forecasting run of all five rules takes ~90 s. AD cannot select on validation (§1.12), so
its headline is each rule at the alpha where it travels exactly as far from the base as task
arithmetic does at its committed alpha = 1.0; the grid supplies only an upper bound, while a SWaT
evaluation costs ~5 minutes per alpha. So AD gets five points bracketing both scales — reaching
5.0, because a smoke run had TIES still improving at 3.0 — plus the matched point, and no val.

**Job layout.** Forecasting: one job per run, all rules (fast, and one model load). AD: one job
per (run, rule), so the slow SWaT evaluations spread across GPUs rather than queueing inside one.
Output paths are keyed by experiment, run and rule, never by SLURM_JOB_ID, so several rules in
one job cannot overwrite each other.
"""

import argparse
import csv
from pathlib import Path

RULES = ("ta", "dare", "ties", "iso_c", "tsv")
FORECAST_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
AD_GRID = (0.5, 1.0, 2.0, 3.0, 5.0)
# TA's distance-matched alpha is 1.0 by definition, so AD's headline point for every other rule
# is the alpha that travels as far as TA does at 1.0.
AD_MATCH_ALPHA = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--remerge_out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    forecasting, ad, missing = [], [], []
    with args.spec.open() as fh:
        for row in csv.DictReader(fh):
            group = args.runs_root / row["merge_experiment"]
            runs = ([p for p in sorted(group.iterdir())
                     if p.is_dir() and (p / "merged" / "checkpoints" / "best.pt").is_file()]
                    if group.is_dir() else [])
            if not runs:
                missing.append(f"{row['dataset']} n={row['n']}: {row['merge_experiment']}")
                continue
            is_ad = not row["metric"].startswith("forecast/")
            for run in runs:
                base = (f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                        f"--out {args.remerge_out}")
                if is_ad:
                    grid = " ".join(f"{a:g}" for a in AD_GRID)
                    for rule in RULES:
                        match = ("" if rule == "ta"
                                 else f" --distance_match_alpha {AD_MATCH_ALPHA:g}")
                        ad.append(f"{base} --baseline_rule {rule} --alpha_grid {grid} "
                                  f"--test_only{match}")
                else:
                    grid = " ".join(f"{a:g}" for a in FORECAST_GRID)
                    forecasting.append(" && ".join(
                        f"{base} --baseline_rule {rule} --alpha_grid {grid}" for rule in RULES))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, commands in (("merge_baselines_forecast.sh", forecasting),
                           ("merge_baselines_ad.sh", ad)):
        (args.out / name).write_text("\n".join(commands) + "\n")
        print(f"  {name:28s} {len(commands):>4} job(s)")
    for item in missing:
        print(f"  ⚠️  no run of record: {item}")


if __name__ == "__main__":
    main()
