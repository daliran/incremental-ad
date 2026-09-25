"""Emit §1.40b's follow-up jobs: Iso-C on its grid-edge cells, and TIES/DARE sensitivity.

    python scripts/generate_merge_followups.py --runs_root $RUNS_ROOT \\
        --main_out $WORK/merge_baselines --sensitivity_out $WORK/merge_sensitivity --out $WORK/sweeps

Registered in EXPERIMENTS.md §1.40b before any of this ran. Forecasting only, training-free.

**Iso-C re-runs go into the MAIN output directory** on an extended grid, so
`merge_baselines_report` takes them as the newest grid for those (run, rule) pairs and §1.40's
table updates in place. The cells are the ones §1.40's report flagged `alpha_at_edge` for Iso-C —
listed here explicitly, not re-derived, so the scope is the registered one.

**Sensitivity runs go into their OWN directory.** Their tags carry the setting (`ties-k0.5_a…`,
`dare-p0.9_a…`) and are read by `merge_sensitivity_report`, never by the main report, so a
non-default setting can never be mistaken for the reference rule's result.
"""

import argparse
import csv
from pathlib import Path

BASE_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
ISO_EXTENSION = (5.0, 8.0, 12.0)
ISO_EDGE_CELLS = {("ETTh2", "2"), ("ETTh2", "3"), ("exchange", "2"), ("exchange", "3"),
                  ("exchange", "5"), ("PSM-forecast", "2")}
SENSITIVITY_CELLS = {("ETTh1", "3"), ("ETTh2", "3"), ("ETTm2", "3"), ("exchange", "3"),
                     ("PSM-forecast", "3")}
TIES_DENSITIES = (0.1, 0.5, 1.0)
DARE_DROP_RATES = (0.3, 0.5, 0.9)


def runs_of(runs_root: Path, experiment: str) -> list[Path]:
    group = runs_root / experiment
    return ([p for p in sorted(group.iterdir())
             if p.is_dir() and (p / "merged" / "checkpoints" / "best.pt").is_file()]
            if group.is_dir() else [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--main_out", type=Path, required=True)
    parser.add_argument("--sensitivity_out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    iso, sensitivity = [], []
    grid = " ".join(f"{a:g}" for a in BASE_GRID)
    iso_grid = " ".join(f"{a:g}" for a in BASE_GRID + ISO_EXTENSION)
    with args.spec.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["n"])
            for run in runs_of(args.runs_root, row["merge_experiment"]):
                if key in ISO_EDGE_CELLS:
                    iso.append(f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                               f"--out {args.main_out} --baseline_rule iso_c "
                               f"--alpha_grid {iso_grid}")
                if key in SENSITIVITY_CELLS:
                    base = (f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                            f"--out {args.sensitivity_out} --alpha_grid {grid}")
                    parts = ([f"{base} --baseline_rule ties --ties_density {k:g}"
                              for k in TIES_DENSITIES]
                             + [f"{base} --baseline_rule dare --dare_drop_rate {p:g}"
                                for p in DARE_DROP_RATES])
                    sensitivity.append(" && ".join(parts))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, commands in (("merge_followup_iso.sh", iso),
                           ("merge_followup_sensitivity.sh", sensitivity)):
        (args.out / name).write_text("\n".join(commands) + "\n")
        print(f"  {name:32s} {len(commands):>3} job(s)")


if __name__ == "__main__":
    main()
