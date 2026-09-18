"""Emit §1.38's P5 sweep: the paper's projection at matched DISTANCE, forecasting only.

    python scripts/generate_p5_sweep.py --runs_root $RUNS_ROOT \\
        --remerge_out $WORK/remerge_closeout --out $WORK/sweeps

§1.37's P4 matched per-vector *coefficients* to the committed alpha*n. That is the treatment P2
gave BECAME, and it is only equivalent to matching the merge's distance from theta_0 when the rule
leaves the task vectors intact. OPCM's projection shrinks them, so P4's forecasting merges landed
at 0.18-0.66x the intended distance and could not separate "the projection is harmful" from "the
merge travelled too little". This sweep fixes that control.

**Forecasting only, and deliberately.** On SWaT and PSM the committed alpha is already 1.0, the P4
rescale was a near no-op (distance ratio 0.96-1.15) and the loss was unchanged to within 0.17pp —
so the AD half of `C34` is already settled and re-running it would be re-measuring a closed claim,
which CLAUDE.md's freeze forbids. The six datasets here are the ones the confound actually touched.

Strength is never chosen here: `remerge.py` reads the committed alpha from `merge_scale/selected`,
falling back to `config.json`. Runs of record come from `analysis_specs/method_comparison_spec.csv`.
"""

import argparse
import csv
from pathlib import Path

# The datasets whose P4 cells were confounded. SWaT and PSM are excluded on purpose: see above.
FORECASTING = {"ETTh1", "ETTh2", "ETTm2", "exchange", "PSM-forecast", "SWaT-forecast"}


def runs_of(runs_root: Path, experiment: str) -> list[Path]:
    group = runs_root / experiment
    if not group.is_dir():
        return []
    return [p for p in sorted(group.iterdir())
            if p.is_dir() and (p / "merged" / "checkpoints" / "best.pt").is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--remerge_out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    args = parser.parse_args()

    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    commands, missing, skipped = [], [], []
    for row in spec:
        if row["dataset"] not in FORECASTING:
            skipped.append(row["dataset"])
            continue
        runs = runs_of(args.runs_root, row["merge_experiment"])
        if not runs:
            missing.append(f"{row['dataset']} n={row['n']}: {row['merge_experiment']}")
            continue
        for run in runs:
            for threshold in args.thresholds:
                commands.append(
                    f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                    f"--out {args.remerge_out} --merge_rule opcm_paper_distance "
                    f"--opcm_threshold {threshold} "
                    f"--tag opcm_distance_t{int(threshold * 100):03d}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "remerge_opcm_distance.sh"
    path.write_text("\n".join(commands) + "\n" if commands else "")
    print(f"  {path.name:32s} {len(commands):>4} commands")
    print(f"  skipped (AD, C34 already settled there): {sorted(set(skipped))}")
    if missing:
        print(f"\n  ⚠️  {len(missing)} spec row(s) have no run with a merged checkpoint:")
        for item in missing:
            print(f"      {item}")
    print(f"\n  total {len(commands)} re-merges, no training")


if __name__ == "__main__":
    main()
