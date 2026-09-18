"""Emit the §1.37 sweep: the paper's OPCM projection at each run's committed magnitude.

    python scripts/generate_c34_sweep.py --runs_root $RUNS_ROOT \\
        --remerge_out $WORK/remerge_closeout --out $WORK/sweeps

The last permitted merging experiment (CLAUDE.md's freeze rule): it closes `C34`, the only open
question §1.36 left. P1 found the paper's operator losing everywhere, with two candidate causes it
could not separate — the projection, or the Thm-5.2 norm rule that put every cell 1.1-1.7x above
this project's order-1 alpha*.n. `--merge_rule opcm_paper_committed` holds the projection
bit-for-bit fixed (asserted collinear to 3.7e-15 in `verify_merge_rules.py`) and merges at the
committed alpha instead, so the comparison against plain summation at that same alpha isolates the
projection alone.

**Strength is never chosen here.** `remerge.py` reads it from `merge_scale/selected`, falling back
to `config.json`, exactly as every other sweep in this project does. Authoritative runs come from
`analysis_specs/method_comparison_spec.csv`, never from experiment-name patterns.

Every dataset with a run of record, n in {2,3,5}, thresholds 0.3/0.5/0.7 — the same grid §1.36
used for P1, so the two tables are directly comparable cell by cell.
"""

import argparse
import csv
from pathlib import Path


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

    commands, missing = [], []
    for row in spec:
        runs = runs_of(args.runs_root, row["merge_experiment"])
        if not runs:
            missing.append(f"{row['dataset']} n={row['n']}: {row['merge_experiment']}")
            continue
        for run in runs:
            for threshold in args.thresholds:
                commands.append(
                    f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                    f"--out {args.remerge_out} --merge_rule opcm_paper_committed "
                    f"--opcm_threshold {threshold} "
                    f"--tag opcm_committed_t{int(threshold * 100):03d}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "remerge_opcm_committed.sh"
    path.write_text("\n".join(commands) + "\n" if commands else "")
    print(f"  {path.name:34s} {len(commands):>4} commands")
    if missing:
        print(f"\n  ⚠️  {len(missing)} spec row(s) have no run with a merged checkpoint — "
              f"reported missing, not retrained:")
        for item in missing:
            print(f"      {item}")
    print(f"\n  total {len(commands)} re-merges, no training")


if __name__ == "__main__":
    main()
