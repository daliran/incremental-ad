"""Build the re-merge commands for OPCM and BECAME across every dataset with checkpoints.

    python scripts/generate_remerge_sweep.py --runs_root $RUNS_ROOT --out $WORK/sweeps

OPCM and BECAME were only ever evaluated on PSM-forecast n=3, because its 1.16% floor is the
only one small enough to resolve a ~3% effect. That was right for the *published* result and a
poor reason not to look elsewhere: both methods are **training-free**, so re-running them
everywhere costs an evaluation pass per configuration, not a retrain.

**Which runs are authoritative comes from `analysis_specs/method_comparison_spec.csv`**, never
from experiment-name patterns. The naming is genuinely inconsistent — `window_etth1_W3` in one
family, `etth2_window_W3` in another — and a prefix rule silently missed ten groups the last time
one was used (see `checkpoint_manifest.py`). The spec is the same file §1.26 reads, so this sweep
cannot drift from the table it will be compared against.

**Strength is never chosen here.** Each re-merge runs at the strength the source run committed to,
which `remerge.py` reads from `merge_scale/selected` first and `config.json` second. Nothing in
this sweep is tuned; the comparison is "same strength, different rule".

The OPCM threshold sweep (0.3 / 0.7 alongside the main 0.5) is emitted only for the cheap
datasets: SWaT-forecast scores 390k windows per pass and PSM/SWaT AD are nearly as large, so
tripling them would dominate the runtime for a hyperparameter §1.31 already showed does not
resolve.
"""

import argparse
import csv
import json
from pathlib import Path

# Datasets whose test sets are small enough that three thresholds are cheap.
CHEAP = {"ETTh1", "ETTh2", "ETTm2", "exchange"}
# BECAME has never been run on anomaly detection. These are the AD rows of the spec.
AD_DATASETS = {"SWaT", "PSM"}


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
    parser.add_argument("--remerge_out", type=Path, required=True,
                        help="where remerge.py writes its per-run output")
    parser.add_argument("--out", type=Path, required=True, help="directory for the command files")
    parser.add_argument("--fisher_batches", type=int, default=256,
                        help="§1.32 found the estimate saturates at the full pass (146 batches "
                             "per shard), so 256 is 'all' for these loaders")
    args = parser.parse_args()

    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    opcm_main, opcm_extra, became, missing = [], [], [], []
    for row in spec:
        dataset, n, experiment = row["dataset"], row["n"], row["merge_experiment"]
        runs = runs_of(args.runs_root, experiment)
        if not runs:
            missing.append(f"{dataset} n={n}: {experiment}")
            continue
        for run in runs:
            base = (f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                    f"--out {args.remerge_out}")
            opcm_main.append(f"{base} --merge_rule opcm --opcm_threshold 0.5 "
                             f"--tag opcm_t050")
            if dataset in CHEAP:
                for threshold in (0.3, 0.7):
                    tag = f"opcm_t{int(threshold * 100):03d}"
                    opcm_extra.append(f"{base} --merge_rule opcm "
                                      f"--opcm_threshold {threshold} --tag {tag}")
            if dataset in AD_DATASETS:
                became.append(f"{base} --coefficient_source became "
                              f"--fisher_batches {args.fisher_batches} --fisher_seed 0 "
                              f"--tag became_fb{args.fisher_batches}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, commands in (("remerge_opcm_main.sh", opcm_main),
                           ("remerge_opcm_thresholds.sh", opcm_extra),
                           ("remerge_became_ad.sh", became)):
        (args.out / name).write_text("\n".join(commands) + "\n" if commands else "")
        print(f"  {name:32s} {len(commands):>4} commands")
    if missing:
        print(f"\n  ⚠️  {len(missing)} spec row(s) have no run with a merged checkpoint — "
              f"reported, not retrained:")
        for item in missing:
            print(f"      {item}")
    print(f"\n  total {len(opcm_main) + len(opcm_extra) + len(became)} re-merges, "
          f"no training")


if __name__ == "__main__":
    main()
