"""Build the closing advanced-merging sweep: rescaled BECAME, and OPCM under reversed order.

    python scripts/generate_remerge_closeout.py --runs_root $RUNS_ROOT \\
        --remerge_out $WORK/remerge_closeout --out $WORK/sweeps

Three falsification tests, registered as P1/P2/P3 in EXPERIMENTS.md §1.36 **before** any of this
runs. Everything here is **training-free** — each command recombines checkpoints that already
exist, at the strength the source run committed to. Which runs are authoritative comes from
`analysis_specs/method_comparison_spec.csv`, never from experiment-name patterns.

**P2 — BECAME's weighting at a chosen strength.** `became_rescaled` keeps BECAME's *relative*
per-period weights and rescales their sum to the source run's validation-selected α·n, which
separates "is Fisher weighting useful" from "is the convex fold's α·n = 1.0 the right magnitude".
The 1/n control at the *same* α·n is emitted as its own tag rather than borrowed from the run's
stored `merged/test/result.json`, so both columns come from the same evaluation code and a
difference cannot be an artefact of two different paths. (It doubles as a check that `remerge.py`
reproduces the published merge's *metrics*, not just its weights.)

**P1 — the paper's OPCM.** `--merge_rule opcm_paper` is Tang et al. 2025 Algorithm 1, distinct
from the simplified `opcm` of §1.31/§1.35 and never reported as it. It takes **no merge scale**:
its lambda pins the merged model at the mean task-vector norm from the base, so the comparison is
"the paper's rule at the magnitude it chooses" against "plain summation at the magnitude the run
committed to", and `implied_alpha_times_n` is emitted so the two magnitudes are visible rather
than conflated.

**P3 — order reversal.** OPCM projects each incoming vector out of its predecessors' span, so the
first period is never projected and the last is projected most. Reversed, that ordering inverts.
A `sum --reverse_order` null check is emitted for one seed per dataset: plain summation is
order-inert, so those rows **must** match the forward ones exactly. If they do not, the reversal
plumbing is wrong and every P3 row is void — which is why the check is generated, not assumed.
"""

import argparse
import csv
from pathlib import Path

# P2: BECAME has a fixed-magnitude problem worth separating on the AD pair, plus the one
# forecasting case where it was actually published (§1.31).
BECAME_SCOPE = {("SWaT", None), ("PSM", None), ("PSM-forecast", "3")}
# P3: the four datasets where OPCM produced a non-tie in §1.35, in either direction.
REVERSAL_DATASETS = {"exchange", "ETTh2", "ETTh1", "ETTm2"}


def in_scope(scope: set, dataset: str, n: str) -> bool:
    return (dataset, None) in scope or (dataset, n) in scope


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
    parser.add_argument("--fisher_batches", type=int, default=256,
                        help="§1.32 found the estimate saturates at the full pass, so 256 is "
                             "'all' for these loaders")
    parser.add_argument("--opcm_threshold", type=float, default=0.5)
    parser.add_argument("--paper_thresholds", type=float, nargs="*", default=[0.3, 0.5, 0.7],
                        help="projection thresholds for the paper's OPCM; its published optimum "
                             "is a flat 0.4-0.6, so 0.3/0.7 bracket it on both sides")
    args = parser.parse_args()

    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    became, reversal, nullcheck, paper, paper_rev, missing = [], [], [], [], [], []
    for row in spec:
        dataset, n, experiment = row["dataset"], row["n"], row["merge_experiment"]
        runs = runs_of(args.runs_root, experiment)
        if not runs:
            missing.append(f"{dataset} n={n}: {experiment}")
            continue
        for index, run in enumerate(runs):
            base = (f"python -m incremental_ad.analysis.remerge --run_dir {run} "
                    f"--out {args.remerge_out}")
            if in_scope(BECAME_SCOPE, dataset, n):
                became.append(f"{base} --coefficient_source became_rescaled "
                              f"--fisher_batches {args.fisher_batches} --fisher_seed 0 "
                              f"--tag became_rescaled")
                # 1/n at the same alpha*n, through the same evaluation path.
                became.append(f"{base} --tag control_uniform")
            # P1: the paper's operator, every dataset and n, at all three thresholds. It takes
            # no merge scale (Algorithm 1 line 14 sets its own), so the only knob swept is the
            # projection threshold, whose published optimum is 0.4-0.6.
            for threshold in args.paper_thresholds:
                paper.append(f"{base} --merge_rule opcm_paper --opcm_threshold {threshold} "
                             f"--tag opcm_paper_t{int(threshold * 100):03d}")
            if dataset in REVERSAL_DATASETS:
                # P3 applies to the paper's operator too — more sharply, because it projects out
                # the accumulated merge rather than the individual predecessors, so what the
                # first period contributes is never projected at all.
                paper_rev.append(f"{base} --merge_rule opcm_paper "
                                 f"--opcm_threshold {args.opcm_threshold} --reverse_order "
                                 f"--tag opcm_paper_t{int(args.opcm_threshold * 100):03d}_rev")
                reversal.append(f"{base} --merge_rule opcm "
                                f"--opcm_threshold {args.opcm_threshold} --reverse_order "
                                f"--tag opcm_t{int(args.opcm_threshold * 100):03d}_rev")
                if index == 0:
                    nullcheck.append(f"{base} --reverse_order --tag sum_rev_nullcheck")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, commands in (("remerge_became_rescaled.sh", became),
                           ("remerge_order_reversal.sh", reversal),
                           ("remerge_reversal_nullcheck.sh", nullcheck),
                           ("remerge_opcm_paper.sh", paper),
                           ("remerge_opcm_paper_reversed.sh", paper_rev)):
        (args.out / name).write_text("\n".join(commands) + "\n" if commands else "")
        print(f"  {name:34s} {len(commands):>4} commands")
    if missing:
        print(f"\n  ⚠️  {len(missing)} spec row(s) have no run with a merged checkpoint — "
              f"reported missing, not retrained:")
        for item in missing:
            print(f"      {item}")
    print(f"\n  total {len(became) + len(reversal) + len(nullcheck) + len(paper) + len(paper_rev)}"
          f" re-merges, no training")


if __name__ == "__main__":
    main()
