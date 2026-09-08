"""Build the §1.31 2×2 ablation commands for a given segment count.

    python scripts/generate_ablation_grid.py --runs_root $RUNS_ROOT \\
        --n_segments 3 --seeds 7 42 123 --out /tmp/opcm_cells.sh

Four cells — {plain sum, OPCM} × {swept α, BECAME λ\\*} — from the published PSM-forecast merge
config at the requested n, so hyperparameters cannot drift from §1.30's runs. Every command is
validated against the real parser before it is written.

Two composition rules are enforced here rather than discovered in the queue:

- **BECAME cells drop `--pipeline_select_merge_scale_on_val` and the extra-scale grid.** The
  coefficient is derived, so a validation sweep would be ignored; the pipeline asserts on the
  combination, and the grid is meaningless without it.
- **The baseline cell keeps the source run's α-selection settings unchanged**, because §1.31 is
  read entirely as differences from that cell and `check_ablation_baseline` asserts it equals
  §1.30's published merge.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_rolling_origin import command, validate  # noqa: E402

CELLS = (
    ("sum", "scale", "sum_scale"),
    ("opcm", "scale", "opcm_scale"),
    ("sum", "became", "sum_became"),
    ("opcm", "became", "opcm_became"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--n_segments", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--source", default=None,
                        help="merge experiment to read the config from "
                             "(default: adfc2_psm_merge_n<n>)")
    parser.add_argument("--prefix", default="opcm2_psm",
                        help="experiment-name prefix; the cell tag and n are appended")
    parser.add_argument("--opcm_threshold", type=float, default=0.5)
    parser.add_argument("--fisher_batches", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.source or f"adfc2_psm_merge_n{args.n_segments}"
    configs = sorted(glob.glob(str(args.runs_root / source / "*" / "config.json")))
    if not configs:
        raise SystemExit(f"no runs under {source} — cannot build the grid from it")
    base = json.loads(Path(configs[0]).read_text())["args"]

    commands = []
    for rule, coefficient, tag in CELLS:
        # n=3 keeps the historical names so §1.31's existing runs and checks still resolve.
        suffix = "" if args.n_segments == 3 else f"_n{args.n_segments}"
        overrides = {
            "experiment_name": f"{args.prefix}_{tag}{suffix}",
            "dataset_n_finetune_segments": args.n_segments,
            "dataset_baseline_fraction": 0.5,
            "dataset_series_fraction": 1.0,
            "pipeline_merge_rule": rule,
            "pipeline_coefficient_source": coefficient,
            "pipeline_opcm_threshold": args.opcm_threshold,
            "pipeline_fisher_batches": args.fisher_batches,
        }
        if coefficient == "became":
            overrides["pipeline_select_merge_scale_on_val"] = False
            overrides["pipeline_extra_merge_scales"] = []
        for seed in args.seeds:
            cmd = command(base, {**overrides, "seed": seed})
            validate(cmd)
            commands.append(" ".join(cmd))

    args.out.write_text("\n".join(commands) + "\n")
    print(f"wrote {args.out} — {len(commands)} commands "
          f"({len(CELLS)} cells x {len(args.seeds)} seeds at n={args.n_segments}), "
          f"all parser-validated")


if __name__ == "__main__":
    main()
