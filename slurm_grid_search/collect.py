"""Gather results for a sweep from $RUNS_ROOT into a CSV.

Usage:
    python collect.py --dataset swat --stage model
    python collect.py --dataset swat --stage train_incremental

Safe to re-run any time -- reads every run directory under
$RUNS_ROOT/<experiment_name>/ fresh each time (matching them back to trial params via
each run's config.json), so it reflects partial progress if some SLURM jobs haven't
finished yet, and catches up automatically as more do. Writes
$SLURM_GRID_OUTPUT_ROOT/<sweep_name>_results.csv with every metric found in any
result.json under each run (not a hand-picked subset).
"""

from __future__ import annotations

import argparse

from harness import Sweep, collect_sweep_results
from sweeps import etth_forecast, psm, swat

_SWEEPS: dict[str, dict[str, Sweep]] = {
    "swat": {
        "model": swat.MODEL_SWEEP,
        "train_standard": swat.TRAIN_STANDARD_SWEEP,
        "train_incremental": swat.TRAIN_INCREMENTAL_SWEEP,
    },
    "psm": {
        "model": psm.MODEL_SWEEP,
        "train_standard": psm.TRAIN_STANDARD_SWEEP,
        "train_incremental": psm.TRAIN_INCREMENTAL_SWEEP,
    },
    "etth_forecast": {
        "model": etth_forecast.MODEL_SWEEP,
        "train_standard": etth_forecast.TRAIN_STANDARD_SWEEP,
        "train_incremental": etth_forecast.TRAIN_INCREMENTAL_SWEEP,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(_SWEEPS.keys()))
    parser.add_argument("--stage", required=True, choices=["model", "train_standard", "train_incremental"])
    args = parser.parse_args()

    collect_sweep_results(_SWEEPS[args.dataset][args.stage])


if __name__ == "__main__":
    main()
