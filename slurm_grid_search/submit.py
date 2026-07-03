#!/usr/bin/env python3
"""Submit one sweep's trials as SLURM jobs.

Usage:
    python submit.py --dataset swat --stage model
    python submit.py --dataset swat --stage train_incremental --arch-args '{"mae_tx_patch_len": "10", ...}'
    python submit.py --dataset swat --stage model --dry-run

Run this on the SLURM cluster itself (where `sbatch` and the repo checkout both live),
not on a local dev machine. Submission never waits for jobs to finish -- use collect.py
afterward, any time, to gather whatever has completed so far.

--arch-args is required for the train_standard/train_incremental stages (not model,
which searches the architecture): the winning architecture from --stage model isn't
picked automatically -- collect.py's results are ranked by every metric, not one, so
review them yourself (or via report.py) and pass the winner's mae_tx_* flags here.
"""

from __future__ import annotations

import argparse
import json

from harness import Sweep, params_to_args, submit_sweep
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
    parser.add_argument(
        "--arch-args", default=None,
        help='JSON dict of the winning mae_tx_* architecture flags from --stage model, '
             'e.g. \'{"mae_tx_patch_len": "10", "mae_tx_encoder_embed_dim": "256", ...}\'. '
             "Required for train_standard/train_incremental; ignored for model.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sweep = _SWEEPS[args.dataset][args.stage]

    extra_args: list[str] = []
    if args.stage != "model":
        if not args.arch_args:
            parser.error(f"--arch-args is required for --stage {args.stage}")
        extra_args = params_to_args(json.loads(args.arch_args))
    elif args.arch_args:
        parser.error("--arch-args is ignored for --stage model (that stage searches the architecture)")

    submit_sweep(sweep, extra_args=extra_args, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
