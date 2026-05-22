import argparse
import logging
import platform

from incremental_ad.datasets import swat
from incremental_ad.models import mae_tx
from incremental_ad.phases.eval import run_eval_phase
from incremental_ad.phases.pretrain_full import run_pretrain_full
from incremental_ad.phases.pretrain_partial import run_pretrain_partial

# Number of workers used by the data loaders.
NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# Registry entries.
DATASETS = {
    "swat": (swat.add_args, swat.make_config, swat.SWaTConfig),
}

MODELS = {
    "mae_tx": (mae_tx.add_args, mae_tx.make_config, mae_tx.MaeTxConfig),
}


def dispatch() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--phase", required=True)
    known, _ = pre.parse_known_args()

    if known.phase == "pretrain_full":
        run_pretrain_full(datasets=DATASETS, models=MODELS, num_workers=NUM_WORKERS)
    elif known.phase == "pretrain_partial":
        run_pretrain_partial(datasets=DATASETS, models=MODELS, num_workers=NUM_WORKERS)
    elif known.phase == "eval":
        run_eval_phase(datasets=DATASETS, models=MODELS, num_workers=NUM_WORKERS)
    else:
        raise SystemExit(f"Unknown phase '{known.phase}'")


def main():

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dispatch()


if __name__ == "__main__":
    main()
