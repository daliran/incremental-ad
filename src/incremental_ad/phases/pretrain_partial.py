import argparse
import logging
import os

from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import setup_run_dir
from incremental_ad.core.seed import set_seed
from incremental_ad.ops.train import run_train
from incremental_ad.training import trainer

log = logging.getLogger(__name__)


def run_pretrain_partial(*, datasets: dict, models: dict, num_workers: int) -> None:
    """Train a base model on the partial-pretrain slice — train-only, no eval.

    The partial slice is the first ``--swat-partial-ratio`` of the train series; a
    validation tail is carved from it (for early stopping and loss curves). The
    remaining ``1 - partial_ratio`` is reserved as ``--swat-n-finetune`` equal
    chunks for later fine-tunings — those chunks are recomputed deterministically
    from the same split params (recorded in this run's checkpoint).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=datasets, required=True)
    parser.add_argument("--model", choices=models, required=True)

    # Split params (phase-level, not part of the dataset config): the partial
    # pretrain trains on the first --partial-ratio of the train series; the rest is
    # reserved as --n-finetune equal chunks for later fine-tunings. n_finetune is
    # given here too so the full split is validated up front (fail fast if infeasible).
    parser.add_argument("--partial-ratio", type=float, required=True)
    parser.add_argument("--n-finetune", type=int, required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset and model.
    datasets[known.dataset][0](parser)
    models[known.model][0](parser)

    # Add trainer specific args (no evaluator — this phase does not eval).
    trainer.add_args(parser)

    args = parser.parse_args()

    # Load dataset, model and trainer config.
    dataset_cfg = datasets[args.dataset][1](args)
    model_cfg = models[args.model][1](args)
    train_cfg = trainer.make_config(args)

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(args.experiment, args.phase, args.run_tag)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(args.device)

    log.info(f"Phase: {args.phase}  (experiment={args.experiment})")

    set_seed(train_cfg.seed)

    run_train(
        experiment=args.experiment,
        phase=args.phase,
        run_tag=args.run_tag,
        device=device,
        run_dir=run_dir,
        run_id=run_id,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_slice="partial",
        partial_ratio=args.partial_ratio,
        n_finetune=args.n_finetune,
        num_workers=num_workers,
    )
