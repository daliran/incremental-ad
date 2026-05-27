import argparse
import logging
import os
from pathlib import Path

from incremental_ad.core import checkpoint
from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import setup_run_dir, save_phase_snapshot
from incremental_ad.core.seed import set_seed
from incremental_ad.ops.val_eval import run_val_eval
from incremental_ad.ops.test_eval import run_test_eval
from incremental_ad.training import test_evaluator, val_evaluator

log = logging.getLogger(__name__)


def run_eval_phase(*, datasets: dict, models: dict, num_workers: int) -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=datasets, required=True)
    parser.add_argument("--split", choices=["test", "val"], required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset.
    datasets[known.dataset][0](parser)

    # Add the evaluator-specific args based on the requested split.
    if known.split == "test":
        test_evaluator.add_args(parser)
    else:
        val_evaluator.add_args(parser)

    args = parser.parse_args()

    # Load checkpoint and rebuild the model config it was produced with.
    ckpt = checkpoint.load_checkpoint(args.checkpoint)
    model_name = ckpt["configs"]["model_name"]
    model_cfg = models[model_name][2](**ckpt["configs"]["model"])

    # The eval lives in its own 'eval' phase folder under the checkpoint's
    # experiment, but is grouped in wandb with the phase that produced the
    # checkpoint so re-evals sit next to the model.
    experiment = ckpt["experiment"]

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(experiment, args.phase, args.run_tag)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(args.device)

    log.info(f"Phase: {args.phase}  (experiment={experiment}, split={args.split})")

    eval_dataset_cfg = datasets[args.dataset][1](args)

    save_phase_snapshot(
        run_dir,
        experiment=experiment,
        phase=args.phase,
        run_id=run_id,
        split=args.split,
        checkpoint=str(args.checkpoint),
    )

    if args.split == "test":
        eval_cfg = test_evaluator.make_config(args)

        set_seed(eval_cfg.seed)

        run_test_eval(
            experiment=experiment,
            phase=ckpt["phase"],
            run_tag=args.run_tag,
            run_id=run_id,
            group_run_id=ckpt["run_id"],
            device=device,
            run_dir=run_dir,
            eval_dataset_name=args.dataset,
            eval_dataset_cfg=eval_dataset_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
            ckpt=ckpt,
            checkpoint_path=args.checkpoint,
            eval_cfg=eval_cfg,
            num_workers=num_workers,
        )

    else:
        val_eval_cfg = val_evaluator.make_config(args)

        set_seed(val_eval_cfg.seed)

        run_val_eval(
            experiment=experiment,
            phase=ckpt["phase"],
            run_tag=args.run_tag,
            run_id=run_id,
            group_run_id=ckpt["run_id"],
            device=device,
            run_dir=run_dir,
            eval_dataset_name=args.dataset,
            eval_dataset_cfg=eval_dataset_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
            ckpt=ckpt,
            checkpoint_path=args.checkpoint,
            val_eval_cfg=val_eval_cfg,
            num_workers=num_workers,
        )
