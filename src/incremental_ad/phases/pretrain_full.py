import argparse
import logging
import os
from dataclasses import replace

from incremental_ad.core import checkpoint
from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import setup_run_dir
from incremental_ad.core.seed import set_seed
from incremental_ad.ops.eval import run_eval
from incremental_ad.ops.train import run_train
from incremental_ad.training import evaluator, trainer

log = logging.getLogger(__name__)


def run_pretrain_full(*, datasets: dict, models: dict, num_workers: int) -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--phase", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=datasets, required=True)
    parser.add_argument("--model", choices=models, required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset and model.
    datasets[known.dataset][0](parser)
    models[known.model][0](parser)

    # Add trainer and evaluator specific args (the bundled eval needs its own).
    trainer.add_args(parser)
    evaluator.add_args(parser)

    args = parser.parse_args()

    # Load dataset, model, trainer and evaluator config.
    dataset_cfg = datasets[args.dataset][1](args)
    model_cfg = models[args.model][1](args)
    train_cfg = trainer.make_config(args)
    eval_cfg = evaluator.make_config(args)

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
        num_workers=num_workers,
    )

    # Evaluate the model just trained. Eval metrics assume stride=1, so the eval
    # dataset reuses the train dataset config with the stride overridden.
    best_ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = checkpoint.load_checkpoint(best_ckpt_path)
    eval_dataset_cfg = replace(dataset_cfg, stride=1)

    set_seed(eval_cfg.seed)

    run_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag=args.run_tag,
        device=device,
        run_dir=run_dir,
        run_id=run_id,
        group_run_id=ckpt["run_id"],
        eval_dataset_name=args.dataset,
        eval_dataset_cfg=eval_dataset_cfg,
        eval_cfg=eval_cfg,
        ckpt=ckpt,
        checkpoint_path=best_ckpt_path,
        model_name=args.model,
        model_cfg=model_cfg,
        num_workers=num_workers,
    )
