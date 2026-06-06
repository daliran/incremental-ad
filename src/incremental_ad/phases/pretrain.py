import argparse
import logging
import os

from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core import checkpoint
from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import setup_run_dir, save_phase_snapshot
from incremental_ad.core.seed import set_seed
from incremental_ad.ops.test_eval import run_test_eval
from incremental_ad.ops.train import run_train
from incremental_ad.ops.train_slice_val_eval import run_train_slice_val_eval
from incremental_ad.training import test_evaluator, trainer, val_evaluator

log = logging.getLogger(__name__)


def run_pretrain(*, datasets: dict, models: dict, num_workers: int) -> None:

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

    # Add trainer, val evaluator and test evaluator specific args (the bundled evals need their own).
    trainer.add_args(parser)
    val_evaluator.add_args(parser)
    test_evaluator.add_args(parser)

    args = parser.parse_args()

    # Load dataset, model, trainer and evaluator config.
    dataset_cfg = datasets[args.dataset][1](args)
    model_cfg = models[args.model][1](args)
    train_cfg = trainer.make_config(args)
    val_eval_cfg = val_evaluator.make_config(args)
    test_eval_cfg = test_evaluator.make_config(args)

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(args.experiment, args.phase, args.run_tag)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(args.device)

    save_phase_snapshot(
        run_dir,
        experiment=args.experiment,
        phase=args.phase,
        run_id=run_id,
    )

    log.info(f"Phase: {args.phase}  (experiment={args.experiment})")

    # Build datasets once; reuse the val_dataset for both training monitoring and
    # the post-training val eval (different batch sizes, same data).
    train_ds, val_ds = factory.build_datasets(
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        train_slice="full",
        partial_ratio=1.0,
        n_finetune=0,
        base_data_ratio=1.0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=num_workers
    )
    
    eval_val_loader = DataLoader(
        val_ds, batch_size=val_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    set_seed(train_cfg.seed)

    run_train(
        experiment=args.experiment,
        phase=args.phase,
        run_tag=args.run_tag,
        run_id=run_id,
        device=device,
        run_dir=run_dir,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    best_ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = checkpoint.load_checkpoint(best_ckpt_path)

    # Evaluate reconstruction quality on the val tail of the full training slice.
    set_seed(val_eval_cfg.seed)

    run_train_slice_val_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag=args.run_tag,
        run_id=run_id,
        group_run_id=ckpt["run_id"],
        device=device,
        run_dir=run_dir,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        ckpt=ckpt,
        train_slice="full",
        val_loader=eval_val_loader,
        val_eval_cfg=val_eval_cfg,
    )

    # Evaluate the model just trained.
    test_train_ds, test_eval_ds = factory.build_eval_datasets(
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        split="test",
    )

    test_train_loader = DataLoader(
        test_train_ds, batch_size=test_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_eval_loader = DataLoader(
        test_eval_ds, batch_size=test_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    set_seed(test_eval_cfg.seed)

    run_test_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag=args.run_tag,
        run_id=run_id,
        group_run_id=ckpt["run_id"],
        device=device,
        run_dir=run_dir,
        eval_dataset_name=args.dataset,
        eval_dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        ckpt=ckpt,
        checkpoint_path=best_ckpt_path,
        eval_cfg=test_eval_cfg,
        train_loader=test_train_loader,
        eval_loader=test_eval_loader,
    )
