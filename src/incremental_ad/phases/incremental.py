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
from incremental_ad.ops.merge import run_merge
from incremental_ad.ops.train import run_train
from incremental_ad.ops.train_slice_val_eval import run_train_slice_val_eval
from incremental_ad.training import test_evaluator, trainer, val_evaluator

log = logging.getLogger(__name__)


def run_incremental(*, datasets: dict, models: dict, num_workers: int) -> None:
    """Whole incremental pipeline in one job on a static dataset:

      1. pretrain a base model on the `partial` slice,
      2. fine-tune it on each of the `n_finetune` chunks (ft_0 .. ft_{N-1}),
      3. merge the fine-tunings into the base via task arithmetic,
      4. evaluate the merged model on the test set.

    Everything lives in one run dir (base/, ft_i/, merged/) and one wandb group, so
    no parameters/paths are passed between jobs. Changing the split = re-run this phase.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=datasets, required=True)
    parser.add_argument("--model", choices=models, required=True)

    # Split + merge params (phase-level).
    parser.add_argument("--partial-ratio", type=float, required=True)
    parser.add_argument("--n-finetune", type=int, required=True)
    parser.add_argument("--merge-scale", type=float, required=True)
    parser.add_argument("--base-data-ratio", type=float, required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset and model.
    datasets[known.dataset][0](parser)
    models[known.model][0](parser)

    # Two trainer configs (base pretrain + fine-tuning), val evaluator and test evaluator.
    trainer.add_args(parser, "train")
    trainer.add_args(parser, "finetune")
    val_evaluator.add_args(parser)
    test_evaluator.add_args(parser)

    args = parser.parse_args()

    dataset_cfg = datasets[args.dataset][1](args)
    model_cfg = models[args.model][1](args)
    pretrain_cfg = trainer.make_config(args, "train")
    finetune_cfg = trainer.make_config(args, "finetune")
    val_eval_cfg = val_evaluator.make_config(args)
    test_eval_cfg = test_evaluator.make_config(args)

    phase_cfg_dict = {
        "partial_ratio": args.partial_ratio,
        "n_finetune": args.n_finetune,
        "merge_scale": args.merge_scale,
        "base_data_ratio": args.base_data_ratio,
    }

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
        incremental=phase_cfg_dict,
    )

    log.info(f"Phase: {args.phase}  (experiment={args.experiment})")

    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Primary loaders for base training.
    base_train_ds, base_val_ds = factory.build_datasets(
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        train_slice="base",
        partial_ratio=args.partial_ratio,
        n_finetune=args.n_finetune,
        base_data_ratio=args.base_data_ratio,
    )

    base_train_loader = DataLoader(
        base_train_ds, batch_size=pretrain_cfg.batch_size, shuffle=True, num_workers=num_workers
    )

    base_val_loader = DataLoader(
        base_val_ds, batch_size=pretrain_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    base_eval_val_loader = DataLoader(
        base_val_ds, batch_size=val_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    # Secondary val loaders: train portion of each ft split, logged each epoch
    # so we can watch base-model generalisation to unseen data during training.
    secondary_val_loaders = {
        f"ft_{i}/train": DataLoader(
            factory.build_datasets(
                dataset_name=args.dataset,
                dataset_cfg=dataset_cfg,
                train_slice=f"ft_{i}",
                partial_ratio=args.partial_ratio,
                n_finetune=args.n_finetune,
                base_data_ratio=args.base_data_ratio,
            )[0],
            batch_size=pretrain_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for i in range(args.n_finetune)
    }

    set_seed(pretrain_cfg.seed)

    run_train(
        experiment=args.experiment,
        phase=args.phase,
        run_tag="base",
        run_id=run_id,
        device=device,
        run_dir=base_dir,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=pretrain_cfg,
        train_loader=base_train_loader,
        val_loader=base_val_loader,
        secondary_val_loaders=secondary_val_loaders,
        phase_config=phase_cfg_dict,
    )

    base_ckpt_path = base_dir / "checkpoints" / "best.pt"
    base_ckpt = checkpoint.load_checkpoint(base_ckpt_path)

    set_seed(val_eval_cfg.seed)

    run_train_slice_val_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag="base",
        run_id=run_id,
        group_run_id=run_id,
        device=device,
        run_dir=base_dir,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        ckpt=base_ckpt,
        train_slice="base",
        val_loader=base_eval_val_loader,
        val_eval_cfg=val_eval_cfg,
        phase_config=phase_cfg_dict,
    )

    base_state = base_ckpt["model_state"]
    ft_ckpt_paths = []

    for i in range(args.n_finetune):

        ft_dir = run_dir / f"ft_{i}"
        ft_dir.mkdir(parents=True, exist_ok=True)

        ft_train_ds, ft_val_ds = factory.build_datasets(
            dataset_name=args.dataset,
            dataset_cfg=dataset_cfg,
            train_slice=f"ft_{i}",
            partial_ratio=args.partial_ratio,
            n_finetune=args.n_finetune,
            base_data_ratio=1.0,
        )

        ft_train_loader = DataLoader(
            ft_train_ds, batch_size=finetune_cfg.batch_size, shuffle=True, num_workers=num_workers
        )

        ft_val_loader = DataLoader(
            ft_val_ds, batch_size=finetune_cfg.batch_size, shuffle=False, num_workers=num_workers
        )

        ft_eval_val_loader = DataLoader(
            ft_val_ds, batch_size=val_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
        )

        set_seed(finetune_cfg.seed)

        run_train(
            experiment=args.experiment,
            phase=args.phase,
            run_tag=f"ft_{i}",
            run_id=run_id,
            device=device,
            run_dir=ft_dir,
            dataset_name=args.dataset,
            dataset_cfg=dataset_cfg,
            model_name=args.model,
            model_cfg=model_cfg,
            train_cfg=finetune_cfg,
            train_loader=ft_train_loader,
            val_loader=ft_val_loader,
            init_model_state=base_state,
            phase_config=phase_cfg_dict,
        )

        ft_ckpt_path = ft_dir / "checkpoints" / "best.pt"
        ft_ckpt = checkpoint.load_checkpoint(ft_ckpt_path)

        set_seed(val_eval_cfg.seed)

        run_train_slice_val_eval(
            experiment=args.experiment,
            phase=args.phase,
            run_tag=f"ft_{i}",
            run_id=run_id,
            group_run_id=run_id,
            device=device,
            run_dir=ft_dir,
            dataset_name=args.dataset,
            dataset_cfg=dataset_cfg,
            model_name=args.model,
            model_cfg=model_cfg,
            ckpt=ft_ckpt,
            train_slice=f"ft_{i}",
            val_loader=ft_eval_val_loader,
            val_eval_cfg=val_eval_cfg,
            phase_config=phase_cfg_dict,
        )

        ft_ckpt_paths.append(ft_ckpt_path)

    merged_dir = run_dir / "merged"
    merged_ckpt_path = merged_dir / "checkpoints" / "best.pt"

    merged_ckpt = run_merge(
        base_checkpoint_path=base_ckpt_path,
        ft_checkpoint_paths=ft_ckpt_paths,
        scale=args.merge_scale,
        out_path=merged_ckpt_path,
    )

    model_name = merged_ckpt["configs"]["model_name"]
    merged_model_cfg = models[model_name][2](**merged_ckpt["configs"]["model"])

    # Val reconstruction eval on the full train val — the merged model has seen all data.
    _, merged_val_ds = factory.build_datasets(
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        train_slice="full",
        partial_ratio=args.partial_ratio,
        n_finetune=args.n_finetune,
        base_data_ratio=1.0,
    )

    merged_eval_val_loader = DataLoader(
        merged_val_ds, batch_size=val_eval_cfg.batch_size, shuffle=False, num_workers=num_workers
    )

    set_seed(val_eval_cfg.seed)

    run_train_slice_val_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag="merged",
        run_id=run_id,
        group_run_id=run_id,
        device=device,
        run_dir=merged_dir,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=model_name,
        model_cfg=merged_model_cfg,
        ckpt=merged_ckpt,
        train_slice="full",
        val_loader=merged_eval_val_loader,
        val_eval_cfg=val_eval_cfg,
        phase_config=phase_cfg_dict,
    )

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
        run_tag="merged",
        run_id=run_id,
        group_run_id=run_id,
        device=device,
        run_dir=merged_dir,
        eval_dataset_name=args.dataset,
        eval_dataset_cfg=dataset_cfg,
        model_name=model_name,
        model_cfg=merged_model_cfg,
        ckpt=merged_ckpt,
        checkpoint_path=merged_ckpt_path,
        eval_cfg=test_eval_cfg,
        train_loader=test_train_loader,
        eval_loader=test_eval_loader,
        phase_config=phase_cfg_dict,
    )
