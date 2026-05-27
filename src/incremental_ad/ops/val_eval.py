import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core.run import save_eval_snapshot
from incremental_ad.core.tracking import init_wandb
from incremental_ad.training import val_evaluator

log = logging.getLogger(__name__)


def run_val_eval(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    device,
    run_dir: Path,
    run_id: str,
    group_run_id: str,
    dataset_name: str,
    dataset_cfg,
    val_eval_cfg,
    ckpt: dict,
    model_name: str,
    model_cfg,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    num_workers: int,
) -> None:
    """Evaluate reconstruction quality on the val tail of the training slice.

    Uses the same slice that was used during training so the val set is identical
    to the one the trainer monitored for early stopping.
    """

    init_wandb(
        experiment=experiment,
        phase=phase,
        op="val_eval",
        run_id=run_id,
        group_run_id=group_run_id,
        run_tag=run_tag,
        config={
            "dataset_name": dataset_name,
            "dataset": asdict(dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "val_eval": asdict(val_eval_cfg),
            "train_run_id": ckpt["run_id"],
            "train_slice": train_slice,
        },
    )

    try:
        log.info(f"Run dir:     {run_dir}")
        log.info(f"Device:      {device}")
        log.info(f"Dataset:     {dataset_name} -> {dataset_cfg}")
        log.info(f"Model:       {model_name} -> {model_cfg}")
        log.info(f"Train slice: {train_slice}")

        result = factory.build_for_training(
            model_name=model_name,
            dataset_name=dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            train_slice=train_slice,
            partial_ratio=partial_ratio,
            n_finetune=n_finetune,
        )

        model = result.model.to(device)

        val_loader = DataLoader(
            result.val_dataset,
            batch_size=val_eval_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        e = val_evaluator.ValEvaluator(
            model=model,
            device=device,
            val_loader=val_loader,
            run_dir=run_dir,
            run_id=run_id,
        )

        e.load_checkpoint(ckpt)
        e.evaluate()

    finally:
        wandb.finish()


def run_standalone_val_eval(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    device,
    run_dir: Path,
    run_id: str,
    group_run_id: str,
    eval_dataset_name: str,
    eval_dataset_cfg,
    val_eval_cfg,
    ckpt: dict,
    checkpoint_path: Path,
    model_name: str,
    model_cfg,
    num_workers: int,
) -> None:
    """Evaluate reconstruction quality on the full training val set.

    Uses build_for_eval (eval_stride, full train series) so no slice parameters
    are required — appropriate for standalone re-evaluation after training.
    """

    save_eval_snapshot(
        run_dir,
        checkpoint_path=checkpoint_path,
        ckpt=ckpt,
        eval_dataset_name=eval_dataset_name,
        eval_dataset_cfg=eval_dataset_cfg,
        eval_cfg=val_eval_cfg,
    )

    init_wandb(
        experiment=experiment,
        phase=phase,
        op="val_eval",
        run_id=run_id,
        group_run_id=group_run_id,
        run_tag=run_tag,
        config={
            "dataset_name": eval_dataset_name,
            "dataset": asdict(eval_dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "val_eval": asdict(val_eval_cfg),
            "train_run_id": ckpt["run_id"],
        },
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Device:  {device}")
        log.info(f"Dataset: {eval_dataset_name} -> {eval_dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")

        result = factory.build_for_eval(
            model_name=model_name,
            dataset_name=eval_dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=eval_dataset_cfg,
            split="val",
        )

        model = result.model.to(device)

        val_loader = DataLoader(
            result.eval_dataset,
            batch_size=val_eval_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        e = val_evaluator.ValEvaluator(
            model=model,
            device=device,
            val_loader=val_loader,
            run_dir=run_dir,
            run_id=run_id,
        )

        e.load_checkpoint(ckpt)
        e.evaluate()

    finally:
        wandb.finish()
