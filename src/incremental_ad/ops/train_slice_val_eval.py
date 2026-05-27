import json
import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core.tracking import init_wandb
from incremental_ad.training import val_evaluator

log = logging.getLogger(__name__)


def _write_train_slice_val_eval_info(
    run_dir: Path,
    *,
    ckpt: dict,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_slice: str,
    val_eval_cfg,
) -> None:
    snapshot = {
        "op": "train_slice_val_eval",
        "checkpoint": {
            "run_id": ckpt["run_id"],
            "epoch": ckpt["epoch"],
            "best_val_loss": ckpt["metrics"]["best_val_loss"],
        },
        "eval": {
            "train_slice": train_slice,
            "dataset_name": dataset_name,
            "dataset": asdict(dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "config": asdict(val_eval_cfg),
        },
    }
    with (run_dir / "train_slice_val_eval_info.json").open("w") as f:
        json.dump(snapshot, f, indent=2)


def run_train_slice_val_eval(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    run_id: str,
    group_run_id: str,
    device,
    run_dir: Path,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    ckpt: dict,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    val_eval_cfg,
    num_workers: int,
    base_data_ratio: float,
    phase_config: dict | None = None,
) -> None:
    """Evaluate reconstruction quality on the val tail of the training slice.

    Uses the same slice and stride that were used during training, so the val
    set is identical to the one the trainer monitored for early stopping.
    """

    _write_train_slice_val_eval_info(
        run_dir,
        ckpt=ckpt,
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        model_name=model_name,
        model_cfg=model_cfg,
        train_slice=train_slice,
        val_eval_cfg=val_eval_cfg,
    )

    wandb_config = {
        "checkpoint": {
            "run_id": ckpt["run_id"],
            "epoch": ckpt["epoch"],
            "best_val_loss": ckpt["metrics"]["best_val_loss"],
        },
        "eval": {
            "train_slice": train_slice,
            "dataset_name": dataset_name,
            "dataset": asdict(dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "config": asdict(val_eval_cfg),
        },
    }
    if phase_config is not None:
        wandb_config["phase"] = phase_config

    init_wandb(
        experiment=experiment,
        phase=phase,
        op="train_slice_val_eval",
        run_id=run_id,
        group_run_id=group_run_id,
        run_tag=run_tag,
        config=wandb_config,
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
            base_data_ratio=base_data_ratio,
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
