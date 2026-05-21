import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core.run import save_config_snapshot
from incremental_ad.core.tracking import init_wandb
from incremental_ad.training import trainer

log = logging.getLogger(__name__)


def run_train(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    device,
    run_dir: Path,
    run_id: str,
    checkpoint_dir: Path,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_cfg,
    num_workers: int,
    resume_ckpt: dict | None = None,
) -> None:
    """Train (or, with resume_ckpt, continue training) a model.

    Checkpoints are written into run_dir and best/last are promoted to the
    deterministic checkpoint_dir.
    """

    save_config_snapshot(
        run_dir,
        experiment=experiment,
        phase=phase,
        op="train",
        run_id=run_id,
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        model_name=model_name,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    config = {
        "dataset_name": dataset_name,
        "dataset": asdict(dataset_cfg),
        "model_name": model_name,
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
    }
    if resume_ckpt is not None:
        config["resumed_from"] = resume_ckpt["run_id"]

    # Init wandb.
    init_wandb(
        experiment=experiment,
        phase=phase,
        op="train",
        run_id=run_id,
        run_tag=run_tag,
        config=config,
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {dataset_name} -> {dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")
        log.info(f"Train:   {train_cfg}")
        if resume_ckpt is not None:
            log.info(f"Resuming from epoch {resume_ckpt['epoch']}")

        # build the model and the datasets based on the selected model-dataset pair
        result = factory.build_for_training(
            model_name=model_name,
            dataset_name=dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
        )

        model = result.model.to(device)

        train_loader = DataLoader(
            result.train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = DataLoader(
            result.val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        t = trainer.Trainer(
            model=model,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            run_id=run_id,
            experiment=experiment,
            phase=phase,
            op="train",
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
        )

        if resume_ckpt is not None:
            t.load_checkpoint(resume_ckpt)

        t.train()

    finally:
        wandb.finish()
