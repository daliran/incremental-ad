import json
import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core.tracking import init_wandb
from incremental_ad.training import trainer

log = logging.getLogger(__name__)


def _write_config(
    run_dir: Path,
    *,
    experiment: str,
    phase: str,
    op: str,
    run_id: str,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_cfg,
) -> None:
    snapshot = {
        "experiment": experiment,
        "phase": phase,
        "op": op,
        "run_id": run_id,
        "dataset_name": dataset_name,
        "dataset": asdict(dataset_cfg),
        "model_name": model_name,
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(snapshot, f, indent=2)


def run_train(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    run_id: str,
    device,
    run_dir: Path,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    secondary_val_loaders: dict[str, DataLoader] | None = None,
    init_model_state: dict | None = None,
    phase_config: dict | None = None,
) -> None:
    """Train a model, optionally initialized from given weights (for fine-tuning).

    The caller is responsible for building train_loader and val_loader (and any
    secondary_val_loaders for bonus monitoring). If init_model_state is given,
    those weights are loaded into the freshly built model before training (fresh
    optimizer/seed — this is a fine-tune, not a resume). Checkpoints go into
    run_dir/checkpoints/.
    """

    _write_config(
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

    wandb_config = {
        "dataset_name": dataset_name,
        "dataset": asdict(dataset_cfg),
        "model_name": model_name,
        "model": asdict(model_cfg),
        "train": asdict(train_cfg),
    }

    if init_model_state is not None:
        wandb_config["finetuned"] = True

    if phase_config is not None:
        wandb_config["phase"] = phase_config

    init_wandb(
        experiment=experiment,
        phase=phase,
        op="train",
        run_id=run_id,
        group_run_id=run_id,
        run_tag=run_tag,
        config=wandb_config,
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {dataset_name} -> {dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")
        log.info(f"Train:   {train_cfg}")

        if init_model_state is not None:
            log.info("Initializing model weights from base checkpoint (fine-tune)")

        model = factory.build_model(
            model_name=model_name,
            dataset_name=dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
        )

        if init_model_state is not None:
            model.load_state_dict(init_model_state)

        model = model.to(device)

        t = trainer.Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            secondary_val_loaders=secondary_val_loaders,
            run_id=run_id,
            run_dir=run_dir,
            experiment=experiment,
            phase=phase,
            op="train",
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
            config=train_cfg,
        )

        t.train()

    finally:
        wandb.finish()
