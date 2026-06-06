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


def _write_val_eval_info(
    run_dir: Path,
    *,
    checkpoint_path: Path,
    ckpt: dict,
    eval_dataset_name: str,
    eval_dataset_cfg,
    model_name: str,
    model_cfg,
    val_eval_cfg,
) -> None:
    snapshot = {
        "op": "val_eval",
        "checkpoint": {
            "path": str(checkpoint_path),
            "phase": ckpt["phase"],
            "run_id": ckpt["run_id"],
            "epoch": ckpt["epoch"],
            "best_val_loss": ckpt["metrics"]["best_val_loss"],
            "dataset_name": ckpt["configs"]["dataset_name"],
            "dataset": ckpt["configs"]["dataset"],
            "model_name": ckpt["configs"]["model_name"],
            "model": ckpt["configs"]["model"],
            "train": ckpt["configs"]["train"],
        },
        "eval": {
            "dataset_name": eval_dataset_name,
            "dataset": asdict(eval_dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "config": asdict(val_eval_cfg),
        },
    }
    with (run_dir / "val_eval_info.json").open("w") as f:
        json.dump(snapshot, f, indent=2)


def run_val_eval(
    *,
    experiment: str,
    phase: str,
    run_tag: str | None,
    run_id: str,
    group_run_id: str,
    device,
    run_dir: Path,
    eval_dataset_name: str,
    eval_dataset_cfg,
    model_name: str,
    model_cfg,
    ckpt: dict,
    checkpoint_path: Path,
    val_eval_cfg,
    val_loader: DataLoader,
    phase_config: dict | None = None,
) -> None:
    """Evaluate reconstruction quality on the full training val series.

    The caller is responsible for building val_loader at eval_stride on the
    full train series. Appropriate for on-demand re-evaluation after training,
    not tied to any training slice.
    """

    _write_val_eval_info(
        run_dir,
        checkpoint_path=checkpoint_path,
        ckpt=ckpt,
        eval_dataset_name=eval_dataset_name,
        eval_dataset_cfg=eval_dataset_cfg,
        model_name=model_name,
        model_cfg=model_cfg,
        val_eval_cfg=val_eval_cfg,
    )

    wandb_config = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "phase": ckpt["phase"],
            "run_id": ckpt["run_id"],
            "epoch": ckpt["epoch"],
            "best_val_loss": ckpt["metrics"]["best_val_loss"],
            "dataset_name": ckpt["configs"]["dataset_name"],
            "dataset": ckpt["configs"]["dataset"],
            "model_name": ckpt["configs"]["model_name"],
            "model": ckpt["configs"]["model"],
        },
        "eval": {
            "dataset_name": eval_dataset_name,
            "dataset": asdict(eval_dataset_cfg),
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
        op="val_eval",
        run_id=run_id,
        group_run_id=group_run_id,
        run_tag=run_tag,
        config=wandb_config,
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Device:  {device}")
        log.info(f"Dataset: {eval_dataset_name} -> {eval_dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")

        model = factory.build_model(
            model_name=model_name,
            dataset_name=eval_dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=eval_dataset_cfg,
        ).to(device)

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
