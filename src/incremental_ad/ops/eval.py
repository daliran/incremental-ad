import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from torch.utils.data import DataLoader

from incremental_ad import model_dataset_factory as factory
from incremental_ad.core.run import save_eval_snapshot
from incremental_ad.core.tracking import init_wandb
from incremental_ad.training import evaluator

log = logging.getLogger(__name__)


def run_eval(
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
    eval_cfg,
    ckpt: dict,
    checkpoint_path: Path,
    model_name: str,
    model_cfg,
    num_workers: int,
) -> None:
    """Evaluate a checkpoint, writing eval artifacts into run_dir.

    phase is the phase that produced the checkpoint, so the eval lands in the
    same wandb group as that model (whether bundled with training or run on
    demand later).
    """

    save_eval_snapshot(
        run_dir,
        checkpoint_path=checkpoint_path,
        ckpt=ckpt,
        eval_dataset_name=eval_dataset_name,
        eval_dataset_cfg=eval_dataset_cfg,
        eval_cfg=eval_cfg,
    )

    log.info(f"Eval info saved to {run_dir / 'eval_info.json'}")

    # Init wandb.
    init_wandb(
        experiment=experiment,
        phase=phase,
        op="eval",
        run_id=run_id,
        group_run_id=group_run_id,
        run_tag=run_tag,
        config={
            "dataset_name": eval_dataset_name,
            "dataset": asdict(eval_dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "eval": asdict(eval_cfg),
            "train_run_id": ckpt["run_id"],
        },
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {eval_dataset_name} -> {eval_dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")
        log.info(f"Eval:    {eval_cfg}")

        result = factory.build_for_eval(
            model_name=model_name,
            dataset_name=eval_dataset_name,
            model_cfg=model_cfg,
            dataset_cfg=eval_dataset_cfg,
            split=eval_cfg.split,
        )

        model = result.model.to(device)

        eval_loader = DataLoader(
            result.eval_dataset,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        train_loader = DataLoader(
            result.train_dataset,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        e = evaluator.Evaluator(
            model=model,
            device=device,
            eval_loader=eval_loader,
            train_loader=train_loader,
            run_dir=run_dir,
            run_id=run_id,
            config=eval_cfg,
        )

        e.load_checkpoint(ckpt)

        e.evaluate()

    finally:
        wandb.finish()
