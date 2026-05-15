import argparse
import logging
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import wandb
from torch.utils.data import DataLoader

from incremental_ad.core import checkpoint
from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import save_config_snapshot, setup_run_dir
from incremental_ad.core.seed import set_rng_state, set_seed
from incremental_ad.core.tracking import init_wandb
from incremental_ad.datasets import swat
from incremental_ad.models import mae_tx
from incremental_ad.training import evaluator, trainer

log = logging.getLogger(__name__)

# Number of worker used by the data loader.
NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# Registry entries.
DATASETS = {
    "swat": (swat.add_args, swat.make_config, swat.SWaTConfig, swat.N_FEATURES),
}

MODELS = {
    "mae_tx": (
        mae_tx.add_args,
        mae_tx.make_config,
        mae_tx.MaeTxConfig,
        mae_tx.build_model,
    ),
}

Op = Literal["train", "resume", "eval"]


@dataclass
class GlobalConfig:
    op: Op
    experiment_name: str
    device: str


def run_train() -> None:

    # Add global args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["train", "resume", "eval"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--device", default="auto")

    # Add train op specific args.
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset and model.
    DATASETS[known.dataset][0](parser)
    MODELS[known.model][0](parser)

    # Add trainer specific args.
    trainer.add_args(parser)

    args = parser.parse_args()

    # Load global config.
    global_cfg = GlobalConfig(
        op=args.op, experiment_name=args.experiment_name, device=args.device
    )

    # Load dataset and model config.
    dataset_cfg = DATASETS[args.dataset][1](args)
    model_cfg = MODELS[args.model][1](args)

    # Load trainer config.
    train_cfg = trainer.make_config(args)

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(global_cfg.experiment_name)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(global_cfg.device)

    # Sets the seed.
    set_seed(train_cfg.seed)

    save_config_snapshot(
        run_dir,
        experiment_name=global_cfg.experiment_name,
        run_id=run_id,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    # Init wandb.
    init_wandb(
        group=run_id,
        job_type="train",
        name=run_id,
        config={
            "dataset_name": args.dataset,
            "dataset": asdict(dataset_cfg),
            "model_name": args.model,
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
        },
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Op: {global_cfg.op}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {args.dataset} -> {dataset_cfg}")
        log.info(f"Model:   {args.model} -> {model_cfg}")
        log.info(f"Train:   {train_cfg}")

        # TODO: build train_loader, val_loader from dataset_cfg
        train_loader = cast(DataLoader, None)
        val_loader = cast(DataLoader, None)

        n_features = DATASETS[args.dataset][3]

        # builds the model
        model = MODELS[args.model][3](dataset_cfg.window_len, n_features, model_cfg).to(
            device
        )

        t = trainer.Trainer(
            model=model,
            run_dir=run_dir,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            run_id=run_id,
            wandb_group=run_id,
            dataset_name=args.dataset,
            dataset_cfg=dataset_cfg,
            model_name=args.model,
            model_cfg=model_cfg,
        )

        t.train()

    finally:
        wandb.finish()


def run_resume() -> None:

    # Add global args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["train", "resume", "eval"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--device", default="auto")

    # Add resume op specific args.
    parser.add_argument("--checkpoint", type=Path, required=True)

    args = parser.parse_args()

    # Load global config.
    global_cfg = GlobalConfig(
        op=args.op, experiment_name=args.experiment_name, device=args.device
    )

    # Load checkpoint.
    ckpt = checkpoint.load_checkpoint(args.checkpoint)

    # Get config embedded in the checkpoint.
    ckpt_cfgs = ckpt["configs"]

    # Load the dataset and model names from the checkpoint.
    dataset_name = ckpt_cfgs["dataset_name"]
    model_name = ckpt_cfgs["model_name"]

    # Load dataset and model config from the checkpoint.
    dataset_cfg = DATASETS[dataset_name][2](**ckpt_cfgs["dataset"])
    model_cfg = MODELS[model_name][2](**ckpt_cfgs["model"])

    # Load trainer config from the checkpoint.
    train_cfg = trainer.TrainingConfig(**ckpt_cfgs["train"])

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(global_cfg.experiment_name)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(global_cfg.device)

    # Restore the seed and the RNG state.
    set_seed(train_cfg.seed)

    if ckpt.get("rng_state"):
        set_rng_state(ckpt["rng_state"])

    # Init wandb.
    # In this context the continuation creates a separate group.
    init_wandb(
        group=run_id,
        job_type="train",
        name=run_id,
        config={
            "dataset_name": dataset_name,
            "dataset": asdict(dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
            "resumed_from": ckpt["run_id"],
        },
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Op: {global_cfg.op}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {dataset_name} -> {dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")
        log.info(f"Train:   {train_cfg}")
        log.info(f"Resuming from epoch {ckpt['epoch']}")

        # TODO: build train_loader, val_loader from dataset_cfg
        train_loader = cast(DataLoader, None)
        val_loader = cast(DataLoader, None)

        n_features = DATASETS[dataset_name][3]

        # builds the model
        model = MODELS[model_name][3](dataset_cfg.window_len, n_features, model_cfg).to(
            device
        )

        t = trainer.Trainer(
            model=model,
            run_dir=run_dir,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            run_id=run_id,
            wandb_group=run_id,
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
        )

        t.load_checkpoint(ckpt)

        t.train()

    finally:
        wandb.finish()


def run_eval() -> None:

    # Add global args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["train", "resume", "eval"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--device", default="auto")

    # Add eval op specific args.
    parser.add_argument("--checkpoint", type=Path, required=True)

    # Dataset can differ from training.
    parser.add_argument("--dataset", choices=DATASETS, required=True)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset.
    DATASETS[known.dataset][0](parser)

    # Add the evaluator specific args.
    evaluator.add_args(parser)

    args = parser.parse_args()

    # Load global config.
    global_cfg = GlobalConfig(
        op=args.op, experiment_name=args.experiment_name, device=args.device
    )

    # Load eval config.
    eval_cfg = evaluator.make_config(args)

    # Load dataset config.
    dataset_cfg = DATASETS[args.dataset][1](args)

    # Load checkpoint.
    ckpt = checkpoint.load_checkpoint(args.checkpoint)

    # Get config embedded in the checkpoint.
    ckpt_cfgs = ckpt["configs"]

    # Load the model name and config from the checkpoint.
    model_name = ckpt_cfgs["model_name"]
    model_cfg = MODELS[model_name][2](**ckpt_cfgs["model"])

    # Define the run directory and the run id.
    run_dir, run_id = setup_run_dir(global_cfg.experiment_name)

    # Set the wandb directory under the specific run.
    os.environ["WANDB_DIR"] = str(run_dir)

    # Resolve the device.
    device = resolve_device(global_cfg.device)

    # Sets the seed.
    set_seed(eval_cfg.seed)

    # Init wandb.
    # In this context the group will be the same used during training.
    init_wandb(
        group=ckpt["wandb_group"],
        job_type="eval",
        name=run_id,
        config={
            "dataset_name": args.dataset,
            "dataset": asdict(dataset_cfg),
            "model_name": model_name,
            "model": asdict(model_cfg),
            "eval": asdict(eval_cfg),
            "train_run_id": ckpt["run_id"],
        },
    )

    try:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Op: {global_cfg.op}")
        log.info(f"Device: {device}")
        log.info(f"Dataset: {args.dataset} -> {dataset_cfg}")
        log.info(f"Model:   {model_name} -> {model_cfg}")
        log.info(f"Eval:    {eval_cfg}")

        # TODO dispatch to evaluator.evaluate(...)
    finally:
        wandb.finish()


def dispatch() -> None:
    """Parse --op and dispatch to the appropriate operation handler."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--op", choices=["train", "resume", "eval"], required=True)
    known, _ = pre.parse_known_args()

    if known.op == "train":
        run_train()
    elif known.op == "resume":
        run_resume()
    elif known.op == "eval":
        run_eval()


def main():

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dispatch()


if __name__ == "__main__":
    main()
