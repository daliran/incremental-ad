import dataclasses
from pathlib import Path
from typing import Any

import torch

from incremental_ad.core.seed import get_rng_state


@dataclasses.dataclass
class Metrics:
    best_val_loss: float
    val_loss: float
    train_loss: float


def save_checkpoint(
    path: Path,
    *,  # forces the next arguments to be provided by keyword and not by position.
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    run_id: str,
    wandb_group: str,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_cfg,
    metrics: Metrics,
) -> None:
    """Save a checkpoint with everything needed to resume training or run eval."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "run_id": run_id,
        "wandb_group": wandb_group,
        "rng_state": get_rng_state(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "configs": {
            "dataset_name": dataset_name,
            "dataset": dataclasses.asdict(dataset_cfg),
            "model_name": model_name,
            "model": dataclasses.asdict(model_cfg),
            "train": dataclasses.asdict(train_cfg),
        },
        "metrics": dataclasses.asdict(metrics),
    }

    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint on the CPU."""

    # Using weights_only=false to save all the states into a single file.
    return torch.load(path, map_location="cpu", weights_only=False)
