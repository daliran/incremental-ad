import os
from typing import Any

import wandb


def init_wandb(
    *, # forces the next arguments to be provided by keyword and not by position.
    group: str,
    job_type: str,
    name: str,
    config: dict[str, Any],
) -> None:

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        group=group,
        job_type=job_type,
        name=name,
        config=config,
    )
