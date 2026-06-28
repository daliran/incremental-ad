import logging
import os
from typing import Any

import wandb

log = logging.getLogger(__name__)


def init_wandb(
    *,
    experiment_name: str,
    run_id: str,
    config: dict[str, Any],
) -> None:
    """Initialise a wandb run for one experiment execution.

    Reads WANDB_PROJECT and WANDB_ENTITY from env vars. No-ops if WANDB_PROJECT is not set.
    SLURM_JOB_ID is automatically captured into config when present.
    """
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        log.debug("WANDB_PROJECT not set — skipping wandb init")
        return

    entity = os.environ.get("WANDB_ENTITY")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    full_config = dict(config)

    if slurm_job_id is not None:
        full_config["slurm_job_id"] = slurm_job_id

    wandb.init(
        project=project,
        entity=entity,
        group=experiment_name,
        name=f"{experiment_name}-{run_id}",
        config=full_config,
    )

    run = wandb.run
    url = run.url if run is not None else None
    log.info(f"wandb run initialised: {url or '(no url — offline mode)'}")
