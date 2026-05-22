import os
from typing import Any

import wandb


def init_wandb(
    *,  # forces the next arguments to be provided by keyword and not by position.
    experiment: str,
    phase: str,
    op: str,
    run_id: str,
    group_run_id: str,
    config: dict[str, Any],
    run_tag: str | None = None,
) -> None:
    """Start a wandb run identified by experiment / phase / op.

    The grouping unit is one trained model and its evaluations:
    ``group = <experiment>/<phase>/<group_run_id>`` where group_run_id is the run
    id of the producing job. A training run uses its own run id; an eval uses the
    run id of the checkpoint it evaluates, so re-evals land in the model's group.
    The op is the job_type, and the run name encodes op[_tag] plus this run's id.
    """

    op_label = f"{op}_{run_tag}" if run_tag else op

    # Identity is recorded in the config so runs can be filtered/queried by it.
    meta = {"experiment": experiment, "phase": phase, "op": op}

    if run_tag:
        meta["run_tag"] = run_tag

    # Adds the slurm id explicitly to wandb if available.
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if slurm_job_id is not None:
        meta["slurm_job_id"] = slurm_job_id

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        group=f"{experiment}/{phase}/{group_run_id}",
        job_type=op,
        name=f"{op_label}-{run_id}",
        config={**config, **meta},
    )
