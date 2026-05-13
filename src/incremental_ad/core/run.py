import dataclasses
import json
import os
import platform
import socket
from datetime import datetime
from pathlib import Path


def setup_run_dir(experiment_name: str) -> tuple[Path, str]:
    """Create <RUNS_ROOT>/<experiment_name>_<run_id>[_task<N>]/ and return (run_dir, run_id)."""

    runs_root = Path(os.environ.get("RUNS_ROOT", "runs"))

    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    suffix = f"_task{task_id}" if task_id else ""

    run_dir = runs_root / f"{experiment_name}_{run_id}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, run_id


def _git_commit() -> str | None:
    """Return the current HEAD commit hash by reading .git/HEAD directly."""
    try:
        # Walk up from this file to find the .git directory
        search = Path(__file__).resolve().parent
        while search != search.parent:
            head = search / ".git" / "HEAD"
            if head.exists():
                content = head.read_text().strip()
                if content.startswith("ref: "):
                    # Normal branch: resolve the ref to its commit hash
                    ref_file = search / ".git" / content[5:]
                    return ref_file.read_text().strip() if ref_file.exists() else None
                # Detached HEAD: content is the commit hash directly
                return content
            search = search.parent
        return None
    except Exception:
        return None


def save_config_snapshot(
    run_dir: Path,
    *,  # forces the next arguments to be provided by keyword and not by position.
    experiment_name: str,
    run_id: str,
    dataset_name: str,
    dataset_cfg,
    model_name: str,
    model_cfg,
    train_cfg,
) -> None:
    """Dump the resolved configuration to <run_dir>/config.json for reproducibility."""
    snapshot = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "dataset_name": dataset_name,
        "dataset": dataclasses.asdict(dataset_cfg),
        "model_name": model_name,
        "model": dataclasses.asdict(model_cfg),
        "train": dataclasses.asdict(train_cfg),
    }

    with (run_dir / "config.json").open("w") as f:
        json.dump(snapshot, f, indent=2)