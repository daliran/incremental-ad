import dataclasses
import json
import os
import platform
import socket
from datetime import datetime
from pathlib import Path


def _runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", "experiments"))


def setup_run_dir(
    experiment: str, phase: str, run_tag: str | None = None
) -> tuple[Path, str]:
    """Create <RUNS_ROOT>/<experiment>/<phase>/<run_label>/ and return (run_dir, run_id).

    A phase is one SLURM job; all of its output lives in run_dir. What that output
    is varies per phase, so this only creates the folder — checkpoint promotion,
    eval artifacts, etc. are the phase handler's concern.
    """

    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_label = f"{run_tag}_{run_id}" if run_tag else run_id
    run_dir = _runs_root() / experiment / phase / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, run_id


def phase_checkpoint_dir(
    experiment: str, phase: str, run_tag: str | None = None
) -> Path:
    """Deterministic dir where a producer phase promotes best/last across jobs.

    Tagged producers (e.g. the fine-tunings ft_1..ft_N) get their own subdir so
    several checkpoint-producing runs can coexist within one phase.
    """
    ckpt_dir = _runs_root() / experiment / phase / "checkpoints"
    return ckpt_dir / run_tag if run_tag else ckpt_dir


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


def save_eval_snapshot(
    run_dir: Path,
    *,
    checkpoint_path: Path,
    ckpt: dict,
    eval_dataset_name: str,
    eval_dataset_cfg,
    eval_cfg,
) -> None:
    """Dump eval provenance to <run_dir>/eval_info.json for reproducibility."""

    snapshot = {
        "checkpoint": str(checkpoint_path),
        "train_run_id": ckpt["run_id"],
        "train_epoch": ckpt["epoch"],
        "train_best_val_loss": ckpt["metrics"]["best_val_loss"],
        "train_configs": ckpt["configs"],
        "eval_dataset": eval_dataset_name,
        "eval_dataset_cfg": dataclasses.asdict(eval_dataset_cfg),
        "eval_cfg": dataclasses.asdict(eval_cfg),
    }

    with (run_dir / "eval_info.json").open("w") as f:
        json.dump(snapshot, f, indent=2)


def save_config_snapshot(
    run_dir: Path,
    *,  # forces the next arguments to be provided by keyword and not by position.
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
    """Dump the resolved configuration to <run_dir>/config.json for reproducibility."""

    snapshot = {
        "experiment": experiment,
        "phase": phase,
        "op": op,
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
