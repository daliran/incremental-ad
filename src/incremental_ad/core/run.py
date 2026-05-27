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

    A phase is one SLURM job; all of its output (checkpoints, eval artifacts, …)
    lives in run_dir.
    """

    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_label = f"{run_tag}_{run_id}" if run_tag else run_id
    run_dir = _runs_root() / experiment / phase / run_label
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


def save_phase_snapshot(run_dir: Path, *, experiment: str, phase: str, run_id: str, **phase_params) -> None:
    """Dump global identity + provenance + phase-specific params to <run_dir>/phase_config.json.

    phase_params should contain only parameters that are specific to this phase (e.g.
    partial_ratio/n_finetune/merge_scale for incremental). Dataset, model, and op configs
    belong at the op level (config.json / eval_info.json) and must NOT be passed here.
    """

    snapshot = {
        "experiment": experiment,
        "phase": phase,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        **phase_params,
    }

    with (run_dir / "phase_config.json").open("w") as f:
        json.dump(snapshot, f, indent=2)
