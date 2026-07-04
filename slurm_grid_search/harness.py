"""Shared, dataset/pipeline-agnostic harness for SLURM-based grid search.

Unlike a local sequential harness (one trial at a time, on one GPU), this submits every
trial in a sweep as its own SLURM job via `sbatch`, then gets out of the way — SLURM's own
scheduler handles parallelism, not this code. Submission and result collection are
separate, decoupled steps:

- `submit_sweep()` never waits for a job to finish. It renders one sbatch script per
  trial, submits it, and records the job ID in a manifest — fire and forget.
- `collect_sweep_results()` can be run any time afterward (repeatedly, to see partial
  progress) and reconstructs results by reading every matching run directory's
  `config.json` (to recover which trial it is) and every `result.json` found anywhere
  under it (to capture *all* metrics, not a hand-picked subset) — it doesn't rely on
  submission order, so it stays correct even when many SLURM jobs finish in the same
  window.

Everything generated — per-trial sbatch scripts, submission manifests, collected results
CSVs — is written under `output_root()` (`$SLURM_GRID_OUTPUT_ROOT`, default
`$WORK/slurm_grid_search`), never inside this repo checkout. This mirrors the existing
`scripts/sbatch_mae_tx_*.sh` convention of keeping the git checkout (`PROJECT_ROOT`)
separate from cluster scratch/output (`WORK`).

This module is meant to run on the SLURM cluster itself (where `sbatch` and the repo
checkout both live), not on a local dev machine.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches the hardcoded values already in every scripts/sbatch_mae_tx_*.sh — override via
# env var if your checkout/scratch paths differ.
_DEFAULT_PROJECT_ROOT = "/homes/ddellacasaventurelli01/workspace/incremental-ad"
_DEFAULT_WORK = "/work/tesi_ddellacasaventurelli01/incremental-ad"


def project_root() -> str:
    return os.environ.get("SLURM_GRID_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)


def work_root() -> str:
    return os.environ.get("WORK", _DEFAULT_WORK)


def output_root() -> Path:
    """Where every generated artifact (sbatch scripts, manifests, results CSVs) lives.
    Never inside the repo — override with $SLURM_GRID_OUTPUT_ROOT if needed."""
    env = os.environ.get("SLURM_GRID_OUTPUT_ROOT")
    return Path(env) if env else Path(work_root()) / "slurm_grid_search"


def runs_root() -> Path:
    """Matches $RUNS_ROOT as set by the generated sbatch scripts themselves."""
    return Path(work_root()) / "runs"


# ── Sweep definition ────────────────────────────────────────────────────────────


@dataclass
class SlurmConfig:
    job_name: str
    account: str = "tesi_ddellacasaventurelli01"
    partition: str = "all_usr_prod"
    gres: str = "gpu:1"
    mem: str = "18G"
    time: str = "01:00:00"
    nodes: int = 1


@dataclass
class Sweep:
    name: str  # short id — used as the results/manifest filename and (with experiment_name) for job names
    pipeline: str  # informational only (base_args already sets --pipeline)
    experiment_name: str  # --experiment_name for every trial; also the $RUNS_ROOT subfolder to collect from
    base_args: list[str]  # fixed CLI args shared by every trial, e.g. ["--model", "MaeTx", ...]
    slurm: SlurmConfig
    grid: dict[str, list[str]] = field(default_factory=dict)  # swept args: cartesian product
    trials: list[dict[str, str]] | None = None  # explicit override dicts; used instead of `grid` if set
    seeds: list[str] = field(default_factory=lambda: ["42"])  # repeat every trial per seed


def _iter_grid(grid: dict[str, list[str]]):
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def resolve_trials(sweep: Sweep) -> list[dict[str, str]]:
    """A Sweep's parameter combinations: `trials` verbatim if given, else the cartesian
    product of `grid`."""
    return sweep.trials if sweep.trials is not None else list(_iter_grid(sweep.grid))


def expand_trials(
    cross: dict[str, list[str]], one_at_a_time: dict[str, list[str]] | None = None
) -> list[dict[str, str]]:
    """Cartesian product of `cross` (the 1-2 axes worth checking jointly), plus one extra
    trial per (axis, value) in `one_at_a_time` — each such trial overrides only that one
    axis, leaving every other param (including the `cross` axes) at whatever `base_args`
    already has (i.e. the dataset's current proven value). Keeps a small grid from
    exploding into a full cross-product across every axis while still checking each
    axis's alternatives/extremes individually. Pass the result as a Sweep's `trials=`.
    """
    trials = list(_iter_grid(cross))
    for axis, values in (one_at_a_time or {}).items():
        for v in values:
            trials.append({axis: v})
    return trials


def params_to_args(params: dict[str, str]) -> list[str]:
    args: list[str] = []
    for k, v in params.items():
        args += [f"--{k}", str(v)]
    return args


# ── Submission ──────────────────────────────────────────────────────────────────

_SBATCH_TEMPLATE = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --gres={gres}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={work}/logs/job_%j.log
#SBATCH --error={work}/logs/job_%j.log

set -euo pipefail

# -- Paths --------------------------------------------------------------------
PROJECT_ROOT={project_root}
WORK={work}

# -- Environment ----------------------------------------------------------------
export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=$WORK/hf_cache
export RUNS_ROOT=$WORK/runs
export WANDB_MODE=online
export WANDB_PROJECT=incremental_ad
export WANDB_ENTITY=kirrel-research
export TMPDIR=/tmp

mkdir -p $WORK/logs

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"
nvidia-smi

cd $PROJECT_ROOT
source .venv/bin/activate

# Generated by slurm_grid_search/harness.py -- trial: {trial_desc}
python -m incremental_ad.main \\
{argv_lines}

echo "Finished at: $(date)"
"""


def render_sbatch_script(
    sweep: Sweep, trial_index: int, params: dict[str, str], seed: str,
    extra_args: list[str] | None = None,
) -> str:
    argv = [
        "--experiment_name", sweep.experiment_name,
        *sweep.base_args, *(extra_args or []), *params_to_args(params),
        "--seed", seed,  # last flag wins (argparse) -- overrides base_args' --seed
    ]
    argv_lines = " \\\n".join(f"    {argv[i]} {argv[i+1]}" for i in range(0, len(argv), 2))
    trial_desc = ", ".join(f"{k}={v}" for k, v in params.items()) or "(base config)"
    job_name = f"{sweep.name}_{trial_index}"
    return _SBATCH_TEMPLATE.format(
        job_name=job_name,
        account=sweep.slurm.account,
        partition=sweep.slurm.partition,
        nodes=sweep.slurm.nodes,
        gres=sweep.slurm.gres,
        mem=sweep.slurm.mem,
        time=sweep.slurm.time,
        work=work_root(),
        project_root=project_root(),
        trial_desc=f"{trial_desc}, seed={seed}",
        argv_lines=argv_lines,
    )


def _manifest_path(sweep: Sweep) -> Path:
    return output_root() / f"{sweep.name}_manifest.csv"


def _load_manifest(sweep: Sweep) -> list[dict[str, str]]:
    path = _manifest_path(sweep)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def submit_sweep(
    sweep: Sweep, extra_args: list[str] | None = None, dry_run: bool = False
) -> list[dict[str, str]]:
    """Submits every not-yet-submitted (params, seed) combo in `sweep` as its own SLURM
    job. `extra_args` is appended to every trial's argv before the swept params (e.g. a
    winning architecture's mae_tx_* flags for a training-parameter sweep). Never waits
    for a job to finish. Returns the manifest rows (existing + new)."""
    combos = list(itertools.product(resolve_trials(sweep), sweep.seeds))
    existing = _load_manifest(sweep)
    extra_args_key = json.dumps(extra_args or [])
    already_submitted = {
        (row["params"], row["seed"], row.get("extra_args", "[]")) for row in existing
    }

    out_dir = output_root()
    sbatch_dir = out_dir / "_generated_sbatch"
    if not dry_run:
        sbatch_dir.mkdir(parents=True, exist_ok=True)

    new_rows: list[dict[str, str]] = []
    print(f"=== Sweep '{sweep.name}' ({sweep.experiment_name}) -- {len(combos)} trial(s) in the grid ===")
    for i, (params, seed) in enumerate(combos):
        params_key = json.dumps(params, sort_keys=True)
        if (params_key, seed, extra_args_key) in already_submitted:
            print(f"  [{i}] {params} seed={seed} -- already submitted (same arch-args), skipping")
            continue

        script = render_sbatch_script(sweep, i, params, seed, extra_args=extra_args)
        script_path = sbatch_dir / f"{sweep.name}_trial_{i}.sh"

        print(f"  [{i}] {params} seed={seed}")
        if dry_run:
            print(f"      would write {script_path} and run: sbatch {script_path}")
            continue

        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"      sbatch FAILED: {proc.stderr.strip()}")
            continue
        job_id = proc.stdout.strip().rsplit(" ", 1)[-1]
        print(f"      submitted job {job_id}")
        new_rows.append({
            "trial": str(i), "params": params_key, "seed": seed, "extra_args": extra_args_key,
            "job_id": job_id, "script_path": str(script_path),
        })

    if not dry_run and new_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _manifest_path(sweep)
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["trial", "params", "seed", "extra_args", "job_id", "script_path"]
            )
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"Manifest updated: {manifest_path}")

    return existing + new_rows


# ── Collection ──────────────────────────────────────────────────────────────────


def _flatten_metrics(run_dir: Path) -> dict[str, Any]:
    """Every metrics dict found in any result.json under run_dir, flattened into
    {"<relpath-with-underscores>_<metric_key>": value} columns. E.g.
    baseline/test/result.json's "pa_f1" -> "baseline_test_pa_f1". Captures *all* metrics
    without the harness needing to know a task's metric names ahead of time."""
    flat: dict[str, Any] = {}
    for result_file in sorted(run_dir.glob("**/result.json")):
        rel = result_file.relative_to(run_dir).parent
        prefix = "root" if str(rel) == "." else str(rel).replace("/", "_").replace("\\", "_")
        try:
            metrics = json.loads(result_file.read_text(encoding="utf-8")).get("metrics", {})
        except (json.JSONDecodeError, OSError):
            continue
        for k, v in metrics.items():
            flat[f"{prefix}_{k}"] = v
    return flat


def collect_sweep_results(sweep: Sweep) -> Path:
    """Scans every run directory under $RUNS_ROOT/<experiment_name>/, matches each one
    back to its trial via config.json's recorded args, and writes one row per run with
    every metric found anywhere in it. Idempotent -- safe to re-run any time to see
    partial progress or refresh after more jobs finish. Returns the results CSV path."""
    group_dir = runs_root() / sweep.experiment_name
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = ["run_id", "seed", "status"]

    if not group_dir.exists():
        print(f"No runs found yet under {group_dir}")
    else:
        for run_dir in sorted(group_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue  # job hasn't started writing output yet, or crashed before doing so
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            args = cfg.get("args", {})

            row: dict[str, Any] = {"run_id": run_dir.name, "seed": args.get("seed")}
            # Every mae_tx_* (architecture) arg actually recorded for this run -- not derived
            # from the sweep's *current* trials/grid, which may since have been trimmed to
            # only the trials still worth (re-)submitting and would otherwise silently drop
            # columns for historical rows that swept other axes.
            for k, v in args.items():
                if k.startswith("mae_tx_"):
                    if k not in fieldnames:
                        fieldnames.append(k)
                    row[k] = v

            metrics = _flatten_metrics(run_dir)
            row.update(metrics)
            row["status"] = "complete" if metrics else "incomplete"
            for k in metrics:
                if k not in fieldnames:
                    fieldnames.append(k)
            rows.append(row)

    output_root().mkdir(parents=True, exist_ok=True)
    results_path = output_root() / f"{sweep.name}_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    n_complete = sum(1 for r in rows if r["status"] == "complete")
    print(f"Collected {len(rows)} run(s) ({n_complete} complete) -> {results_path}")
    return results_path
