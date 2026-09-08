#!/bin/bash
# The §1.31 2x2: {plain sum, OPCM} x {swept alpha, BECAME lambda*} on PSM-forecast.
#
#     bash scripts/sbatch_opcm_became_ablation.sh [n_segments] [seeds...]
#
# One sbatch per (cell, seed) — never a loop inside one job. `run_id` is `SLURM_JOB_ID`, so
# several runs in one job share a directory and overwrite each other silently (EXECUTION_PLAN
# §3.15). Commands are built by `scripts/generate_adfc_grid.py`-style config readback so the
# hyperparameters cannot drift from the published PSM-forecast runs.
set -euo pipefail
N="${1:-3}"; shift || true
SEEDS=("${@:-7 42 123}")
: "${RUNS_ROOT:?export RUNS_ROOT before submitting — an unset value writes ./runs inside the repo}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT="${SLURM_ACCOUNT:-tesi_ddellacasaventurelli01}"
LOGS="${SWEEP_LOGS:-$RUNS_ROOT/../sweeps/logs}"; mkdir -p "$LOGS"

python "$REPO/scripts/generate_ablation_grid.py" --runs_root "$RUNS_ROOT" \
    --n_segments "$N" --seeds ${SEEDS[@]} --out /tmp/opcm_cells.sh
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  n=$((n+1))
  sbatch --account="$ACCOUNT" --partition=all_usr_prod --gres=gpu:1 --time=03:00:00 \
         --mem=32G --cpus-per-task=4 --job-name="opcm_$n" \
         --output="$LOGS/opcm_%j.log" \
         --wrap "bash -lc 'cd $REPO && source .venv/bin/activate && export RUNS_ROOT=$RUNS_ROOT && $cmd'" >/dev/null
done < /tmp/opcm_cells.sh
echo "submitted $n independent jobs (n=$N, seeds ${SEEDS[*]})"
