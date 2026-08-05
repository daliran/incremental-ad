#!/bin/bash

#SBATCH --job-name=run_command
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=04:00:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log

set -euo pipefail

# Generic runner for a pre-generated `python -m incremental_ad.main ...` invocation.
#
# Exists because replication arms (extra seeds, one-off parameter repairs) are best
# generated *from a finished run's own config.json* rather than retyped -- see
# EXECUTION_PLAN.md. The generator writes a command file; this script runs it.
#
#   CMD=$WORK/replication/cmd_nf_swat_s7.sh sbatch --time=04:00:00 \
#       --job-name=nf_swat_s7 scripts/sbatch_run_command.sh
#
# CMD *must* live on $WORK. The per-session scratchpad under /tmp is node-local and is
# not visible to compute nodes -- a job pointed there dies with "No such file".
#
# Pass CMD as an env prefix, NOT via --export: on this cluster any explicit
# `--export=VAR=value` gets the job CANCELLED by root ~2s in, with no log written.
CMD=${CMD:-}

PROJECT_ROOT=/homes/ddellacasaventurelli01/workspace/incremental-ad
WORK=/work/tesi_ddellacasaventurelli01/incremental-ad

export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=$WORK/hf_cache
export RUNS_ROOT=$WORK/runs
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=incremental_ad
export WANDB_ENTITY=kirrel-research
export TMPDIR=$WORK/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p $WORK/logs

if [[ -z "$CMD" || ! -f "$CMD" ]]; then
    echo "ERROR: CMD is unset or not a file: '${CMD}'" >&2
    echo "  CMD=\$WORK/replication/cmd_<label>.sh sbatch $0" >&2
    exit 1
fi

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Command:    $CMD"
echo "Started at: $(date)"
nvidia-smi -L

cd $PROJECT_ROOT
source .venv/bin/activate
echo "──────── command ────────"; cat "$CMD"; echo "─────────────────────────"
bash "$CMD"

echo "Finished at: $(date)"
