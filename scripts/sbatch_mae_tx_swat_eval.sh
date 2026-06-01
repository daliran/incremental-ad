#!/bin/bash

#SBATCH --job-name=incremental_ad_eval
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=01:00:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/homes/ddellacasaventurelli01/workspace/incremental-ad
WORK=/work/tesi_ddellacasaventurelli01/incremental-ad

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=$WORK/hf_cache
export RUNS_ROOT=$WORK/experiments
export WANDB_MODE=online
export WANDB_PROJECT=incremental_ad
export WANDB_ENTITY=kirrel-research
export TMPDIR=/tmp

# ── Pre-create the log dir (required before SLURM writes the log file) ────────
mkdir -p /work/tesi_ddellacasaventurelli01/incremental-ad/logs

# ── Diagnostic info ───────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"
nvidia-smi

# ── Activate venv (uv) ────────────────────────────────────────────────────────
cd $PROJECT_ROOT
source .venv/bin/activate

# ── Eval (standalone, on-demand) ────────────────────────────────────────────────
# Evaluate an existing checkpoint, e.g. to try a different threshold. The
# experiment/phase are read back from the checkpoint, so this run lands in the
# same wandb group as the model it evaluates; its output folder is
# experiments/<experiment>/eval/<run_tag>_<job>/.
#
# CHECKPOINT: full path to the checkpoint to evaluate. Deterministic layout is
#   $RUNS_ROOT/<experiment>/<phase>/<run_label>/checkpoints/best.pt
# RUN_TAG: a short label for this eval (also used in the run folder and name).
CHECKPOINT=$RUNS_ROOT/mae_tx_swat/pretrain/<run_label>/checkpoints/best.pt
RUN_TAG=oracle

python -m incremental_ad.main \
    --phase eval \
    --run-tag $RUN_TAG \
    --checkpoint $CHECKPOINT \
    --device auto \
    --dataset swat \
    --swat-window-len 100 \
    --swat-stride 1 \
    --swat-eval-stride 1 \
    --swat-normalization standard \
    --swat-val-ratio 0.15 \
    --split test \
    --test-eval-seed 0 \
    --test-eval-batch-size 512 \
    --test-eval-threshold-strategy oracle \
    --test-eval-threshold-percentile 99

echo "Finished at: $(date)"