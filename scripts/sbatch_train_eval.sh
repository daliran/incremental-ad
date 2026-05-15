#!/bin/bash

#SBATCH --job-name=incremental_ad_train
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=06:00:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental_ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental_ad/logs/job_%j.log

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/homes/ddellacasaventurelli01/workspace/incremental_ad
WORK=/work/tesi_ddellacasaventurelli01/incremental_ad

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=$WORK/hf_cache
export RUNS_ROOT=$WORK/runs
export WANDB_MODE=online
export WANDB_PROJECT=incremental_ad
export WANDB_ENTITY=kirrel-research
export TMPDIR=/tmp

# ── Pre-create the log dir (required before SLURM writes the log file) ────────
mkdir -p /work/tesi_ddellacasaventurelli01/incremental_ad/logs

# ── Diagnostic info ───────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"
nvidia-smi

# ── Activate venv (uv) ────────────────────────────────────────────────────────
cd $PROJECT_ROOT
source .venv/bin/activate

# ── Experiment ────────────────────────────────────────────────────────────────
# Tag meaningful hyperparameter variations (e.g. mae_tx_wd01, mae_tx_lr1e3).
# For pure seed repeats the job ID already differentiates the run directories.
EXPERIMENT=mae_tx

# ── Train ─────────────────────────────────────────────────────────────────────
python -m incremental_ad.main \
    --op train \
    --experiment-name $EXPERIMENT \
    --device auto \
    --dataset swat \
    --model mae_tx \
    --swat-window-len 100 \
    --swat-stride 50 \
    --swat-normalization standard \
    --mae-tx-patch-len 10 \
    --mae-tx-encoder-embed-dim 256 \
    --mae-tx-encoder-layers 2 \
    --mae-tx-encoder-heads 2 \
    --mae-tx-decoder-embed-dim 128 \
    --mae-tx-decoder-layers 1 \
    --mae-tx-decoder-heads 2 \
    --mae-tx-mask-ratio 0.90 \
    --mae-tx-patch-norm \
    --train-seed 42 \
    --train-epochs 300 \
    --train-patience 30 \
    --train-batch-size 64 \
    --train-optimizer adamw \
    --train-weight-decay 1e-2 \
    --train-learning-rate 1e-4 \
    --train-grad-clip 0.5 \
    --train-scheduler cosine \
    --train-warmup-ratio 0.1

# ── Eval ──────────────────────────────────────────────────────────────────────
# Checkpoint path is deterministic: $RUNS_ROOT/<experiment>_<job_id>/checkpoints/best.pt
python -m incremental_ad.main \
    --op eval \
    --experiment-name ${EXPERIMENT}_eval \
    --device auto \
    --checkpoint $RUNS_ROOT/${EXPERIMENT}_${SLURM_JOB_ID}/checkpoints/best.pt \
    --dataset swat \
    --swat-window-len 100 \
    --swat-stride 1 \
    --swat-normalization standard \
    --eval-seed 0 \
    --eval-split test \
    --eval-batch-size 512

echo "Finished at: $(date)"