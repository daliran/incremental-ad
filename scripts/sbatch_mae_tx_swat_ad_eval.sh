#!/bin/bash

#SBATCH --job-name=incremental_ad_eval
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=00:30:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/homes/ddellacasaventurelli01/workspace/incremental-ad
WORK=/work/tesi_ddellacasaventurelli01/incremental-ad

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH=$PROJECT_ROOT/src
export HF_HOME=$WORK/hf_cache
export RUNS_ROOT=$WORK/runs
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

# ── EvalPipeline: evaluate an existing checkpoint, no training ────────────────
# Use this to re-evaluate with a different threshold strategy, or to run the
# debug visualisations on an already-trained model.
#
# CHECKPOINT: full path to the checkpoint to evaluate. Deterministic layout is:
#   $RUNS_ROOT/<experiment_name>/<run_id>/<phase>/checkpoints/best.pt
CHECKPOINT=$RUNS_ROOT/mae_tx_swat/<run_id>/merged/checkpoints/best.pt

python -m incremental_ad.main \
    --experiment_name mae_tx_swat \
    --model MaeTx \
    --dataset Swat \
    --task ad \
    --pipeline EvalPipeline \
    --seed 0 \
    \
    --mae_tx_patch_len 5 \
    --mae_tx_encoder_embed_dim 256 \
    --mae_tx_encoder_layers 2 \
    --mae_tx_encoder_heads 2 \
    --mae_tx_decoder_embed_dim 128 \
    --mae_tx_decoder_layers 1 \
    --mae_tx_decoder_heads 2 \
    --mae_tx_patch_norm false \
    --mae_tx_mask_ratio 0.8 \
    --mae_tx_n_eval_passes 30 \
    --mae_tx_training_mode random_mask \
    \
    --dataset_window_len 100 \
    --dataset_stride 1 \
    --dataset_normalization standard \
    --dataset_val_fraction 0.15 \
    \
    --loader_batch_size 512 \
    --loader_num_workers 4 \
    \
    --pipeline_checkpoint_path $CHECKPOINT \
    \
    --configurator_threshold_strategy oracle \
    --configurator_threshold_percentile 99

echo "Finished at: $(date)"
