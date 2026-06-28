#!/bin/bash

#SBATCH --job-name=incremental_ad_standard
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

# ── StandardPipeline: train on all training data, then evaluate ───────────────
# PSM anomaly rate is ~28%, so threshold_percentile ~72 for train_percentile strategy.
python -m incremental_ad.main \
    --experiment_name mae_tx_psm \
    --model MaeTx \
    --dataset Psm \
    --task ad \
    --pipeline StandardPipeline \
    --seed 42 \
    \
    --mae_tx_patch_len 10 \
    --mae_tx_encoder_embed_dim 256 \
    --mae_tx_encoder_layers 2 \
    --mae_tx_encoder_heads 2 \
    --mae_tx_decoder_embed_dim 128 \
    --mae_tx_decoder_layers 1 \
    --mae_tx_decoder_heads 2 \
    --mae_tx_patch_norm false \
    --mae_tx_mask_ratio 0.80 \
    --mae_tx_n_eval_passes 30 \
    --mae_tx_training_mode random_mask \
    \
    --dataset_window_len 100 \
    --dataset_stride 50 \
    --dataset_normalization standard \
    --dataset_val_fraction 0.15 \
    \
    --loader_batch_size 64 \
    --loader_num_workers 4 \
    \
    --trainer_n_epochs 300 \
    --trainer_patience 30 \
    --trainer_optimizer adamw \
    --trainer_weight_decay 1e-2 \
    --trainer_learning_rate 1e-4 \
    --trainer_grad_clip 0.5 \
    --trainer_scheduler cosine \
    --trainer_warmup_ratio 0.1 \
    --trainer_checkpoint_interval 0 \
    \
    --configurator_threshold_strategy oracle \
    --configurator_threshold_percentile 72

echo "Finished at: $(date)"
