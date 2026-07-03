#!/bin/bash

#SBATCH --job-name=traffic_forecast_standard
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
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
# Same recipe as ETTh1's winning config (see sbatch_mae_tx_etth_forecast_standard.sh),
# except a smaller batch size and larger --mem/--time: Traffic has 862 features (vs
# ETTh1's 7 / Weather's 21) — patch_embedding's input dim scales with n_features, so
# this is noticeably heavier per epoch. A 1-epoch smoke test took ~1m45s on a single
# GPU with batch_size=64. Reduce --loader_batch_size further if this OOMs.
# window_len = context_len + forecast_len (120 = 96 + 24).
# Val sizing: val = val_fraction × train must exceed window_len (120).
#   train = 0.8 × 17544 = 14036;  0.1 × 14036 = 1403 ≥ 120 — OK.
# Output lands in $RUNS_ROOT/<experiment_name>/<run_id>/.
python -m incremental_ad.main \
    --experiment_name mae_tx_traffic_forecast \
    --model MaeTx \
    --dataset TrafficForecastDataset \
    --task forecast \
    --pipeline StandardPipeline \
    --seed 42 \
    \
    --mae_tx_patch_len 4 \
    --mae_tx_encoder_embed_dim 128 \
    --mae_tx_encoder_layers 3 \
    --mae_tx_encoder_heads 4 \
    --mae_tx_decoder_embed_dim 64 \
    --mae_tx_decoder_layers 2 \
    --mae_tx_decoder_heads 4 \
    --mae_tx_patch_norm false \
    --mae_tx_mask_ratio 0.5 \
    --mae_tx_n_eval_passes 1 \
    --mae_tx_training_mode causal_mask \
    --mae_tx_instance_norm false \
    \
    --dataset_window_len 120 \
    --dataset_forecast_len 24 \
    --dataset_stride 1 \
    --dataset_normalization standard \
    --dataset_val_fraction 0.1 \
    --dataset_test_fraction 0.2 \
    \
    --loader_batch_size 64 \
    --loader_num_workers 4 \
    \
    --trainer_n_epochs 100 \
    --trainer_patience 15 \
    --trainer_optimizer adamw \
    --trainer_weight_decay 1e-4 \
    --trainer_learning_rate 1e-3 \
    --trainer_grad_clip 1.0 \
    --trainer_scheduler cosine \
    --trainer_warmup_ratio 0.1 \
    --trainer_checkpoint_interval 0

echo "Finished at: $(date)"
