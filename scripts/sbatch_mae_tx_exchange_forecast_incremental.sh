#!/bin/bash

#SBATCH --job-name=exchange_forecast_incremental
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=03:00:00
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
# Node-local /tmp is shared with every other job on that node and frequently runs out
# of space -- when it does, Python's tempfile falls back through its candidate list to
# the current working directory (PROJECT_ROOT, since we cd there below), littering the
# repo with empty pymp-* dirs; when it doesn't fall back cleanly, DataLoader workers can
# spin retrying a failed mkdtemp instead of failing fast, burning the whole time limit.
# Use a per-job dir on $WORK (BeeGFS, effectively never full) instead, cleaned up on exit.
export TMPDIR=$WORK/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT

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

# ── IncrementalTaskArithmeticPipeline ─────────────────────────────────────────
# Same recipe as ETTh1's winning config (see sbatch_mae_tx_etth_forecast_incremental.sh),
# except a smaller window_len/forecast_len: ExchangeRate has only 8 features and DAILY
# (not hourly/10-min) frequency, ~7588 rows — much smaller than ETTh1/Weather/Traffic.
# window_len=120/forecast_len=24 (ETTh1's values) do NOT fit safely here: with
# baseline_fraction=0.5 and n_finetune_segments=3, each fine-tune segment is only
# ~1012 rows, and val_fraction=0.1 of that (~101 rows) would still exceed 120? No —
# 101 < 120, which would leave an EMPTY val loader (see standard_trainer.py's "0
# windows" guard) and crash. Uses window_len=48/forecast_len=12 instead — verified
# against an actual run: baseline val=303 rows (256 windows), each fine-tune segment
# val=101 rows (54 windows), all comfortably ≥ 48.
# Output: $RUNS_ROOT/<experiment_name>/<run_id>/{baseline,finetune_0..N,merged}/.
# --pipeline_extra_merge_scales traces the merge-scale/metric curve in the same job:
# each listed scale is evaluated from the already-trained task vectors (no extra
# training, no extra checkpoints; merged/ still comes from --pipeline_merge_scale),
# and merged/merge_scale_curve.csv covers the primary scale plus these. Empty = off.
# Because every point shares one set of checkpoints, the curve carries none of the
# run-to-run training noise a curve built from separate runs would (EXPERIMENTS.md §4).
python -m incremental_ad.main \
    --experiment_name mae_tx_exchange_forecast \
    --model MaeTx \
    --dataset ExchangeRateForecastDataset \
    --task forecast \
    --pipeline IncrementalTaskArithmeticPipeline \
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
    --dataset_window_len 48 \
    --dataset_forecast_len 12 \
    --dataset_stride 1 \
    --dataset_normalization standard \
    --dataset_baseline_fraction 0.5 \
    --dataset_baseline_use_fraction 1.0 \
    --dataset_n_finetune_segments 3 \
    --dataset_val_fraction 0.1 \
    --dataset_test_fraction 0.2 \
    \
    --loader_batch_size 128 \
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
    --trainer_checkpoint_interval 0 \
    \
    --finetune_trainer_n_epochs 30 \
    --finetune_trainer_patience 10 \
    --finetune_trainer_weight_decay 1e-4 \
    --finetune_trainer_learning_rate 1e-4 \
    --finetune_trainer_grad_clip 1.0 \
    --finetune_trainer_scheduler cosine \
    --finetune_trainer_reg_lambda 0.0 \
    \
    --pipeline_merge_scale 0.5 \
    --pipeline_extra_merge_scales

echo "Finished at: $(date)"
