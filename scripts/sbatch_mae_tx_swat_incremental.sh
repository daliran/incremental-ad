#!/bin/bash

#SBATCH --job-name=incremental_ad_incremental
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

# ── Phase ─────────────────────────────────────────────────────────────────────
# "incremental" is the whole pipeline in one job: pretrain the base on the first
# --partial-ratio of the train series, fine-tune it on each of the --n-finetune
# remaining chunks, merge the fine-tunings into the base (task arithmetic, scaled
# by --merge-scale) and evaluate the merged model. Output goes to
# experiments/<exp>/incremental/<job>/{base,ft_0..,merged}/.
EXPERIMENT=mae_tx_swat
PHASE=incremental

python -m incremental_ad.main \
    --phase $PHASE \
    --experiment $EXPERIMENT \
    --device auto \
    --dataset swat \
    --model mae_tx \
    --partial-ratio 0.5 \
    --n-finetune 3 \
    --merge-scale 0.5 \
    --base-data-ratio 1.0 \
    --swat-window-len 100 \
    --swat-stride 50 \
    --swat-eval-stride 1 \
    --swat-normalization standard \
    --swat-val-ratio 0.15 \
    --mae-tx-patch-len 10 \
    --mae-tx-encoder-embed-dim 256 \
    --mae-tx-encoder-layers 2 \
    --mae-tx-encoder-heads 2 \
    --mae-tx-decoder-embed-dim 128 \
    --mae-tx-decoder-layers 1 \
    --mae-tx-decoder-heads 2 \
    --mae-tx-patch-norm false \
    --mae-tx-mask-ratio 0.80 \
    --mae-tx-n-eval-passes 30 \
    --mae-tx-training-mode random_mask \
    --train-seed 42 \
    --train-epochs 300 \
    --train-patience 30 \
    --train-batch-size 64 \
    --train-optimizer adamw \
    --train-weight-decay 1e-2 \
    --train-learning-rate 1e-4 \
    --train-grad-clip 0.5 \
    --train-scheduler cosine \
    --train-warmup-ratio 0.1 \
    --train-checkpoint-interval 0 \
    --finetune-seed 42 \
    --finetune-epochs 50 \
    --finetune-patience 10 \
    --finetune-batch-size 64 \
    --finetune-optimizer adamw \
    --finetune-weight-decay 1e-2 \
    --finetune-learning-rate 1e-5 \
    --finetune-grad-clip 0.5 \
    --finetune-scheduler cosine \
    --finetune-warmup-ratio 0.1 \
    --finetune-checkpoint-interval 0 \
    --val-eval-seed 0 \
    --val-eval-batch-size 512 \
    --test-eval-seed 0 \
    --test-eval-batch-size 512 \
    --test-eval-threshold-strategy oracle \
    --test-eval-threshold-percentile 99

echo "Finished at: $(date)"