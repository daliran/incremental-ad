#!/bin/bash

#SBATCH --job-name=etth_forecast_diagnostics
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=01:00:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log

set -euo pipefail

# ── Source run to analyse ─────────────────────────────────────────────────────
# A completed IncrementalTaskArithmeticPipeline run directory. Override per submission:
#   sbatch --export=ALL,SOURCE_RUN=$WORK/runs/mae_tx_etth_forecast/<run_id> \
#          scripts/sbatch_mae_tx_etth_forecast_diagnostics.sh
# Optional: STANDARD_RUN, a StandardPipeline run added as a reference row (test block
# only -- Standard trains on everything but the global tail, so the interior val slices
# sit inside its training data and its val cells would be training performance).
SOURCE_RUN=${SOURCE_RUN:-}
STANDARD_RUN=${STANDARD_RUN:-}

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

# ── Fail before queueing work if the source run wasn't given ──────────────────
if [[ -z "$SOURCE_RUN" ]]; then
    echo "ERROR: SOURCE_RUN is not set. Pass a completed incremental run directory:" >&2
    echo "  sbatch --export=ALL,SOURCE_RUN=\$WORK/runs/mae_tx_etth_forecast/<run_id> $0" >&2
    exit 1
fi
if [[ ! -f "$SOURCE_RUN/config.json" ]]; then
    echo "ERROR: $SOURCE_RUN is not a run directory (no config.json)" >&2
    exit 1
fi

# ── Diagnostic info ───────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Source run: $SOURCE_RUN"
nvidia-smi

# ── Activate venv (uv) ────────────────────────────────────────────────────────
cd $PROJECT_ROOT
source .venv/bin/activate

# ── MergeDiagnosticsPipeline ──────────────────────────────────────────────────
# Training-free: reloads the source run's checkpoints and cross-evaluates every model
# (base, each specialist theta_0 + tau_i, merged) against every column.
#
# What this adds over the source run itself: that run already scored each specialist on
# *its own* shard's val slice and on the full test set -- the matrix diagonal and the
# test column. The off-diagonal (theta_0 + tau_i on shard j) is what's missing, and
# without it the diagonal proves nothing: "tau_1 improved on shard 1" cannot be told
# apart from "tau_1 improved everywhere". Only improving *more* on its own shard than on
# the others shows anything shard-specific was learned.
#
# Two column blocks: the val slices (loss-shaped -- MSE for forecasting -- reported as a
# ratio to the base model on the same column, so shards of differing difficulty compare
# and lower is better), and the full unpartitioned test set with every task metric,
# identical for every row.
#
# EVERY dataset arg below must match the source run -- the pipeline hard-errors on any
# difference, because mismatched splits would evaluate regions that are not the shards
# these checkpoints were fine-tuned on, and every number would look plausible while
# meaning nothing. Keep this block in sync with sbatch_mae_tx_etth_forecast_incremental.sh.
#
# --pipeline_merge_scales additionally traces test metrics against the merge scale, from
# these same checkpoints. The source run fixed its scale before training, so tracing the
# curve that way costs a full retrain per point and confounds the shape with seed-to-seed
# variance (EXPERIMENTS.md §4); here only the scale varies. alpha=0 and the source run's
# own scale are reused from the matrix, so two of the points are free.
#
# Output: $RUNS_ROOT/<experiment_name>/<run_id>/merge_diagnostics/
#   transfer_matrix.csv    long format: model, column, block, metric, value,
#                          ratio_to_base, n_windows, eval_seed
#   merge_scale_curve.csv  merge_scale, split, metric, value, n_windows, eval_seed
#   result.json            per-metric summary scalars (diag/offdiag/specialisation/
#                          merged/base_slice ratios, GRR + gap, scale_at_min/max)
#                          -- picked up by collect.py
#   source.json            which run was analysed, its merge_scale, columns, window counts
# The source run is never written to; it stays an immutable artifact.
python -m incremental_ad.main \
    --experiment_name mae_tx_etth_forecast_diagnostics \
    --model MaeTx \
    --dataset EtthForecastDataset \
    --task forecast \
    --pipeline MergeDiagnosticsPipeline \
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
    --dataset_baseline_fraction 0.5 \
    --dataset_baseline_use_fraction 1.0 \
    --dataset_n_finetune_segments 3 \
    --dataset_val_fraction 0.1 \
    --dataset_test_fraction 0.2 \
    \
    --loader_batch_size 128 \
    \
    --pipeline_source_run_dir "$SOURCE_RUN" \
    --pipeline_checkpoint_name best \
    --pipeline_merge_scales 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 \
    ${STANDARD_RUN:+--pipeline_standard_run_dir "$STANDARD_RUN"}

echo "Finished at: $(date)"
