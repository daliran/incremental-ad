#!/bin/bash

#SBATCH --job-name=merge_diagnostics
#SBATCH --account=tesi_ddellacasaventurelli01
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --time=02:00:00
#SBATCH --output=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log
#SBATCH --error=/work/tesi_ddellacasaventurelli01/incremental-ad/logs/job_%j.log

set -euo pipefail

# ── What to analyse ───────────────────────────────────────────────────────────
# SOURCE_RUN   a completed IncrementalTaskArithmeticPipeline run directory (required)
# STANDARD_RUN a StandardPipeline run, added as the GRR reference row (optional). Scored
#              on the test block only: Standard trains on everything but the global tail,
#              so the interior val slices sit inside its training data.
# MERGE_SCALES scales for the merge-scale curve; empty skips it. AD is far more expensive
#              per point than forecasting (n_eval_passes random masks over a much larger
#              test set), so a coarser grid there is reasonable.
# EXTRA_ARGS   passed through to the pipeline verbatim, e.g. "--pipeline_eval_seeds 43 44"
#
#   sbatch --export=ALL,SOURCE_RUN=$WORK/runs/mae_tx_etth_forecast/<run_id> \
#          scripts/sbatch_merge_diagnostics.sh
#
# This script is dataset-agnostic on purpose. Every model and dataset argument is read
# back out of the source run's own config.json, so one script covers SWaT, PSM and ETTh1,
# and it works on runs from any past sweep -- including trials that varied
# baseline_fraction / val_fraction / n_finetune_segments, which a script with hardcoded
# args could not analyse at all.
SOURCE_RUN=${SOURCE_RUN:-}
STANDARD_RUN=${STANDARD_RUN:-}
MERGE_SCALES=${MERGE_SCALES:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"}
EXTRA_ARGS=${EXTRA_ARGS:-}

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
    echo "  sbatch --export=ALL,SOURCE_RUN=\$WORK/runs/<experiment>/<run_id> $0" >&2
    exit 1
fi
if [[ ! -f "$SOURCE_RUN/config.json" ]]; then
    echo "ERROR: $SOURCE_RUN is not a run directory (no config.json)" >&2
    exit 1
fi

# ── Diagnostic info ───────────────────────────────────────────────────────────
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Started at:   $(date)"
echo "Source run:   $SOURCE_RUN"
echo "Standard run: ${STANDARD_RUN:-<none>}"
echo "Merge scales: ${MERGE_SCALES:-<curve skipped>}"
nvidia-smi

# ── Activate venv (uv) ────────────────────────────────────────────────────────
cd $PROJECT_ROOT
source .venv/bin/activate

# ── MergeDiagnosticsPipeline ──────────────────────────────────────────────────
# Training-free: reloads the source run's checkpoints and cross-evaluates every model
# (base, each specialist theta_0 + tau_i, merged, optionally Standard) against every
# column.
#
# What this adds over the source run itself: that run already scored each specialist on
# *its own* shard's val slice and on the full test set -- the matrix diagonal and the
# test column. The off-diagonal (theta_0 + tau_i on shard j) is what's missing, and
# without it the diagonal proves nothing: "tau_1 improved on shard 1" cannot be told
# apart from "tau_1 improved everywhere". Only improving *more* on its own shard than on
# the others shows anything shard-specific was learned.
#
# The merge-scale curve answers the other question a training run cannot: it fixes its
# scale before training, so tracing the curve that way costs a full retrain per point and
# confounds the shape with seed-to-seed variance (EXPERIMENTS.md §4). Here every point
# shares one set of checkpoints, so only the scale varies.
#
# Output: $RUNS_ROOT/<source experiment>_diagnostics/<run_id>/merge_diagnostics/
#   transfer_matrix.csv    model, column, block, metric, value, ratio_to_base,
#                          n_windows, eval_seed
#   merge_scale_curve.csv  merge_scale, split, metric, value, n_windows, eval_seed
#   result.json            summary scalars: diag/offdiag/specialisation/merged/
#                          base_slice ratios, GRR + the gap it is divided by, and
#                          scale_at_min/max -- picked up by collect.py
#   source.json            which run was analysed, its merge_scale, columns, window counts
# The source run is never written to; it stays an immutable artifact.
python -m incremental_ad.analysis.diagnose \
    --source_run_dir "$SOURCE_RUN" \
    ${STANDARD_RUN:+--standard_run_dir "$STANDARD_RUN"} \
    ${MERGE_SCALES:+--merge_scales $MERGE_SCALES} \
    ${EXTRA_ARGS:+-- $EXTRA_ARGS}

echo "Finished at: $(date)"
