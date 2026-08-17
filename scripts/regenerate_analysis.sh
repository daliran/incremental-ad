#!/bin/bash
# Regenerate every analysis output the documents are checked against, into ONE directory.
#
#     srun --account=... --partition=all_usr_prod --time=00:30:00 --mem=16G \
#          bash scripts/regenerate_analysis.sh $WORK/audit_full
#
# Run it under srun, not on the login node. It is only CSV aggregation, but `results_audit`
# walks every run directory and takes over a minute on a full $RUNS_ROOT — past what a shared
# login node should be asked to do.
#     python scripts/check_tables_against_csv.py --audit_dir $WORK/audit_full --runs_root $RUNS_ROOT --strict
#     python scripts/archive_results.py --runs_root $RUNS_ROOT --audit_dir $WORK/audit_full \
#         --geometry_root $WORK/audit_full/geometry --out results_archive
#
# **Why this exists.** `results_archive/audit/` was assembled by hand across several sessions —
# each subdirectory written by a different invocation, remembered in a different transcript. That
# made the archive unreproducible as a whole even though every individual number in it was
# reproducible, and it meant adding one dataset required rediscovering ten command lines. This is
# the command line, once, for all of them.
#
# Everything here is pure aggregation over finished runs: it reads `config.json`, `result.json`,
# `merge_scale_curve.csv` and `transfer_matrix.csv` only. No GPU, no model, no dataset download,
# so it is safe on a login node and can be re-run at any time.
#
# The exception is `oracle_router` and `error_concentration`, which load checkpoints and score the
# test set. They need a GPU and are NOT run here — their outputs are carried forward from the
# previous archive by `--carry`. Regenerating those is a separate, deliberate, sbatch-ed step.
set -euo pipefail

OUT="${1:?usage: regenerate_analysis.sh <output_dir> [runs_root]}"
RUNS="${2:-${RUNS_ROOT:?set RUNS_ROOT or pass it as the second argument}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARRY="$REPO/results_archive/audit"

cd "$REPO"
mkdir -p "$OUT"

echo "== results_audit (run_metrics / derived / floors) =="
python -m incremental_ad.analysis.results_audit --runs_root "$RUNS" --out "$OUT"

echo "== merge-scale reports =="
python -m incremental_ad.analysis.scale_report "$RUNS"/{etth2,ettm2}_merge_n{2,3,5}_diagnostics/* \
    "$RUNS"/exch_incremental_diagnostics/* "$RUNS"/noisefloor_etth_diagnostics/* \
    "$RUNS"/segsweep_{etth1,exchange}_merge_n{2,5}_diagnostics/* --out "$OUT/scale_forecast"
python -m incremental_ad.analysis.scale_report "$RUNS"/noisefloor_{psm,swat}_diagnostics/* \
    "$RUNS"/segsweep_{psm,swat}_merge_n{2,5}_diagnostics/* \
    --val_metric reconstruction/score_mean --test_metric window_auroc --out "$OUT/scale_ad"
# PSM-forecast kept in its own directory: it is the only forecasting group whose curve grid is
# 0.05 rather than 0.1, and pooling grids is what §2.23 had to retract once.
python -m incremental_ad.analysis.scale_report "$RUNS"/adfc2_psm_merge_n{2,3,5}_diagnostics/* \
    --out "$OUT/scale_psm_forecast"

echo "== routing reports =="
python -m incremental_ad.analysis.routing_report "$RUNS"/{etth2,ettm2}_merge_n{2,3,5}_diagnostics/* \
    "$RUNS"/segsweep_{etth1,exchange}_merge_n{2,5}_diagnostics/* \
    --metric forecast/mse --out "$OUT/routing_forecast"
# AD routes on `reconstruction/score_mean`, NOT on a detection metric: the per-regime columns
# of an AD transfer matrix carry no AUROC (§1.16), so asking for window_auroc yields "no usable
# cells" rather than a number. That is the intended refusal, not a bug — do not "fix" it by
# picking a metric that happens to parse.
python -m incremental_ad.analysis.routing_report "$RUNS"/noisefloor_{psm,swat}_diagnostics/* \
    --metric reconstruction/score_mean --out "$OUT/routing_ad"
python -m incremental_ad.analysis.routing_report "$RUNS"/adfc2_psm_merge_n{2,3,5}_diagnostics/* \
    --metric forecast/mse --out "$OUT/routing_psm_forecast"

echo "== method comparison (primary metric, then every metric) =="
python -m incremental_ad.analysis.method_comparison --runs_root "$RUNS" \
    --spec analysis_specs/method_comparison_spec.csv --routing_dir "$OUT/routing_forecast" \
    --floors "$OUT/floors.csv" --run_metrics "$OUT/run_metrics.csv" --out "$OUT/methods"
cp "$OUT/methods/method_comparison.csv" "$OUT/method_comparison.csv"
python -m incremental_ad.analysis.method_comparison --runs_root "$RUNS" \
    --spec analysis_specs/method_comparison_spec.csv \
    --metrics forecast/mse forecast/mae window_auroc window_auprc point_auroc point_auprc \
    --routing_dir "$OUT/routing_forecast" \
    --floors "$OUT/floors.csv" --run_metrics "$OUT/run_metrics.csv" --out "$OUT/methods_all"

echo "== prefix reports =="
# prefix_report reads `prefix_merges.csv`, written only by diagnostics runs launched with
# --pipeline_prefix_merges. Those live in `prefix_etth1` / `prefix_exchange`, NOT in the
# `*_diagnostics` groups — pointing it at the latter gives "no prefix_merges.csv found".
python -m incremental_ad.analysis.prefix_report "$RUNS"/prefix_etth1/* \
    --label ETTh1 --out "$OUT/prefix_ETTh1"
python -m incremental_ad.analysis.prefix_report "$RUNS"/prefix_exchange/* \
    --label exchange --out "$OUT/prefix_exchange"

echo "== geometry / novelty (checkpoint readers — GPU node) =="
# geometry_report loads every checkpoint to compute tau norms and overlaps, and novelty's two
# subcommands read its output. That is minutes of I/O over gigabytes, so it is NOT run on a login
# node: submit it, then re-run this script to pick the results up. Until then the archived copies
# are carried forward and reported as carried.
if [ "${WITH_GEOMETRY:-0}" = "1" ]; then
    python -m incremental_ad.analysis.geometry_report \
        "$RUNS"/noisefloor_{etth,psm,swat}/* "$RUNS"/exch_incremental/* \
        "$RUNS"/{etth2,ettm2}_merge_n{2,3,5}/* \
        "$RUNS"/segsweep_{etth1,exchange,psm,swat}_merge_n{2,5}/* --out "$OUT/geometry"
    python -m incremental_ad.analysis.novelty_report geometry_table \
        --geometry_root "$OUT/geometry" --spec analysis_specs/geometry_table_spec.csv \
        --out "$OUT/novelty"
    python -m incremental_ad.analysis.novelty_report alignment \
        --geometry "$OUT/geometry/geometry_summary.csv" \
        --scale "$OUT/scale_forecast/scale_summary.csv" "$OUT/scale_ad/scale_summary.csv" \
        --spec analysis_specs/alignment_spec.csv --out "$OUT/alignment"
else
    echo "  skipped (set WITH_GEOMETRY=1 on a compute node to regenerate)"
fi

echo "== carrying forward GPU-only outputs (not regenerated here) =="
for sub in oracle_router concentration novelty_swat selection_probe drift \
           geometry novelty alignment; do
    if [ -d "$CARRY/$sub" ] && [ ! -d "$OUT/$sub" ]; then
        cp -r "$CARRY/$sub" "$OUT/$sub"
        echo "  carried $sub from results_archive (regenerate with a GPU job if its runs changed)"
    fi
done
# Anything else the archive has and this run did not produce — carried, and reported, so a
# missing generator is visible rather than silently inherited.
for path in "$CARRY"/*; do
    name="$(basename "$path")"
    [ -e "$OUT/$name" ] && continue
    cp -r "$path" "$OUT/$name"
    echo "  ⚠️  carried $name with no generator in this script — add one or drop it"
done

echo
echo "wrote $OUT"
