# results_archive — the evidence behind every published number

This directory exists so that **losing `$WORK` does not turn the documents back into a set of
unverifiable assertions.** `$WORK` is scratch: not backed up, purged on a schedule nobody
controls. It holds 11 GB of runs, but the *evidence* — the CSVs every table in
[EXPERIMENTS.md](../EXPERIMENTS.md) is checked against — is about 3.4 MB, so it lives here in
the repo instead.

Regenerate with:

```bash
python scripts/archive_results.py --runs_root $RUNS_ROOT \
    --audit_dir $OUT --geometry_root $OUT/geometry
```

## The test that matters

The whole point is that verification works with **no `$WORK` at all**:

```bash
python scripts/check_tables_against_csv.py \
    --audit_dir results_archive/audit \
    --runs_root results_archive/run_diagnostics --strict
```

That reproduces all **276 checks over 19 sections**, plus the 80-cell transfer-matrix pass
(§1.4), the 92-cell merge-scale-curve pass (§1.3) and the §1.11↔§1.12 reconciliation — from
this directory alone. If it passes, the numbers in the markdown are still backed by data.

## Layout

| path | what it is | used by |
|---|---|---|
| `audit/derived.csv` | per-experiment floor, headroom, GRR, retention, committed α | §0.1b, §1.9, §1.11, §1.25 |
| `audit/run_metrics.csv` | mean/sd per (experiment, **n_segments**, block, metric), including `finetune_i/test`, with an `of_record` column | §1.17, §1.21, §1.23, §1.24 |
| `audit/scale_forecast/`, `audit/scale_ad/` | α\*, α\*·n, GRR, honest-α cost | §1.11, §1.12, §1.18 |
| `audit/routing_forecast/`, `audit/routing_ad/` | routing headroom, merge cost, specialisation | §1.16 |
| `audit/alignment/` | alignment vs α\*·n, within/between correlations | §1.18 |
| `audit/geometry/geometry_by_dataset.csv` | per-dataset ρ, cosine, rank, τ norms | §1.8 |
| `audit/novelty/<dataset>/` | per-step ρ and new_k | §1.7 |
| `audit/methods/`, `audit/outcomes.csv` | five-method comparison, ρ-indicator outcomes | §1.26, THEORY §5.4 |
| `audit/methods_all/` | **every technique on every metric** — all six metrics, per-W windowed | §1.5b |
| `audit/prefix_ETTh1/`, `audit/prefix_exchange/` | forward transfer, accumulate vs materialise | §1.19, §1.22 |
| `audit/oracle_router/` | per-window routing ceiling on test, per run + seed-aggregated summary | §1.16b |

⚠️ **Two different quantities, two different scales — never pool them.** `methods_all/method_comparison.csv:routing_headroom_pct` is a **percentage** (5.6, 22.6, 108.3): how far merging sits from the per-regime optimum on validation slices, §1.16. `oracle_router/oracle_router_summary.csv:oracle_router` is an **absolute test metric** (0.4057, 0.2354): the per-window ceiling of §1.16b. Both were briefly called `oracle_router`; the first was renamed on 2026-08-08 so a percentage cannot be misread as an MSE.
| `geometry/` | `geometry_summary.csv` + small per-run geometry | §1.7, §1.8, §1.15, §1.18 |
| `run_diagnostics/<group>/<run>/merge_diagnostics/` | `transfer_matrix.csv`, `merge_scale_curve.csv` | §1.3, §1.4 |
| `MANIFEST.csv` | every file with size and SHA-256 | integrity check |

## What is deliberately not here

- **Checkpoints (`*.pt`) and `wandb/`** — gigabytes, and reproducible from each run's
  `config.json`. Note the consequence: the **bitwise merge-reproduction check** (CLAUDE.md) needs
  the checkpoints and therefore cannot be re-run from this archive.
- **`principal_angles.csv`, `per_tensor_cosine.csv`, `geometry.json`** — 33 MB combined, and no
  published number reads them.
- **Raw datasets** — downloaded from HuggingFace (`thuml/Time-Series-Library`); SWaT and PSM are
  access-restricted and were never in the repo.

## Reading `run_metrics.csv` safely

Two things a consumer needs to know, both fixed on 2026-08-08 after a reader hit them:

- **Rows are keyed by (experiment, n_segments), not by experiment.** Three
  `slurm_grid_*_train_incremental` groups hold runs at n = 2, **3 and 5** under one directory.
  Keying on the name alone stamped the row with whichever `n_segments` was read last (3, the
  majority) while carrying `finetune_0..finetune_4` blocks from the n=5 runs — so filtering on
  `(dataset, n_segments, block)` built a five-shard row out of a three-shard experiment. Worse,
  because blocks are keyed by seed, same-seed runs at different n silently *overwrote* each
  other instead of averaging. Both are gone; those three groups now appear as three rows each.
- **`of_record` names the experiment the published numbers read.** Eighteen (dataset, n) keys
  have more than one experiment, and "the one with the most seeds" is a heuristic, not a
  guarantee. The column carries `merge`, `sequential` or `joint` for the 42 experiments the
  documents actually use, from `analysis_specs/experiment_of_record.csv`; blank means the run
  exists but no published number reads it. **Filter on it** rather than guessing.

## Caveats worth knowing

- The archive is a **snapshot**, not a live mirror. Re-run `archive_results.py` after any run
  that changes a published number, or the checker will verify the documents against stale
  evidence — which is worse than not checking, because it looks like it passed.
- Four sections cannot be checked from here or anywhere else yet, because no script emits the
  quantity they use: **§1.2 and §1.10** (block-mean α, EXECUTION_PLAN.md §3.13) and **§1.19 and
  §1.22** (prefix-merge α, §3.14). The checker names them on every run.
- `audit/` is copied wholesale rather than by filename, so output from a newly added analysis
  tool lands here automatically.
