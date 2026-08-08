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
| `audit/run_metrics.csv` | mean/sd per (experiment, block, metric), **including `finetune_i/test`** | §1.17, §1.21, §1.23, §1.24 |
| `audit/scale_forecast/`, `audit/scale_ad/` | α\*, α\*·n, GRR, honest-α cost | §1.11, §1.12, §1.18 |
| `audit/routing_forecast/`, `audit/routing_ad/` | routing headroom, merge cost, specialisation | §1.16 |
| `audit/alignment/` | alignment vs α\*·n, within/between correlations | §1.18 |
| `audit/geometry/geometry_by_dataset.csv` | per-dataset ρ, cosine, rank, τ norms | §1.8 |
| `audit/novelty/<dataset>/` | per-step ρ and new_k | §1.7 |
| `audit/methods/`, `audit/outcomes.csv` | five-method comparison, ρ-indicator outcomes | §1.26, THEORY §5.4 |
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

## Caveats worth knowing

- The archive is a **snapshot**, not a live mirror. Re-run `archive_results.py` after any run
  that changes a published number, or the checker will verify the documents against stale
  evidence — which is worse than not checking, because it looks like it passed.
- Four sections cannot be checked from here or anywhere else yet, because no script emits the
  quantity they use: **§1.2 and §1.10** (block-mean α, EXECUTION_PLAN.md §3.13) and **§1.19 and
  §1.22** (prefix-merge α, §3.14). The checker names them on every run.
- `audit/` is copied wholesale rather than by filename, so output from a newly added analysis
  tool lands here automatically.
