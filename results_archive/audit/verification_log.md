# Verification log

Committed output of the checks that cannot be run from the archive alone — they need either the
checkpoints or a GPU, so they are run on the cluster and their results recorded here. Everything
else (`check_tables_against_csv.py --strict`, `--self-test`) runs from the repo and is not
duplicated.

Generated at 2026-09-08 19:59 UTC · commit `e20a993`

## Merge-rule semantics — `scripts/verify_merge_rules.py`

The four properties that would silently corrupt §1.31's ablation if they failed. Run **before**
any number from a new merge rule is quoted, per CLAUDE.md's convention.

```
MERGE RULE SELF-CHECKS
  ok  plain sum reproduces apply_task_vectors bitwise at 6 scales
  ok  BECAME fold equals the convex weight update
  ok  OPCM residual is orthogonal, removes exactly rho, identity at n=1, and passes 1-D tensors through
  ok  equal Fishers give lambda_t = 1/t, and the fold equals alpha = 1/n exactly

0 failure(s)
```

## Bitwise merge reproduction — `scripts/verify_merge_reproduction.py`

Every archived merge recomputed from its own `baseline/` + `finetune_i/` checkpoints at the
committed α, requiring `torch.equal` on every tensor. CLAUDE.md calls this the project's
strongest regression test and cited **87/87**; it had no committed script until now, and the
current run covers far more:

| status | runs |
|---|---|
| `bitwise` | 454 |
| `skipped_became` | 21 |

**454/454 reproduce bitwise** (re-run 2026-09-25; max abs difference 0.0 on every tensor). Up from 412 because the strategy-6 and §1.40 run groups now exist.

α was read from `merge_scale/selected` for **183** runs and
from `config.json` for 271. That split is the point of the check's
ordering rule: with validation selection on, `config.json` records the value that was *asked
for* while the checkpoint was built at the one that *won*, and reading the config first once
produced 12 spurious mismatches — exactly the selecting runs.

`skipped_became` are the §1.31 BECAME cells: λ\* is derived from per-shard Fishers that are not
stored with the run, so the merge cannot be rebuilt from checkpoints alone. Reported as skipped
rather than counted as passing — a check that was not performed must not read as a pass.

Full per-run detail: `results_archive/audit/merge_reproduction.csv`.

## Unit gates and the full verifier pass — 2026-09-25

`scripts/verify_core.py` (new): **64/64 gates pass** — splitting, windowing, metrics against
sklearn and hand-worked cases, the trainer's checkpoint restore and L2-SP exclusion, seeded
evaluation, results storage, and the forecasting **future-leakage test**: poisoning the horizon
of a window leaves the prediction bit-identical, with and without instance norm.

`scripts/verify_merge_baselines.py` (new): **49/49** — the supervisor's four merge rules as
ported equal his reference (`other/`, imported read-only) except by exactly the four documented
defects, each of which is also reproduced on the reference.

Every pre-existing verifier and self-test was re-run in one job (SLURM 116616), all exit 0:
`verify_adaptive_lambda`, `verify_merge_rules`, `verify_train_only`, `verify_checkpoints`
(**3,921/3,921** checkpoints verified, 0 missing, 0 mismatched), `verify_mask_span`,
`verify_merge_reproduction` (above), and the `--self-test` of every analysis script, the claims
register and the table checker.

**Evaluation-seed fix (`remerge.py`).** Re-merges now score at the run's own `eval_seed`
(seed + 1), as the pipeline does. Before the fix the same model scored 0.0026% (SWaT) and 0.018%
(PSM) AUROC away from its stored value; after it, a PSM re-merge reproduces the stored score to
~1e-9 (`window_auroc` 0.8028227483 stored, 0.8028227477 re-merged; `pa_f1` identical). All 54 AD
re-merge rows published before the fix were re-read with both arms at one seed: **0 verdicts
changed**, largest delta shift 0.006 percentage points.

## Evaluation-gap batch — 2026-09-26

`scripts/verify_core.py`: **68/68 gates pass**. That is the 64 above plus four new
**Fisher-sample-count** gates: `diagonal_fisher` records the samples it actually used, not the
product of its flags. A full pass at B = 1, a partial last batch, a batch cap, and a cap larger
than the shard (512 requested, 10 available → 10 recorded).

`scripts/verify_merge_baselines.py`: **49/49** (unchanged code, re-run).
`scripts/verify_adaptive_lambda.py`: the 5 locally runnable gates pass, 0 failures. Gates 1 and 3
run inside the pipeline on the cluster and were last recorded in SLURM 116616.
`analysis/calibration_split.py` self-test passes. That includes the refusal of a reordered score
vector, which is what binds the §1.42 split to the evaluator's window order.

**New evaluation code, checked against published values before any result was quoted:**
- **§1.42 (calibration split):**
  - The base at α = 0 reproduces every AD run's own `baseline/test` window AUROC to ≤ 8e-9.
  - Every job asserts that the split, applied to the whole test set, reproduces the published
    `window_auroc` to 1e-9.
- **§1.43 (prequential):** every job's rescored chain matched the chain run's own recorded
  per-period value at every k (tolerance 1e-5).
- **§1.39 full-pass Fisher:** the recorded sample counts (712 per period, 2,230 base) match the
  window counts the dataset defines.

**Two summaries that had no generator** now reproduce the archived files byte for byte:
`oracle_router/oracle_router_summary.csv` (`oracle_router_report.py`) and
`remerge/fisher_sweep_summary.csv` (`fisher_sweep_report.py`).

`verify_merge_reproduction` and `verify_checkpoints` were not re-run: no merge of record changed.

## Full GPU regeneration (item F) — 2026-09-26

SLURM 120740 (`WITH_GEOMETRY=1`): **every CSV the checker reads is regenerated from the runs and
reproduces the archive** — 107 files byte for byte, and `geometry/geometry_summary.csv` within a
relative 1e-9. That last file is GPU float noise: the largest difference is 1.7e-13, with the same
rows and the same text cells. 0 differ, 0 not produced. Nothing the checker reads is carried any
more.

**The first attempt (SLURM 120705) "passed" and was wrong.** Checked file by file, it showed
that the guard skipped every file under a carried-type directory even when the GPU run had
regenerated it. Behind that were three pipeline defects, showing up as four failing files:
- `geometry_report` was invoked on a hand-written list covering **54 of the archived table's 220
  runs**. Its 54 rows matched to 1.7e-13; the other 166 were never rebuilt.
- `geometry_by_dataset.csv` was written to `novelty/`, while the archive keeps it in `geometry/`.
- As a consequence, `alignment_correlation.csv` came out slightly different (within_r 0.4595
  against 0.4617), inside the checker's 0.02 tolerance, so no number check could see it.

**Fixed:**
- The run list is now read from the archived table itself.
- The output path matches the archive.
- The regeneration records which directories it copied (`<out>.carried`), and the guard
  compares everything else, including a file missing from a produced directory, which now counts
  as NOT PRODUCED.
- Replayed on the first attempt's output, the fixed guard fails it with exactly those four
  files: three that differ and one not produced.

**Still carried**, none read by the checker:
- the GPU re-merge and grid-search per-run trees, which the pipeline reads as inputs;
- the verification records;
- five older GPU analyses: `geometry_gap`, `geometry_aeft`, `novelty_swat`, `mask_span`,
  `subblocks_origin`.
