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
