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
| `bitwise` | 412 |
| `skipped_became` | 6 |

**412/412 reproduce bitwise.**

α was read from `merge_scale/selected` for **153** runs and
from `config.json` for 265. That split is the point of the check's
ordering rule: with validation selection on, `config.json` records the value that was *asked
for* while the checkpoint was built at the one that *won*, and reading the config first once
produced 12 spurious mismatches — exactly the selecting runs.

`skipped_became` are the §1.31 BECAME cells: λ\* is derived from per-shard Fishers that are not
stored with the run, so the merge cannot be rebuilt from checkpoints alone. Reported as skipped
rather than counted as passing — a check that was not performed must not read as a pass.

Full per-run detail: `results_archive/audit/merge_reproduction.csv`.
