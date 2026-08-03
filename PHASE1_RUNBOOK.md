# Phase 1 runbook — merge diagnostics on the cluster

Handoff for a session running on the SLURM server, where the real checkpoints live. All
of Phase 1 is **training-free**: it re-reads checkpoints that already exist under
`$WORK/runs` and cross-evaluates them. Nothing here trains a model.

Read `CLAUDE.md` first for repo conventions, then this.

---

## 1. Why this exists

Every incremental result so far is two rows: the frozen base θ₀, and the merged model
θ₀ + scale·Στᵢ. When merging fails to help, **three different mechanisms produce that same
signature** and nothing on disk distinguishes them:

- **redundancy** — the task vectors carry little the base doesn't already have
- **interference** — each τᵢ helps on its own shard, but the sum destroys the gains
- **degenerate fine-tuning** — the per-shard fine-tunes never learned anything

Only the second licenses any claim about non-orthogonality, which is the thesis's central
argument. Separating them needs one measurement that does not currently exist anywhere:
**θ₀ + τᵢ evaluated on shard j**, for j ≠ i.

A training run already produces the diagonal (each specialist on its own shard's val
slice) and the test column. It structurally cannot produce the off-diagonal. Without it,
"τ₁ improved on shard 1" cannot be told apart from "τ₁ improved everywhere" — and only
improving *more* on its own shard than on the others shows anything shard-specific was
learned.

**The goal of this runbook is to produce those numbers and report them back.** Do not
draw the research conclusions here; emit the measurements and summarise honestly.

---

## 2. What the code gives you

Three entry points, all new. None writes into the run it analyses — source runs stay
immutable and every output lands in a fresh run directory.

| tool | cost | what it answers |
|---|---|---|
| `python -m incremental_ad.analysis.geometry_report <run dirs>` | seconds, CPU, no dataset | where the task vectors point relative to each other |
| `python -m incremental_ad.analysis.diagnose --source_run_dir <dir>` | GPU, ~(n+3) eval passes + one per merge scale | the transfer matrix, GRR, and the merge-scale curve |
| `sbatch scripts/sbatch_merge_diagnostics.sh` | as above | the same, as a batch job |

Also new, but only affecting *future* training runs: `--pipeline_extra_merge_scales` on
`IncrementalTaskArithmeticPipeline` traces the scale curve at the end of training for
free. You do not need it for Phase 1 — the post-hoc path covers runs already on disk.

### Outputs, per diagnosed run

```
$RUNS_ROOT/<source experiment>_diagnostics/<run_id>/merge_diagnostics/
  transfer_matrix.csv    model, column, block, metric, value, ratio_to_base, n_windows, eval_seed
  merge_scale_curve.csv  merge_scale, split, metric, value, n_windows, eval_seed
  result.json            summary scalars (collected by collect.py)
  source.json            which run was analysed, its merge_scale, columns, window counts
```

`transfer_matrix.csv` rows are `base`, `ft_0..ft_{n-1}`, `merged`, and `standard` if you
supplied one. Columns are two blocks: the per-shard val slices (`val_base`, `val_0`, …)
and the full unpartitioned test set. **Val cells are loss-shaped** — reconstruction error
for AD (training data is unlabeled, so AUROC is undefined there), MSE for forecasting —
and are reported as `ratio_to_base`, so **lower is better and 1.0 means "no different from
the base model"**. Detection metrics come from the test column, which is identical for
every row.

---

## 3. Procedure

### Step 0 — sync and sanity-check

```bash
cd $PROJECT_ROOT && git pull && source .venv/bin/activate
python -c "import incremental_ad.project.datasets, incremental_ad.project.models, \
  incremental_ad.framework.trainers, incremental_ad.framework.pipelines, \
  incremental_ad.framework.evaluators; print('OK')"
```

**Regression check worth running once** — recompute every `merged/checkpoints/best.pt`
under `$WORK/runs` from its baseline + finetune checkpoints and the recorded
`pipeline_merge_scale`, and confirm `torch.equal` on every tensor. It passes 47/47 on the
local runs; if it fails on the cluster the merge is not reproducible there and everything
downstream is suspect. Use `framework.merging.merge_task_arithmetic` and
`framework.core.checkpoints.load_model_state`.

### Step 1 — inventory what is actually on disk

I do not know what runs exist on the server. Before anything else, enumerate them:

```bash
for f in $WORK/runs/*/*/config.json; do
  python - "$f" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
if c.get("pipeline") != "IncrementalTaskArithmeticPipeline": raise SystemExit
print(f"{sys.argv[1].rsplit('/',2)[0]:70s} {c['dataset']:20s} "
      f"seg={a['dataset_n_finetune_segments']} bf={a['dataset_baseline_fraction']} "
      f"vf={a['dataset_val_fraction']} patch={a.get('mae_tx_patch_len')} "
      f"scale={a['pipeline_merge_scale']}")
PY
done
```

Then check each candidate actually has `baseline/checkpoints/best.pt` and every
`finetune_*/checkpoints/best.pt` — a job that hit its time limit may have partial output.

**Priority targets**, in order:

1. The **proven-recipe incremental run per dataset** — SWaT, PSM at `patch_len=5`, ETTh1.
   These are what the thesis reports, so their matrices are the headline result.
2. Their matching **`StandardPipeline` runs**, as `--standard_run_dir`. Needed for GRR.
   They must agree on everything except `baseline_fraction` / `baseline_use_fraction` /
   `n_finetune_segments`, which `StandardPipeline` necessarily differs on.
3. The **`train_incremental` sweep trials**, for the correlation analysis (§5).

### Step 2 — geometry first

Cheapest thing here, no GPU, no dataset, and it can invalidate the framing before you
spend GPU time. Run it over **every** incremental run you found:

```bash
python -m incremental_ad.analysis.geometry_report $WORK/runs/<experiment>/* \
    --out $WORK/runs/analysis/geometry
```

Read `geometry_summary.csv`. The load-bearing check is `cosine_vs_distance.csv` per run:
**if cosine does not decay as `|i−j|` grows, temporal distribution shift is not what
differentiates the task vectors**, and the thesis framing needs revisiting regardless of
what the matrices show. Report the decay per dataset either way.

Also flag any run where `mean_tau_over_base` is near zero, or where one segment's
`tau_norm` is an order of magnitude below its siblings — that is degenerate fine-tuning
for that segment, and it makes an n-vector merge effectively an (n−1)-vector one. On the
local ETTh1 fixture, `finetune_0` had a τ 25× smaller than its siblings and
`best_epoch=1`; check `finetune_*/result.json` for `best_epoch` whenever you see it.

### Step 3 — diagnostics per target run

```bash
sbatch --export=ALL,SOURCE_RUN=$WORK/runs/<experiment>/<run_id>,\
STANDARD_RUN=$WORK/runs/<std_experiment>/<run_id> \
    scripts/sbatch_merge_diagnostics.sh
```

Env vars the script accepts: `SOURCE_RUN` (required), `STANDARD_RUN`, `MERGE_SCALES`
(defaults to `0.0 0.1 … 1.5`), `EXTRA_ARGS` (passed through verbatim).

**Cost control — read this before submitting AD jobs.** The default 16-point curve is
trivial for ETTh1 (~3.4k test windows, 1 eval pass) and expensive for SWaT/PSM: PSM's test
set is ~88k windows and AD scoring averages `n_eval_passes=30` random masks per window, so
each curve point is ~2.6M window-forwards. Start AD with a coarse grid and widen only if
the walltime allows:

```
MERGE_SCALES="0.0 0.25 0.5 0.75 1.0 1.25 1.5"
```

**Evaluation noise on AD.** Because AD scoring is stochastic, pass several seeds and
report the spread — on a local check with `n_eval_passes=1` the worst cell varied **47%
between seeds**, on `event_precision`. At the production 30 passes most of that averages
out, but do not quote a per-shard event metric without a spread:

```
EXTRA_ARGS="--pipeline_eval_seeds 43 44 45"
```

Also record `n_windows` per val slice from `source.json` — if the slices are small, the
per-shard numbers deserve the seed spread even more.

**If a job fails with "args differ from …", that is the guard working, not a bug.** The
pipeline refuses to analyse checkpoints with mismatched `dataset_*`/`mae_tx_*` args,
because the val columns would not be the shards those checkpoints were trained on, and
every number would look plausible while meaning nothing. `diagnose.py` reads the args from
the source run so this should not happen; if it does, you have pointed at the wrong
`STANDARD_RUN`. Do not reach for `--pipeline_allow_config_mismatch` to make it go away.

### Step 4 — collect

`merge_diagnostics/result.json` is `EvalStepResult`-shaped, so the existing collector
picks it up and yields one row per diagnosed run:

```bash
python slurm_grid_search/collect.py --dataset <ds> --stage <stage>   # if wired to a sweep
```

Otherwise read the per-run CSVs directly with pandas. Join diagnostics runs back to their
source via `source.json`'s `source_run_id`.

---

## 4. How to read the matrix

Pivot `transfer_matrix.csv` on the val block, using `ratio_to_base` (lower is better):

```
           val_base   val_0   val_1   val_2  |  test
base          1.000   1.000   1.000   1.000  |  ...
ft_0          0.978   0.913   0.884   1.005  |  ...
ft_1          0.951   0.469   0.649   0.984  |  ...
ft_2          1.288   0.719   0.506   0.416  |  ...
merged        0.941   0.497   0.604   0.533  |  ...
```

Questions to answer per dataset:

- **Is the diagonal the column minimum?** If `ft_i` is not the best model on `val_i`, that
  shard's specialist did not specialise. In the fixture above it holds only for segment 2.
- **`specialisation` in `result.json`** = `offdiag_ratio_mean − diag_ratio_mean`. Positive
  means specialists help more on their own shard than on others.
- **Is merged worse than each specialist on that specialist's shard, while still beating
  base everywhere?** That combination — and only that one — is the interference signature.
- **`base_slice_ratio_mean` > 1** means adapting to later shards costs performance on the
  regime the base model was trained for. That is forgetting, now measurable.
- **`grr`** — but only if `gap` clears the noise floor. Where the base already sits near
  the ceiling, GRR divides one noise term by another. The pipeline warns when `|gap|` is
  under 2% of base (the size of the measured same-seed ETTh1 reproducibility gap,
  EXPERIMENTS.md §4). **On SWaT this warning is expected to fire** — that dataset is
  saturation-limited and should be reported as a control, not as evidence about merging.

Reference decision table:

| diag ratio | offdiag ratio | ‖τ‖/‖θ₀‖ | curve shape | reading |
|---|---|---|---|---|
| ≈ 1 | ≈ 1 | small but nonzero | flat | redundancy |
| ≪ 1 | > diag | moderate | peaked, falls after | **interference** |
| ≈ 1 | ≈ 1 | ≈ 0 | flat at every scale | degenerate fine-tuning |

For the curve, `merge_scale_curve.csv`: a peak near 1.0 means the vectors combine at full
magnitude, 0.3–0.7 that they overlap and the full sum overshoots, and a monotone decline
that any contribution is harmful. On the ETTh1 fixture the optimum was 0.6 with a clean U.

---

## 5. Where this goes next

Once the per-run matrices exist, the cross-run question is whether the **cheap geometry
predicts the expensive outcome**: join `geometry_summary.csv` to the diagnostics
`result.json` scalars on run id, and check whether `mean_offdiag_cosine` /
`effective_rank` / `mean_sequential_overlap` correlate with `specialisation`, `grr`, or
`scale_at_min`. If they do, geometry becomes a cheap online diagnostic — which is exactly
what the model-materialisation question in Part B needs. If they do not, that is itself a
result, and it is the concrete evidence for the caveat that parameter-space cosine is not
weight disentanglement.

This is why the sweep trials matter as targets, not just the proven recipes: the
correlation only exists across configurations that differ.

---

## 6. Gotchas

- **The tokenisation guard is new.** `MaeTx` now refuses any pretext task leaving fewer
  than 4 visible patches. It rejects **10 of 15 PSM and 4 of 9 SWaT `MODEL_SWEEP` cross
  trials** — resubmitting those sweeps will fail fast by design. It rejects **zero** of the
  runs in the local `runs/` directory, so diagnosing existing checkpoints is unaffected;
  if a diagnostics job hits it on the cluster, that run was trained with a degenerate
  configuration and its results should be read with that in mind (see EXPERIMENTS.md §2.3).
- **The `standard` row has no val cells, by design.** `StandardPipeline` trains on
  everything but the global tail, so the interior val slices sit inside its training data.
  It is honest only in the test column.
- **α = 0 reproduces the base model bitwise** and α = the source run's own scale
  reproduces `merged/test` exactly — both are reused from the matrix rather than
  recomputed. If either ever disagrees, something is wrong; check before trusting the rest.
- **GPU is not strictly needed** for geometry, and `--runner_device cpu` works for the
  diagnostics too, but AD on CPU is impractically slow.
- **Report what you did not run.** If walltime forced a coarser `MERGE_SCALES` or fewer
  seeds, say so alongside the numbers rather than leaving it implicit.

---

## 7. What to report back

For each dataset: the pivoted val-block matrix, the test column, the summary scalars, the
merge-scale curve shape and its optimum, the cosine-vs-temporal-distance decay, and any
segment whose τ is anomalously small. State plainly which of the three mechanisms the
evidence supports — including "cannot tell" where the gap is inside the noise floor.
Append the findings to `EXPERIMENTS.md` as a new section rather than editing existing ones.
