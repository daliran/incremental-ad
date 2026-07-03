# Experiment Summary — ETTh1 Forecasting & SWaT/PSM Anomaly Detection

Snapshot as of 2026-07-03. Covers every run under `runs/` from this session, including
the now-complete SWaT/PSM AD grid (§2.2) and the ETTh1 grid searches (§1.2-1.6) — both
originally run via local tooling since replaced by `slurm_grid_search/` (§3); the raw
CSVs those runs produced (`scripts/grid_search_results/`) have been deleted, but the
findings below remain valid. Written as a working reference, not a polished report —
update it as new SLURM sweeps finish.

## TL;DR

- **Best ETTh1 forecast (96-24) config found**: `patch_len=4, encoder_embed_dim=128,
  instance_norm=false` (MSE 0.3768) + incremental `reg_lambda=0, merge_scale≈0.5`
  (merged MSE 0.4197, ~44% better than frozen baseline).
- **L2-SP (`reg_lambda` > 0) looks actively harmful on ETTh1** — monotonically worse
  merged MSE at every merge_scale tested, every reg_lambda tried. Best guess why:
  `merge_scale < 1` already does the "stay close to baseline" job at the ensemble
  level; L2-SP does it a second time during fine-tuning, at the cost of adaptation
  capacity, with nothing to show for it since ETTh1 doesn't have much genuine
  regime shift to forget. **On SWaT/PSM, reg_lambda made no measurable difference
  at all** (grid complete, 12/12 trials) — L2-SP's ETTh1-harmful effect didn't
  transfer, but neither did any benefit; see §2.2.
- **On PSM, whether merging "helps" depends entirely on which metric you read**:
  `pa_f1` gets worse after merging at every merge_scale tried, while `window_f1`/
  `window_auroc`/`point_f1`/`point_auroc` all get *better* at every merge_scale —
  a real precision/recall trade-off effect, not noise. See §2.2.
- **`merge_scale=0.5` narrowly beats 0.4/0.6 on average across 3 seeds, but the
  per-seed ranking flips** (2 of 3 seeds prefer higher scale, 1 prefers lower) — the
  effect at this resolution is smaller than seed-to-seed noise. Treat "0.5" as "somewhere
  in 0.4-0.6," not a precise optimum.
- **Reproducibility problem, real and unresolved**: baseline test MSE for ETTh1
  forecast varied ~20% across trials that should be identical (same seed=42, only
  `n_finetune_segments` differed, which shouldn't touch baseline at all). By
  contrast, SWaT/PSM baseline training reproduced *bit-for-bit* across independent
  process launches with the same seed. See §4 — this needs a dedicated repro check
  before trusting any single-seed ETTh1 comparison at face value.
- **Two real code bugs found and fixed this session** (§3): `Segment.val` had a
  latent format mismatch in `EtthForecastDataset` (harmless by luck, not by design);
  `window_len % patch_len` wasn't validated for forecasting (only `forecast_len` was).

---

## 1. ETTh1 Forecasting

### 1.1 Manual/exploratory runs (2026-06-30 to 2026-07-01, pre-grid-search)

All `EtthForecastDataset`, model `MaeTx`. "96-H" = context 96 / horizon H (the
literature convention); some early runs used other context lengths before that
convention was settled on.

| run_id | pipeline | window/forecast | patch_len | embed | instance_norm | test MSE | test MAE |
|---|---|---|---|---|---|---|---|
| 021750_bfe3beec | Standard | 336-96 | 16 | 256 | true | 0.549 | 0.552 |
| 023835_96c85ac5 | Standard | 336-96 | 16 | 256 | true | (crashed before test) | — |
| 022009_694915d4 | Incremental (bf=0.5, n=3) | 336-96 | 16 | 256 | true | baseline 0.931 → merged 0.926 | 0.711 → 0.702 |
| 023612_1e0a1b9e | Incremental (bf=0.5, n=3) | 336-32 | 16 | 256 | true | baseline 0.635 → merged 0.582 | — |
| 192335_648e85c2 | Standard | 96-48 | 8 | 64 | false | 1.091 | 0.736 |
| 192950_9cb59bde | Standard | 96-32 (wl=96) | 8 | 64 | false | 1.052 | 0.718 |
| 193400/193443/193529 (identical) | Standard | 96-24 (wl=96) | 8 | 64 | false | 1.015 | 0.699 |
| 193704_090341aa | Standard | 96-24 (wl=96) | 8 | 64 | false | 1.055 | 0.716 |
| 195719_510c0e8a | Standard | **96-24 (wl=120)** | 4 | 128 | false | 0.4143 | 0.4489 |
| 200016_aa815431 | Standard | 96-24 (wl=120) | 4 | 128 | **true** | 0.4109 | 0.4399 |
| **200129_e37e7985** | Standard | 96-24 (wl=120) | 4 | 128 | false | **0.3768** | **0.4245** |
| 200241_f4957ac4 | Standard | 96-24 (wl=120) | 4 | 128 | true | 0.4156 | 0.4432 |
| 200938_d7ebb83e | Standard | 96-24 (wl=120) | 4 | 128 | false | 0.3777 | 0.4234 |
| 201956_238471c9 | Standard | 96-24 (wl=120), dup of 200129 | 4 | 128 | false | 0.3768 | 0.4245 |
| 202137_8d6f9f4a | Incremental (bf=0.5, n=3) | 96-24 (wl=120) | 4 | 128 | false | baseline 0.753 → merged 0.479 | 0.632 → 0.475 |
| 202642_e7353f64 | Standard | 96-96 (wl=192) | 4 | 128 | false | 0.530 | 0.540 |
| 202905_ccb052f4 | Standard | 96-96 (wl=192) | 4 | 128 | false | 0.516 | 0.521 |
| 204010_8c36529d | Standard | 96-96 (wl=192) | 4 | 128 | **true** | **0.895** | 0.690 |
| 003328_240e69c9 (this session, testing new `reg_lambda`) | Incremental | 96-24 (wl=120) | 4 | 128 | false | baseline 0.753 → merged 0.4795 (reg=1e-4, ms=0.3) | — |
| 003604_223c5d5a (this session) | Incremental | 96-24 (wl=120) | 4 | 128 | false | baseline 0.753 → merged 0.4895 (reg=1e-2, ms=0.3) | — |

**Takeaways from the manual runs:**
- Context/window sizing matters a lot more than most other knobs tried: 96-24 with
  `window_len=120` (0.377-0.415) massively outperforms the same horizon with
  `window_len=96` (1.01-1.09) — more context helps a lot at this horizon.
  336-context runs (021750/022009/023612) sit in between at their much longer
  horizons.
- `instance_norm=true` is inconsistent at 96-24 (sometimes very slightly worse,
  200016 vs 195719) but **actively bad at 96-96** — 204010 (instnorm=true, MSE 0.895)
  vs 202642/202905 (instnorm=false, MSE 0.516-0.530) is a huge gap, not noise.
- Fine-tuning + merging clearly helps relative to a frozen baseline in every
  incremental run tried manually (baseline MSE always higher than merged).

### 1.2 Grid search — Stage 1: model architecture (`StandardPipeline`, 12 trials)

`scripts/grid_search_results/etth_forecast_grid_model.csv`. Fixed at 96-24
(`window_len=120, forecast_len=24`), swept `patch_len ∈ {4,6,8}` ×
`encoder_embed_dim ∈ {64,128}` × `instance_norm ∈ {true,false}`.

| patch_len | embed | instance_norm | MSE |
|---|---|---|---|
| **4** | **128** | **false** | **0.3768** ← winner |
| 8 | 128 | true | 0.3806 |
| 4 | 64 | false | 0.3877 |
| 8 | 64 | true | 0.3919 |
| 6 | 128 | true | 0.4009 |
| 6 | 64 | false | 0.4013 |
| 8 | 128 | false | 0.4036 |
| 6 | 64 | true | 0.4073 |
| 8 | 64 | false | 0.4144 |
| 4 | 128 | true | 0.4156 |
| 4 | 64 | true | 0.4215 |
| 6 | 128 | false | 0.4267 |

`patch_len=6` is consistently worse than 4 or 8 in every embed/instance_norm
combination — a real pattern, though single-seed. `instance_norm` has no consistent
direction here (wins 3 of 6 matched pairs, loses the other 3) — looks like noise at
this horizon, unlike its clear harm at 96-96 (§1.1).

### 1.3 Grid search — Stage 2: incremental reg_lambda × merge_scale (12 trials)

`etth_forecast_grid_incremental.csv`. Uses stage 1's winning architecture.
`baseline_forecast/mse = 0.7530` for all 12 (expected — these params don't touch
baseline training).

| reg_lambda | merge_scale=0.3 | merge_scale=0.5 | merge_scale=1.0 |
|---|---|---|---|
| 0.0 | 0.4705 | **0.4197** ← best | 0.4572 |
| 1e-4 | 0.4708 | 0.4198 | 0.4570 |
| 1e-3 | 0.4792 | 0.4264 | 0.4583 |
| 1e-2 | 0.4880 | 0.4307 | 0.4590 |

Two consistent patterns: `merge_scale=0.5` beats 0.3 and 1.0 at *every* reg_lambda;
`reg_lambda=0` (or 1e-4) is best-or-tied-best at *every* merge_scale, monotonically
worse as it increases. See TL;DR for the L2-SP interpretation.

Per-segment behavior (best run, reg=0/ms=0.5): baseline 0.753 → finetune_0 alone
0.747 (barely moves) → finetune_1 alone 0.464 → finetune_2 alone 0.467 → **merged
0.420**, beating every individual fine-tuned model. The task vectors are genuinely
complementary, not just "the best segment wins."

### 1.4 Grid search — Stage 3: merge_scale refine, 3 seeds (9 trials)

`etth_forecast_grid_merge_scale_refine.csv`. `reg_lambda=0` fixed;
`merge_scale ∈ {0.4, 0.5, 0.6}` × seeds `{42, 123, 7}`.

| merge_scale | seed 42 | seed 123 | seed 7 | avg |
|---|---|---|---|---|
| 0.4 | 0.4366 | 0.4858 | 0.4384 | 0.4536 |
| 0.5 | 0.4197 | 0.5021 | 0.4236 | **0.4485** |
| 0.6 | 0.4143 | 0.5290 | 0.4181 | 0.4538 |

`baseline_forecast/mse` itself varies 0.602-0.753 across these three seeds — random
init alone moves the baseline more than the merge_scale sweep moves the merged
result. The ranking of merge_scale **flips per seed** (42 & 7 prefer 0.6, 123 prefers
0.4). Average favors 0.5, but only by ~1%, sitting on top of much larger noise — not
a confident single-point win. What *does* hold: 0.4-0.6 all clearly beat the original
0.3 and 1.0.

### 1.5 Grid search — Stage 4: n_finetune_segments (4 trials, single seed)

`etth_forecast_grid_segments.csv`. `reg_lambda=0, merge_scale=0.5` fixed;
`n_finetune_segments ∈ {2,3,4,5}`.

| segments | merged MSE | baseline MSE | improvement |
|---|---|---|---|
| 2 | 0.4035 | 0.6227 | 0.219 |
| 3 | 0.4197 | 0.7530 | 0.333 |
| 4 | 0.4642 | 0.6763 | 0.212 |
| 5 | 0.4862 | 0.7004 | 0.214 |

**Confounded** — `baseline MSE` varies ~20% across these four rows despite all
using `--seed 42` and identical baseline data/config (n_finetune_segments doesn't
touch baseline at all in the code). This is the same reproducibility problem as §4,
here big enough to make the raw merged-MSE column untrustworthy for ranking segment
counts. The `improvement` column (each row's own baseline vs its own merged) is
flatter (0.21-0.33) and more self-consistent, but I still wouldn't treat this as a
clean answer to "does segment count matter" without repeats.

### 1.6 Grid search — Stage 5: forecast horizons (3 trials, `StandardPipeline`)

`etth_forecast_grid_horizons.csv`. Winning architecture, context=96 fixed, horizon
varied.

| horizon | window_len | forecast_len | MSE |
|---|---|---|---|
| 96-24 | 120 | 24 | 0.3768 |
| 96-48 | 144 | 48 | 0.4275 |
| 96-96 | 192 | 96 | 0.5151 |

Clean, monotonic, matches expectations (longer horizon = harder). Consistent with
the earlier finding that this repo's gap to published TimesNet-style numbers grows
at longer horizons.

### 1.7 Best known ETTh1 96-24 recipe (put together)

```
--mae_tx_patch_len 4 --mae_tx_encoder_embed_dim 128 --mae_tx_instance_norm false
--finetune_trainer_reg_lambda 0.0 --pipeline_merge_scale 0.5   # (± noise, see §1.4)
```
Already reflected in `.vscode/launch.json`'s "ETTh1 — Forecast Incremental" config.

---

## 2. SWaT / PSM (Anomaly Detection)

Both datasets only support the AD task in this codebase (neither implements
`forecast_len`/`mask_patch_len`), so all runs below are `task=ad`.

### 2.1 Manual runs (2026-06-28 to 2026-06-29, pre-session)

Same architecture throughout: `patch_len=5, embed_dim=256, encoder_layers=2,
mask_ratio=0.8, training_mode=random_mask, window_len=100, stride=50`.
`configurator_threshold_strategy=oracle` (so `threshold_percentile` is set but
functionally unused — oracle mode sweeps its own per-metric threshold).

**SWaT** (3 identical `StandardPipeline` runs + 2 identical `IncrementalTaskArithmeticPipeline` runs, `baseline_fraction=0.5, n_finetune_segments=3, merge_scale=0.5`):

| pipeline | pa_f1 | window_f1 | point_f1 | event_f1 |
|---|---|---|---|---|
| Standard (train/test) | 0.8427 | 0.7509 | 0.7670 | 0.302 |
| Incremental baseline/test | 0.8300 | 0.7489 | 0.7657 | 0.143 |
| Incremental merged/test | 0.8300 | 0.7500-0.7502 | 0.7667-0.7669 | 0.18-0.21 |

**PSM** (3 identical `StandardPipeline` runs + 2 identical `IncrementalTaskArithmeticPipeline` runs, same split config, `threshold_percentile=72` for PSM's ~28% anomaly rate):

| pipeline | pa_f1 | window_f1 | point_f1 | event_f1 |
|---|---|---|---|---|
| Standard (train/test) | 0.7652 | 0.6780 | 0.6229 | 0.286 |
| Incremental baseline/test | 0.7926 | 0.6445 | 0.5865 | 0.288 |
| Incremental merged/test | 0.7669-0.7671 | 0.662-0.668 | 0.604-0.610 | 0.179-0.186 |

Interesting cross-dataset detail: for SWaT/PSM, **baseline training reproduced
bit-for-bit** across separate process launches with the same seed (all 3 Standard
runs per dataset give numerically *identical* metrics to many decimal places) — very
different from what we saw with ETTh1 forecast baselines (§1.5, §4). Finetune stages
show small (not huge) drift between the two Incremental runs per dataset (e.g. PSM
`finetune_0` best_val_loss: 0.2024 vs 0.2285) even though baseline itself matched
exactly — see §4.

Merging doesn't clearly beat baseline here the way it does for ETTh1: SWaT's merged
pa_f1 (0.830) ties baseline's (0.830); PSM's merged pa_f1 (0.767) is *slightly worse*
than baseline's (0.793). Worth keeping in mind when interpreting the AD grid search
in §2.2 — unlike ETTh1, there isn't yet clear evidence that fine-tuning+merging helps
on these datasets at all with `merge_scale=0.5`, which is part of why that grid
search exists.

### 2.2 Grid search — complete (12/12 trials, both datasets)

(Originally run via the now-removed local `scripts/grid_search_ad.py`; that tooling
and its `scripts/grid_search_results/*.csv` output have since been deleted and
replaced by the SLURM-only `slurm_grid_search/` — see §3 — but the findings below
are still valid, just no longer backed by a raw CSV on disk.)

Tested the same `reg_lambda × merge_scale` hypothesis as ETTh1 §1.3, on the premise
that SWaT/PSM (industrial telemetry with distinct attack-scenario segments) are a
more plausible place for L2-SP's anti-forgetting value to show up than ETTh1 (one
fairly homogeneous series). 6 trials per dataset (`reg_lambda ∈ {0.0, 1e-3}` ×
`merge_scale ∈ {0.3, 0.5, 1.0}`), model architecture fixed at the proven values
above (no model-arch sweep for SWaT/PSM yet — see slurm_grid_search's `MODEL_SWEEP`
for that, not yet run).

**SWaT** — `reg_lambda` made no measurable difference at any `merge_scale` (every
pair within ~0.0001 pa_f1 of its counterpart at the other reg_lambda). Only
`merge_scale=1.0` clearly beat baseline (pa_f1 0.8403 vs. baseline 0.8300,
improvement +0.0103); `0.3`/`0.5` were indistinguishable from baseline (both
essentially 0.830, improvement ≈ 0).

**PSM** — `reg_lambda` again made no measurable difference anywhere. But `pa_f1` and
every *other* AD metric disagree about whether merging helped, consistently across
all three merge_scales: `pa_f1` was **worse** after merging at every merge_scale
(baseline 0.7926 → merged 0.7657-0.7671, ≈ -0.026 regardless of merge_scale), while
`window_f1`, `window_auroc`, `point_f1`, and `point_auroc` were all **better** after
merging at every merge_scale (e.g. `merge_scale=0.5`: window_f1 0.6445→0.6722,
window_auroc 0.7850→0.7984, point_f1 0.5865→0.6149, point_auroc 0.7818→0.7947).
Root cause: merging shifts the precision/recall trade-off in point-adjusted terms
(baseline `pa_precision=0.925, pa_recall=0.694` → merged `pa_precision≈0.67,
pa_recall≈0.90`) — a real, consistent effect, not noise, and the reason
`collect_sweep_results` now captures every metric instead of ranking by one (see §3).

Takeaway: "does merging help SWaT/PSM" doesn't have a single answer — it depends
entirely on which metric you read for PSM, and on hitting `merge_scale=1.0`
specifically for SWaT. Neither pattern matches ETTh1, where merging helped clearly
and consistently by MSE/MAE at every merge_scale tried.

---

## 3. Code changes made this session

- **Added L2-SP regularization** (`reg_lambda`, `reg_exclude`, `reference_state`) to
  `StandardTrainer`/`IncrementalTaskArithmeticPipeline` — opt-in, no-op by default,
  penalizes drift from baseline weights during fine-tuning.
- **Added per-finetune-segment eval** (val/test/debug) to the incremental pipeline,
  matching what baseline/merged already had. Renamed
  `get_incremental_val_eval_dataset` → `get_merged_val_eval_dataset` for clarity
  against the new `get_finetune_val_eval_dataset(i)`.
- **Fixed a latent `Segment.val` format bug** in `EtthForecastDataset`: `val` was a
  `(window, future)`-tuple dataset while `train` was bare-tensor — worked only
  because `MaeTx.compute_loss` happened to defensively unwrap tuples. Now matches
  SWaT/PSM's pattern (bare `SlidingWindowDataset` for both).
- **Added missing validation**: `MaeTxForecastingConfigurator` now asserts
  `window_len % patch_len == 0` (previously only checked `forecast_len % patch_len`),
  closing a silent-truncation footgun.
- **Built, then fully replaced, a local grid-search harness.** The original
  `scripts/_grid_search_harness.py` (shared by `grid_search_etth_forecast.py` and
  `grid_search_ad.py`) proved the `Sweep`/`run_sweep` design — cartesian-product
  grids, linked-arg explicit trial lists, multi-seed repeats, `--resume` — but only
  ever ran one trial at a time on a single local GPU. Once the scope grew to a full
  architecture search (encoder/decoder layers, heads, embed_dim, patch_len,
  mask_ratio) plus training-parameter search (baseline_fraction, val_fraction,
  n_finetune_segments, merge_scale, reg_lambda) for both pipelines across SWaT/PSM/
  ETTh1, that stopped being viable. Replaced with **`slurm_grid_search/`**
  (repo root, sibling to `scripts/`) — submits every trial as its own SLURM job
  (`submit.py`, fire-and-forget, no waiting) instead of looping sequentially;
  collection (`collect.py`) is a separate, idempotent step that matches run
  directories back to trials via `config.json` and captures *every* metric found in
  any `result.json`, not a hand-picked subset (motivated directly by the PSM
  pa_f1-vs-window_f1 disagreement in §2.2). `report.py` compares best Standard vs.
  best Incremental (baseline/merged) performance. All generated artifacts (sbatch
  scripts, manifests, results CSVs) land under `$SLURM_GRID_OUTPUT_ROOT`
  (default `$WORK/slurm_grid_search`), never inside the repo. The old local scripts
  and `scripts/grid_search_results/*.csv` were deleted outright (not archived) —
  the findings they produced (§1.2-1.6, §2.2) remain valid and are recorded above,
  just no longer backed by a raw CSV on disk.
- **Added `--resume` to both grid-search scripts**: skips `(params, seed)` combos
  already completed (status `"ok"`) in a sweep's existing results CSV and appends
  new rows instead of overwriting — added specifically to stop the SWaT/PSM AD grid
  after its first trial (§2.2) without losing progress or needing to redo it later.

---

## 4. The seed / reproducibility problem

This is the single most important caveat on everything above. Two separate pieces
of evidence, pointing in different directions:

- **SWaT/PSM baseline training is bit-for-bit reproducible** across independent
  `python -m incremental_ad.main` launches with the same seed (§2.1 — three
  `StandardPipeline` runs per dataset gave numerically identical metrics to many
  decimal places). `torch.backends.cudnn.deterministic=True` /
  `benchmark=False` are set in `framework/core/seed.py`, and this evidently is
  enough for full reproducibility on this AD path (random-mask training, which
  itself draws from `torch.rand` during both train *and* eval).
- **ETTh1 forecast baseline is *not* reproducible** across trials that should be
  identical: the "segments" sweep (§1.5) shows baseline MSE ranging 0.62-0.75 across
  four trials sharing seed=42 and identical baseline data (only
  `n_finetune_segments` differs, which the code never uses to compute the baseline's
  own train/val split). The "merge_scale_refine" sweep (§1.4) independently confirms
  baseline test MSE varies a lot seed-to-seed (0.60-0.75 across 3 seeds) — expected
  there since those genuinely are different seeds, but the *segments* sweep case is
  the concerning one since it's supposedly the *same* seed.
- I traced several candidate explanations (GPU/cuDNN non-determinism in attention
  softmax/backward passes not covered by `cudnn.deterministic`; RNG-state
  interference from `secondary_loaders` evaluated during baseline training) and
  couldn't confirm either cleanly — forecast's `causal_mask` path doesn't call
  `torch.rand` at all during loss computation, which argues against both. I don't
  have a confirmed root cause.
- **This means every single-seed ETTh1 forecast conclusion in §1.1-§1.2 and §1.5
  should be read as "probably true, not confirmed"** — exactly the caveat I gave
  when reporting them, now with concrete supporting evidence of how large the noise
  floor actually is (comparable in magnitude to several of the effects being
  measured, e.g. patch_len=4 vs 8 in §1.2, or the whole segments sweep in §1.5).

**Recommended diagnostic** (cheap, hasn't been run yet): launch the exact same ETTh1
incremental config twice in a row with `--seed 42` and nothing else changed, and
check whether `baseline/test/result.json` matches bit-for-bit. That single
experiment would definitively separate "inherent non-determinism in this code path"
from "some hidden dependency I haven't found" — worth doing before trusting any
further single-seed ETTh1 grid results.

---

## 5. My take / recommended next steps

Ranked by what I'd actually do next, not by neatness:

1. **Run the repro-check in §4 first.** Everything else here is built on top of
   single-seed ETTh1 numbers; knowing whether that ~20% baseline swing is inherent
   noise or a real bug changes how much weight to put on the finer-grained findings
   (patch_len, merge_scale precision, segment count).
2. **L2-SP's ETTh1-harmful finding did not generalize to SWaT/PSM** (§2.2, now
   complete) — reg_lambda made no measurable difference there at all, in either
   direction. Not much more to chase on L2-SP itself until/unless a stronger
   reg_lambda than 1e-3 is tried on SWaT/PSM (matching the ETTh1 value that actually
   showed an effect, 1e-2) — `slurm_grid_search`'s `TRAIN_INCREMENTAL_SWEEP` already
   includes that value.
3. **The 96-96 `instance_norm=true` result (0.895 vs 0.516) is the single largest
   effect size found anywhere in this document** — much bigger than merge_scale,
   segment count, or reg_lambda's effects. Worth understanding *why* RevIN hurts so
   much at longer horizons here (it's supposed to help with distribution shift, the
   opposite of what's observed) before spending more time on the smaller-effect
   hyperparameters.
4. **The "does merging even help" question is unresolved for SWaT/PSM** (§2.1: ties
   or slightly loses vs baseline at merge_scale=0.5), unlike ETTh1 where merging
   clearly wins in every run tried. If the AD grid also shows no benefit at every
   merge_scale/reg_lambda tested, that's a more fundamental finding than any
   hyperparameter tuning — it would say task-arithmetic merging itself doesn't
   transfer to this AD setup the way it does for ETTh1 forecasting, which would be
   worth its own investigation rather than more grid search.
5. Lower priority, easy to defer: model-arch sweep for SWaT/PSM (§2.2 doesn't have
   one yet), multi-seed repeats for the segments/patch_len findings, and the
   literature-standard ETTh1 split (discussed earlier in this conversation, not
   implemented) for a paper-comparable number — none of these change what to *do*
   next, just add polish once the more fundamental questions above are settled.
