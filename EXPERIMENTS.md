# Experiment Summary — ETTh1 Forecasting & SWaT/PSM Anomaly Detection

**This file is the source of truth for every experiment in this project — not the
artifact page, not chat history, not $WORK's CSVs.** §1-2 are the narrative (what was
tried, what worked, what didn't); §6 is the complete per-trial data backing that
narrative — every value, every parameter combination, nothing summarized away. §7 has
the interactive artifact URL. Update this file (prose *and* §6's tables) every time a
new sweep is collected — don't let results live only on the cluster or in memory.

Snapshot as of 2026-07-05. Covers every run under `runs/` from this session, including
the now-complete SWaT/PSM AD grid (§2.2), the model-architecture grids (§1.8, §2.3),
the training-parameter grids (§1.9, §2.4) — all four run via `slurm_grid_search/` — and
the ETTh1 grid searches (§1.2-1.6) — the latter originally run via local tooling since
replaced by `slurm_grid_search/` (§3); the raw CSVs those local runs produced
(`scripts/grid_search_results/`) have been deleted, but the findings below remain valid.

All four SLURM stages (`model`, `train_standard`, `train_incremental` × SWaT/PSM/ETTh1)
are now complete. Architecture decisions (§2.3) are already reflected in
`scripts/sbatch_mae_tx_{swat,psm}_ad_*.sh` and `.vscode/launch.json` (SWaT
`decoder_heads=4`; PSM `patch_len=5, encoder_embed_dim=128` — revised from the initial
`patch_len=25` pick after the head-to-head comparison in §2.4; ETTh1 unchanged).

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
- **SWaT model-arch grid (18/18, §2.3): architecture barely matters for window/point/
  AUROC metrics** (spreads <1%) **except `event_f1`, where it matters a lot** (spread
  0.144-0.372). `mae_tx_decoder_heads=4` looks like a free win: ties-or-beats the base
  recipe on every other metric while lifting `event_f1` from 0.291 to 0.372 (+28%
  relative).
- **PSM model-arch grid (24/24 complete, §2.3): `patch_len` trades off window-level
  vs. event-level detection, monotonically and in opposite directions** as it grows
  from 5→10→20→25 (window_auroc 0.795→0.817), peaking at **`patch_len=25`** then
  slightly reversing at 50 — while `event_f1` falls the whole way (0.275→0.139 at
  patch_len=50). No single winner — `patch_len=25/mask_ratio=0.8` is best on
  window/point/AUROC, `encoder_embed_dim=128` (at the base `patch_len=5`) is best on
  `event_f1` (0.396) *without* costing anything on the other four metrics, making it
  the safer default of the two. `pa_f1` tracks neither trend, peaking instead at
  `patch_len=50` — a third, independent pattern. `patch_len=25` was carried through
  training-param sweeps first (§2.4), but once run end-to-end through
  `train_incremental` it lost on nearly every metric to `patch_len=5` (see §2.4's
  confirmation batch) — **final decision: `patch_len=5`**, now the "proven recipe"
  in scripts/launch.json/ARCH_ARGS.
- **Training-parameter grids, all three datasets complete (§1.9, §2.4)**: SWaT/PSM
  both get a real `pa_f1` win from `dataset_val_fraction=0.20` (SWaT +0.9%, PSM +5.4%
  relative vs. the base recipe) — though `window_auroc` prefers `val_fraction=0.15`
  on PSM, the same metric-disagreement pattern as architecture. SWaT's incremental
  sweep reconfirms `merge_scale=1.0` as best-in-cross; three one-at-a-time trials edge
  it out further (best: `n_finetune_segments=2`, pa_f1 0.8438) — they inherit the same
  `merge_scale=1.0` default (not a different "0.5" as an earlier pass at this doc
  claimed), so this is `merge_scale=1.0` plus a slightly better split, not a mystery.
  PSM's merge_scale-vs-window-metrics relationship **flipped direction** now that the
  architecture is `patch_len=25` instead of 5: low `merge_scale=0.3` now helps
  window_auroc/event_f1 and high `merge_scale=1.0` now hurts them — the opposite of
  §2.2's old-architecture finding. ETTh1's training-param sweep reconfirms
  `merge_scale=0.5, reg_lambda=0` (MSE 0.4236, consistent with §1.3's 0.4197), but
  found one dramatic new failure mode: `dataset_baseline_fraction=0.3` makes merging
  **actively catastrophic** (MSE 1.0076→1.9766, nearly doubling) — the only case in
  this entire document where merging made things clearly worse rather than better or
  neutral.
- **ETTh1 model-arch grid, extended and seed-checked (§1.8): the apparent
  `decoder_embed_dim=128` winner does NOT hold up across seeds** — ranking flips at
  seed 123 (128 scores 15% *worse* there), and the 3-seed average slightly favors the
  original `decoder_embed_dim=64` recipe (0.3924 vs. 0.3970). **Recommendation:
  keep the original §1.2/§1.7 recipe as-is**, no architecture change. Bonus: an exact
  same-seed repeat of the base recipe gave a different result (0.3829 vs. 0.3911,
  ~2.1%), now directly confirming the previously-unconfirmed reproducibility problem
  in §4 with hard evidence. `patch_len=4` and `encoder_embed_dim ∈ {64,128}` remain
  best (256 and patch_len 6/8 both consistently worse, confirming §1.2).

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

### 1.8 SLURM grid search — extended model architecture (complete, 22/24 usable rows)

`slurm_grid_search/sweeps/etth_forecast.py::MODEL_SWEEP`,
`etth_forecast_model_results.csv`. Goes beyond stage 1 (§1.2, which only varied
patch_len/embed_dim/instance_norm): cross of `patch_len ∈ {4,6,8}` ×
`encoder_embed_dim ∈ {64,128,256}` (9 trials) plus one-at-a-time
`encoder_layers ∈ {2,4}`, `encoder_heads=2`, `decoder_layers ∈ {1,3}`,
`decoder_heads=2`, `decoder_embed_dim ∈ {32,128}` (8 trials), plus a follow-up 3-seed
repro check on the apparent `decoder_embed_dim` winner (5 more trials) — 24 submitted,
22 complete (2 stale `TIMEOUT` rows superseded by their retries, see below).

| run | seed | patch_len | encoder_embed_dim | decoder_embed_dim | other changes | test MSE |
|---|---|---|---|---|---|---|
| 58391 | 42 | 4 | 128 | 128 | — | 0.3663 |
| 58384 | 42 | 4 | 128 | 64 | encoder_layers=2 | 0.3738 |
| 58376 (base recipe) | 42 | 4 | 128 | 64 | — | 0.3829 |
| 58838 | 123 | 4 | 128 | 64 | — | 0.3831 |
| 58841 | 7 | 4 | 128 | 128 | — | 0.3844 |
| 58837 | 42 | 4 | 128 | 64 | — (exact repeat of 58376) | 0.3911 |
| 58839 | 7 | 4 | 128 | 64 | — | 0.4071 |
| 58840 | 123 | 4 | 128 | 128 | — | 0.4402 (worst of this group) |

Same two monotonic patterns as before hold up: `patch_len=4` still beats 6/8 at every
`encoder_embed_dim`, and `encoder_embed_dim=256` is still consistently worse than
64/128. **The `decoder_embed_dim=128` "winner" from the first pass does not survive
multi-seed testing.** Isolating `decoder_embed_dim ∈ {64,128}` across seeds 42/123/7
(everything else fixed at the base recipe):

| decoder_embed_dim | seed 42 | seed 123 | seed 7 | avg |
|---|---|---|---|---|
| 64 (base recipe) | 0.3829 / 0.3911 | 0.3831 | 0.4071 | **0.3924** |
| 128 | 0.3663 | 0.4402 | 0.3844 | 0.3970 |

The ranking **flips at seed 123** (128 is 15% *worse* than 64 there), and on average
across seeds `decoder_embed_dim=64` (the simpler, original recipe) is marginally
*better*, not worse. Same pattern as the `merge_scale` seed-flip in §1.4 — the
single-seed 4.3% "improvement" was noise, not a real architecture effect.
**Recommendation: keep the original §1.2/§1.7 recipe (`decoder_embed_dim=64`)** rather
than adopting 128; there's no evidence it's actually better.

**Bonus finding, confirms §4**: running the *exact same config and seed* (42) twice —
run 58376 (0.3829) vs. its verbatim repeat, run 58837 (0.3911) — gives measurably
different results (~2.1% relative). This directly confirms §4's previously
"suspected but unconfirmed" ETTh1 non-reproducibility with hard evidence: even a
literal same-seed rerun isn't bit-for-bit here, unlike SWaT/PSM. See updated §4.

**The 2 retried trials** (`decoder_layers=3` → run 58835, MSE 0.3858;
`decoder_embed_dim=32` → run 58836, MSE 0.4270) both landed after retrying past a
`TIMEOUT` — the original attempts (jobs 58388/58390, still showing as `incomplete`
rows in the CSV) both landed on compute node `rezzonico` and died in an identical
`OSError: [Errno 28] No space left on device` loop while `DataLoader` workers
(`--loader_num_workers 4`) tried to create their multiprocessing temp dir under
`$TMPDIR=/tmp` — that node's local `/tmp` was full, and PyTorch's
`multiprocessing.resource_sharer` retries `mkdtemp` in a loop instead of failing fast,
so the job spun until the wall-clock limit killed it; not a real compute-time or
architecture issue. Neither retried config beats 0.3663/0.3829 — no change to the
recipe recommendation above.

**Framework fix made in passing**: `collect_sweep_results` (harness.py) used to derive
which `mae_tx_*` columns to show from the sweep's *current* `trials`/`grid`, so trimming
a sweep file down to just its remaining/follow-up trials (done here and for PSM, §2.3)
silently dropped architecture columns for already-collected historical rows on the next
`collect.py` run — the values were still in each run's `config.json`, just no longer
surfaced. Fixed to scan every `mae_tx_*`-prefixed arg actually recorded per run instead,
independent of the sweep object's current state — same "capture everything found, don't
hand-pick" principle CLAUDE.md already documents for metrics (§3).

### 1.9 SLURM grid search — training parameters (`train_standard` 7/7, `train_incremental` 15/16)

`slurm_grid_search/sweeps/etth_forecast.py::TRAIN_STANDARD_SWEEP` /
`TRAIN_INCREMENTAL_SWEEP`, using the unchanged §1.2/§1.7/§1.8 architecture recipe
(`decoder_embed_dim=64` — the apparent 128 winner didn't survive §1.8's seed check).

**`train_standard`**: `dataset_val_fraction ∈ {0.05,0.1,0.15}` × `trainer_learning_rate
∈ {1e-3,3e-4}` + `trainer_weight_decay=1e-3` one-at-a-time. Best: `val_fraction=0.1,
learning_rate=3e-4` (MSE 0.3838) vs. the base recipe's own row (`val_fraction=0.1,
learning_rate=1e-3`, MSE 0.3911) — only a ~1.9% improvement, i.e. **smaller than the
~2.1% same-seed noise floor confirmed in §1.8/§4**, so treat this as "plausible, not
confirmed" rather than a real win. `val_fraction=0.15` is consistently worst regardless
of learning rate (0.43-0.45) — that part of the trend looks real, not noise-sized.

**`train_incremental`**: `pipeline_merge_scale ∈ {0.3,0.5,1.0}` ×
`finetune_trainer_reg_lambda ∈ {0.0,1e-3,1e-2}` cross + `dataset_baseline_fraction ∈
{0.3,0.7}` / `dataset_val_fraction ∈ {0.1,0.15}` / `dataset_n_finetune_segments ∈
{2,5}` one-at-a-time (14/15 valid — the `val_fraction=0.05` row is the degenerate-split
failure from earlier, excluded). Reconfirms §1.3's `merge_scale=0.5, reg_lambda=0` as
best-in-cross (MSE 0.4236 here vs. 0.4197 there — consistent). `reg_lambda` mostly
confirms §1.3 (higher is worse) **except one exception at `merge_scale=1.0`,
`reg_lambda=1e-3` (MSE 0.4486, beating `reg_lambda=0`'s 0.4684)** — but that row's own
*baseline* MSE (0.6873) differs from its cross-mates (0.6132) despite nominally
identical config, i.e. it's the same unresolved ETTh1 non-reproducibility problem
(§4) confounding the comparison, not a clean reversal of the reg_lambda trend.

**Notable one-at-a-time finding**: `dataset_baseline_fraction=0.3` (baseline gets only
30% of the data, 3 finetune segments cover the rest) makes merging **actively
catastrophic** — MSE goes from baseline 1.0076 to merged 1.9766, nearly *doubling* —
the only row in this entire session where merging made things dramatically worse
rather than better or flat. Plausible mechanism: with too little baseline data, the
baseline task vector itself is unreliable, so merging in that direction actively hurts
rather than helps. Worth a dedicated follow-up if low-baseline-fraction incremental
setups are ever actually used, since every other finding in this document assumes
merging helps or is neutral.

**Follow-up**: paired `dataset_val_fraction=0.05` with `dataset_n_finetune_segments=2`
(instead of the default 3) as a one-off check — this clears the degenerate-split guard
that killed the plain `val_fraction=0.05` trial, and gives a normal result (baseline
MSE 0.6301 → merged 0.4562), in line with the rest of the `train_incremental` grid.
Confirms the earlier failure was specifically about segment count interacting with
`val_fraction`, not `val_fraction=0.05` being unworkable on its own.

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
above (the model-arch sweep for SWaT/PSM has since been run — see §2.3).

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

### 2.3 SLURM grid search — model architecture (SWaT 18/18 complete; PSM 24/24 complete)

> **Caveat added later — most of the `patch_len` axis in this section is degenerate.**
> `MaeTx` now refuses to build a pretext task leaving fewer than 4 visible patches
> (`_assert_pretext_non_degenerate`). At `dataset_window_len=100` that rejects **10 of
> PSM's 15 cross trials and 4 of SWaT's 9**. The one that matters: PSM's reported
> window/point-metric winner, `patch_len=25 / mask_ratio=0.8`, tokenises to **4 patches
> with 1 visible** — the model is asked to reconstruct three quarters of the window from a
> single patch. Its good ranking scores are consistent with a model collapsed toward the
> global prior, since anomalies deviate from that prior more than normal data does; what
> such a model has *not* learned is anything shard-specific, which is why its task vectors
> carried so little (see §2.4, where `patch_len=25` lost end-to-end to `patch_len=5`).
> **No recipe changes** — `patch_len=5` was chosen on `event_f1` and merge behaviour
> anyway — but the `patch_len` comparison here should not be read as a clean architecture
> result. To vary `patch_len` honestly, hold the token count fixed by scaling the window
> with it (100/5, 200/10, 500/25 all give 20 tokens). Note also that the chosen recipe
> sits at *exactly* 4 visible patches, i.e. with no margin.

`slurm_grid_search/sweeps/{swat,psm}.py::MODEL_SWEEP`,
`{swat,psm}_model_results.csv`. Same shape both datasets: cross of
`patch_len ∈ {5,10,20}` × `mask_ratio ∈ {0.5,0.65,0.8}` (9 trials) plus one-at-a-time
`encoder_layers ∈ {3,4}`, `encoder_heads=4`, `encoder_embed_dim ∈ {128,512}`,
`decoder_layers=2`, `decoder_heads=4`, `decoder_embed_dim ∈ {64,256}` (9 trials)
around the proven recipe (`patch_len=5, mask_ratio=0.8, encoder_layers=2,
encoder_heads=2, encoder_embed_dim=256, decoder_layers=1, decoder_heads=2,
decoder_embed_dim=128`).

**SWaT (18/18 complete)** — architecture barely moves window/point-level metrics:

| metric | min | max | spread |
|---|---|---|---|
| window_auroc | 0.804 | 0.813 | 0.009 |
| window_f1 | 0.747 | 0.752 | 0.005 |
| point_f1 | 0.764 | 0.768 | 0.004 |
| pa_f1 | 0.826 | 0.850 | 0.024 |
| event_f1 | 0.144 | 0.372 | 0.228 |

`event_f1` is the one axis where architecture has real signal, and
`decoder_heads=4` (run 58266) is essentially a free win: window_auroc 0.8089 (base
0.8088), window_f1 0.7514 (base 0.7509), pa_f1 0.8427 (exactly tied with base),
`event_f1` 0.372 vs. the base recipe's 0.291 (+28% relative, best of all 18 trials) —
no cost anywhere else.

**PSM (24/24 complete)** — much more architecture-sensitive than SWaT, and
`patch_len` drives a genuine trade-off (averaged over the mask_ratio cross, rest at
base recipe; extended with the `{25,50}` follow-up batch):

| patch_len | window_auroc | window_f1 | point_f1 | event_f1 |
|---|---|---|---|---|
| 5 | 0.795 | 0.664 | 0.606 | **0.275** |
| 10 | 0.805 | 0.678 | 0.624 | 0.182 |
| 20 | 0.812 | 0.697 | 0.648 | 0.194 |
| 25 | **0.817** | **0.712** | **0.660** | 0.194 |
| 50 | 0.813 | 0.704 | 0.654 | 0.139 (worst) |

The trend **peaks at `patch_len=25`, then slightly reverses at 50** on
window_auroc/window_f1/point_f1 — so it does saturate rather than climbing forever, and
`event_f1` keeps getting worse the whole way, hitting its overall floor at
`patch_len=50`. Best single trial overall: run 58831, `patch_len=25, mask_ratio=0.8`
(window_auroc=0.8205, the best of all 24 trials); `pa_f1` behaves differently again,
peaking instead at `patch_len=50` (0.811, its own overall best) — a third pattern that
doesn't track window/point/AUROC or event_f1, reinforcing §2.2's point that no single
metric tells the whole story here.

No single winner, same as before: `patch_len=25, mask_ratio=0.8` (run 58831) is best
on window/point/AUROC; `encoder_embed_dim=128` (run 58337, at base `patch_len=5,
mask_ratio=0.8`) is still the best `event_f1` across all 24 trials (0.396) while
staying at-or-above the base recipe on every other metric — a strict win-or-tie, and
still the safer default absent a specific reason to prioritize window-level metrics.
`patch_len=25` (window-optimized) was carried forward *initially* into
`train_standard`/`train_incremental` below to test it end-to-end. **That decision was
later reversed** (§2.4's confirmation batch) once the full incremental pipeline showed
`patch_len=5` winning on nearly every metric except a wash on `pa_f1` — final recipe is
`patch_len=5, encoder_embed_dim=128`, reflected in `scripts/sbatch_mae_tx_psm_ad_*.sh`
and `.vscode/launch.json`'s "PSM" configs. SWaT's `decoder_heads=4` *was* a clean free
win and needed no reversal (same files updated for SWaT too).

### 2.4 SLURM grid search — training parameters (SWaT 7/7 + 15/15; PSM 14/14 + 30/30, two architectures)

`slurm_grid_search/sweeps/{swat,psm}.py::TRAIN_STANDARD_SWEEP` / `TRAIN_INCREMENTAL_SWEEP`,
using the §2.3 architecture decisions (SWaT `decoder_heads=4`; PSM `patch_len=25`
initially — see the confirmation batch below for why it was revised to `patch_len=5`).

**`train_standard`** (`dataset_val_fraction ∈ {0.10,0.15,0.20}` × `trainer_learning_rate
∈ {1e-4,3e-4}` + `trainer_weight_decay=1e-3` one-at-a-time, ranked by `pa_f1`):

- **SWaT**: best `val_fraction=0.20, learning_rate=1e-4` (pa_f1=0.8503) vs. the base
  recipe's own row (`val_fraction=0.15, learning_rate=1e-4`, pa_f1=0.8427) — a real,
  if modest, +0.9% relative improvement. `val_fraction=0.20` appears twice in the top 4.
- **PSM**: same combo wins more decisively — `val_fraction=0.20, learning_rate=1e-4`
  (pa_f1=0.8141) clearly beats the base recipe's `val_fraction=0.15` row (0.7725,
  +5.4% relative) and every other trial (next-best 0.7729). Note `window_auroc`
  disagrees again (peaks at `val_fraction=0.15` combos, 0.8205/0.8197) — the same
  "pa_f1 vs. window-level metrics" split from §2.2/§2.3, now showing up on a training
  hyperparameter too, not just architecture.

**`train_incremental`** (`pipeline_merge_scale ∈ {0.3,0.5,1.0}` ×
`finetune_trainer_reg_lambda ∈ {0.0,1e-3,1e-2}` cross + `dataset_baseline_fraction ∈
{0.3,0.7}` / `dataset_val_fraction ∈ {0.1,0.2}` / `dataset_n_finetune_segments ∈
{2,5}` one-at-a-time):

- **SWaT** (15/15 valid): `merge_scale=1.0, reg_lambda=0.0` best-in-cross (pa_f1
  0.8373→0.8408), confirming §2.2's SWaT finding that `merge_scale=1.0` is the one
  value that clearly beats baseline there. `reg_lambda` again makes only a tiny,
  slightly-negative difference (0.8408→0.8407→0.8402 as it increases) — consistent
  with §2.2's "no measurable difference" call, just barely visible here. Correction
  to an earlier read of this data: the one-at-a-time trials (`dataset_baseline_fraction`,
  `dataset_val_fraction`, `dataset_n_finetune_segments`) don't override
  `pipeline_merge_scale`, so they run at its argparse *default* of **1.0** — the same
  value the cross already found best, not a different "0.5 default" as previously
  stated here. Three of them (`val_fraction=0.10` → pa_f1 0.8436,
  `n_finetune_segments=2` → 0.8438 the best of all 15, `val_fraction=0.20` → 0.8434)
  edge out the cross's own best (0.8408) despite worse baselines — a small further gain
  from combining `merge_scale=1.0` with a slightly different split, not a mystery.
- **PSM** (15/15 valid): `pa_f1` is worse after merging at every merge_scale (baseline
  0.798 → merged 0.771-0.776), same direction as §2.2 but the *window-level* pattern
  has flipped from §2.2 now that the architecture is `patch_len=25` instead of 5:
  `window_auroc` and `event_f1` are now **best at low `merge_scale=0.3`** (auroc
  0.7904→0.7853, event_f1 0.1652→0.2096) and **worst at `merge_scale=1.0`** (auroc
  →0.7689, event_f1 →0.1630, actually below baseline) — the opposite ranking from
  §2.2's old-architecture result, where every merge_scale improved window-level
  metrics. The architecture change appears to have changed which merge_scale is
  "good," not just the absolute numbers — another reason architecture and
  training-hyperparameter decisions aren't fully separable here.

**Confirmation batch — PSM re-run at `patch_len=5` (event-optimized arch, same
`train_standard`/`train_incremental` grids, 22 more trials)**: this makes the
`patch_len` comparison a controlled one — same training-hyperparameter grid, only
`patch_len` differs (25 vs. 5), settling the "why did merging used to help PSM"
question from a genuine architecture/merge_scale interaction, not a fluke:

| | `patch_len=25` | `patch_len=5` |
|---|---|---|
| `train_standard` best pa_f1 | **0.8141** | 0.8035 |
| `train_standard` best window_auroc | **0.8185** | 0.8033 |
| `train_standard` best event_f1 | 0.1891 | **0.2582** |
| `train_incremental` avg window_auroc: baseline→merged | 0.7891→0.7722 (**worse**) | 0.7748→**0.7950** (better) |
| `train_incremental` avg event_f1: baseline→merged | 0.1606→0.1741 | 0.2272→**0.2459** |
| `train_incremental` avg pa_f1: baseline→merged | 0.8015→0.7745 | 0.8029→0.7706 |

`patch_len=5`'s incremental `window_auroc` improves after merging (matching §2.2's
original finding, now reproduced with the current codebase/session); `patch_len=25`'s
gets worse (matching §2.4's finding above). Same training-hyperparameter grid, same
codebase, only `patch_len` differs — about as clean a confirmation as this kind of
comparison gets. `pa_f1` is a wash either way (merging hurts it regardless of
`patch_len`). **Working hypothesis for the mechanism** (not directly tested): at
`patch_len=25` PSM only has 4 total patches per window, and at `mask_ratio=0.8` that
leaves just 1 visible patch during MAE training (`n_patches=100//25=4`,
`n_masked=int(4×0.8)=3`) — a far coarser, lower-context reconstruction task than
`patch_len=5`'s 20 patches (4 visible), which may make the coarse-patch model more
fragile to the weight-space perturbation that task-arithmetic merging performs.
Untested: intermediate `patch_len` values against the same `merge_scale` cross, which
would show whether this is a gradual effect or a sharp threshold.

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
- **Fixed a `collect_sweep_results` column-selection bug** (`slurm_grid_search/
  harness.py`): it used to derive which `mae_tx_*` columns to display from the
  sweep's *current* `trials`/`grid`, so trimming a sweep file down to just its
  remaining/follow-up trials (done for both `psm.py` and `etth_forecast.py` this
  session, §1.8/§2.3) silently dropped architecture columns for already-collected
  historical rows on the next `collect.py` run — values were still in each run's
  `config.json`, just no longer surfaced into the CSV. Fixed to scan every
  `mae_tx_*`-prefixed arg actually recorded per run instead, independent of the sweep
  object's current state — matches the "capture everything found, don't hand-pick"
  principle already applied to metrics below. `_all_trial_keys` (now unused) removed.
  **Follow-up fix**: restricting to `mae_tx_*` was itself too narrow — it silently
  dropped the `dataset_*`/`trainer_*`/`pipeline_*`/`finetune_trainer_*` columns that
  `train_standard`/`train_incremental` sweep (only `MODEL_SWEEP` varies `mae_tx_*`).
  Fixed again to capture every recorded arg (minus `seed`, which already has its own
  column), not just `mae_tx_*` ones.
- **Data-quality gotcha found while analyzing §2.4/§1.9 results**: `collect_sweep_results`
  marks a row `"complete"` if *any* `result.json` is found anywhere under the run dir —
  for `IncrementalTaskArithmeticPipeline` runs that crash partway through (e.g. the
  wandb `ENOSPC` failures below), this means a row can show `status=complete` with a
  fully populated `baseline_test_*` but **empty `merged_test_*`** columns, since
  baseline/finetune stages already wrote their own `result.json` before the crash.
  Two SWaT `train_incremental` rows (58944, 58946) are exactly this — stale partial
  duplicates of their successful retries (59055, 59056). Anyone re-analyzing these
  CSVs should filter on the target metric being non-empty, not just `status=="complete"`.
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

**Recommended diagnostic — now done, confirms the problem is real** (§1.8): submitted
the exact same ETTh1 `StandardPipeline` config (base recipe, `--seed 42`) a second
time, verbatim. Run 58376 (original) gave test MSE 0.3829; its exact repeat, run
58837, gave 0.3911 — a ~2.1% difference with nothing at all changed between the two
submissions. This settles the question this diagnostic was designed to answer:
**it's inherent non-determinism in this code path, not a hidden config/seed-handling
bug** (a hidden dependency would more plausibly have produced identical results
whenever the actual inputs happened to match, which they did here). Root cause within
that non-determinism (cuDNN attention kernels, RNG interference, or something else)
is still not pinned down, but "is this reproducible at all" is now answered: no. A
3-seed check of the `decoder_embed_dim=64` vs. `128` architectures (§1.8) reinforces
this at a coarser grain — `128` wins at seeds 42 and 7 but loses badly at seed 123,
confirming the noise floor is large enough to flip real-looking single-seed
conclusions.

---

## 5. My take / recommended next steps

Ranked by what I'd actually do next, not by neatness:

1. **Done: the repro-check in §4 confirms the noise is inherent, not a bug** — an
   exact same-seed repeat gave a different result, and the ranking between
   `decoder_embed_dim=64` and `128` flips across seeds. Treat every single-seed ETTh1
   architecture finding smaller than ~5% MSE as unconfirmed; the base recipe
   (`decoder_embed_dim=64`, i.e. no change from §1.2/§1.7) is the one to carry
   forward, since the apparent 128 improvement didn't survive multi-seed testing.
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
5. Lower priority, easy to defer: multi-seed repeats for the segments/patch_len
   findings, and the literature-standard ETTh1 split (discussed earlier in this
   conversation, not implemented) for a paper-comparable number — neither changes
   what to *do* next, just adds polish once the more fundamental questions above are
   settled.
6. **Done: PSM's `patch_len` decision (§2.3) was made — `patch_len=25`** carried
   through `train_standard`/`train_incremental` (§2.4) and into
   `scripts/sbatch_mae_tx_psm_ad_*.sh`/`launch.json`. Worth remembering this was a
   deliberate window-optimized choice, not a free win like SWaT's `decoder_heads=4` —
   `encoder_embed_dim=128` at the base `patch_len=5` remains the better pick if
   event-level detection ever becomes the priority.
7. **Correction, not a mystery (§2.4)**: SWaT's one-at-a-time trials
   (`val_fraction=0.10/0.20`, `n_finetune_segments=2`) don't override
   `pipeline_merge_scale`, so they inherit its argparse default of **1.0** — the same
   value the cross already found best, not the "0.5 default" earlier drafts of this
   doc claimed. Their small edge over the cross's own best (0.8438 vs. 0.8408) is just
   `merge_scale=1.0` combined with a slightly different split, not an unexplained
   effect — no follow-up sweep needed here.
8. **New, higher-priority finding (§2.4/§1.9)**: `dataset_baseline_fraction=0.3` made
   ETTh1 incremental merging catastrophically worse (MSE nearly doubled) — the only
   case anywhere in this document where merging clearly hurt rather than helped or
   stayed neutral. Caveat: this one-at-a-time trial also inherited the `merge_scale=1.0`
   default rather than ETTh1's own cross-optimal `merge_scale=0.5` (see point 7) — the
   cross itself shows `merge_scale=1.0` is only mildly worse than 0.5 for ETTh1
   (MSE 0.4684 vs. 0.4236, nowhere near a doubling), so the catastrophic effect is very
   unlikely to be a `merge_scale` artifact, but re-testing `baseline_fraction=0.3` at
   `merge_scale=0.5` would close that gap. Before trusting task-arithmetic merging in
   any low-baseline-data regime, this deserves its own targeted check (repeat at a
   couple of seeds, and try the same low-baseline-fraction setting on SWaT/PSM) rather
   than being left as a single data point.
9. **PSM's merge_scale ranking flipped when the architecture changed** (§2.4): at
   `patch_len=5`, window-level metrics improve at *every* `merge_scale` tested (§2.2),
   and improve *more* as `merge_scale` rises toward 1.0 (§2.4's repeat confirms this).
   At `patch_len=25` it's the opposite — window-level metrics get *worse* as
   `merge_scale` rises, with low `merge_scale=0.3` now winning. This means architecture
   and incremental-pipeline hyperparameters aren't cleanly separable for PSM; if
   `patch_len` is ever revisited, the training-param sweep should be re-run rather than
   assumed to transfer.
---

## 6. Appendix — complete per-trial results

Every trial submitted this session, in full — the summarized tables and prose in §1-2
pick out winners and patterns, but every row behind those calls is here. `—` means that
metric wasn't available (the run never reached that stage, e.g. `incomplete`/timed-out
rows, or the `merged_test_*` columns for a run that crashed between baseline and merge —
see the data-quality note in §3). Columns match the metrics used throughout this doc;
architecture columns are named without the `mae_tx_` prefix for width. Retried trials
(after a `TIMEOUT`/`FAILED`) keep their original row *and* their retry's new row side by
side, so the failure history isn't lost.

Regenerating this from scratch (e.g. after more sweeps run): `collect.py --dataset
<swat|psm|etth_forecast> --stage <model|train_standard|train_incremental>` against
$RUNS_ROOT — idempotent, safe to re-run any time (see CLAUDE.md). The underlying CSVs
live on `$WORK/slurm_grid_search` (cluster scratch), not in this repo, by the project's
own convention (CLAUDE.md: "everything generated... never inside the repo") — this
appendix is the durable copy.

### A.1 SWaT — model architecture (18 trials)

| run_id | patch_len | mask_ratio | encoder_layers | encoder_heads | encoder_embed_dim | decoder_layers | decoder_heads | decoder_embed_dim | pa_f1 | window_auroc | window_auprc | window_f1 | point_auroc | point_auprc | point_f1 | event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 58251 | 5 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8459 | 0.8039 | 0.7041 | 0.7520 | 0.8190 | 0.7069 | 0.7678 | 0.3636 |
| 58252 | 5 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8433 | 0.8075 | 0.7052 | 0.7515 | 0.8226 | 0.7067 | 0.7676 | 0.3636 |
| 58253 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8427 | 0.8088 | 0.7022 | 0.7509 | 0.8232 | 0.7008 | 0.7670 | 0.2909 |
| 58254 | 10 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8378 | 0.8087 | 0.7054 | 0.7515 | 0.8234 | 0.7058 | 0.7676 | 0.3636 |
| 58255 | 10 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8259 | 0.8110 | 0.7072 | 0.7499 | 0.8257 | 0.7080 | 0.7658 | 0.1441 |
| 58256 | 10 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8445 | 0.8127 | 0.7053 | 0.7474 | 0.8268 | 0.7061 | 0.7642 | 0.3182 |
| 58257 | 20 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8427 | 0.8063 | 0.7050 | 0.7514 | 0.8208 | 0.7064 | 0.7673 | 0.3478 |
| 58258 | 20 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8451 | 0.8078 | 0.7041 | 0.7473 | 0.8221 | 0.7028 | 0.7641 | 0.3256 |
| 58259 | 20 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8450 | 0.8112 | 0.7099 | 0.7467 | 0.8254 | 0.7231 | 0.7638 | 0.3111 |
| 58260 | 5 | 0.8 | 3 | 2 | 256 | 1 | 2 | 128 | 0.8424 | 0.8081 | 0.7021 | 0.7510 | 0.8226 | 0.6991 | 0.7673 | 0.3636 |
| 58261 | 5 | 0.8 | 4 | 2 | 256 | 1 | 2 | 128 | 0.8425 | 0.8095 | 0.7026 | 0.7508 | 0.8239 | 0.7007 | 0.7671 | 0.3265 |
| 58262 | 5 | 0.8 | 2 | 4 | 256 | 1 | 2 | 128 | 0.8431 | 0.8092 | 0.7030 | 0.7500 | 0.8235 | 0.7014 | 0.7662 | 0.2162 |
| 58263 | 5 | 0.8 | 2 | 2 | 128 | 1 | 2 | 128 | 0.8433 | 0.8083 | 0.7028 | 0.7514 | 0.8230 | 0.7031 | 0.7673 | 0.3404 |
| 58264 | 5 | 0.8 | 2 | 2 | 512 | 1 | 2 | 128 | 0.8430 | 0.8106 | 0.7033 | 0.7490 | 0.8252 | 0.7040 | 0.7652 | 0.2769 |
| 58265 | 5 | 0.8 | 2 | 2 | 256 | 2 | 2 | 128 | 0.8499 | 0.8069 | 0.7036 | 0.7505 | 0.8222 | 0.7117 | 0.7669 | 0.3265 |
| 58266 | 5 | 0.8 | 2 | 2 | 256 | 1 | 4 | 128 | 0.8427 | 0.8089 | 0.7028 | 0.7514 | 0.8235 | 0.7027 | 0.7674 | 0.3721 |
| 58267 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 64 | 0.8423 | 0.8071 | 0.7049 | 0.7468 | 0.8217 | 0.7069 | 0.7639 | 0.3182 |
| 58268 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 256 | 0.8499 | 0.8089 | 0.7030 | 0.7511 | 0.8235 | 0.7005 | 0.7673 | 0.3137 |

### A.2 SWaT — train_standard (7 trials)

| run_id | dataset_val_fraction | trainer_learning_rate | trainer_weight_decay | pa_f1 | window_auroc | window_auprc | window_f1 | point_auroc | point_auprc | point_f1 | event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 58928 | 0.1 | 0.0001 | 0.01 | 0.8449 | 0.8093 | 0.7035 | 0.7509 | 0.8239 | 0.7056 | 0.7671 | 0.3265 |
| 58929 | 0.1 | 0.0003 | 0.01 | 0.8492 | 0.8107 | 0.7020 | 0.7485 | 0.8254 | 0.6999 | 0.7648 | 0.2250 |
| 58930 | 0.15 | 0.0001 | 0.01 | 0.8427 | 0.8089 | 0.7028 | 0.7514 | 0.8235 | 0.7027 | 0.7674 | 0.3721 |
| 58931 | 0.15 | 0.0003 | 0.01 | 0.8490 | 0.8097 | 0.7011 | 0.7486 | 0.8243 | 0.6985 | 0.7649 | 0.2571 |
| 58932 | 0.2 | 0.0001 | 0.01 | 0.8503 | 0.8100 | 0.7029 | 0.7511 | 0.8245 | 0.7025 | 0.7672 | 0.3200 |
| 58933 | 0.2 | 0.0003 | 0.01 | 0.8486 | 0.8108 | 0.7025 | 0.7478 | 0.8254 | 0.7028 | 0.7640 | 0.2500 |
| 58934 | 0.15 | 0.0001 | 0.001 | 0.8452 | 0.8091 | 0.7031 | 0.7512 | 0.8236 | 0.7034 | 0.7674 | 0.3721 |

### A.3 SWaT — train_incremental (19 rows: 15 trial slots + 4 pre-retry failures/timeouts)

| run_id | status | pipeline_merge_scale | finetune_trainer_reg_lambda | dataset_baseline_fraction | dataset_val_fraction | dataset_n_finetune_segments | base_pa_f1 | base_window_auroc | base_window_auprc | base_window_f1 | base_point_auroc | base_point_auprc | base_point_f1 | base_event_f1 | merg_pa_f1 | merg_window_auroc | merg_window_auprc | merg_window_f1 | merg_point_auroc | merg_point_auprc | merg_point_f1 | merg_event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 58935 | complete | 0.3 | 0.0 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8371 | 0.8008 | 0.7036 | 0.7510 | 0.8161 | 0.7077 | 0.7674 | 0.3556 |
| 58936 | complete | 0.3 | 0.001 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8370 | 0.8007 | 0.7036 | 0.7510 | 0.8161 | 0.7075 | 0.7673 | 0.3556 |
| 58937 | complete | 0.3 | 0.01 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8370 | 0.8007 | 0.7035 | 0.7509 | 0.8161 | 0.7076 | 0.7673 | 0.3556 |
| 58938 | complete | 0.5 | 0.0 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8386 | 0.8031 | 0.7039 | 0.7511 | 0.8183 | 0.7077 | 0.7674 | 0.3556 |
| 58939 | complete | 0.5 | 0.001 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8388 | 0.8030 | 0.7038 | 0.7511 | 0.8181 | 0.7073 | 0.7673 | 0.3556 |
| 58940 | complete | 0.5 | 0.01 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8384 | 0.8023 | 0.7037 | 0.7510 | 0.8174 | 0.7071 | 0.7673 | 0.3556 |
| 58941 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8408 | 0.8060 | 0.7046 | 0.7513 | 0.8209 | 0.7085 | 0.7674 | 0.3478 |
| 58942 | complete | 1.0 | 0.001 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8407 | 0.8059 | 0.7043 | 0.7513 | 0.8208 | 0.7074 | 0.7674 | 0.3478 |
| 58943 | complete | 1.0 | 0.01 | 0.5 | 0.15 | 3 | 0.8373 | 0.8005 | 0.7035 | 0.7508 | 0.8159 | 0.7086 | 0.7673 | 0.3478 | 0.8402 | 0.8054 | 0.7041 | 0.7512 | 0.8202 | 0.7065 | 0.7674 | 0.3556 |
| 58944 | complete | 1.0 | 0.0 | 0.3 | 0.15 | 3 | 0.8408 | 0.7993 | 0.7004 | 0.7498 | 0.8143 | 0.7019 | 0.7668 | 0.3478 | — | — | — | — | — | — | — | — |
| 58945 | complete | 1.0 | 0.0 | 0.7 | 0.15 | 3 | 0.8434 | 0.8034 | 0.7020 | 0.7507 | 0.8185 | 0.7050 | 0.7669 | 0.3333 | 0.8428 | 0.8065 | 0.7021 | 0.7510 | 0.8211 | 0.7030 | 0.7671 | 0.3721 |
| 58946 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 3 | 0.8301 | 0.8022 | 0.7045 | 0.7482 | 0.8178 | 0.7107 | 0.7646 | 0.1875 | — | — | — | — | — | — | — | — |
| 58947 | incomplete | 1.0 | 0.0 | 0.5 | 0.2 | 3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 58948 | incomplete | 1.0 | 0.0 | 0.5 | 0.15 | 2 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 58949 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 5 | 0.8303 | 0.8011 | 0.7034 | 0.7505 | 0.8164 | 0.7085 | 0.7670 | 0.2388 | 0.8408 | 0.8057 | 0.7039 | 0.7514 | 0.8206 | 0.7071 | 0.7674 | 0.3636 |
| 59055 | complete | 1.0 | 0.0 | 0.3 | 0.15 | 3 | 0.8424 | 0.7993 | 0.7016 | 0.7491 | 0.8143 | 0.7089 | 0.7660 | 0.2807 | 0.8422 | 0.7908 | 0.6940 | 0.7510 | 0.8078 | 0.7089 | 0.7664 | 0.3404 |
| 59056 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 3 | 0.8301 | 0.8022 | 0.7045 | 0.7482 | 0.8178 | 0.7107 | 0.7646 | 0.1875 | 0.8436 | 0.8082 | 0.7063 | 0.7506 | 0.8232 | 0.7115 | 0.7672 | 0.3265 |
| 59080 | complete | 1.0 | 0.0 | 0.5 | 0.2 | 3 | 0.8312 | 0.7958 | 0.7003 | 0.7477 | 0.8114 | 0.7020 | 0.7645 | 0.1684 | 0.8434 | 0.8040 | 0.7034 | 0.7498 | 0.8188 | 0.7064 | 0.7666 | 0.1928 |
| 59081 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 2 | 0.8373 | 0.8009 | 0.7034 | 0.7498 | 0.8164 | 0.7103 | 0.7663 | 0.1798 | 0.8438 | 0.8065 | 0.7047 | 0.7509 | 0.8214 | 0.7098 | 0.7673 | 0.3333 |

### A.4 PSM — model architecture (24 trials: 18 original + 6 patch_len∈{25,50} follow-up)

| run_id | patch_len | mask_ratio | encoder_layers | encoder_heads | encoder_embed_dim | decoder_layers | decoder_heads | decoder_embed_dim | pa_f1 | window_auroc | window_auprc | window_f1 | point_auroc | point_auprc | point_f1 | event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 58325 | 5 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7883 | 0.7890 | 0.6521 | 0.6579 | 0.7790 | 0.5566 | 0.5967 | 0.2425 |
| 58326 | 5 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7665 | 0.7945 | 0.6547 | 0.6582 | 0.7870 | 0.5619 | 0.5993 | 0.2967 |
| 58327 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7660 | 0.8017 | 0.6511 | 0.6761 | 0.7987 | 0.5645 | 0.6209 | 0.2857 |
| 58328 | 10 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7670 | 0.7993 | 0.6555 | 0.6675 | 0.7915 | 0.5603 | 0.6074 | 0.1647 |
| 58329 | 10 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7670 | 0.8031 | 0.6570 | 0.6723 | 0.7975 | 0.5647 | 0.6154 | 0.1729 |
| 58330 | 10 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7707 | 0.8119 | 0.6508 | 0.6955 | 0.8105 | 0.5647 | 0.6476 | 0.2095 |
| 58331 | 20 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7858 | 0.8076 | 0.6597 | 0.6859 | 0.8021 | 0.5616 | 0.6321 | 0.1987 |
| 58332 | 20 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7694 | 0.8144 | 0.6553 | 0.7062 | 0.8135 | 0.5677 | 0.6590 | 0.1867 |
| 58333 | 20 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7715 | 0.8129 | 0.6461 | 0.6994 | 0.8134 | 0.5596 | 0.6517 | 0.1966 |
| 58334 | 5 | 0.8 | 3 | 2 | 256 | 1 | 2 | 128 | 0.7665 | 0.8048 | 0.6553 | 0.6726 | 0.8014 | 0.5649 | 0.6215 | 0.2068 |
| 58335 | 5 | 0.8 | 4 | 2 | 256 | 1 | 2 | 128 | 0.7652 | 0.8024 | 0.6538 | 0.6678 | 0.7980 | 0.5625 | 0.6152 | 0.2676 |
| 58336 | 5 | 0.8 | 2 | 4 | 256 | 1 | 2 | 128 | 0.7661 | 0.8021 | 0.6512 | 0.6728 | 0.7986 | 0.5638 | 0.6164 | 0.2340 |
| 58337 | 5 | 0.8 | 2 | 2 | 128 | 1 | 2 | 128 | 0.7676 | 0.8002 | 0.6532 | 0.6918 | 0.7977 | 0.5643 | 0.6391 | 0.3960 |
| 58338 | 5 | 0.8 | 2 | 2 | 512 | 1 | 2 | 128 | 0.7654 | 0.8078 | 0.6596 | 0.6925 | 0.8044 | 0.5686 | 0.6404 | 0.2132 |
| 58339 | 5 | 0.8 | 2 | 2 | 256 | 2 | 2 | 128 | 0.7740 | 0.8096 | 0.6606 | 0.6896 | 0.8049 | 0.5672 | 0.6349 | 0.1731 |
| 58340 | 5 | 0.8 | 2 | 2 | 256 | 1 | 4 | 128 | 0.7660 | 0.8036 | 0.6497 | 0.6852 | 0.8003 | 0.5610 | 0.6311 | 0.2803 |
| 58341 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 64 | 0.7647 | 0.8063 | 0.6518 | 0.6905 | 0.8034 | 0.5634 | 0.6378 | 0.3287 |
| 58342 | 5 | 0.8 | 2 | 2 | 256 | 1 | 2 | 256 | 0.8022 | 0.8050 | 0.6625 | 0.6726 | 0.8003 | 0.5701 | 0.6188 | 0.2736 |
| 58829 | 25 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7742 | 0.8155 | 0.6684 | 0.7083 | 0.8110 | 0.5750 | 0.6553 | 0.1977 |
| 58830 | 25 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7742 | 0.8155 | 0.6684 | 0.7083 | 0.8110 | 0.5750 | 0.6553 | 0.1977 |
| 58831 | 25 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.7725 | 0.8205 | 0.6631 | 0.7191 | 0.8193 | 0.5753 | 0.6697 | 0.1879 |
| 58832 | 50 | 0.5 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8103 | 0.8126 | 0.6516 | 0.7031 | 0.8103 | 0.5625 | 0.6538 | 0.1336 |
| 58833 | 50 | 0.65 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8116 | 0.8131 | 0.6528 | 0.7042 | 0.8105 | 0.5637 | 0.6546 | 0.1413 |
| 58834 | 50 | 0.8 | 2 | 2 | 256 | 1 | 2 | 128 | 0.8116 | 0.8131 | 0.6528 | 0.7042 | 0.8105 | 0.5637 | 0.6546 | 0.1413 |

### A.5 PSM — train_standard (14 trials: 7 at patch_len=25, 7 at patch_len=5)

| run_id | dataset_val_fraction | trainer_learning_rate | trainer_weight_decay | pa_f1 | window_auroc | window_auprc | window_f1 | point_auroc | point_auprc | point_f1 | event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 58950 | 0.1 | 0.0001 | 0.01 | 0.7729 | 0.8173 | 0.6691 | 0.7074 | 0.8152 | 0.5824 | 0.6552 | 0.1440 |
| 58951 | 0.1 | 0.0003 | 0.01 | 0.7724 | 0.8081 | 0.6509 | 0.6881 | 0.8059 | 0.5602 | 0.6364 | 0.1507 |
| 58952 | 0.15 | 0.0001 | 0.01 | 0.7725 | 0.8205 | 0.6631 | 0.7191 | 0.8193 | 0.5753 | 0.6697 | 0.1879 |
| 58953 | 0.15 | 0.0003 | 0.01 | 0.7714 | 0.8059 | 0.6503 | 0.6902 | 0.8065 | 0.5597 | 0.6372 | 0.2065 |
| 58954 | 0.2 | 0.0001 | 0.01 | 0.8141 | 0.8185 | 0.6691 | 0.7159 | 0.8171 | 0.5802 | 0.6682 | 0.1891 |
| 58955 | 0.2 | 0.0003 | 0.01 | 0.7726 | 0.8087 | 0.6594 | 0.6897 | 0.8065 | 0.5661 | 0.6358 | 0.1680 |
| 58956 | 0.15 | 0.0001 | 0.001 | 0.7724 | 0.8197 | 0.6619 | 0.7172 | 0.8183 | 0.5738 | 0.6674 | 0.1934 |
| 59088 | 0.1 | 0.0001 | 0.01 | 0.7653 | 0.8046 | 0.6521 | 0.6948 | 0.8020 | 0.5620 | 0.6453 | 0.3654 |
| 59089 | 0.1 | 0.0003 | 0.01 | 0.7664 | 0.8032 | 0.6540 | 0.6745 | 0.7994 | 0.5625 | 0.6186 | 0.2243 |
| 59090 | 0.15 | 0.0001 | 0.01 | 0.7676 | 0.8002 | 0.6532 | 0.6918 | 0.7977 | 0.5643 | 0.6391 | 0.3960 |
| 59091 | 0.15 | 0.0003 | 0.01 | 0.7799 | 0.8011 | 0.6567 | 0.6826 | 0.7979 | 0.5613 | 0.6278 | 0.3012 |
| 59092 | 0.2 | 0.0001 | 0.01 | 0.7727 | 0.7996 | 0.6509 | 0.6857 | 0.7958 | 0.5585 | 0.6311 | 0.3288 |
| 59093 | 0.2 | 0.0003 | 0.01 | 0.8035 | 0.8033 | 0.6571 | 0.6681 | 0.7980 | 0.5607 | 0.6190 | 0.2582 |
| 59094 | 0.15 | 0.0001 | 0.001 | 0.7678 | 0.8002 | 0.6532 | 0.6918 | 0.7977 | 0.5643 | 0.6391 | 0.4027 |

### A.6 PSM — train_incremental, patch_len=25 (15 trials, initial architecture pick)

| run_id | status | pipeline_merge_scale | finetune_trainer_reg_lambda | dataset_baseline_fraction | dataset_val_fraction | dataset_n_finetune_segments | base_pa_f1 | base_window_auroc | base_window_auprc | base_window_f1 | base_point_auroc | base_point_auprc | base_point_f1 | base_event_f1 | merg_pa_f1 | merg_window_auroc | merg_window_auprc | merg_window_f1 | merg_point_auroc | merg_point_auprc | merg_point_f1 | merg_event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 58957 | complete | 0.3 | 0.0 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7710 | 0.7853 | 0.6360 | 0.6417 | 0.7795 | 0.5527 | 0.5864 | 0.2096 |
| 58958 | complete | 0.3 | 0.001 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7710 | 0.7853 | 0.6360 | 0.6416 | 0.7795 | 0.5527 | 0.5864 | 0.2077 |
| 58959 | complete | 0.3 | 0.01 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7710 | 0.7852 | 0.6361 | 0.6415 | 0.7795 | 0.5529 | 0.5866 | 0.1971 |
| 58960 | complete | 0.5 | 0.0 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7740 | 0.7761 | 0.6255 | 0.6343 | 0.7671 | 0.5394 | 0.5750 | 0.2017 |
| 58961 | complete | 0.5 | 0.001 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7740 | 0.7761 | 0.6255 | 0.6343 | 0.7671 | 0.5394 | 0.5750 | 0.2013 |
| 58962 | complete | 0.5 | 0.01 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7740 | 0.7761 | 0.6256 | 0.6342 | 0.7672 | 0.5395 | 0.5748 | 0.2044 |
| 58963 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7763 | 0.7689 | 0.6153 | 0.6270 | 0.7695 | 0.5345 | 0.5739 | 0.1630 |
| 58964 | complete | 1.0 | 0.001 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7763 | 0.7690 | 0.6154 | 0.6271 | 0.7696 | 0.5346 | 0.5739 | 0.1594 |
| 58965 | complete | 1.0 | 0.01 | 0.5 | 0.15 | 3 | 0.7977 | 0.7904 | 0.6480 | 0.6534 | 0.7931 | 0.5721 | 0.6050 | 0.1652 | 0.7763 | 0.7697 | 0.6159 | 0.6279 | 0.7699 | 0.5348 | 0.5745 | 0.1913 |
| 58966 | complete | 1.0 | 0.0 | 0.3 | 0.15 | 3 | 0.8132 | 0.7747 | 0.6451 | 0.6380 | 0.7799 | 0.5772 | 0.5930 | 0.1541 | 0.7754 | 0.7484 | 0.6188 | 0.6232 | 0.7401 | 0.5335 | 0.5628 | 0.1493 |
| 58967 | complete | 1.0 | 0.0 | 0.7 | 0.15 | 3 | 0.7983 | 0.8002 | 0.6417 | 0.6760 | 0.8007 | 0.5606 | 0.6259 | 0.1478 | 0.7759 | 0.7708 | 0.6147 | 0.6380 | 0.7629 | 0.5278 | 0.5780 | 0.1733 |
| 58968 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 3 | 0.8093 | 0.7895 | 0.6444 | 0.6536 | 0.7911 | 0.5663 | 0.6068 | 0.1637 | 0.7764 | 0.7769 | 0.6215 | 0.6379 | 0.7787 | 0.5405 | 0.5836 | 0.1189 |
| 58969 | complete | 1.0 | 0.0 | 0.5 | 0.2 | 3 | 0.7959 | 0.7917 | 0.6483 | 0.6612 | 0.7944 | 0.5717 | 0.6146 | 0.1700 | 0.7760 | 0.7708 | 0.6293 | 0.6414 | 0.7672 | 0.5460 | 0.5876 | 0.1659 |
| 58970 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 2 | 0.8133 | 0.7827 | 0.6443 | 0.6429 | 0.7834 | 0.5670 | 0.5911 | 0.1475 | 0.7744 | 0.7775 | 0.6217 | 0.6425 | 0.7688 | 0.5323 | 0.5813 | 0.1581 |
| 58971 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 5 | 0.8140 | 0.7846 | 0.6449 | 0.6433 | 0.7847 | 0.5673 | 0.5915 | 0.1385 | 0.7759 | 0.7469 | 0.6107 | 0.6074 | 0.7440 | 0.5284 | 0.5560 | 0.1105 |

### A.7 PSM — train_incremental, patch_len=5 (15 trials, final architecture pick)

| run_id | status | pipeline_merge_scale | finetune_trainer_reg_lambda | dataset_baseline_fraction | dataset_val_fraction | dataset_n_finetune_segments | base_pa_f1 | base_window_auroc | base_window_auprc | base_window_f1 | base_point_auroc | base_point_auprc | base_point_f1 | base_event_f1 | merg_pa_f1 | merg_window_auroc | merg_window_auprc | merg_window_f1 | merg_point_auroc | merg_point_auprc | merg_point_f1 | merg_event_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 59095 | complete | 0.3 | 0.0 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7667 | 0.7899 | 0.6404 | 0.6507 | 0.7851 | 0.5586 | 0.5958 | 0.2857 |
| 59096 | complete | 0.3 | 0.001 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7666 | 0.7899 | 0.6404 | 0.6507 | 0.7851 | 0.5586 | 0.5958 | 0.2842 |
| 59097 | complete | 0.3 | 0.01 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7666 | 0.7894 | 0.6401 | 0.6498 | 0.7846 | 0.5584 | 0.5949 | 0.2689 |
| 59098 | complete | 0.5 | 0.0 | 0.5 | 0.15 | 3 | 0.8074 | 0.7718 | 0.6375 | 0.6331 | 0.7683 | 0.5635 | 0.5785 | 0.2208 | 0.7709 | 0.7939 | 0.6418 | 0.6566 | 0.7904 | 0.5598 | 0.5983 | 0.3448 |
| 59099 | complete | 0.5 | 0.001 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7699 | 0.7973 | 0.6435 | 0.6619 | 0.7940 | 0.5601 | 0.6037 | 0.2084 |
| 59100 | complete | 0.5 | 0.01 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7699 | 0.7968 | 0.6431 | 0.6598 | 0.7934 | 0.5597 | 0.6016 | 0.2060 |
| 59101 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7716 | 0.7991 | 0.6467 | 0.6790 | 0.8019 | 0.5663 | 0.6289 | 0.2749 |
| 59102 | complete | 1.0 | 0.001 | 0.5 | 0.15 | 3 | 0.8074 | 0.7718 | 0.6375 | 0.6331 | 0.7683 | 0.5635 | 0.5785 | 0.2208 | 0.7733 | 0.7988 | 0.6452 | 0.6788 | 0.8006 | 0.5647 | 0.6268 | 0.2007 |
| 59103 | complete | 1.0 | 0.01 | 0.5 | 0.15 | 3 | 0.8067 | 0.7740 | 0.6379 | 0.6361 | 0.7711 | 0.5630 | 0.5817 | 0.2294 | 0.7719 | 0.8000 | 0.6474 | 0.6796 | 0.8025 | 0.5667 | 0.6296 | 0.2392 |
| 59104 | complete | 1.0 | 0.0 | 0.3 | 0.15 | 3 | 0.8140 | 0.7743 | 0.6365 | 0.6420 | 0.7813 | 0.5656 | 0.5974 | 0.2609 | 0.7696 | 0.7719 | 0.6296 | 0.6501 | 0.7728 | 0.5455 | 0.5923 | 0.2764 |
| 59105 | complete | 1.0 | 0.0 | 0.7 | 0.15 | 3 | 0.7745 | 0.7887 | 0.6447 | 0.6486 | 0.7840 | 0.5629 | 0.5917 | 0.1729 | 0.7721 | 0.7941 | 0.6399 | 0.6679 | 0.7926 | 0.5525 | 0.6128 | 0.1519 |
| 59106 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 3 | 0.7982 | 0.7770 | 0.6397 | 0.6387 | 0.7718 | 0.5636 | 0.5793 | 0.2053 | 0.7729 | 0.8015 | 0.6491 | 0.6898 | 0.8057 | 0.5705 | 0.6433 | 0.2434 |
| 59107 | complete | 1.0 | 0.0 | 0.5 | 0.2 | 3 | 0.8092 | 0.7730 | 0.6368 | 0.6410 | 0.7710 | 0.5621 | 0.5873 | 0.2412 | 0.7724 | 0.8003 | 0.6496 | 0.6796 | 0.8025 | 0.5689 | 0.6301 | 0.1890 |
| 59108 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 2 | 0.8060 | 0.7713 | 0.6347 | 0.6324 | 0.7681 | 0.5608 | 0.5774 | 0.2361 | 0.7692 | 0.8053 | 0.6562 | 0.7114 | 0.8086 | 0.5761 | 0.6579 | 0.2759 |
| 59109 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 5 | 0.7804 | 0.7753 | 0.6339 | 0.6362 | 0.7717 | 0.5585 | 0.5761 | 0.2432 | 0.7751 | 0.7973 | 0.6449 | 0.6911 | 0.8009 | 0.5638 | 0.6376 | 0.2391 |

### A.8 ETTh1 — model architecture (24 rows: 17 original + 2 retries + 5-row seed-repro check)

| run_id | seed | patch_len | encoder_embed_dim | encoder_layers | encoder_heads | decoder_embed_dim | decoder_layers | decoder_heads | mse | rmse | mae |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 58375 | 42 | 4 | 64 | 3 | 4 | 64 | 2 | 4 | 0.3877 | 0.6227 | 0.4419 |
| 58376 | 42 | 4 | 128 | 3 | 4 | 64 | 2 | 4 | 0.3829 | 0.6188 | 0.4285 |
| 58377 | 42 | 4 | 256 | 3 | 4 | 64 | 2 | 4 | 0.4110 | 0.6411 | 0.4577 |
| 58378 | 42 | 6 | 64 | 3 | 4 | 64 | 2 | 4 | 0.3944 | 0.6280 | 0.4467 |
| 58379 | 42 | 6 | 128 | 3 | 4 | 64 | 2 | 4 | 0.3915 | 0.6257 | 0.4404 |
| 58380 | 42 | 6 | 256 | 3 | 4 | 64 | 2 | 4 | 0.4311 | 0.6566 | 0.4518 |
| 58381 | 42 | 8 | 64 | 3 | 4 | 64 | 2 | 4 | 0.4133 | 0.6429 | 0.4462 |
| 58382 | 42 | 8 | 128 | 3 | 4 | 64 | 2 | 4 | 0.3919 | 0.6260 | 0.4490 |
| 58383 | 42 | 8 | 256 | 3 | 4 | 64 | 2 | 4 | 0.4170 | 0.6458 | 0.4590 |
| 58384 | 42 | 4 | 128 | 2 | 4 | 64 | 2 | 4 | 0.3738 | 0.6114 | 0.4250 |
| 58385 | 42 | 4 | 128 | 4 | 4 | 64 | 2 | 4 | 0.3817 | 0.6178 | 0.4274 |
| 58386 | 42 | 4 | 128 | 3 | 2 | 64 | 2 | 4 | 0.4061 | 0.6373 | 0.4439 |
| 58387 | 42 | 4 | 128 | 3 | 4 | 64 | 1 | 4 | 0.4071 | 0.6381 | 0.4411 |
| 58388 | 42 | 4 | 128 | 3 | 4 | 64 | 3 | 4 | — | — | — |
| 58389 | 42 | 4 | 128 | 3 | 4 | 64 | 2 | 2 | 0.3974 | 0.6304 | 0.4348 |
| 58390 | 42 | 4 | 128 | 3 | 4 | 32 | 2 | 4 | — | — | — |
| 58391 | 42 | 4 | 128 | 3 | 4 | 128 | 2 | 4 | 0.3663 | 0.6052 | 0.4235 |
| 58835 | 42 | 4 | 128 | 3 | 4 | 64 | 3 | 4 | 0.3858 | 0.6211 | 0.4299 |
| 58836 | 42 | 4 | 128 | 3 | 4 | 32 | 2 | 4 | 0.4270 | 0.6534 | 0.4564 |
| 58837 | 42 | 4 | 128 | 3 | 4 | 64 | 2 | 4 | 0.3911 | 0.6253 | 0.4342 |
| 58838 | 123 | 4 | 128 | 3 | 4 | 64 | 2 | 4 | 0.3831 | 0.6189 | 0.4328 |
| 58839 | 7 | 4 | 128 | 3 | 4 | 64 | 2 | 4 | 0.4071 | 0.6380 | 0.4474 |
| 58840 | 123 | 4 | 128 | 3 | 4 | 128 | 2 | 4 | 0.4402 | 0.6635 | 0.4720 |
| 58841 | 7 | 4 | 128 | 3 | 4 | 128 | 2 | 4 | 0.3844 | 0.6200 | 0.4283 |

### A.9 ETTh1 — train_standard (7 trials)

| run_id | dataset_val_fraction | trainer_learning_rate | trainer_weight_decay | mse | rmse | mae |
|---|---|---|---|---|---|---|
| 59058 | 0.05 | 0.001 | 0.0001 | 0.4171 | 0.6458 | 0.4398 |
| 59059 | 0.05 | 0.0003 | 0.0001 | 0.3881 | 0.6230 | 0.4288 |
| 59060 | 0.1 | 0.001 | 0.0001 | 0.3911 | 0.6253 | 0.4342 |
| 59061 | 0.1 | 0.0003 | 0.0001 | 0.3838 | 0.6195 | 0.4363 |
| 59062 | 0.15 | 0.001 | 0.0001 | 0.4271 | 0.6535 | 0.4573 |
| 59063 | 0.15 | 0.0003 | 0.0001 | 0.4547 | 0.6743 | 0.4707 |
| 59064 | 0.1 | 0.001 | 0.001 | 0.3916 | 0.6257 | 0.4344 |

### A.10 ETTh1 — train_incremental (16 rows: 15 trial slots + 1 degenerate-split failure)

`val_fraction=0.05` alone (default `n_finetune_segments=3`) fails the split-size guard
(§1.9) — that row is included below with `status=incomplete` and empty merged metrics,
since it's a real, deterministic, reproducible outcome, not noise to discard.

| run_id | status | pipeline_merge_scale | finetune_trainer_reg_lambda | dataset_baseline_fraction | dataset_val_fraction | dataset_n_finetune_segments | base_mse | base_rmse | base_mae | merg_mse | merg_rmse | merg_mae |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 59065 | complete | 0.3 | 0.0 | 0.5 | 0.1 | 3 | 0.6873 | 0.8291 | 0.6013 | 0.4565 | 0.6757 | 0.4631 |
| 59066 | complete | 0.3 | 0.001 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4540 | 0.6738 | 0.4647 |
| 59067 | complete | 0.3 | 0.01 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4620 | 0.6797 | 0.4706 |
| 59068 | complete | 0.5 | 0.0 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4236 | 0.6508 | 0.4528 |
| 59069 | complete | 0.5 | 0.001 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4249 | 0.6519 | 0.4536 |
| 59070 | complete | 0.5 | 0.01 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4313 | 0.6567 | 0.4580 |
| 59071 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4684 | 0.6844 | 0.5035 |
| 59072 | complete | 1.0 | 0.001 | 0.5 | 0.1 | 3 | 0.6873 | 0.8291 | 0.6013 | 0.4486 | 0.6698 | 0.4950 |
| 59073 | complete | 1.0 | 0.01 | 0.5 | 0.1 | 3 | 0.6132 | 0.7831 | 0.5675 | 0.4778 | 0.6912 | 0.5133 |
| 59074 | complete | 1.0 | 0.0 | 0.3 | 0.1 | 3 | 1.0076 | 1.0038 | 0.7197 | 1.9766 | 1.4059 | 1.0558 |
| 59075 | complete | 1.0 | 0.0 | 0.7 | 0.1 | 3 | 0.4977 | 0.7055 | 0.5038 | 0.5170 | 0.7190 | 0.5586 |
| 59076 | complete | 1.0 | 0.0 | 0.5 | 0.05 | 3 | 0.7490 | 0.8655 | 0.6280 | — | — | — |
| 59077 | complete | 1.0 | 0.0 | 0.5 | 0.15 | 3 | 0.7023 | 0.8380 | 0.6131 | 0.5918 | 0.7693 | 0.5661 |
| 59078 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 2 | 0.6741 | 0.8211 | 0.5872 | 0.4404 | 0.6636 | 0.4934 |
| 59079 | complete | 1.0 | 0.0 | 0.5 | 0.1 | 5 | 0.6820 | 0.8258 | 0.5964 | 0.7261 | 0.8521 | 0.6386 |
| 59110 | complete | 1.0 | 0.0 | 0.5 | 0.05 | 2 | 0.6301 | 0.7938 | 0.5874 | 0.4562 | 0.6754 | 0.5001 |

---

## 7. Resources

- **Interactive results overview** (Standard vs. Incremental, every dataset/variant,
  grid-average and best-config side by side, mobile-friendly):
  https://claude.ai/code/artifact/b9478404-0c3d-4629-a43b-0aedb1ee24bf
- **This file** is the source of truth for every experiment run this session — keep it
  updated (§6) whenever a new sweep is collected, rather than letting results live only
  in the artifact or in memory.

---

## 8. Phase 1 — merge diagnostics (transfer matrix, GRR, scale curve), 2026-08-03/04

Training-free cross-evaluation of checkpoints already on disk, per `PHASE1_RUNBOOK.md`.
These are measurements; the research reading is stated where the evidence supports one and
withheld where it does not.

### 8.0 What was run

| dataset | source run | standard ref | diagnostics run | walltime |
|---|---|---|---|---|
| SWaT | `slurm_grid_swat_train_incremental/58941` | `slurm_grid_swat_train_standard/58930` | `..._diagnostics/78070` | 1:24:44 |
| PSM (patch_len=5) | `slurm_grid_psm_train_incremental/59101` | `slurm_grid_psm_train_standard/59090` | `..._diagnostics/78071` | 0:15:36 |
| ETTh1 | `slurm_grid_etth_forecast_train_incremental/59071` | `slurm_grid_etth_forecast_train_standard/59060` | `..._diagnostics/78072` | 0:00:34 |

All three are the `reg_lambda=0`, `merge_scale=1.0`, `baseline_fraction=0.5`,
`n_finetune_segments=3` trial of their sweep, with the Standard run matched on trainer args
(same lr / weight decay / epochs / seed) as well as on the guarded `dataset_*`/`mae_tx_*`.

**Not run, and it matters:** a single `eval_seed` throughout, and a coarse 7-point scale
grid on SWaT/PSM (ETTh1 got the full 16-point grid). The runbook asks for
`--pipeline_eval_seeds 43 44 45` on AD; at the production `n_eval_passes=30` a 3-seed SWaT
matrix costs ~4.2 h and needs the 2 h `--time` raised. **No per-shard event metric below
should be quoted without that spread.** The readings here rest on the reconstruction val
block and on AUROC/F1, not on event metrics.

**Prerequisite check:** all **60/60** `merged/checkpoints/best.pt` under `$WORK/runs`
recompute bitwise from baseline + finetunes (`torch.equal` on every tensor). Merging is
reproducible on the cluster.

**Parameter-space coverage:** every complete incremental run on disk (60) has at least one
complete Standard run passing the config guard. **No missing training** — Phase 1 needed no
new jobs.

### 8.1 Geometry (all 62 incremental runs, no GPU)

Cosine **does** decay with temporal distance on all three datasets, pooled over runs
(distances 3–4 come from the single 5-segment run each):

| distance | SWaT | PSM | ETTh1 |
|---|---|---|---|
| 1 | 0.776 | 0.361 | 0.169 |
| 2 | 0.676 | 0.262 | 0.112 |
| 3 | 0.631 | 0.195 | 0.183 |
| 4 | 0.551 | 0.104 | −0.006 |

So temporal distribution shift does differentiate the task vectors, monotonically on
SWaT and PSM. The thesis framing survives this check.

Per-dataset character (headline runs), quoting **mean off-diagonal cosine** (not the
distance-1 value tabulated above): SWaT's vectors are near-collinear (0.737, effective rank
**1.63/3**, ‖τ‖/‖θ₀‖ 0.0044–0.0074); PSM is the most spread (0.399, eff. rank **2.51/3**);
ETTh1 looks nearly orthogonal (0.095) but its rank is only 1.88/3 — low because
`finetune_0` is dead, not because two vectors are parallel. Note that near-orthogonality is
the *default* in this many dimensions: for random vectors cosine would be ~0 ± 1/√D, so
even ETTh1's 0.095 sits ~80σ above chance and none of these sets is meaningfully orthogonal.

**Degenerate fine-tuning on ETTh1.** In **11 of 15** ETTh1 runs one τ is 6–24× smaller than
its siblings. In headline run 59071, `finetune_0` has ‖τ‖/‖θ₀‖ = **0.0012** vs 0.0197 /
0.0283, with `best_epoch=1` — its third singular value carries 0.1% of the energy. **That
run's 3-vector merge is effectively a 2-vector merge**, and its `ft_0` row below is inert by
construction. One PSM run (59109, 5 segments) shows the same at 5.8×. No SWaT run does.

### 8.2 ETTh1 — interference, plus one dead segment

Val block, `forecast/mse` as ratio to base (lower better); test column raw:

```
         val_base   val_0   val_1   val_2  |  test MSE
base       1.000    1.000   1.000   1.000  |   0.6132
ft_0       0.966    0.958   0.908   0.959  |   0.6334
ft_1       0.940    0.541   0.757   1.063  |   0.4721
ft_2       1.222    0.791   0.583   0.481  |   0.4575
merged     1.207    1.127   0.738   0.871  |   0.4684
standard      —        —       —       —   |   0.3911
```

- `specialisation` **+0.076** (diag 0.732 < offdiag 0.808) — specialists do help more on
  their own shard than on others.
- **The diagonal is the column minimum only for segment 2.** `ft_1` beats `ft_0` on `val_0`
  (0.541 vs 0.958) and `ft_2` beats `ft_1` on `val_1`. `ft_0` is inert everywhere — the
  degenerate τ from §8.1.
- `base_slice_ratio_mean` **1.043**, and `ft_2` costs **1.222** on `val_base` — forgetting,
  now measured.
- Merged at scale 1.0 is *worse than base* on `val_0` (1.127) and `val_base` (1.207) while
  better on `val_1`/`val_2`, and worse than each specialist on that specialist's shard.
- `grr` = **0.652** on a gap of −0.222 (**36% of base**, far above the 2% noise floor).
- Curve: clean U, minimum at **scale 0.6** (MSE 0.4228 vs 0.4684 at the run's own 1.0),
  monotone worsening past it. At 0.6 the implied GRR would be 0.857.

**Reading: interference** (decision-table row 2 — diag ≪ 1, offdiag > diag, moderate ‖τ‖,
curve peaked below 1.0 and falling after), *contaminated by* degenerate fine-tuning on
segment 0. The run's own `merge_scale=1.0` overshoots by a measurable margin.

### 8.3 PSM — interference, the cleanest AD case

Val block, `reconstruction/score_mean` ratio to base:

```
         val_base   val_0   val_1   val_2  |  test AUROC  window_f1   pa_f1  event_f1
base       1.000    1.000   1.000   1.000  |     0.7740     0.6361  0.8067    0.2294
ft_0       1.039    0.774   0.845   0.909  |     0.7832     0.6517  0.7667    0.2957
ft_1       1.125    0.763   0.770   0.875  |     0.7936     0.6546  0.7675    0.1971
ft_2       1.090    0.950   0.665   0.714  |     0.7932     0.6557  0.7667    0.2145
merged     1.650    0.859   0.734   1.096  |     0.7991     0.6790  0.7716    0.2749
standard      —        —       —       —   |     0.8002     0.6918  0.7676    0.3960
```

- `specialisation` **+0.082** (diag 0.753 < offdiag 0.835).
- Same pattern as ETTh1: **the diagonal is the column minimum only for segment 2.**
- `base_slice_ratio_mean` **1.085**, merged **1.650** on `val_base` — the merge forgets the
  base regime substantially more than any single specialist does.
- Merged beats base on `val_0`/`val_1` but is *worse than base* on `val_2` (1.096), and
  worse than the best specialist on every column.
- `grr`: window_auroc **0.958** (gap 0.0262 = 3.4% of base — above the floor but not by
  much), window_f1 **0.770** (gap 8.8% of base — solidly informative).
- Curve peaks at **scale 0.75** (AUROC 0.8018) and falls to 0.7801 by 1.5; window_f1 peaks
  at 0.75 as well.

**Metrics disagree, as they have before on PSM (§2.4).** `pa_f1` has a *negative* gap
(standard 0.7676 < base 0.8067), so its GRR of 0.897 describes recovery toward a worse
model; on `event_f1` Standard is far ahead (0.396 vs 0.229) and the merge recovers only 27%.
Ranking PSM by a single metric remains unsafe.

**Reading: interference.** Each τᵢ improves reconstruction on every shard; the sum at full
scale overshoots (worse than base on `val_2`, 1.65× on `val_base`) and the test curve peaks
below 1.0.

### 8.4 SWaT — the merge wrecks reconstruction and the detector does not notice

Val block, `reconstruction/score_mean` ratio to base:

```
         val_base   val_0   val_1   val_2  |  test AUROC  window_f1   pa_f1
base       1.000    1.000   1.000   1.000  |     0.8005     0.7508  0.8373
ft_0       1.026    0.740   0.698   0.750  |     0.7991     0.7508  0.8370
ft_1       1.235    0.770   0.603   0.538  |     0.8016     0.7510  0.8373
ft_2       1.413    0.861   0.638   0.481  |     0.8029     0.7511  0.8383
merged     5.122    3.180   2.487   1.625  |     0.8060     0.7513  0.8407
standard      —        —       —       —   |     0.8089     0.7514  0.8428
```

- `specialisation` **+0.101** (diag 0.608 < offdiag 0.709) — the *strongest* specialisation
  of the three datasets. Here the diagonal **is** the column minimum for **all three**
  segments — the only dataset where every specialist wins on its own shard.
- `base_slice_ratio_mean` **1.225**, rising monotonically with segment index (1.03 → 1.24 →
  1.41) — clean, monotone forgetting.
- `merged_ratio_mean` **2.43**, and **5.12 on `val_base`**. Summing three near-collinear τ
  (cosine 0.67–0.85, eff. rank 1.63/3) at full scale blows reconstruction error up by 2–5×.
  This is the textbook interference signature, in its most extreme form on record here.
- **And yet every test metric barely moves**: AUROC 0.8005 → 0.8060, window_f1 0.7508 →
  0.7513. The curve rises monotonically to 1.5 (0.8067) and never peaks.
- **Every GRR warning fired.** All test gaps are ≤1% of base (window_auroc 0.0084 = 1.0%,
  window_f1 0.0006 = 0.1%, point_f1 0.0001 = 0.0%). Every SWaT GRR in `result.json` is a
  ratio of two noise terms and **must not be quoted**.

**Reading: cannot tell from the test column — report SWaT as a control, as planned.** But
the val block is unambiguous and is new information: the merge destroys the reconstruction
model (2–5× worse) while AUROC/F1 shift by <1%. That is direct evidence that **SWaT's
detection metrics are insensitive to large changes in model quality**, i.e. the dataset is
saturation-limited, sharpening the earlier finding (§2.4, and the "benchmarks too
stationary" note) from "merging gives no gain" to "the metric cannot see a 5× change in the
underlying model."

### 8.5 Cross-dataset summary

| | SWaT | PSM | ETTh1 |
|---|---|---|---|
| mean off-diag cosine | 0.737 | 0.399 | 0.095 |
| cosine at temporal distance 1 | 0.771 | 0.466 | 0.121 |
| effective rank (of 3) | 1.63 | 2.51 | 1.88 |
| mean ‖τ‖/‖θ₀‖ | 0.0061 | 0.0059 | 0.0164 |
| `specialisation` | +0.101 | +0.082 | +0.076 |
| `diag_ratio_mean` | 0.608 | 0.753 | 0.732 |
| `offdiag_ratio_mean` | 0.709 | 0.835 | 0.808 |
| `base_slice_ratio_mean` | 1.225 | 1.085 | 1.043 |
| `merged_ratio_mean` | 2.431 | 0.896 | 0.912 |
| curve optimum | none (rises to 1.5) | 0.75 | 0.60 |
| primary gap vs base | 1.0% (below floor) | 3.4% | 36% |
| GRR usable? | **no** | marginally (0.958) | yes (0.652) |
| mechanism | interference in val; test uninformative | **interference** | **interference** + dead segment |

Specialisation is positive on all three, so **degenerate fine-tuning is ruled out as the
global explanation** — with the documented exception of ETTh1 `finetune_0` and PSM 59109.
Redundancy is ruled out too: diagonal ratios of 0.61–0.75 are far from 1.0. **Interference
is the mechanism the evidence supports on all three**, which is the reading that licenses
the non-orthogonality argument. Note the ordering: specialisation is *highest* where the
vectors are *most collinear* (SWaT), and the merge damage tracks collinearity
(`merged_ratio_mean` 2.43 at cosine 0.74, ~0.90 at cosine 0.47/0.12) — consistent with
interference being driven by overlap, and the natural next test for §5's
geometry-predicts-outcome question.

Every source run used its own `merge_scale=1.0`, and on both datasets with a usable curve
the optimum is **below** 1.0 (0.75 PSM, 0.60 ETTh1). The reported incremental numbers are
therefore not the best this merge can do.

### 8.6 The headline question: does merging reach joint training?

The question the project exists to answer — *can a base model plus task-arithmetic merging
of incrementally fine-tuned models match a model trained on all the data at once?* — is
exactly what GRR measures. The comparison is fair: the incremental base trains on
`baseline_fraction=0.5` of the training data and the Standard reference on `1.0`, with the
same `val_fraction` held out on both (verified from both `config.json`s).

Reading the answer off the merge-scale curve rather than off the single α the run happened
to use:

| dataset | metric | base (frozen, 50% data) | merged @α=1 | merged @best α | joint training | gap closed @best |
|---|---|---|---|---|---|---|
| **PSM** | window_auroc | 0.7740 | 0.7991 | **0.8018** @0.75 | 0.8002 | **~100%** |
| **PSM** | window_f1 | 0.6361 | 0.6790 | 0.6802 @0.75 | 0.6918 | 79% |
| **ETTh1** | forecast/mse | 0.6132 | 0.4684 | **0.4228** @0.60 | 0.3911 | 86% |
| **ETTh1** | forecast/mae | 0.5675 | 0.5035 | 0.4528 @0.50 | 0.4342 | 86% |
| SWaT | window_auroc | 0.8005 | 0.8060 | 0.8067 @1.5 | 0.8089 | *gap itself is noise* |

**Which shortfalls are real.** Applying the same 2%-of-base floor used for the gap to the
*remaining* shortfall (`standard − merged@best`):

| | gap vs base | shortfall @best α | verdict |
|---|---|---|---|
| PSM window_auroc | 3.4% | **0.2%** | inside floor — **merging matches joint training** |
| PSM window_f1 | 8.8% | **1.8%** | inside floor — cannot claim a real remaining gap |
| ETTh1 forecast/mse | 36.2% | **5.2%** | **REAL** — a genuine shortfall remains |
| ETTh1 forecast/mae | 23.5% | **3.3%** | **REAL** |
| SWaT (all) | ≤1.0% | ≤0.3% | the gap never cleared the floor; unanswerable |

**So the answer is yes on PSM and not-quite on ETTh1.** On PSM, merging half-data +
task vectors reaches a model trained on everything, on both usable metrics. On ETTh1 it
closes 86% of a large gap, and the remaining 5.2%-of-base shortfall clears the noise floor,
so that one is a genuine limitation rather than measurement error. SWaT cannot answer the
question at all — its frozen base is already within 1% of joint training, so there is no gap
for any method to recover, which is a statement about the benchmark, not about merging.

**Three caveats that bound all of the above, none of them small.**

1. **α was selected on the test set.** `merge_scale_curve.csv` traces *test* metrics, so
   "best α" is chosen by looking at the number being reported. The *shape* of the curve is
   trustworthy and "α = 1.0 is suboptimal" is safe, but the @best-α values are optimistic and
   are not a defensible headline until α is picked on validation data and then reported on
   test. This is the single biggest methodological hole in §8.6.
2. **The 2% floor is imported, not measured here.** It comes from the same-seed ETTh1 MSE
   repeat (§4) and is applied to AUROC and F1 by assumption. No reproducibility floor has
   ever been measured for the AD metrics, so every "inside floor" verdict above rests on that
   transfer.
3. **One evaluation seed and one training seed.** There are no error bars anywhere in §8.

**Consistency checks run against this table** (all pass, all 19 test metrics per AD dataset,
3 for ETTh1): the curve at α = the run's own scale reproduces the matrix `merged` test row
exactly; the curve at α = 0 reproduces the `base` row exactly; and GRR recomputed by hand
matches `result.json` to 1e-9.
