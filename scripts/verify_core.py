"""Unit gates for the parts of the project nothing else tests.

    python scripts/verify_core.py

Before this script the only executable checks in the repo covered merging and analysis
(`verify_merge_*`, `verify_adaptive_lambda`, the `--self-test` flags, the table checker). The
data path, the model's masking, the metrics, the trainer, the evaluation runner and the results
storage had no test at all — they were verified only by reading them, and by the fact that real
runs produced plausible numbers, which is exactly the kind of evidence the mask-span bug survived
for a week. Each gate below checks one property **against an independent computation** (sklearn,
a hand-worked example, or an identity), never against the code under test.

Plain gates rather than pytest, matching the project's other `verify_*` scripts and adding no
dependency. Runs on CPU in well under a minute.
"""

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# ── A. splitting ───────────────────────────────────────────────────────────────────────────────
def test_splitting() -> None:
    from incremental_ad.framework.contracts.dataset import SplitConfig
    from incremental_ad.framework.datasets.splitting import (
        all_segment_ranges, baseline_range, finetune_ranges, val_tail_split,
    )
    print("A. SPLITTING")
    for n, bf, k in ((1000, 0.5, 3), (997, 0.4, 5), (13936, 0.5, 2)):
        cfg = SplitConfig(baseline_fraction=bf, n_finetune_segments=k, val_fraction=0.15)
        base = baseline_range(n, cfg)
        fts = finetune_ranges(n, cfg)
        anchor = int(n * bf)
        gate(f"n={n} baseline is [0, int(n*bf))", base == (0, anchor))
        contiguous = all(fts[i][1] == fts[i + 1][0] for i in range(len(fts) - 1))
        gate(f"n={n} {k} finetune ranges are contiguous and start at the anchor",
             contiguous and fts[0][0] == anchor and len(fts) == k)
        sizes = {e - s for s, e in fts}
        gate(f"n={n} finetune ranges are equal-sized and never pass n",
             len(sizes) == 1 and fts[-1][1] <= n and n - fts[-1][1] < k)
        gate(f"n={n} all_segment_ranges = baseline + finetunes",
             all_segment_ranges(n, cfg) == [base, *fts])
    (tr, va) = val_tail_split(100, 200, 0.15)
    gate("val is the TAIL of the range, and train + val tile it",
         tr == (100, 185) and va == (185, 200))
    gate("validate rejects baseline_fraction 0",
         raises(lambda: SplitConfig(baseline_fraction=0.0).validate(100), ValueError))
    gate("validate rejects val_fraction 1",
         raises(lambda: SplitConfig(val_fraction=1.0).validate(100), ValueError))


# ── B. windows ─────────────────────────────────────────────────────────────────────────────────
def test_windows() -> None:
    from incremental_ad.framework.datasets.forecast_window import ForecastWindowDataset
    from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset
    print("\nB. WINDOWS")
    data = torch.arange(50, dtype=torch.float32).reshape(25, 2)
    labels = torch.arange(25)
    ds = SlidingWindowDataset(data, window_len=6, stride=4, labels=labels)
    gate("sliding length = (T - W) // stride + 1", len(ds) == (25 - 6) // 4 + 1)
    window, lab = ds[2]
    gate("sliding window i covers [i*stride, i*stride + W)",
         torch.equal(window, data[8:14]) and torch.equal(lab, labels[8:14]))
    gate("a series shorter than the window yields no windows",
         len(SlidingWindowDataset(data[:3], 6, 1)) == 0)
    fds = ForecastWindowDataset(data, window_len=8, forecast_len=3, stride=1)
    full, future = fds[4]
    # CLAUDE.md's convention: assert the derived span against the dataset's own definition.
    gate("len(ds[i][0]) == window_len", len(full) == fds.window_len == 8)
    gate("future is exactly the last forecast_len steps of the window",
         torch.equal(future, data[4 + 5:4 + 8]) and fds.context_len == 5)


# ── C. metrics ─────────────────────────────────────────────────────────────────────────────────
def test_metrics() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    from incremental_ad.framework.evaluators._metrics import (
        best_f1_threshold, eval_classification, eval_event, eval_point_adjusted, find_segments,
    )
    from incremental_ad.framework.evaluators.ad_test_evaluator import AdTestEvaluator
    from incremental_ad.framework.evaluators.forecasting_evaluator import ForecastingEvaluator
    print("\nC. METRICS")
    gate("find_segments on a hand-worked example",
         [(int(s), int(e)) for s, e in find_segments(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1]))]
         == [(1, 3), (5, 6), (7, 9)])
    rng = np.random.default_rng(0)
    labels = (rng.random(500) < 0.2).astype(int)
    scores = rng.random(500) + labels * 0.5
    got = eval_classification(scores, labels, None)
    gate("AUROC equals sklearn", abs(got["auroc"] - roc_auc_score(labels, scores)) < 1e-12)
    gate("AUPRC equals sklearn",
         abs(got["auprc"] - average_precision_score(labels, scores)) < 1e-12)
    perfect = eval_classification(labels.astype(float), labels, None)
    gate("perfect separation gives AUROC 1 and F1 1",
         perfect["auroc"] == 1.0 and perfect["f1"] == 1.0)
    gate("best-F1 threshold separates a clean toy case",
         best_f1_threshold(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.8)

    # Point adjustment: detecting ONE point of a segment credits the whole segment.
    lab = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    sc = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    pa = eval_point_adjusted(sc, lab, threshold=0.5)
    gate("PA credits a whole segment from one detected point (recall 3/5, precision 1)",
         abs(pa["recall"] - 0.6) < 1e-12 and pa["precision"] == 1.0)
    ev = eval_event(np.array([0, 1, 0, 0, 0, 0, 0, 0, 1.0]), lab, threshold=0.5)
    gate("event metrics on a hand-worked case: 1 TP, 1 FN, 1 FP",
         (ev["tp"], ev["fn"], ev["fp"]) == (1, 1, 1) and abs(ev["f1"] - 0.5) < 1e-12)

    ev_ad = AdTestEvaluator()
    window_labels = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    ev_ad.update((torch.tensor([0.9, 0.1, 0.8, 0.2]), window_labels))
    out = ev_ad.compute()
    gate("AD window label = any anomaly in the window (max)",
         list(ev_ad._last_labels) == [1, 0, 1, 1])
    gate("AD point label = the window's LAST timestep", list(ev_ad._last_point_labels)
         == [1, 0, 0, 0])
    gate("AD evaluator reports the full suite", {"window_auroc", "pa_f1", "event_f1"} <= set(out))

    fe = ForecastingEvaluator()
    pred = torch.randn(4, 3, 2)
    target = torch.randn(4, 3, 2)
    fe.update((pred[:2], target[:2]))
    fe.update((pred[2:], target[2:]))
    m = fe.compute()
    mse = float(((pred - target) ** 2).mean())
    gate("forecast MSE/MAE/RMSE equal a direct computation over all batches",
         abs(m["forecast/mse"] - mse) < 1e-6 and abs(m["forecast/rmse"] - mse ** 0.5) < 1e-6
         and abs(m["forecast/mae"] - float((pred - target).abs().mean())) < 1e-6)
    gate("forecast selection metric is MSE, minimised",
         fe.selection_metric() == ("forecast/mse", "min"))


# ── D. model ───────────────────────────────────────────────────────────────────────────────────
def build_model(mode: str, *, instance_norm: bool = False, mask_ratio: float = 0.5,
                patch_len: int = 4, n_patches: int = 12, forecast_patches: int = 3):
    from incremental_ad.project.models.mae_tx.mae import (
        InferenceMode, MaeTx, MaeTxConfig, TrainingMode,
    )
    training = {"forecast": TrainingMode.CAUSAL_MASK, "ad": TrainingMode.RANDOM_MASK}[mode]
    cfg = MaeTxConfig(patch_len=patch_len, encoder_embed_dim=32, encoder_layers=2,
                      encoder_heads=2, decoder_embed_dim=16, decoder_layers=1, decoder_heads=2,
                      mask_ratio=mask_ratio, n_eval_passes=5, training_mode=training,
                      patch_norm=False, instance_norm=instance_norm)
    model = MaeTx(cfg)
    model.n_features = 3
    model.seq_len = n_patches * patch_len
    if mode == "forecast":
        model.forecast_patches = forecast_patches
        model.inference_mode = InferenceMode.FORECAST
    else:
        model.inference_mode = InferenceMode.AD
    model._build()
    model.eval()
    return model


def test_model() -> None:
    print("\nD. MODEL — masking and FUTURE LEAKAGE")
    torch.manual_seed(0)
    for instance_norm in (False, True):
        model = build_model("forecast", instance_norm=instance_norm)
        x = torch.randn(8, model.seq_len, 3)
        ctx = (model.n_patches - model.forecast_patches) * model.config.patch_len
        with torch.no_grad():
            before = model.score(x)
            poisoned = x.clone()
            poisoned[:, ctx:, :] = torch.randn_like(poisoned[:, ctx:, :]) * 1e3 + 500.0
            after = model.score(poisoned)
            moved = x.clone()
            moved[:, :ctx, :] += 1.0
            shifted = model.score(moved)
        gate(f"instance_norm={instance_norm}: poisoning the FUTURE leaves the forecast "
             f"bit-identical", torch.equal(before, after))
        gate(f"instance_norm={instance_norm}: changing the CONTEXT changes the forecast",
             not torch.equal(before, shifted))
        gate(f"instance_norm={instance_norm}: forecast shape is [B, horizon, F]",
             tuple(before.shape) == (8, model.forecast_patches * model.config.patch_len, 3))

    model = build_model("forecast", instance_norm=True)
    x = torch.randn(4, model.seq_len, 3)
    ctx = (model.n_patches - model.forecast_patches) * model.config.patch_len
    _, mean, std = model._instance_normalize(x)
    gate("instance-norm statistics are the CONTEXT's, not the window's",
         torch.allclose(mean, x[:, :ctx].mean(1, keepdim=True))
         and not torch.allclose(mean, x.mean(1, keepdim=True)))
    masked, visible = model._create_mask(4, torch.device("cpu"))
    gate("forecast training masks exactly the horizon patches",
         torch.equal(masked[0], torch.arange(model.n_patches - model.forecast_patches,
                                             model.n_patches)))

    ad = build_model("ad", mask_ratio=0.5)
    torch.manual_seed(1)
    masked, visible = ad._create_mask(16, torch.device("cpu"))
    union = torch.cat([masked, visible], dim=1).sort(dim=1).values
    gate("AD random mask: masked and visible partition every patch",
         torch.equal(union, torch.arange(ad.n_patches).expand(16, -1)))
    gate("AD random mask: masked count = int(n_patches * mask_ratio)",
         masked.shape[1] == int(ad.n_patches * 0.5))
    gate("AD random masks differ across rows (not one mask for the batch)",
         len({tuple(r.tolist()) for r in masked.sort(dim=1).values}) > 1)
    x = torch.randn(6, ad.seq_len, 3)
    torch.manual_seed(7)
    s1 = ad.score(x)
    torch.manual_seed(7)
    s2 = ad.score(x)
    torch.manual_seed(8)
    s3 = ad.score(x)
    gate("AD score is reproducible under the same seed", torch.equal(s1, s2))
    gate("AD score depends on the mask draw (different seed differs)", not torch.equal(s1, s3))
    gate("AD score is one finite value per window",
         s1.shape == (6,) and bool(torch.isfinite(s1).all()))
    gate("a pretext with fewer than 4 visible patches is refused",
         raises(lambda: build_model("ad", mask_ratio=0.9, n_patches=12), ValueError))


# ── E. evaluation runner ───────────────────────────────────────────────────────────────────────
def test_runner() -> None:
    from incremental_ad.framework.contracts.dataset import DataLoaderConfig
    from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset
    from incremental_ad.framework.evaluators.ad_test_evaluator import AdTestEvaluator
    from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
    print("\nE. EVALUATION RUNNER")
    model = build_model("ad")
    data = torch.randn(200, 3)
    labels = (torch.rand(200) < 0.1).long()
    ds = SlidingWindowDataset(data, model.seq_len, 1, labels=labels)
    runner = EvaluationRunner(DataLoaderConfig(batch_size=32, num_workers=0), device="cpu")
    a = runner.run(model, AdTestEvaluator(), ds, seed=5)
    b = runner.run(model, AdTestEvaluator(), ds, seed=5)
    c = runner.run(model, AdTestEvaluator(), ds, seed=6)
    gate("same eval seed -> identical AD metrics", a == b)
    gate("different eval seed -> different AD metrics (the mask draw is part of the score)",
         a["window_auroc"] != c["window_auroc"])


# ── F. trainer ─────────────────────────────────────────────────────────────────────────────────
def test_trainer() -> None:
    from incremental_ad.framework.contracts.dataset import DataLoaderConfig, Segment
    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.datasets.sliding_window import SlidingWindowDataset
    from incremental_ad.framework.trainers.standard_trainer import StandardTrainer
    print("\nF. TRAINER")
    torch.manual_seed(0)
    model = build_model("forecast")
    series = torch.sin(torch.linspace(0, 60, 1200)).unsqueeze(1).repeat(1, 3)
    segment = Segment(train=SlidingWindowDataset(series[:900], model.seq_len, 4),
                      val=SlidingWindowDataset(series[900:], model.seq_len, 4))
    loader = DataLoaderConfig(batch_size=32, num_workers=0)

    def trainer(**kw):
        base = dict(n_epochs=6, patience=2, optimizer="adamw", weight_decay=0.0,
                    learning_rate=1e-3, grad_clip=1.0, scheduler="constant", warmup_ratio=0.0,
                    loader_config=loader, device="cpu")
        return StandardTrainer(**{**base, **kw})

    with tempfile.TemporaryDirectory() as tmp:
        summary = trainer().fit(model, segment, checkpoint_dir=Path(tmp))
        best = load_model_state(Path(tmp) / "best.pt")
        live = model.state_dict()
        gate("fit() returns the model AT the best checkpoint",
             all(torch.equal(best[k], live[k].cpu()) for k in best))
        gate("best epoch is where the val loss was lowest",
             summary.best_val_loss is not None and summary.best_epoch is not None
             and summary.best_val_loss <= summary.final_val_loss + 1e-12)

    t0 = trainer(reg_lambda=0.0)
    t5 = trainer(reg_lambda=5.0)
    val_loader = loader.make_loader(segment.val, shuffle=False)
    torch.manual_seed(3)
    l0 = t0._compute_loader_loss(model, val_loader)
    torch.manual_seed(3)
    l5 = t5._compute_loader_loss(model, val_loader)
    gate("val loss excludes the L2-SP penalty (identical at lambda 0 and 5)", l0 == l5)

    frozen = build_model("forecast")
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    trainer(n_epochs=2, weight_decay=0.1, train_only=["decoder"]).fit(frozen, segment)
    after = frozen.state_dict()
    untouched = [k for k in before if "decoder" not in k and before[k].is_floating_point()]
    gate("train_only: every frozen tensor is bitwise unchanged under weight decay",
         all(torch.equal(before[k], after[k]) for k in untouched))
    gate("train_only: the trained part did move",
         any(not torch.equal(before[k], after[k]) for k in before if "decoder" in k))


# ── G. merging primitives ──────────────────────────────────────────────────────────────────────
def test_merging() -> None:
    from incremental_ad.framework.merging.task_vectors import (
        apply_task_vectors, merge_sequential, merge_task_arithmetic, task_vector,
    )
    print("\nG. MERGING PRIMITIVES")
    g = torch.Generator().manual_seed(0)
    base = {"w": torch.randn(4, 3, generator=g), "step": torch.tensor(7)}
    fts = [{"w": base["w"] + torch.randn(4, 3, generator=g), "step": torch.tensor(9)}
           for _ in range(3)]
    taus = [task_vector(base, ft) for ft in fts]
    gate("task vectors cover floating-point tensors only", set(taus[0]) == {"w"})
    merged = merge_task_arithmetic(base, fts, 0.4)
    expected = base["w"] + 0.4 * sum(ft["w"] - base["w"] for ft in fts)
    gate("TA merge = base + alpha * sum(tau)", torch.allclose(merged["w"], expected, atol=1e-6))
    gate("non-float tensors are copied from the base, never merged",
         int(merged["step"]) == 7)
    gate("merge_sequential with (decay 1, coefficient alpha) IS task arithmetic",
         torch.equal(merge_sequential(base, taus, [(1.0, 0.4)] * 3)["w"],
                     apply_task_vectors(base, taus, 0.4)["w"]))


# ── H. results storage ─────────────────────────────────────────────────────────────────────────
def test_storage() -> None:
    from incremental_ad.analysis.remerge_provenance import load_result, write_result
    from incremental_ad.framework.core.checkpoints import (
        load_checkpoint_metadata, load_model_state, save_model_state,
    )
    print("\nH. RESULTS STORAGE")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        state = {"a": torch.randn(3, 2), "b": torch.tensor([1, 2])}
        save_model_state(tmp / "x.pt", state, epoch=4, val_loss=0.5)
        back = load_model_state(tmp / "x.pt")
        gate("checkpoint round-trip preserves every tensor exactly",
             all(torch.equal(state[k], back[k]) for k in state))
        gate("checkpoint metadata survives the round-trip",
             load_checkpoint_metadata(tmp / "x.pt").get("epoch") == 4)

        (tmp / "r").mkdir()          # callers create the directory; write_result does not
        path = write_result(tmp / "r", {"metrics": {"test/x": 1.0}})
        payload, why = load_result(path, require_metric="test/x")
        gate("provenance: a written result loads, stamped with schema and fingerprint",
             payload is not None and "schema" in payload and "code_fingerprint" in payload)
        gate("provenance: no temporary file is left behind (atomic write)",
             sorted(p.name for p in (tmp / "r").iterdir()) == ["result.json"])
        gate("provenance: a missing metric is refused, with a reason",
             load_result(path, require_metric="test/y")[0] is None)
        (tmp / "old").mkdir()
        (tmp / "old" / "result.json").write_text(json.dumps({"metrics": {"test/x": 1}}))
        gate("provenance: an unversioned result is refused",
             load_result(tmp / "old" / "result.json")[0] is None)

    archive = REPO / "results_archive"
    rows = list(csv.DictReader((archive / "MANIFEST.csv").open(encoding="utf-8")))
    bad = [r["path"] for r in rows
           if not (archive / r["path"]).is_file()
           or hashlib.sha256((archive / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    gate(f"MANIFEST: all {len(rows)} archived files exist and match their SHA-256", not bad,
         f"{len(bad)} mismatched: {bad[:3]}")
    listed = {r["path"] for r in rows}
    on_disk = {p.relative_to(archive).as_posix() for p in archive.rglob("*")
               if p.is_file() and p.name != "MANIFEST.csv"}
    unlisted = sorted(on_disk - listed)
    gate("MANIFEST: every file in the archive is listed", not unlisted,
         f"{len(unlisted)} unlisted, e.g. {unlisted[:3]}")


def main() -> None:
    import os
    os.environ.setdefault("TQDM_DISABLE", "1")
    torch.set_num_threads(4)
    for test in (test_splitting, test_windows, test_metrics, test_model, test_runner,
                 test_trainer, test_merging, test_storage):
        try:
            test()
        except Exception as exc:                                    # noqa: BLE001
            gate(f"{test.__name__} crashed", False, f"{type(exc).__name__}: {exc}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} gate(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("all gates pass")


if __name__ == "__main__":
    main()
