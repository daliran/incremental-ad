"""Isolate the effect of the anomaly-mask span fix, without retraining anything.

    python scripts/verify_mask_span.py --runs_root $RUNS_ROOT --out $OUT/mask_span

`PsmForecastDataset` / `SwatForecastDataset` keep a test window only when every point it spans
is labelled normal. The span was written as `window_len + forecast_len`, but
`ForecastWindowDataset` already defines `context_len = window_len - forecast_len` and window i
covers exactly `[start, start + window_len)` — so the mask checked `forecast_len` points *beyond*
the window's own extent. The direction was conservative (every scored window really was clean,
so no published number was invalid) but it discarded legitimate windows and tilted the scored set
toward quieter stretches, because a clean window followed by an anomaly was dropped.

**Why this script rather than just re-running.** Re-running the experiments is what makes
`result.json` consistent with the code, and that is done separately — but a re-run also re-rolls
GPU placement, which §3.2 shows moves ETTh1 by up to 18.8% on its own. The re-run therefore
cannot attribute its own deltas: fix and hardware noise arrive together. This scores **the same
frozen checkpoints** under both masks, so the difference it reports is the fix and nothing else.
That is the same discipline §1.27b applies to §1.27 — when a design moves several things at once,
find the measurement that moves one.

One-purpose and one-time: it exists so §1.30's correction note is attributable, and it is
committed rather than run ad-hoc because no published number in this project may come from a
script that is not in the repository.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("verify_mask_span")

FIELDS = ["dataset", "experiment", "block", "n_seeds", "metric",
          "windows_old", "windows_new", "windows_recovered", "pct_more_windows",
          "mean_old_span", "mean_new_span", "pct_change"]


def scored_indices(labels, n_total: int, stride: int, span: int) -> list[int]:
    return [i for i in range(n_total) if labels[i * stride: i * stride + span].sum() == 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+",
                        default=["adfc2_psm_merge_n3_oldmask", "adfc2_swat_merge_n3_oldmask"],
                        help="experiments holding the checkpoints to re-score")
    parser.add_argument("--block", default="merged", help="which stage's checkpoint")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    from incremental_ad.analysis.oracle_router import _per_window_errors
    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.datasets.forecast_window import ForecastWindowDataset

    rows = []
    for experiment in args.experiments:
        runs = sorted(p for p in (args.runs_root / experiment).glob("*") if p.is_dir())
        old_means, new_means = {"mse": [], "mae": []}, {"mse": [], "mae": []}
        counts = None
        for run in runs:
            checkpoint = run / args.block / "checkpoints" / "best.pt"
            if not checkpoint.exists():
                continue
            dataset, model, _cfg, runner, _c = _load_run(run)
            model.to(runner.device)
            model.load_state_dict(load_model_state(checkpoint))

            labels = dataset._test_labels
            window_len, forecast_len = dataset._window_len, dataset._forecast_len
            stride = dataset._eval_stride
            full = ForecastWindowDataset(dataset._test_data, window_len, forecast_len, stride)
            n_total = len(full)
            keep_old = scored_indices(labels, n_total, stride, window_len + forecast_len)
            keep_new = scored_indices(labels, n_total, stride, window_len)
            counts = (len(keep_old), len(keep_new), n_total)

            # Score every window once, then aggregate over each mask's index set. Two subsets of
            # one pass — scoring twice would let batching differences leak into the comparison.
            loader = runner.loader_config.make_loader(full, shuffle=False)
            mse, mae = _per_window_errors(model, loader, runner.device)
            for key, errors in (("mse", mse), ("mae", mae)):
                old_means[key].append(float(errors[torch.tensor(keep_old)].mean()))
                new_means[key].append(float(errors[torch.tensor(keep_new)].mean()))
            log.info("  %s/%s scored", experiment, run.name)

        if counts is None:
            log.warning("%s: no checkpoints — skipped", experiment)
            continue
        old_n, new_n, total = counts
        for key, metric in (("mse", "forecast/mse"), ("mae", "forecast/mae")):
            a = sum(old_means[key]) / len(old_means[key])
            b = sum(new_means[key]) / len(new_means[key])
            rows.append({
                "dataset": experiment.split("_")[1], "experiment": experiment,
                "block": args.block, "n_seeds": len(old_means[key]), "metric": metric,
                "windows_old": old_n, "windows_new": new_n,
                "windows_recovered": new_n - old_n,
                "pct_more_windows": round(100.0 * (new_n - old_n) / old_n, 3),
                "mean_old_span": round(a, 6), "mean_new_span": round(b, 6),
                "pct_change": round(100.0 * (b - a) / a, 3),
            })
            log.info("%s %s: %.6f -> %.6f (%+.3f%%), %d -> %d windows of %d",
                     experiment, metric, a, b, rows[-1]["pct_change"], old_n, new_n, total)

    if args.out and rows:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "mask_span_delta.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
