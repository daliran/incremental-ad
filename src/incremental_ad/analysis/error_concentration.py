"""Why the forecasting noise floor is 6-14% when the AD floor is 0.07%.

    python -m incremental_ad.analysis.error_concentration \\
        --run_dirs $RUNS_ROOT/noisefloor_etth/* --label ETTh1 --out $OUT/concentration

Loads each seed's **baseline** model, scores the test set window by window, and asks where the
seed-to-seed variance in mean MSE actually comes from. Three quantities, each answering a
distinct candidate explanation:

**Concentration** -- the share of total squared error contributed by the worst 1% and 5% of
windows. MSE is an unbounded mean of squared errors, so if a handful of windows dominate it,
the mean inherits their instability. AUROC cannot behave this way: it is a rank statistic over
the whole test set, bounded in [0, 1], which is the leading hypothesis for why the same code
gives a 0.07% floor on anomaly detection and 8.76% on ETTh1.

**Trimmed floor** -- the floor recomputed after dropping the worst 1% / 5% of windows *per
seed*. If the floor collapses under trimming, the variance lives in the tail and the headline
number is a statement about a few hard windows rather than about model quality. If it barely
moves, the models genuinely differ everywhere and the tail is a red herring. This is the
decisive test, and it is cheap.

**Cross-seed agreement** -- the correlation between per-window errors of different seeds. High
correlation means every seed finds the same windows hard (the difficulty is in the data, and
the variance is in a shared scale factor); low correlation means seeds fail on *different*
windows, which points at optimisation noise -- different local optima, different early-stopping
epochs -- rather than at the data.

Forecasting only: it needs a per-window error, which a ranking metric does not have.
"""

import argparse
import csv
import json
import logging
import statistics as st
from pathlib import Path

log = logging.getLogger("error_concentration")

FIELDS = ["label", "dataset", "n_seeds", "n_windows", "metric",
          "mean", "sd", "floor_pct",
          "top1pct_share", "top5pct_share",
          "trimmed1_mean", "trimmed1_floor_pct", "trimmed5_mean", "trimmed5_floor_pct",
          "cross_seed_pearson"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--block", default="baseline",
                        help="which stage's checkpoint to load (default: baseline)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    from incremental_ad.analysis.oracle_router import _per_window_errors
    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state

    per_seed = []
    dataset_name = None
    for run in [p for p in args.run_dirs if p.is_dir()]:
        checkpoint = run / args.block / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            continue
        config = json.loads((run / "config.json").read_text())
        if "forecast" not in config["args"].get("task", "").lower():
            raise SystemExit("forecasting only — a ranking metric has no per-window error")
        dataset_name = config.get("dataset")
        dataset, model, _cfg, runner, _c = _load_run(run)
        model.to(runner.device)
        model.load_state_dict(load_model_state(checkpoint))
        loader = runner.loader_config.make_loader(dataset.get_test_dataset(), shuffle=False)
        mse, _mae = _per_window_errors(model, loader, runner.device)
        per_seed.append(mse)
        log.info("  %-10s mean=%.6f  max/mean=%.1f", run.name, mse.mean().item(),
                 (mse.max() / mse.mean()).item())

    if len(per_seed) < 2:
        raise SystemExit("need at least two seeds to estimate a floor")

    stacked = torch.stack(per_seed)
    n_windows = stacked.shape[1]

    def floor(values: list[float]) -> float:
        return 100.0 * st.stdev(values) / st.mean(values)

    means = [v.mean().item() for v in per_seed]
    shares, trimmed = {}, {}
    for pct in (1, 5):
        keep = n_windows - max(1, int(n_windows * pct / 100))
        # Share of the total contributed by the worst pct% of windows, per seed.
        shares[pct] = st.mean([float(v.sort().values[keep:].sum() / v.sum()) for v in per_seed])
        # Trim each seed's own worst windows, then recompute the floor.
        trimmed[pct] = [float(v.sort().values[:keep].mean()) for v in per_seed]

    # Do different seeds find the same windows hard?
    pairs = [(i, j) for i in range(len(per_seed)) for j in range(i + 1, len(per_seed))]
    cors = []
    for i, j in pairs:
        a, b = per_seed[i], per_seed[j]
        a, b = a - a.mean(), b - b.mean()
        denom = (a.norm() * b.norm()).item()
        if denom:
            cors.append(float((a @ b).item() / denom))

    row = {
        "label": args.label, "dataset": dataset_name, "n_seeds": len(per_seed),
        "n_windows": n_windows, "metric": "forecast/mse",
        "mean": round(st.mean(means), 6), "sd": round(st.stdev(means), 6),
        "floor_pct": round(floor(means), 3),
        "top1pct_share": round(shares[1], 4), "top5pct_share": round(shares[5], 4),
        "trimmed1_mean": round(st.mean(trimmed[1]), 6),
        "trimmed1_floor_pct": round(floor(trimmed[1]), 3),
        "trimmed5_mean": round(st.mean(trimmed[5]), 6),
        "trimmed5_floor_pct": round(floor(trimmed[5]), 3),
        "cross_seed_pearson": round(st.mean(cors), 4) if cors else "",
    }
    log.info("\n%s — %d seeds, %d test windows", args.label, len(per_seed), n_windows)
    log.info("  floor                     %.2f%%", row["floor_pct"])
    log.info("  worst 1%% of windows carry  %.1f%% of total squared error", 100 * shares[1])
    log.info("  worst 5%% of windows carry  %.1f%% of total squared error", 100 * shares[5])
    log.info("  floor after trimming 1%%    %.2f%%", row["trimmed1_floor_pct"])
    log.info("  floor after trimming 5%%    %.2f%%", row["trimmed5_floor_pct"])
    log.info("  cross-seed per-window r   %.3f", row["cross_seed_pearson"])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"error_concentration_{args.label}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
