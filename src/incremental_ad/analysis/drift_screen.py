"""Model-free drift screen — how much a series changes across its periods, before any training.

Downloads each candidate from `thuml/Time-Series-Library` and computes two statistics on the
raw values. No model, no checkpoints, no training.

    python -m incremental_ad.analysis.drift_screen --out /tmp/drift
    python -m incremental_ad.analysis.drift_screen --n_segments 3   # the older 3-way screen

**Why this file exists.** The drift figures underpin one of the project's live findings — that
routing headroom tracks drift while retention behaviour tracks shard size — and until
2026-08-07 they were produced by a script kept *outside* the repository, on `$WORK`. Numbers
that cannot be regenerated from the tree cannot be defended, and this one was carrying an
argument.

**The statistics.** The series is z-scored per feature, then split the same way the experiments
split it: a baseline block of `--baseline_fraction`, then `--n_segments` equal periods, after
removing the final `--test_fraction` as held-out test.

- **drift** — the standard deviation *across periods* of each feature's per-period mean,
  averaged over features. Zero means every period has the same level; larger means the periods
  sit at systematically different levels. Because the input is z-scored, it is already relative
  to each feature's own spread and so comparable across datasets.
- **KS** — the mean two-sample Kolmogorov–Smirnov statistic between the baseline block and each
  later period, averaged over periods and over the first 25 features. It answers a different
  question from drift: not "did the level move" but "did the whole distribution change". Both
  are reported because a series can drift in level without changing shape, and vice versa.

⚠️ **The segmentation is part of the definition.** `--n_segments 5` and `--n_segments 3` give
different numbers for the same series — ETTh1 reads 0.412 on the 5-way screen and 0.309 on the
3-way one. Values from different screens are **not comparable**, and EXPERIMENTS.md §0.1b keeps
them in separate columns for that reason. The 5-way screen is the current default because it
matches the largest segment count the experiments use.

Sampling for KS is capped at 3000 rows per block with a fixed seed, so the statistic is
deterministic but not exact; drift uses every row.
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("drift_screen")

DEFAULT_CANDIDATES = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "traffic",
    "electricity", "exchange_rate", "national_illness",
]

FIELDS = ["dataset", "rows", "features", "drift", "ks", "n_segments",
          "baseline_fraction", "test_fraction"]


def segment_statistics(values: np.ndarray, baseline_fraction: float, n_segments: int,
                       ks_features: int = 25, ks_sample: int = 3000,
                       seed: int = 0) -> tuple[float, float]:
    """(drift, KS) for one z-scored series split into a baseline block plus n periods."""
    from scipy.stats import ks_2samp

    values = np.asarray(values, dtype=np.float64)
    mean, sd = values.mean(0), values.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (values - mean) / sd

    anchor = int(len(z) * baseline_fraction)
    size = (len(z) - anchor) // n_segments
    blocks = [z[:anchor]] + [z[anchor + i * size: anchor + (i + 1) * size]
                             for i in range(n_segments)]

    # drift: spread of the per-block means, averaged over features
    per_block_means = np.stack([b.mean(0) for b in blocks])
    drift = float(per_block_means.std(0).mean())

    # KS: baseline against each later block, sub-sampled with a fixed seed
    rng = np.random.default_rng(seed)
    stats = []
    for block in blocks[1:]:
        idx_base = rng.choice(len(blocks[0]), min(ks_sample, len(blocks[0])), replace=False)
        idx_block = rng.choice(len(block), min(ks_sample, len(block)), replace=False)
        stats.append(np.mean([
            ks_2samp(blocks[0][idx_base, j], block[idx_block, j]).statistic
            for j in range(min(z.shape[1], ks_features))
        ]))
    return drift, float(np.mean(stats)) if stats else float("nan")


def load_series(name: str, test_fraction: float) -> np.ndarray:
    """The training portion of one HuggingFace series, numeric columns only."""
    import pandas as pd
    from datasets import load_dataset

    frame = load_dataset("thuml/Time-Series-Library", name)["train"].to_pandas()
    frame = frame.drop(columns=[c for c in frame.columns
                                if c.lower() in ("date", "timestamp")])
    values = frame.apply(pd.to_numeric, errors="coerce").ffill().bfill().values
    return values[: len(values) - int(len(values) * test_fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--n_segments", type=int, default=5)
    parser.add_argument("--baseline_fraction", type=float, default=0.5)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--min_rows", type=int, default=500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    rows = []
    for name in args.datasets:
        try:
            values = load_series(name, args.test_fraction)
        except Exception as exc:  # noqa: BLE001 - a missing dataset must not abort the screen
            log.warning("%-20s unavailable: %s: %s", name, type(exc).__name__, str(exc)[:70])
            continue
        if len(values) < args.min_rows:
            log.warning("%-20s too short (%d rows) — skipped", name, len(values))
            continue
        drift, ks = segment_statistics(values, args.baseline_fraction, args.n_segments)
        rows.append({"dataset": name, "rows": len(values), "features": values.shape[1],
                     "drift": round(drift, 4), "ks": round(ks, 4),
                     "n_segments": args.n_segments,
                     "baseline_fraction": args.baseline_fraction,
                     "test_fraction": args.test_fraction})
        log.info("%-20s rows=%7d  feat=%4d  drift=%.3f  KS=%.3f",
                 name, len(values), values.shape[1], drift, ks)

    if not rows:
        raise SystemExit("no datasets could be screened")

    log.info("\nRanked by drift (%d-way segmentation — not comparable to other segmentations):",
             args.n_segments)
    for r in sorted(rows, key=lambda r: -r["drift"]):
        log.info("  %-20s drift=%.3f  KS=%.3f  rows=%7d  features=%4d",
                 r["dataset"], r["drift"], r["ks"], r["rows"], r["features"])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"drift_screen_n{args.n_segments}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
