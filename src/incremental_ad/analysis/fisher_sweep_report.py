"""§1.32's Fisher-sample table from the archived re-merges — pure aggregation.

    python -m incremental_ad.analysis.fisher_sweep_report \\
        --remerge_dir $OUT/remerge --out $OUT/remerge/fisher_sweep_summary.csv

Until 2026-09-25 `fisher_sweep_summary.csv` had no generator in the repository, although the
checker binds six of its cells. Its inputs did exist: one `remerge.py --coefficient_source became`
result per (seed, fisher_batches, fisher_seed), with the λ* sidecar `became_lambdas.csv`. This
is the missing step, written to reproduce the archived table byte for byte.

Columns: `max_dev_from_uniform_pct` = 100 × the sidecar's `max_dev_from_uniform` (one value per
merge, repeated on every step); `weight_newest` = the last step's weight; `test_mse` from the
result. `fisher_batches = 100000` is the "all" row: a cap larger than the data, i.e. a full pass.
"""

import argparse
import csv
import json
from pathlib import Path

FIELDS = ["seed", "fisher_batches", "fisher_seed", "max_dev_from_uniform_pct", "weight_newest",
          "test_mse"]
BATCH_ORDER = {1024: 0, 256: 1, 64: 2, 100000: 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--remerge_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for result in sorted(args.remerge_dir.glob("*/became_fb*_fs*/result.json")):
        payload = json.loads(result.read_text())
        with (result.parent / "became_lambdas.csv").open(encoding="utf-8") as fh:
            steps = list(csv.DictReader(fh))
        rows.append({
            "seed": int(payload["seed"]), "fisher_batches": int(payload["fisher_batches"]),
            "fisher_seed": int(payload["fisher_seed"]),
            "max_dev_from_uniform_pct": round(100.0 * float(steps[-1]["max_dev_from_uniform"]), 1),
            "weight_newest": round(float(steps[-1]["weight"]), 4),
            "test_mse": round(float(payload["metrics"]["test/forecast/mse"]), 6)})
    if not rows:
        raise SystemExit(f"no became_fb*_fs* re-merges under {args.remerge_dir}")
    seed_order = {7: 0, 42: 1, 123: 2}
    rows.sort(key=lambda r: (seed_order.get(r["seed"], 9), BATCH_ORDER.get(r["fisher_batches"], 9),
                             r["fisher_seed"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
