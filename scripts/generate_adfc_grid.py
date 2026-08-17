"""Full evaluation grid for the AD datasets used as forecasting benchmarks, plus the two
published merge cells that were run at a fixed alpha.

    python scripts/generate_adfc_grid.py --runs_root $RUNS_ROOT --out $WORK/sweeps

**Why this exists.** PSM and SWaT are the only datasets here with a genuinely *separate* test
capture rather than a chronological cut of one series (EXPERIMENTS.md 1.27 shows how much the
cut matters). A first pass ran them at four pipelines x three seeds, but inherited a reference
config with `pipeline_merge_scale = 1.0` and no validation selection -- so the merged model sat
at three times the pre-declared alpha = 1/n and the result measured *overshoot*, which 1.16
already warns about. This grid re-runs them the way every other published merge cell is run.

Two parts:

**A. Consistency fix.** `noisefloor_etth` (ETTh1 n=3) merges at a fixed alpha = 1.0 and
`exch_incremental` (exchange n=3) at a fixed 0.5, while `segsweep_*` and `etth2/ettm2_merge_*`
select alpha on validation. 1.26 and 1.5b quote all of them in one column, and the two fixed-alpha
cells are the ones that look worst for merging -- 1.16 already excludes ETTh1 n=3 for exactly
this reason. These re-runs make the column internally consistent for the first time.

**B. Full AD-forecasting grid**, at the same level of detail as the forecasting datasets:
segment counts n in {2,3,5}, window budgets W in {1,2,3}, joint and sequential references, three
seeds throughout, alpha selected on validation over 25 scales. Plus a training-fraction sweep --
the analogue of 1.27's rolling origin, since the test capture is fixed and cannot be moved, so
what varies is how much of the training capture the model sees.

Every command is built from an existing run's `config.json` and validated against the real
parser, so hyperparameters cannot drift and a bad flag fails here rather than in the queue.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_rolling_origin import command, validate  # noqa: E402

log = logging.getLogger("generate_adfc_grid")

DATASETS = {"psm": "PsmForecastDataset", "swat": "SwatForecastDataset"}
SEEDS = (7, 42, 123)
SEGMENT_COUNTS = (2, 3, 5)
# baseline_fraction / baseline_use_fraction per window budget, from the published etth2 runs:
# the base always trains on the first 50% of the series, W periods are retained after it.
WINDOWS = {1: (0.9, 0.555556), 2: (0.8, 0.625), 3: (0.7, 0.714286)}
TRAIN_FRACTIONS = (0.6, 0.8)          # 1.0 is the main grid
REFERENCE = {"merge": "etth2_merge_n3", "sequential": "etth2_continual_n3",
             "joint": "etth2_gate_joint", "window": "etth2_window_W3"}


def load(runs_root: Path, experiment: str) -> dict:
    configs = sorted((runs_root / experiment).glob("*/config.json"))
    if not configs:
        raise SystemExit(f"no runs under {experiment} — cannot build commands from it")
    return dict(json.loads(configs[0].read_text())["args"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    refs = {role: load(args.runs_root, exp) for role, exp in REFERENCE.items()}
    part_a: list[str] = []
    part_b: list[str] = []

    # ── A. the two fixed-alpha cells, re-run with selection ──────────────────────────────
    for tag, source in (("etth1", "noisefloor_etth"), ("exchange", "exch_incremental")):
        base = load(args.runs_root, source)
        merge_ref = refs["merge"]
        # Keep the dataset and its shape; take the alpha-selection settings from the reference.
        base["pipeline_select_merge_scale_on_val"] = True
        base["pipeline_extra_merge_scales"] = merge_ref["pipeline_extra_merge_scales"]
        base["pipeline_merge_scale"] = merge_ref["pipeline_merge_scale"]
        for seed in SEEDS:
            cmd = command(base, {"experiment_name": f"selalpha_{tag}_n3", "seed": seed,
                                 "dataset_baseline_fraction": 0.5,
                                 "dataset_series_fraction": 1.0})
            validate(cmd)
            part_a.append(" ".join(cmd))

    # ── B. the AD-forecasting grid ───────────────────────────────────────────────────────
    for tag, dataset in DATASETS.items():
        for n in SEGMENT_COUNTS:
            for role in ("merge", "sequential"):
                base = dict(refs[role])
                base["dataset"] = dataset
                for seed in SEEDS:
                    cmd = command(base, {"experiment_name": f"adfc2_{tag}_{role}_n{n}",
                                         "seed": seed, "dataset_n_finetune_segments": n,
                                         "dataset_baseline_fraction": 0.5,
                                         "dataset_series_fraction": 1.0})
                    validate(cmd)
                    part_b.append(" ".join(cmd))

        base = dict(refs["joint"]); base["dataset"] = dataset
        for seed in SEEDS:
            cmd = command(base, {"experiment_name": f"adfc2_{tag}_joint", "seed": seed,
                                 "dataset_baseline_fraction": 1.0,
                                 "dataset_series_fraction": 1.0})
            validate(cmd)
            part_b.append(" ".join(cmd))

        for w, (fraction, use_fraction) in WINDOWS.items():
            base = dict(refs["window"]); base["dataset"] = dataset
            for seed in SEEDS:
                cmd = command(base, {"experiment_name": f"adfc2_{tag}_window_W{w}", "seed": seed,
                                     "dataset_baseline_fraction": fraction,
                                     "dataset_baseline_use_fraction": use_fraction,
                                     "dataset_series_fraction": 1.0})
                validate(cmd)
                part_b.append(" ".join(cmd))

        # Training-fraction sweep: the test capture is fixed, so this varies how much of the
        # training capture is used — the analogue of 1.27's rolling origin for a fixed test set.
        base = dict(refs["merge"]); base["dataset"] = dataset
        for fraction in TRAIN_FRACTIONS:
            for seed in SEEDS:
                cmd = command(base, {"experiment_name":
                                     f"adfc2_{tag}_merge_tf{str(fraction).replace('.', '')}",
                                     "seed": seed, "dataset_n_finetune_segments": 3,
                                     "dataset_baseline_fraction": 0.5,
                                     "dataset_series_fraction": fraction})
                validate(cmd)
                part_b.append(" ".join(cmd))

    for name, cmds in (("adfc_partA_consistency.sh", part_a), ("adfc_partB_grid.sh", part_b)):
        (args.out / name).write_text("\n".join(cmds) + "\n")
        log.info("wrote %s (%d commands, parser-validated)", args.out / name, len(cmds))


if __name__ == "__main__":
    main()
