"""§1.40b's script of record: TIES and DARE at settings other than the reference defaults.

    python -m incremental_ad.analysis.merge_sensitivity_report --runs_root $RUNS_ROOT \\
        --sensitivity_dir $WORK/merge_sensitivity --baselines_dir $WORK/merge_baselines \\
        --floors results_archive/audit/floors.csv --out $OUT/merge_sensitivity

Same protocol as §1.40 (`merge_baselines_report`), imported rather than re-implemented: alpha
selected per seed on merged-val `forecast/mse` over the same grid, test read at that alpha, the
floor decision rule, and the delta **paired** against task arithmetic on the same seeds. TA and
the default setting of each rule (TIES k = 0.2, DARE p = 0.7) are read from §1.40's own grid
(`--baselines_dir`), so the default row here is §1.40's number, not a re-run of it.

The variants are read from their own directory and their tags carry the setting
(`ties-k0.5_a…`, `dare-p0.9_a…`), which `merge_baselines_report` never globs — so a non-default
setting cannot leak into §1.40's table.

Predictions P5/P6 were registered in EXPERIMENTS.md §1.40b before the runs; this script scores
them mechanically from the table and prints the outcome, so the verdict in the prose is a copy.
"""

import argparse
import csv
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

from incremental_ad.analysis.merge_baselines_report import (
    FLOOR_LABEL, higher_is_better, load_floors, select, verdict,
)
from incremental_ad.analysis.remerge_provenance import load_result

log = logging.getLogger("merge_sensitivity_report")

METRIC = "forecast/mse"
CELLS = (("ETTh1", "selalpha_etth1_n3"), ("ETTh2", "etth2_merge_n3"),
         ("ETTm2", "ettm2_merge_n3"), ("exchange", "selalpha_exchange_n3"),
         ("PSM-forecast", "adfc2_psm_merge_n3"))
N = 3
# (rule, setting label, tag prefix, directory key). Defaults live in §1.40's grid.
SETTINGS = (("ties", 0.1, "ties-k0.1", "sensitivity"), ("ties", 0.2, "ties", "baselines"),
            ("ties", 0.5, "ties-k0.5", "sensitivity"), ("ties", 1.0, "ties-k1", "sensitivity"),
            ("dare", 0.3, "dare-p0.3", "sensitivity"), ("dare", 0.5, "dare-p0.5", "sensitivity"),
            ("dare", 0.7, "dare", "baselines"), ("dare", 0.9, "dare-p0.9", "sensitivity"))
DEFAULTS = {("ties", 0.2), ("dare", 0.7)}

FIELDS = ["dataset", "n_segments", "metric", "rule", "setting", "is_default", "n_seeds",
          "floor_pct", "value", "value_sd", "ta_value", "delta_pct", "delta_paired_sd",
          "margin_ratio", "verdict", "alpha_values", "alpha_at_edge", "source_experiment"]
PREDICTION_FIELDS = ["prediction", "statement", "observed", "outcome"]


def read_grid(folder: Path, prefix: str) -> dict[float, dict]:
    """alpha -> payload for one (run, tag prefix), newest alpha grid only."""
    by_grid: dict[tuple, dict[float, dict]] = defaultdict(dict)
    newest: dict[tuple, str] = {}
    for tag_dir in sorted(folder.glob(f"{prefix}_a*")) if folder.is_dir() else []:
        payload, _why = load_result(tag_dir / "result.json", require_metric=f"test/{METRIC}")
        if payload is None:
            continue
        key = tuple(payload["alpha_grid"])
        by_grid[key][float(payload["alpha"])] = payload
        newest[key] = max(newest.get(key, ""), payload.get("completed_at", ""))
    if not newest:
        return {}
    grid = max(newest, key=newest.get)
    points = by_grid[grid]
    return points if set(points) == set(grid) else {}


def build(runs_root: Path, dirs: dict[str, Path], floors: dict) -> list[dict]:
    rows = []
    for dataset, experiment in CELLS:
        group = runs_root / experiment
        runs = sorted(p for p in group.iterdir()
                      if (p / "merged" / "checkpoints" / "best.pt").is_file()) \
            if group.is_dir() else []
        floor = floors.get((FLOOR_LABEL.get(dataset, dataset), METRIC))
        ta: dict[str, float] = {}
        chosen: dict[tuple, dict[str, tuple]] = defaultdict(dict)
        for run in runs:
            name = f"{experiment}__{run.name}"
            points = read_grid(dirs["baselines"] / name, "ta")
            picked = select(points, METRIC, True) if points else None
            if picked:
                ta[str(points[picked[0]]["seed"])] = picked[1]
            for rule, setting, prefix, where in SETTINGS:
                points = read_grid(dirs[where] / name, prefix)
                picked = select(points, METRIC, True) if points else None
                if picked:
                    edge = picked[0] in (min(points), max(points))
                    chosen[(rule, setting)][str(points[picked[0]]["seed"])] = (
                        picked[0], picked[1], edge)
        for rule, setting, _prefix, _where in SETTINGS:
            mine = chosen.get((rule, setting), {})
            seeds = sorted(set(mine) & set(ta))
            if not seeds:
                log.warning("[coverage] %s %s=%g: no seed shared with TA", dataset, rule, setting)
                continue
            values = [mine[s][1] for s in seeds]
            ta_value = st.fmean(ta[s] for s in seeds)
            value = st.fmean(values)
            delta = 100.0 * (value - ta_value) / ta_value
            diffs = [100.0 * (mine[s][1] - ta[s]) / ta_value for s in seeds]
            rows.append({
                "dataset": dataset, "n_segments": N, "metric": METRIC, "rule": rule,
                "setting": setting, "is_default": (rule, setting) in DEFAULTS,
                "n_seeds": len(seeds), "floor_pct": floor if floor is not None else "",
                "value": value, "value_sd": st.stdev(values) if len(values) > 1 else 0.0,
                "ta_value": ta_value, "delta_pct": delta,
                "delta_paired_sd": st.stdev(diffs) if len(diffs) > 1 else 0.0,
                "margin_ratio": abs(delta) / floor if floor else "",
                "verdict": verdict(delta, floor, higher_is_better(METRIC)),
                "alpha_values": " ".join(f"{mine[s][0]:g}" for s in seeds),
                "alpha_at_edge": any(mine[s][2] for s in seeds),
                "source_experiment": experiment})
    return rows


def score_predictions(rows: list[dict]) -> list[dict]:
    at = {(r["dataset"], r["rule"], r["setting"]): r for r in rows}
    datasets = [d for d, _ in CELLS]
    out = []

    # P5: TIES's loss vs TA shrinks monotonically with density, and k = 1.0 ties on >= 3 of 5.
    monotone = []
    for d in datasets:
        deltas = [at[(d, "ties", k)]["delta_pct"] for k in (0.1, 0.2, 0.5, 1.0)
                  if (d, "ties", k) in at]
        if len(deltas) == 4:
            monotone.append(all(a >= b for a, b in zip(deltas, deltas[1:])))
    ties_k1 = sum(at[(d, "ties", 1.0)]["verdict"] == "tie" for d in datasets
                  if (d, "ties", 1.0) in at)
    ok = len(monotone) == 5 and all(monotone) and ties_k1 >= 3
    out.append({"prediction": "P5",
                "statement": "TIES delta vs TA falls monotonically in density on every dataset, "
                             "and density 1.0 ties TA on >= 3 of 5",
                "observed": f"monotone on {sum(monotone)} of {len(monotone)}; "
                            f"k=1.0 ties on {ties_k1} of 5",
                "outcome": "confirmed" if ok else "refuted"})

    # P6: DARE ties TA at p = 0.3 and 0.5 on >= 4 of 5, and loses at p = 0.9 on >= 2.
    low = sum(all(at.get((d, "dare", p), {}).get("verdict") == "tie" for p in (0.3, 0.5))
              for d in datasets)
    high = sum(at.get((d, "dare", 0.9), {}).get("verdict") == "worse" for d in datasets)
    out.append({"prediction": "P6",
                "statement": "DARE ties TA at p = 0.3 and 0.5 on >= 4 of 5, and is worse at "
                             "p = 0.9 on >= 2",
                "observed": f"ties at both low rates on {low} of 5; worse at 0.9 on {high}",
                "outcome": "confirmed" if low >= 4 and high >= 2 else "refuted"})
    return out


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d row(s))", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--sensitivity_dir", type=Path, required=True)
    parser.add_argument("--baselines_dir", type=Path, required=True)
    parser.add_argument("--floors", type=Path, default=Path("results_archive/audit/floors.csv"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in (args.sensitivity_dir, args.baselines_dir):
        if not path.is_dir():
            raise SystemExit(f"{path} does not exist")

    rows = build(args.runs_root, {"sensitivity": args.sensitivity_dir,
                                  "baselines": args.baselines_dir}, load_floors(args.floors))
    expected = len(CELLS) * len(SETTINGS)
    if len(rows) != expected:
        raise SystemExit(f"built {len(rows)} of {expected} cells — refusing a partial table")
    for r in rows:
        log.info("%-12s %-4s %-4g%s %+8.2f%%  (%.2fx floor, paired sd %.2f)  %-5s alpha %s%s",
                 r["dataset"], r["rule"], r["setting"], "*" if r["is_default"] else " ",
                 r["delta_pct"], r["margin_ratio"] or 0, r["delta_paired_sd"], r["verdict"],
                 r["alpha_values"], "  EDGE" if r["alpha_at_edge"] else "")
    predictions = score_predictions(rows)
    for p in predictions:
        log.info("[%s] %s — %s: %s", p["prediction"], p["statement"], p["observed"],
                 p["outcome"].upper())
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out / "merge_sensitivity.csv", FIELDS, rows)
        write(args.out / "merge_sensitivity_predictions.csv", PREDICTION_FIELDS, predictions)


if __name__ == "__main__":
    main()
