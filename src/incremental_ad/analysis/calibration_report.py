"""§1.42's script of record: choosing AD's α on a labelled calibration prefix.

    python -m incremental_ad.analysis.calibration_report --calibration_dir $WORK/merge_calibration \\
        --merge_baselines results_archive/audit/merge_baselines/merge_baselines.csv \\
        --floors results_archive/audit/floors.csv --out $OUT/calibration

Reads `remerge.py --calibration_split` outputs (one `result.json` per (run, rule, α), plus each
non-TA rule's `_dm` point) and applies the protocol registered in EXPERIMENTS.md §1.42 before the
runs. Every definition below is that section's, restated so the code can be audited against it:

- **Candidates** per rule: the grid {0.5, 1, 2, 3, 5} plus α = 0, the base model, read from TA's
  `ta_a0.00` (identical for every rule: θ₀ + 0·Δ).
- **Selectors**, each scored on the evaluation part: `calibrated` (best calibration AUROC, ties to
  the smaller α), `validation` (lowest merged-val `reconstruction/score_mean`, §1.12), `committed`
  (TA at α = 1), `distance_matched` (the rule's `_dm` point; TA's is α = 1), `oracle` (best
  evaluation AUROC), `base` (α = 0).
- **Recovered fraction** = (selected − base) ÷ (oracle − base) on seed means; blank when the oracle
  gain is inside the evaluation part's own floor.
- **Own floor** per (dataset, n, c) = sd over seeds of the base's evaluation AUROC, % of its mean.
- **Verdicts** are counts against the own floor, with the published floor carried beside it.
"""

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

RULES = ("ta", "dare", "ties", "iso_c", "tsv")
GRID = (0.5, 1.0, 2.0, 3.0, 5.0)
FRACTIONS = ("c05", "c10", "c20", "c30")
LABEL = {"Swat": "SWaT", "Psm": "PSM"}
SELECTORS = ("calibrated", "validation", "committed", "distance_matched", "oracle", "base")

SPLIT_FIELDS = ["dataset", "n_segments", "c", "cut_requested", "cut", "cut_moved_by",
                "calib_windows", "eval_windows", "dropped_windows", "calib_anomalous_windows",
                "eval_anomalous_windows", "calib_events", "eval_events"]
CELL_FIELDS = ["dataset", "n_segments", "c", "rule", "selector", "n_seeds", "eval_auroc",
               "eval_auroc_sd", "alpha_values", "own_floor_pct", "published_floor_pct",
               "gain_over_base_pct", "oracle_gain_pct", "recovered_fraction",
               "vs_validation", "vs_distance_matched"]
RANK_FIELDS = ["dataset", "n_segments", "c", "calibrated_order", "distance_matched_order",
               "spearman_rho", "matches"]
PRED_FIELDS = ["prediction", "statement", "observed", "outcome"]


def verdict(a: float, b: float, floor_pct: float) -> str:
    """a vs b (AUROC, up is better) against a floor in % of b."""
    delta = 100.0 * (a - b) / b
    if abs(delta) <= floor_pct:
        return "tie"
    return "better" if delta > 0 else "worse"


def spearman(x: list[str], y: list[str]) -> float:
    rank_y = {r: i for i, r in enumerate(y)}
    n = len(x)
    d2 = sum((i - rank_y[r]) ** 2 for i, r in enumerate(x))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def load(calibration_dir: Path):
    """{(dataset, n, seed): {rule: {alpha|'dm': metrics}}} plus split metadata."""
    runs = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for result in sorted(calibration_dir.glob("*/*/result.json")):
        payload = json.loads(result.read_text())
        run = Path(payload["source_run"])
        config = json.loads((run / "config.json").read_text())["args"]
        key = (LABEL[config["dataset"]], int(config["dataset_n_finetune_segments"]),
               int(payload["seed"]))
        tag = result.parent.name
        point = "dm" if tag.endswith("_dm") else float(payload["alpha"])
        runs[key][payload["baseline_rule"]][point] = payload["metrics"]
        meta[key] = payload["metrics"]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibration_dir", type=Path, required=True)
    parser.add_argument("--merge_baselines", type=Path, required=True)
    parser.add_argument("--floors", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.calibration_dir.is_dir():
        raise SystemExit(f"{args.calibration_dir} does not exist")

    published = {(r["dataset"], r["metric"]): float(r["floor_pct"])
                 for r in csv.DictReader(args.floors.open(encoding="utf-8"))
                 if r["role"] == "floor" and r["floor_pct"]}
    dm_value = {(r["dataset"], int(r["n_segments"]), r["rule"]): float(r["value"])
                for r in csv.DictReader(args.merge_baselines.open(encoding="utf-8"))
                if r["metric"] == "window_auroc"}
    runs = load(args.calibration_dir)

    # Completeness: every seed needs every rule's full grid, TA's base point and the dm points.
    complete = {}
    for key, rules in runs.items():
        ok = ("ta" in rules and 0.0 in rules["ta"]
              and all(r in rules and set(GRID) <= set(rules[r]) for r in RULES)
              and all("dm" in rules[r] for r in RULES if r != "ta"))
        if ok:
            complete[key] = rules
        else:
            print(f"[coverage] incomplete {key}: dropped")
    configs = sorted({(d, n) for d, n, _ in complete})

    split_rows, cells, rank_rows = [], [], []
    for dataset, n in configs:
        seeds = sorted(s for d, m, s in complete if (d, m) == (dataset, n))
        first = complete[(dataset, n, seeds[0])]["ta"][0.0]
        for c in FRACTIONS:
            split_rows.append({"dataset": dataset, "n_segments": n, "c": c, **{
                f: int(first[f"test_split/{c}/{f}"]) for f in SPLIT_FIELDS[3:]}})
        for c in FRACTIONS:
            calib, evaluate = f"test_split/{c}/calib_window_auroc", \
                f"test_split/{c}/eval_window_auroc"
            base_values = [complete[(dataset, n, s)]["ta"][0.0][evaluate] for s in seeds]
            base = st.fmean(base_values)
            own_floor = 100.0 * st.stdev(base_values) / base if len(seeds) > 1 else 0.0
            pub = published.get((dataset, "window_auroc"))
            chosen_at = {}
            for rule in RULES:
                picks = defaultdict(list)            # selector -> [(alpha, eval) per seed]
                for s in seeds:
                    points = dict(complete[(dataset, n, s)][rule])
                    points[0.0] = complete[(dataset, n, s)]["ta"][0.0]
                    cands = sorted(a for a in points if a != "dm")
                    best_c = max(cands, key=lambda a: (points[a][calib], -a))
                    best_v = min(cands, key=lambda a: (
                        points[a].get("val/reconstruction/score_mean", math.inf), a))
                    best_o = max(cands, key=lambda a: (points[a][evaluate], -a))
                    picks["calibrated"].append((best_c, points[best_c][evaluate]))
                    picks["validation"].append((best_v, points[best_v][evaluate]))
                    picks["oracle"].append((best_o, points[best_o][evaluate]))
                    picks["base"].append((0.0, points[0.0][evaluate]))
                    dm = points[1.0] if rule == "ta" else points["dm"]
                    picks["distance_matched"].append(("dm" if rule != "ta" else 1.0,
                                                      dm[evaluate]))
                    if rule == "ta":
                        picks["committed"].append((1.0, points[1.0][evaluate]))
                means = {k: st.fmean(v for _, v in vals) for k, vals in picks.items()}
                oracle_gain = 100.0 * (means["oracle"] - base) / base
                chosen_at[rule] = means["calibrated"]
                for selector in SELECTORS:
                    if selector not in picks:
                        continue
                    values = [v for _, v in picks[selector]]
                    mean = means[selector]
                    recovered = ((mean - base) / (means["oracle"] - base)
                                 if oracle_gain > own_floor else "")
                    cells.append({
                        "dataset": dataset, "n_segments": n, "c": c, "rule": rule,
                        "selector": selector, "n_seeds": len(values), "eval_auroc": mean,
                        "eval_auroc_sd": st.stdev(values) if len(values) > 1 else 0.0,
                        "alpha_values": " ".join(str(a) if a == "dm" else f"{a:g}"
                                                 for a, _ in picks[selector]),
                        "own_floor_pct": round(own_floor, 4),
                        "published_floor_pct": pub if pub is not None else "",
                        "gain_over_base_pct": 100.0 * (mean - base) / base,
                        "oracle_gain_pct": oracle_gain,
                        "recovered_fraction": recovered,
                        "vs_validation": (verdict(mean, means["validation"], own_floor)
                                          if selector == "calibrated" else ""),
                        "vs_distance_matched": (verdict(mean, means["distance_matched"],
                                                        own_floor)
                                                if selector == "calibrated" else "")})
            cal_order = sorted(RULES, key=lambda r: -chosen_at[r])
            dm_order = sorted(RULES, key=lambda r: -dm_value[(dataset, n, r)])
            rho = spearman(cal_order, dm_order)
            rank_rows.append({"dataset": dataset, "n_segments": n, "c": c,
                              "calibrated_order": " > ".join(cal_order),
                              "distance_matched_order": " > ".join(dm_order),
                              "spearman_rho": round(rho, 4), "matches": rho >= 0.8})

    predictions = score(cells, rank_rows)
    for p in predictions:
        print(f"[{p['prediction']}] {p['observed']}: {p['outcome'].upper()}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("calibration_split.csv", SPLIT_FIELDS, split_rows),
                                   ("calibration.csv", CELL_FIELDS, cells),
                                   ("calibration_ranking.csv", RANK_FIELDS, rank_rows),
                                   ("calibration_predictions.csv", PRED_FIELDS, predictions)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {args.out / name} ({len(rows)} rows)")


def pooled_recovery(cells, datasets, c, selector) -> float | None:
    """Σ(selected − base) / Σ(oracle − base) over configurations, TA, on seed means."""
    index = {(r["dataset"], r["n_segments"], r["selector"]): r["eval_auroc"]
             for r in cells if r["rule"] == "ta" and r["c"] == c}
    num = den = 0.0
    for (d, n, sel), _ in index.items():
        if sel != "base" or d not in datasets:
            continue
        base = index[(d, n, "base")]
        num += index[(d, n, selector)] - base
        den += index[(d, n, "oracle")] - base
    return num / den if den > 0 else None


def score(cells, rank_rows) -> list[dict]:
    out = []
    swat = {c: pooled_recovery(cells, {"SWaT"}, c, "calibrated") for c in FRACTIONS}
    swat_val = {c: pooled_recovery(cells, {"SWaT"}, c, "validation") for c in FRACTIONS}
    ok = all(swat[c] is not None and swat[c] >= 0.5 for c in ("c10", "c20", "c30"))
    out.append({"prediction": "P1",
                "statement": "TA, SWaT pooled: calibrated recovers >= 0.5 of the oracle gain "
                             "at every c >= 10%",
                "observed": "calibrated " + ", ".join(
                    f"{c}={'n/a' if swat[c] is None else f'{swat[c]:.3f}'}" for c in FRACTIONS)
                + " | validation " + ", ".join(
                    f"{c}={'n/a' if swat_val[c] is None else f'{swat_val[c]:.3f}'}"
                    for c in FRACTIONS),
                "outcome": "confirmed" if ok else "refuted"})
    both = {c: pooled_recovery(cells, {"SWaT", "PSM"}, c, "calibrated") for c in FRACTIONS}
    r = [both[c] for c in FRACTIONS]
    ok = (None not in r and r[0] < r[1] < r[2] and (r[3] - r[2]) < (r[1] - r[0]))
    out.append({"prediction": "P2",
                "statement": "TA pooled over 6 configurations: rises 5->10->20% and the 20->30% "
                             "rise is smaller than the 5->10% rise",
                "observed": ", ".join(f"{c}={'n/a' if v is None else f'{v:.3f}'}"
                                      for c, v in both.items()),
                "outcome": "confirmed" if ok else "refuted"})
    per_c = {c: sum(1 for r in rank_rows if r["c"] == c and r["matches"])
             for c in ("c10", "c20", "c30")}
    out.append({"prediction": "P3",
                "statement": "calibrated ranking matches §1.40's distance-matched ranking "
                             "(rho >= 0.8) on >= 4 of 6 configurations at every c >= 10%",
                "observed": ", ".join(f"{c}: {k} of 6" for c, k in per_c.items()),
                "outcome": "confirmed" if all(k >= 4 for k in per_c.values()) else "refuted"})
    return out


if __name__ == "__main__":
    main()
