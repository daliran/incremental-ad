"""§1.40's script of record: the supervisor's merge rules against task arithmetic, one protocol.

    python -m incremental_ad.analysis.merge_baselines_report --runs_root $RUNS_ROOT \\
        --remerge_dir $WORK/merge_baselines --floors results_archive/audit/floors.csv \\
        --derived results_archive/audit/derived.csv --out $OUT/merge_baselines

Reads the grid `remerge.py --baseline_rule` writes — one `result.json` per (run, rule, alpha) —
and does the one thing that script deliberately does not: **choose alpha**. Every rule, task
arithmetic included, is chosen by the same rule on the same grid, because the rules' alphas are
on different scales (`framework/merging/interference.py`) and a shared fixed alpha would favour
whichever rule it happened to suit.

- **Forecasting:** alpha minimises merged-val `forecast/mse` per seed — the `selection_metric()`
  task arithmetic's own `--pipeline_select_merge_scale_on_val` uses. Ties break toward the
  smaller alpha, as the pipeline's does. The headline is test at that alpha.
- **AD:** val selection is refused (§1.12), so every rule is read at the alpha where it travels
  **exactly as far from the base as task arithmetic does at its committed alpha = 1.0**
  (`remerge.py --distance_match_alpha`, tag `<rule>_dm`). Not at a shared alpha = 1.0: the rules
  put Delta on different scales, and a smoke run showed TIES — which averages task vectors where
  TA sums them — still improving at alpha = 3.0 on PSM, i.e. a shared alpha handed it roughly 1/n
  of TA's strength. Matching distance isolates direction, which is §1.38's lesson. Task
  arithmetic's own matched point is alpha = 1.0 by definition, so it is read off the grid. The
  grid's test-optimal alpha is carried as an upper bound.

**The comparison is against task arithmetic under the same protocol**, not against the run's
stored merge. Both columns come from the same grid and the same evaluation code, so a difference
between them cannot be an artefact of two paths. A seed with any grid point missing is excluded
from its cell and the cell reports how many seeds it kept; a selected alpha on the grid's edge is
flagged, since the true optimum may lie outside it.

The spread quoted beside each delta is **paired** — per seed, rule minus TA from the same
checkpoints — so it cancels the variance the two share, as in `remerge_closeout_report`.
"""

import argparse
import csv
import logging
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from incremental_ad.analysis.remerge_provenance import load_result, report_exclusions

log = logging.getLogger("merge_baselines_report")

RULES = ("ta", "dare", "ties", "iso_c", "tsv")
AD_ALPHA = 1.0
BORDERLINE_RATIO = 1.5
HIGHER_IS_BETTER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")

CELL_FIELDS = ["dataset", "n_segments", "metric", "rule", "protocol", "n_seeds", "n_seeds_expected",
               "floor_pct", "value", "value_sd", "ta_value", "delta_pct", "delta_paired_sd",
               "margin_ratio", "borderline", "verdict", "alpha_mean", "alpha_values",
               "alpha_at_edge", "oracle_value", "oracle_alpha_values", "distance",
               "ta_distance", "grr", "grr_ta", "grr_delta", "source_experiment"]
SEED_FIELDS = ["dataset", "n_segments", "seed", "rule", "metric", "alpha", "value",
               "oracle_alpha", "oracle_value", "distance", "source_experiment", "run"]
SUMMARY_FIELDS = ["rule", "protocol", "n_cells", "better", "tie", "worse", "mean_rank",
                  "mean_improvement_pct"]


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER_IS_BETTER)


def verdict(delta_pct: float, floor: float | None, up: bool) -> str:
    """`delta_pct` is always the raw change of the rule against TA; orientation is passed in."""
    if floor is None:
        return "no_floor"
    if abs(delta_pct) <= floor:
        return "tie"
    return "better" if (delta_pct > 0) == up else "worse"


def load_floors(path: Path) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8") as fh:
        return {(r["dataset"], r["metric"]): float(r["floor_pct"])
                for r in csv.DictReader(fh) if r["role"] == "floor" and r["floor_pct"]}


# `method_comparison_spec.csv` spells it "exchange"; floors.csv and the register say
# "exchange_rate". Named so the floor lookup cannot silently miss a dataset.
FLOOR_LABEL = {"exchange": "exchange_rate"}


def select(points: dict[float, dict], metric: str, forecasting: bool,
           matched: dict | None = None, rule: str = "ta"):
    """(alpha, test value) under the protocol, and (oracle alpha, oracle value).

    `matched` is the rule's distance-matched payload (AD only). For task arithmetic the matched
    alpha is 1.0 by construction — Delta IS sum(tau) — so its grid point is used and no separate
    evaluation exists.
    """
    up = higher_is_better(metric)
    test = {a: p["metrics"][f"test/{metric}"] for a, p in points.items()}
    oracle = (max if up else min)(sorted(test), key=lambda a: test[a])
    if forecasting:
        val = {a: p["metrics"].get("val/forecast/mse") for a, p in points.items()}
        if any(v is None for v in val.values()):
            return None
        # sorted() + min() keeps the FIRST minimum, i.e. ties go to the smaller alpha.
        chosen = min(sorted(val), key=lambda a: val[a])
        return chosen, test[chosen], oracle, test[oracle]
    if rule == "ta":
        if AD_ALPHA not in test:
            return None
        return AD_ALPHA, test[AD_ALPHA], oracle, test[oracle]
    if matched is None or matched["metrics"].get(f"test/{metric}") is None:
        return None
    return (float(matched["alpha"]), matched["metrics"][f"test/{metric}"],
            oracle, test[oracle])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path)
    parser.add_argument("--remerge_dir", type=Path)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--floors", type=Path, default=Path("results_archive/audit/floors.csv"))
    parser.add_argument("--derived", type=Path, default=Path("results_archive/audit/derived.csv"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if args.runs_root is None or args.remerge_dir is None:
        parser.error("--runs_root and --remerge_dir are required (or pass --self-test)")

    if not args.remerge_dir.is_dir():
        # A missing input must not look like "no results": the regeneration script's fallback
        # only fires on a non-zero exit, and an empty CSV exits 0.
        raise SystemExit(f"--remerge_dir {args.remerge_dir} does not exist")
    floors = load_floors(args.floors)
    derived = {}
    with args.derived.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("base") and row.get("joint"):
                derived[(row["experiment"], row["metric"])] = row
    with args.spec.open(encoding="utf-8") as fh:
        spec = list(csv.DictReader(fh))

    seed_rows, cells, reasons, scanned = [], [], defaultdict(int), 0
    gaps: dict[str, int] = defaultdict(int)
    ranks: dict[str, list[int]] = defaultdict(list)
    for entry in spec:
        dataset, n, metric = entry["dataset"], entry["n"], entry["metric"]
        experiment = entry["merge_experiment"]
        forecasting = metric.startswith("forecast/")
        group = args.runs_root / experiment
        runs = sorted(p for p in group.iterdir() if p.is_dir()) if group.is_dir() else []
        runs = [r for r in runs if (r / "merged" / "checkpoints" / "best.pt").is_file()]
        per_rule: dict[str, dict[str, tuple]] = defaultdict(dict)
        grid_seen = None
        for run in runs:
            for rule in RULES:
                folder = args.remerge_dir / f"{experiment}__{run.name}"
                # Group by the grid each payload was written under and keep the NEWEST grid. A
                # directory can hold points from an earlier grid (a smoke run, a re-run with a
                # wider range); `remerge.py` never deletes old tags, and mixing two grids made
                # every such seed look incomplete and drop out with only a warning.
                by_grid: dict[tuple, dict[float, dict]] = defaultdict(dict)
                newest: dict[tuple, str] = {}
                for tag_dir in sorted(folder.glob(f"{rule}_a*")) if folder.is_dir() else []:
                    scanned += 1
                    payload, why = load_result(tag_dir / "result.json",
                                               require_metric=f"test/{metric}")
                    if payload is None:
                        reasons[why] += 1
                        continue
                    key = tuple(payload["alpha_grid"])
                    by_grid[key][float(payload["alpha"])] = payload
                    newest[key] = max(newest.get(key, ""), payload.get("completed_at", ""))
                grid = max(newest, key=newest.get) if newest else None
                points = {a: p for a, p in by_grid.get(grid, {}).items() if a in (grid or ())}
                if grid is not None and len(by_grid) > 1:
                    gaps["older alpha grids present and ignored (newest kept)"] += 1
                # Seed-level gaps are counted apart from file-level exclusions: `reasons` counts
                # result FILES and is reported against `scanned`, so adding (run, rule) pairs to
                # it produced "355 of 46 results excluded" — two units in one ratio.
                if grid is None:
                    gaps["no results at all for this (run, rule)"] += 1
                    continue
                if set(points) != set(grid):
                    gaps["incomplete alpha grid (the seed is dropped from its cell)"] += 1
                    continue
                grid_seen = grid
                matched = None
                if not forecasting and rule != "ta":
                    matched, why = load_result(folder / f"{rule}_dm" / "result.json",
                                               require_metric=f"test/{metric}")
                    if matched is None:
                        gaps[f"no distance-matched point ({why})"] += 1
                        continue
                    # The AD comparison reads TA at AD_ALPHA and the rule at the alpha matched to
                    # TA's distance AT AD_ALPHA. Both halves are checked rather than assumed:
                    # every AD run committed 1.0 today, which makes the assumption true only by
                    # coincidence, and a hand re-run at another target would otherwise be read
                    # as distance-matched while comparing two different magnitudes (§1.38).
                    target = matched.get("distance_matched_to_ta_alpha")
                    committed = matched.get("committed_alpha")
                    if target is None or abs(float(target) - AD_ALPHA) > 1e-9:
                        gaps[f"matched to alpha {target}, not {AD_ALPHA}"] += 1
                        continue
                    if committed is None or abs(float(committed) - AD_ALPHA) > 1e-9:
                        gaps[f"run committed alpha {committed}, not {AD_ALPHA}"] += 1
                        continue
                chosen = select(points, metric, forecasting, matched, rule)
                if chosen is None:
                    gaps["missing the value the protocol selects on"] += 1
                    continue
                alpha, value, oracle_alpha, oracle_value = chosen
                seed = (matched if matched is not None else points[alpha]).get("seed")
                source_payload = matched if matched is not None else points[alpha]
                per_rule[rule][str(seed)] = (alpha, value, oracle_alpha, oracle_value,
                                             source_payload["distance_from_base"])
                seed_rows.append({
                    "dataset": dataset, "n_segments": n, "seed": seed, "rule": rule,
                    "metric": metric, "alpha": alpha, "value": value,
                    "oracle_alpha": oracle_alpha, "oracle_value": oracle_value,
                    "distance": source_payload["distance_from_base"],
                    "source_experiment": experiment, "run": run.name})

        # Rank on the seeds EVERY rule has, so no rule is ranked on a different sample of seeds
        # than another; and never on SWaT-forecast, whose 84% floor makes its ordering noise and
        # which every other summary figure here also excludes.
        common = set.intersection(*(set(per_rule[r]) for r in RULES)) if all(
            per_rule.get(r) for r in RULES) else set()
        if common and dataset != "SWaT-forecast":
            up_rank = higher_is_better(metric)
            means = {r: st.fmean(per_rule[r][s][1] for s in common) for r in RULES}
            for position, rule in enumerate(sorted(RULES, key=means.get, reverse=up_rank), 1):
                ranks[rule].append(position)

        ta = per_rule.get("ta", {})
        floor = floors.get((FLOOR_LABEL.get(dataset, dataset), metric))
        up = higher_is_better(metric)
        source = derived.get((experiment, metric))
        base = float(source["base"]) if source else None
        joint = float(source["joint"]) if source else None
        gap = (base - joint) if source else None
        for rule in RULES:
            mine = per_rule.get(rule, {})
            # Pair on seeds present for BOTH the rule and TA, so neither side averages over a
            # seed the other lacks.
            seeds = sorted(set(mine) & set(ta))
            if not seeds:
                continue
            values = [mine[s][1] for s in seeds]
            ta_values = [ta[s][1] for s in seeds]
            value, ta_value = st.fmean(values), st.fmean(ta_values)
            delta = 100.0 * (value - ta_value) / ta_value
            diffs = [100.0 * (mine[s][1] - ta[s][1]) / ta_value for s in seeds]
            alphas = [mine[s][0] for s in seeds]
            grr = grr_ta = None
            if gap:
                grr = (base - value) / gap
                grr_ta = (base - ta_value) / gap
            cells.append({
                "dataset": dataset, "n_segments": n, "metric": metric, "rule": rule,
                "protocol": ("val-selected alpha" if forecasting
                             else f"distance-matched to TA at alpha = {AD_ALPHA}"),
                "n_seeds": len(seeds), "n_seeds_expected": len(runs),
                "floor_pct": floor if floor is not None else "",
                "value": value, "value_sd": st.stdev(values) if len(values) > 1 else 0.0,
                "ta_value": ta_value, "delta_pct": delta,
                "delta_paired_sd": st.stdev(diffs) if len(diffs) > 1 else 0.0,
                "margin_ratio": abs(delta) / floor if floor else "",
                "borderline": bool(floor and floor < abs(delta) < BORDERLINE_RATIO * floor),
                "verdict": "control" if rule == "ta" else verdict(delta, floor, up),
                "alpha_mean": st.fmean(alphas),
                "alpha_values": " ".join(f"{a:g}" for a in alphas),
                "alpha_at_edge": bool(forecasting and grid_seen
                                      and any(a in (min(grid_seen), max(grid_seen))
                                              for a in alphas)),
                "oracle_value": st.fmean(mine[s][3] for s in seeds),
                "oracle_alpha_values": " ".join(f"{mine[s][2]:g}" for s in seeds),
                "distance": st.fmean(mine[s][4] for s in seeds),
                "ta_distance": st.fmean(ta[s][4] for s in seeds),
                "grr": round(grr, 6) if grr is not None else "",
                "grr_ta": round(grr_ta, 6) if grr_ta is not None else "",
                "grr_delta": round(grr - grr_ta, 6) if grr is not None else "",
                "source_experiment": experiment})

    # --- per-rule tally and mean rank over the configurations every rule covers ------------------
    summary = []
    for protocol in sorted({c["protocol"] for c in cells}) + ["all"]:
        for rule in RULES:
            mine = [c for c in cells if c["rule"] == rule
                    and (protocol == "all" or c["protocol"] == protocol)
                    and c["dataset"] != "SWaT-forecast"]
            if not mine:
                continue
            summary.append({
                "rule": rule, "protocol": protocol, "n_cells": len(mine),
                "better": sum(c["verdict"] == "better" for c in mine),
                "tie": sum(c["verdict"] == "tie" for c in mine),
                "worse": sum(c["verdict"] == "worse" for c in mine),
                "mean_rank": st.fmean(ranks[rule]) if protocol == "all" and ranks[rule] else "",
                # Signed so POSITIVE = the rule beats TA, and only within one protocol: an MSE
                # change (lower is better, ~10%) and an AUROC change (higher is better, ~0.5%)
                # have opposite signs AND different scales, so their average means nothing.
                "mean_improvement_pct": (
                    st.fmean(-c["delta_pct"] if not higher_is_better(c["metric"])
                             else c["delta_pct"] for c in mine)
                    if protocol != "all" else "")})

    for row in summary:
        log.info("[%-6s] %-42s %2d cells: better %2d  tie %2d  worse %2d  %s", row["rule"],
                 row["protocol"], row["n_cells"], row["better"], row["tie"], row["worse"],
                 (f'mean rank {row["mean_rank"]:.2f}' if row["mean_rank"] != ""
                  else f'mean improvement {row["mean_improvement_pct"]:+.2f}%'))
    log.info("(SWaT-forecast excluded from every row above, rank included)")
    edge = [c for c in cells if c["alpha_at_edge"]]
    if edge:
        log.warning("[edge] %d cell(s) selected an alpha on the grid's edge — the optimum may lie "
                    "outside it: %s", len(edge),
                    sorted({(c["dataset"], c["n_segments"], c["rule"]) for c in edge})[:10])
    report_exclusions(log, reasons, scanned)
    for why, count in sorted(gaps.items()):
        log.warning("[coverage] %d (run, rule) pair(s): %s", count, why)

    if not cells:
        raise SystemExit("no cell could be built — every (run, rule) was absent or incomplete; "
                         "refusing to write empty CSVs that would read as a result")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("merge_baselines.csv", CELL_FIELDS, cells),
                                   ("merge_baselines_per_seed.csv", SEED_FIELDS, seed_rows),
                                   ("merge_baselines_summary.csv", SUMMARY_FIELDS, summary)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            log.info("wrote %s (%d row(s))", args.out / name, len(rows))


def _self_test() -> None:
    def pts(vals, tests):
        return {a: {"metrics": {"val/forecast/mse": v, "test/forecast/mse": t}}
                for a, v, t in zip((0.1, 0.3, 1.0), vals, tests)}
    # Val picks the val minimum, not the test minimum — the headline must not peek at test.
    assert select(pts([3, 1, 2], [9, 8, 1]), "forecast/mse", True) == (0.3, 8, 1.0, 1)
    # Ties on val go to the SMALLER alpha, as the pipeline's selection does.
    assert select(pts([1, 1, 2], [5, 4, 3]), "forecast/mse", True)[0] == 0.1
    # AD reads the fixed alpha; its oracle takes the MAX because AUROC is up-is-better.
    ad = {a: {"metrics": {"test/window_auroc": t}} for a, t in ((0.5, 0.7), (1.0, 0.8), (2.0, 0.9))}
    # TA on AD: its distance-matched point IS alpha = 1.0, read off the grid.
    assert select(ad, "window_auroc", False) == (1.0, 0.8, 2.0, 0.9)
    assert select({0.5: ad[0.5]}, "window_auroc", False) is None, "AD TA without alpha=1.0 is void"
    # Any other rule on AD must be read at its OWN matched alpha, never at a shared 1.0.
    dm = {"alpha": 2.7, "metrics": {"test/window_auroc": 0.85}}
    assert select(ad, "window_auroc", False, dm, "ties") == (2.7, 0.85, 2.0, 0.9)
    assert select(ad, "window_auroc", False, None, "ties") is None, "no matched point is void"
    # Orientation: the same raw delta reads opposite ways for an error and an AUROC.
    assert verdict(-20, 5, up=False) == "better" and verdict(-20, 5, up=True) == "worse"
    assert verdict(3, 5, up=False) == "tie" and verdict(3, None, up=False) == "no_floor"
    log.info("self-test OK")


if __name__ == "__main__":
    main()
