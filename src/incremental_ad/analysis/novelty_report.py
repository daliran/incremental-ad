"""Per-step novelty (rho, new_k), alignment, and the merge-vs-sequential indicator test.

Reads `geometry_report` output (`sequential_overlap.csv`, `norms.csv`,
`geometry_summary.csv`) and, for the indicator test, a table of measured outcomes.

    # per-step trajectory for one or more geometry output directories
    python -m incremental_ad.analysis.novelty_report steps $OUT/geometry/noisefloor_etth/*

    # does rho predict which of merging / sequential wins?
    python -m incremental_ad.analysis.novelty_report indicator --outcomes outcomes.csv

**Why this file exists.** THEORY.md 5.4 reached three different verdicts about rho in one
sitting — "no skill", "significant", "no skill on the data that matters" — all correctly
computed from the same numbers, differing only in a threshold chosen by hand and in whether
saturated datasets were included. A threshold rule reported without its threshold, its fitting
procedure and its held-out score is not a result, so the procedure is pinned here.

Definitions:

**rho_k** — the share of step k's task vector already inside the span of its predecessors,
emitted per step by `geometry_report` as `sequential_overlap.csv`. The dataset-level rho quoted
in THEORY.md 5.2 is the **mean over steps** of these.

**new_k** — the part of step k that is genuinely new, as a fraction of the base model's norm:

    new_k = ||tau_k|| * sqrt(1 - rho_k) / ||theta_0||

exact by Pythagoras, no free parameters. Step 0 has no predecessors, so rho_0 is undefined and
new_0 = ||tau_0|| / ||theta_0||, which `norms.csv` records as `tau_over_base`.

**alignment** — ||sum tau|| / sum ||tau||, the quantity THEORY.md calls collinearity. Computed
here from the measured mean pairwise cosine c under equal norms:

    alignment ~= sqrt((1 + (n - 1) * c) / n)

This reproduces the directly measured ETTh1 values (0.824 / 0.737 / 0.633) to within 2-6%. It
is an approximation and labelled as one; the exact value needs the summed vector, which
`geometry_report` does not currently emit.

**The indicator test.** Given per-configuration (dataset, n, rho, winner), it reports:

- accuracy at a **fixed** threshold, and at the threshold **fitted** on the same data;
- the majority-class baseline, which is what any constant predictor achieves;
- a **permutation test** that re-fits the threshold on every shuffle, so the fitting itself is
  inside the null rather than free;
- **leave-one-dataset-out** accuracy, fitting on the other datasets — the only honest number,
  because configurations from one dataset are not independent of each other.

The outcomes CSV needs columns `dataset,n,rho,winner`, and optionally `decisive` (true/false)
so that configurations whose margin sits inside the reproducibility floor can be excluded.
"""

import argparse
import csv
import logging
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("novelty_report")

STEP_FIELDS = ["group", "run", "step", "rho", "tau_over_base", "new_k"]


def alignment_from_cosine(mean_cosine: float, n: int) -> float:
    """||sum tau|| / sum ||tau|| under the equal-norm approximation."""
    return math.sqrt((1.0 + (n - 1) * mean_cosine) / n)


def steps_for_run(run_dir: Path) -> list[dict]:
    """Per-step rho and new_k for one geometry output directory."""
    overlap = run_dir / "sequential_overlap.csv"
    norms = run_dir / "norms.csv"
    if not norms.exists():
        return []
    tau: dict[int, float] = {}
    with norms.open() as fh:
        for row in csv.DictReader(fh):
            tau[int(row["segment"])] = float(row["tau_over_base"])
    rho: dict[int, float] = {}
    if overlap.exists():
        with overlap.open() as fh:
            for row in csv.DictReader(fh):
                rho[int(row["merge_step"])] = float(row["rho"])
    out = []
    for step in sorted(tau):
        r = rho.get(step)
        # Step 0 has no predecessors: all of it is new.
        new = tau[step] * math.sqrt(1.0 - r) if r is not None else tau[step]
        out.append({"run": run_dir.name, "step": step, "rho": r,
                    "tau_over_base": tau[step], "new_k": new})
    return out


def aggregate_steps(run_dirs: list[Path]) -> list[dict]:
    """Mean over seeds of rho_k, ||tau_k||/||theta_0|| and new_k, per step."""
    per_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in run_dirs:
        for row in steps_for_run(run):
            for key in ("rho", "tau_over_base", "new_k"):
                if row[key] is not None:
                    per_step[row["step"]][key].append(row[key])
    rows = []
    for step in sorted(per_step):
        vals = per_step[step]
        rows.append({
            "step": step,
            "rho": round(st.mean(vals["rho"]), 4) if vals["rho"] else None,
            "tau_over_base": round(st.mean(vals["tau_over_base"]), 5),
            "new_k": round(st.mean(vals["new_k"]), 5),
            "n_seeds": len(vals["tau_over_base"]),
        })
    return rows



# Outcome table construction -------------------------------------------------------------

MERGE_BLOCK = "merged/test"


def build_outcomes(runs_root: Path, geometry_root: Path, pairs: list[tuple], floors: dict) -> list[dict]:
    """One row per (dataset, n): rho, the measured winner, the margin and whether it is decisive.

    `pairs` is (dataset, n, merge_experiment, seq_experiment, metric). The winner is whichever
    of merged / continual is better on `metric`; the margin is |merged - continual| / continual
    as a percentage, and `decisive` is that margin clearing the dataset's reproducibility floor.
    Configurations inside the floor are recorded rather than dropped, so the caller can decide.
    """
    import json as _json

    def mean_metric(experiment: str, block: str, metric: str) -> float | None:
        vals = []
        for result in sorted((runs_root / experiment).glob(f"*/{block}/result.json")):
            try:
                metrics = _json.loads(result.read_text()).get("metrics") or {}
            except (OSError, ValueError):
                continue
            if metric in metrics:
                vals.append(metrics[metric])
        return st.mean(vals) if vals else None

    def mean_rho(experiment: str) -> float | None:
        per: dict[int, list[float]] = defaultdict(list)
        for run in sorted((geometry_root / experiment).glob("*/")):
            path = run / "sequential_overlap.csv"
            if not path.exists():
                continue
            with path.open() as fh:
                for row in csv.DictReader(fh):
                    per[int(row["merge_step"])].append(float(row["rho"]))
        if not per:
            return None
        return st.mean([st.mean(v) for v in per.values()])

    rows = []
    for dataset, n, merge_exp, seq_exp, metric in pairs:
        merged = mean_metric(merge_exp, MERGE_BLOCK, metric)
        seq = mean_metric(seq_exp, f"continual_{n - 1}/test", metric)
        rho = mean_rho(merge_exp)
        if merged is None or seq is None or rho is None:
            log.warning("%s n=%s: missing data (merged=%s seq=%s rho=%s)",
                        dataset, n, merged is not None, seq is not None, rho is not None)
            continue
        higher = any(k in metric.lower() for k in
                     ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))
        winner = "merge" if ((merged > seq) if higher else (merged < seq)) else "sequential"
        margin = 100.0 * abs(merged - seq) / seq
        rows.append({"dataset": dataset, "n": n, "rho": round(rho, 4), "winner": winner,
                     "merged": round(merged, 6), "continual": round(seq, 6),
                     "margin_pct": round(margin, 3),
                     "decisive": margin > floors.get(dataset, 0.0)})
    return rows


def _fit_threshold(rows: list[dict]) -> float:
    """The threshold maximising accuracy on `rows` (midpoints between observed rho values)."""
    values = sorted(r["rho"] for r in rows)
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [1.0]
    return max(candidates, key=lambda t: _accuracy(rows, t))


def _accuracy(rows: list[dict], threshold: float) -> int:
    return sum(1 for r in rows
               if ("merge" if r["rho"] > threshold else "sequential") == r["winner"])


def indicator_test(rows: list[dict], fixed: float | None, trials: int, seed: int) -> dict:
    """Fixed-threshold, fitted-threshold, permutation and leave-one-dataset-out scores."""
    n = len(rows)
    majority = max(sum(1 for r in rows if r["winner"] == w) for w in ("merge", "sequential"))
    fitted = _fit_threshold(rows)
    observed = _accuracy(rows, fitted)

    rng = random.Random(seed)
    labels = [r["winner"] for r in rows]
    ge = 0
    for _ in range(trials):
        shuffled = rng.sample(labels, n)
        perm = [{**r, "winner": w} for r, w in zip(rows, shuffled)]
        # Re-fit on every shuffle, so the fitting is part of the null.
        if _accuracy(perm, _fit_threshold(perm)) >= observed:
            ge += 1

    lodo_hit = 0
    thresholds = []
    for dataset in sorted({r["dataset"] for r in rows}):
        train = [r for r in rows if r["dataset"] != dataset]
        test = [r for r in rows if r["dataset"] == dataset]
        if not train:
            continue
        th = _fit_threshold(train)
        thresholds.append(th)
        lodo_hit += _accuracy(test, th)

    return {
        "n": n,
        "majority_baseline": majority,
        "fixed_threshold": fixed,
        "fixed_accuracy": _accuracy(rows, fixed) if fixed is not None else None,
        "fitted_threshold": round(fitted, 4),
        "fitted_accuracy": observed,
        "permutation_p": round(ge / trials, 4),
        "lodo_accuracy": lodo_hit,
        "lodo_thresholds": f"{min(thresholds):.3f}-{max(thresholds):.3f}" if thresholds else "",
    }


def _load_outcomes(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("decisive", "").strip().lower() in ("false", "0", "no"):
                row["_decisive"] = False
            else:
                row["_decisive"] = True
            rows.append({"dataset": row["dataset"], "n": int(row["n"]),
                         "rho": float(row["rho"]), "winner": row["winner"].strip(),
                         "decisive": row["_decisive"]})
    return rows


# Alignment vs the alpha*.n product ------------------------------------------------------

ALIGN_FIELDS = ["dataset", "n", "mean_cosine", "n_seeds", "alignment", "alpha_star_n"]


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    denom = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return float("nan") if denom == 0 else sum(
        (a - mx) * (b - my) for a, b in zip(xs, ys)) / denom


def alignment_rows(geometry_csv: Path, scale_csvs: list[Path], spec_csv: Path) -> list[dict]:
    """Join measured cosines to alpha*.n, per the spec's (dataset, n) -> experiment mapping.

    The mapping is explicit rather than inferred because the geometry summary pools *every* run
    of a dataset -- including grid-search runs that never entered the alpha* estimate. Averaging
    those in silently changes the n=3 rows (ETTh1 0.695 -> 0.674, PSM 0.771 -> 0.745). Several
    geometry experiments may be given for one cell, separated by `|`, and are averaged.
    """
    cosines: dict[tuple[str, int], list[float]] = defaultdict(list)
    with geometry_csv.open() as fh:
        for row in csv.DictReader(fh):
            raw = row.get("mean_offdiag_cosine")
            if not raw:
                continue
            try:
                cosines[(row["experiment_name"], int(row["n_segments"]))].append(float(raw))
            except (ValueError, TypeError):
                continue

    products: dict[str, float] = {}
    for path in scale_csvs:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("alpha_excl_n"):
                    products[row["group"]] = float(row["alpha_excl_n"])

    rows = []
    with spec_csv.open() as fh:
        for spec in csv.DictReader(fh):
            n = int(spec["n"])
            seeds = [c for name in spec["geometry_experiment"].split("|")
                     for c in cosines.get((name, n), [])]
            product = products.get(spec["scale_group"])
            if not seeds or product is None:
                log.warning("%s n=%s: missing %s", spec["dataset"], n,
                            "geometry" if not seeds else f"scale group {spec['scale_group']!r}")
                continue
            cos = st.mean(seeds)
            rows.append({"dataset": spec["dataset"], "n": n,
                         "mean_cosine": round(cos, 4), "n_seeds": len(seeds),
                         "alignment": round(alignment_from_cosine(cos, n), 4),
                         "alpha_star_n": product})
    return rows


def correlation_split(rows: list[dict]) -> dict:
    """Pooled, within-dataset and per-dataset correlation of 1/alignment against alpha*.n.

    Reported separately because they disagree: pooling all points mixes the between-dataset
    component (which is noise here) into the within-dataset trend (which is the actual claim),
    so the pooled value drifts as datasets are added. Reporting only the pooled number is what
    produced the successive +0.44 / +0.31 / -0.02 readings of the same relationship.
    """
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)

    per_dataset = {}
    within_x: list[float] = []
    within_y: list[float] = []
    for dataset, group in by_ds.items():
        if len(group) < 2:
            continue
        xs = [1.0 / r["alignment"] for r in group]
        ys = [r["alpha_star_n"] for r in group]
        per_dataset[dataset] = _pearson(xs, ys)
        mx, my = st.mean(xs), st.mean(ys)
        within_x += [x - mx for x in xs]
        within_y += [y - my for y in ys]

    xs = [1.0 / r["alignment"] for r in rows]
    ys = [r["alpha_star_n"] for r in rows]
    ratios = [y / x for x, y in zip(xs, ys)]
    finite = [v for v in per_dataset.values() if v == v]
    return {
        "n_points": len(rows),
        "pooled_r": _pearson(xs, ys),
        "within_r": _pearson(within_x, within_y) if within_x else float("nan"),
        "per_dataset_r": per_dataset,
        "median_per_dataset_r": st.median(finite) if finite else float("nan"),
        "n_positive": sum(1 for v in finite if v > 0),
        "n_datasets": len(finite),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_align = sub.add_parser("alignment",
                             help="alignment vs alpha*.n — within- and between-dataset")
    p_align.add_argument("--geometry", type=Path, required=True,
                         help="geometry_summary.csv from geometry_report")
    p_align.add_argument("--scale", type=Path, nargs="+", required=True,
                         help="one or more scale_summary.csv from scale_report")
    p_align.add_argument("--spec", type=Path, required=True,
                         help="CSV: dataset,n,geometry_experiment,scale_group")
    p_align.add_argument("--out", type=Path)

    p_steps = sub.add_parser("steps", help="per-step rho / new_k from geometry output")
    p_steps.add_argument("run_dirs", nargs="+", type=Path)
    p_steps.add_argument("--out", type=Path)

    p_out = sub.add_parser("outcomes", help="build the outcomes table from runs + geometry")
    p_out.add_argument("--runs_root", type=Path, required=True)
    p_out.add_argument("--geometry_root", type=Path, required=True)
    p_out.add_argument("--spec", type=Path, required=True,
                       help="CSV: dataset,n,merge_experiment,seq_experiment,metric,floor_pct")
    p_out.add_argument("--out", type=Path, required=True)

    p_ind = sub.add_parser("indicator", help="does rho predict merge vs sequential?")
    p_ind.add_argument("--outcomes", type=Path, required=True)
    p_ind.add_argument("--threshold", type=float, default=0.35,
                       help="the fixed threshold to report alongside the fitted one")
    p_ind.add_argument("--trials", type=int, default=20000)
    p_ind.add_argument("--seed", type=int, default=0)
    p_ind.add_argument("--exclude", action="append", default=[],
                       help="drop a dataset (repeatable) — e.g. saturated ones")
    p_ind.add_argument("--decisive_only", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "alignment":
        rows = alignment_rows(args.geometry, args.scale, args.spec)
        if not rows:
            raise SystemExit("no (dataset, n) cell resolved — check --spec against the CSVs")
        stats = correlation_split(rows)
        log.info(f"{'dataset':>14}{'n':>3}{'cos':>8}{'seeds':>7}{'align':>8}{'a*.n':>8}")
        for row in rows:
            log.info(f"{row['dataset']:>14}{row['n']:>3}{row['mean_cosine']:>8.4f}"
                     f"{row['n_seeds']:>7}{row['alignment']:>8.3f}{row['alpha_star_n']:>8.2f}")
        log.info("\nper-dataset r (1/alignment vs alpha*.n):")
        for dataset, value in sorted(stats["per_dataset_r"].items()):
            log.info("  %-14s r=%+.2f", dataset, value)
        log.info("\n  pooled across %d points : r=%+.2f", stats["n_points"], stats["pooled_r"])
        log.info("  within (dataset-centred): r=%+.2f", stats["within_r"])
        log.info("  median per-dataset      : r=%+.2f  (positive in %d/%d)",
                 stats["median_per_dataset_r"], stats["n_positive"], stats["n_datasets"])
        log.info("  ratio alpha*.n / (1/alignment) spans %.2f-%.2f",
                 stats["ratio_min"], stats["ratio_max"])
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / "alignment_vs_scale.csv"
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ALIGN_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            summary = args.out / "alignment_correlation.csv"
            with summary.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["statistic", "value"])
                for key in ("n_points", "pooled_r", "within_r", "median_per_dataset_r",
                            "n_positive", "n_datasets", "ratio_min", "ratio_max"):
                    writer.writerow([key, round(stats[key], 4)
                                     if isinstance(stats[key], float) else stats[key]])
                for dataset, value in sorted(stats["per_dataset_r"].items()):
                    writer.writerow([f"r_{dataset}", round(value, 4)])
            log.info("wrote %s and %s", path, summary)
        return

    if args.command == "steps":
        rows = aggregate_steps(args.run_dirs)
        log.info(f"{'step':>5}{'rho':>9}{'|tau|/|th0|':>13}{'new_k':>10}{'seeds':>7}")
        for r in rows:
            rho_txt = f"{r['rho']:.3f}" if r["rho"] is not None else "—"
            log.info(f"{r['step']:>5}{rho_txt:>9}{r['tau_over_base']:>13.5f}"
                     f"{r['new_k']:>10.5f}{r['n_seeds']:>7}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / "novelty_steps.csv"
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["step", "rho", "tau_over_base", "new_k", "n_seeds"])
                w.writeheader()
                w.writerows(rows)
            log.info("wrote %s", path)
        return

    if args.command == "outcomes":
        pairs, floors = [], {}
        with args.spec.open() as fh:
            for row in csv.DictReader(fh):
                pairs.append((row["dataset"], int(row["n"]), row["merge_experiment"],
                              row["seq_experiment"], row["metric"]))
                floors[row["dataset"]] = float(row["floor_pct"])
        rows = build_outcomes(args.runs_root, args.geometry_root, pairs, floors)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "n", "rho", "winner", "merged",
                                               "continual", "margin_pct", "decisive"])
            w.writeheader(); w.writerows(rows)
        log.info("wrote %s (%d configurations, %d decisive)", args.out, len(rows),
                 sum(1 for r in rows if r["decisive"]))
        return

    rows = _load_outcomes(args.outcomes)
    if args.decisive_only:
        rows = [r for r in rows if r["decisive"]]
    for dataset in args.exclude:
        rows = [r for r in rows if r["dataset"] != dataset]
    if len(rows) < 3:
        raise SystemExit("too few configurations left to test")

    result = indicator_test(rows, args.threshold, args.trials, args.seed)
    n = result["n"]
    log.info("configurations: %d   (decisive_only=%s, excluded=%s)",
             n, args.decisive_only, args.exclude or "none")
    log.info("  majority-class baseline   %d/%d = %.0f%%",
             result["majority_baseline"], n, 100 * result["majority_baseline"] / n)
    if result["fixed_accuracy"] is not None:
        log.info("  fixed threshold %.3f      %d/%d = %.0f%%", result["fixed_threshold"],
                 result["fixed_accuracy"], n, 100 * result["fixed_accuracy"] / n)
    log.info("  fitted threshold %.3f     %d/%d = %.0f%%   permutation P = %.4f",
             result["fitted_threshold"], result["fitted_accuracy"], n,
             100 * result["fitted_accuracy"] / n, result["permutation_p"])
    log.info("  leave-one-dataset-out     %d/%d = %.0f%%   (thresholds %s)",
             result["lodo_accuracy"], n, 100 * result["lodo_accuracy"] / n,
             result["lodo_thresholds"])
    verdict = ("skill beyond a constant predictor"
               if result["lodo_accuracy"] > result["majority_baseline"]
               else "NO skill beyond a constant predictor")
    log.info("  -> %s", verdict)


if __name__ == "__main__":
    main()
