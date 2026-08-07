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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

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
