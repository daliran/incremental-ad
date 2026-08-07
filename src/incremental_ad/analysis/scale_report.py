"""Everything about the merge scale alpha, computed from the merge-scale curves.

Reads only `merge_diagnostics/merge_scale_curve.csv` and `transfer_matrix.csv`. No model, no
GPU, no dataset.

    python -m incremental_ad.analysis.scale_report $RUNS_ROOT/*_diagnostics/* --out /tmp/scale
    python -m incremental_ad.analysis.scale_report ... \\
        --val_metric reconstruction/score_mean --test_metric window_auroc   # anomaly detection

**Why this file exists.** alpha* was the most error-prone quantity in the project: its
definition was never written down, and two "corrections" to published values were made against
reconstructions that pooled the validation columns differently from the pipeline. SWaT selects
the same alpha either way and so hid the ambiguity; PSM does not. Every alpha-derived number in
EXPERIMENTS.md now comes from here.

The curve must carry validation columns. Diagnostics run without `--pipeline_curve_include_val`
produce a test-only curve, from which none of this is computable; that is reported rather than
silently skipped.

Definitions, fixed here:

**alpha\\*** — the deployable choice. Per seed, pool the validation columns by window count and
take the argmin; then average those per-seed optima:

    alpha*(seed) = argmin_a  sum_c |D_c| * L(a, c) / sum_c |D_c|
    alpha*       = mean over seeds of alpha*(seed)

The pool includes **val_base**, the baseline's own slice, unless `--exclude_val_base`. That is
not a detail: at a 50% baseline it carries 50-68% of the weight, so alpha* is a
*retention-weighted* optimum by construction. Averaging the argmins rather than taking the
argmin of the seed-averaged curve is what puts published values like 0.53 and 0.375 off the
grid; both are emitted (`alpha_star` and `alpha_star_pooled`) so the difference is visible.

**alpha* . n** — the product whose near-constancy is the scale rule (THEORY.md 6.6). Reported
with val_base both in and out, because the flatness is partly an artefact of including it.

**Honest-alpha cost** — what choosing alpha on validation costs against choosing it on test:

    GRR(a) = (L_test(a) - L_test(0)) / (L_joint - L_test(0))
    cost   = 1 - GRR(alpha_val) / GRR(alpha_oracle)

`L_test(0)` is the curve at alpha = 0, which *is* the base model; `L_joint` is the `standard`
row of the transfer matrix. alpha_oracle is the test optimum.

**Pre-declared penalty** — the price of never tuning: the test metric at the grid point nearest
1/n against the test optimum. The comparator is the *test* optimum rather than the validation
choice, so this is an upper bound on what fixing alpha = 1/n costs.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("scale_report")

HIGHER_IS_BETTER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")

FIELDS = [
    "group", "n_segments", "n_seeds", "n_dropped_no_val", "val_base_weight_pct",
    "alpha_star", "alpha_star_pooled", "alpha_star_n",
    "alpha_excl_val_base", "alpha_excl_n",
    "alpha_oracle", "grr_val", "grr_oracle", "honest_alpha_cost_pct",
    "alpha_one_over_n", "penalty_one_over_n_pct",
]


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER_IS_BETTER)


def _rows(path: Path):
    with path.open() as fh:
        reader = csv.DictReader(fh)
        key = "column" if "column" in (reader.fieldnames or []) else "split"
        for row in reader:
            yield row, key


def _curve(run: Path, metric: str, family: str, include_val_base: bool):
    """scale -> [(value, weight)] for family in {"val", "test"}, one run."""
    path = run / "merge_diagnostics" / "merge_scale_curve.csv"
    if not path.exists():
        return {}
    acc: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row, key in _rows(path):
        if row["metric"] != metric:
            continue
        column = row[key]
        if family == "val":
            if not column.startswith("val"):
                continue
            if column == "val_base" and not include_val_base:
                continue
        elif column != "test":
            continue
        acc[float(row["merge_scale"])].append((float(row["value"]), float(row.get("n_windows") or 1)))
    return acc


def _pool(acc) -> dict[float, float]:
    """Window-weighted mean per scale — NOT a mean of per-column means."""
    return {a: sum(v * w for v, w in lst) / sum(w for _, w in lst) for a, lst in acc.items()}


def _argbest(curve: dict[float, float], higher: bool) -> float:
    return (max if higher else min)(curve, key=curve.get)


def _val_base_weight(run: Path) -> float | None:
    """What share of the pooled selection signal the baseline's own slice carries."""
    src = run / "merge_diagnostics" / "source.json"
    if not src.exists():
        return None
    try:
        columns = json.loads(src.read_text()).get("columns") or []
    except (OSError, ValueError):
        return None
    base = sum(c["n_windows"] for c in columns if c["name"] == "val_base")
    seg = sum(c["n_windows"] for c in columns
              if c["name"].startswith("val_") and c["name"] != "val_base")
    return 100.0 * base / (base + seg) if base + seg else None


def has_val_columns(run: Path, metric: str) -> bool:
    """True when this run's curve carries validation columns for `metric`."""
    path = run / "merge_diagnostics" / "merge_scale_curve.csv"
    if not path.exists():
        return False
    for row, key in _rows(path):
        if row["metric"] == metric and row[key].startswith("val"):
            return True
    return False


def analyse(runs: list[Path], val_metric: str, test_metric: str) -> dict | None:
    """One row per group of runs (same configuration, different seeds).

    Runs whose curve lacks validation columns are dropped **before** anything is computed, not
    skipped per quantity. A group can legitimately hold both kinds — diagnostics produced
    without `--pipeline_curve_include_val` sit alongside ones produced with it — and letting
    the test half average over more seeds than the validation half silently compares two
    different populations.
    """
    val_higher, test_higher = higher_is_better(val_metric), higher_is_better(test_metric)

    usable = [r for r in runs if has_val_columns(r, val_metric)]
    dropped = len(runs) - len(usable)
    if not usable:
        return {"error": f"none of {len(runs)} run(s) carry validation columns for "
                         f"{val_metric!r} — rerun diagnostics with --pipeline_curve_include_val"}
    runs = usable

    per_seed_alpha, per_seed_alpha_excl = [], []
    pooled_val: dict[float, list[tuple[float, float]]] = defaultdict(list)
    pooled_test: dict[float, list[float]] = defaultdict(list)
    joint, weights, n_seg = [], [], set()

    for run in runs:
        acc_in = _curve(run, val_metric, "val", include_val_base=True)
        acc_out = _curve(run, val_metric, "val", include_val_base=False)
        if acc_in:
            per_seed_alpha.append(_argbest(_pool(acc_in), val_higher))
            for a, lst in acc_in.items():
                pooled_val[a].extend(lst)
        if acc_out:
            per_seed_alpha_excl.append(_argbest(_pool(acc_out), val_higher))
        for a, lst in _curve(run, test_metric, "test", True).items():
            pooled_test[a].extend(v for v, _ in lst)
        matrix = run / "merge_diagnostics" / "transfer_matrix.csv"
        if matrix.exists():
            for row, key in _rows(matrix):
                if row["model"] == "standard" and row[key] == "test" and row["metric"] == test_metric:
                    joint.append(float(row["value"]))
        w = _val_base_weight(run)
        if w is not None:
            weights.append(w)
        src = run / "merge_diagnostics" / "source.json"
        if src.exists():
            try:
                cols = json.loads(src.read_text()).get("columns") or []
                n_seg.add(sum(1 for c in cols
                              if c["name"].startswith("val_") and c["name"] != "val_base"))
            except (OSError, ValueError):
                pass

    if not per_seed_alpha:
        return {"error": "no validation columns in the curve — rerun diagnostics with "
                         "--pipeline_curve_include_val"}

    n = next(iter(n_seg)) if len(n_seg) == 1 else None
    alpha = st.mean(per_seed_alpha)
    alpha_pooled = _argbest(_pool(pooled_val), val_higher)
    alpha_excl = st.mean(per_seed_alpha_excl) if per_seed_alpha_excl else None

    out = {
        "n_segments": n, "n_seeds": len(runs), "n_dropped_no_val": dropped,
        "val_base_weight_pct": round(st.mean(weights), 1) if weights else None,
        "alpha_star": round(alpha, 4), "alpha_star_pooled": round(alpha_pooled, 4),
        "alpha_star_n": round(alpha * n, 3) if n else None,
        "alpha_excl_val_base": round(alpha_excl, 4) if alpha_excl is not None else None,
        "alpha_excl_n": round(alpha_excl * n, 3) if (alpha_excl is not None and n) else None,
    }

    test = {a: st.mean(v) for a, v in pooled_test.items()} if pooled_test else {}
    if test and 0.0 in test and joint:
        base_val, joint_val = test[0.0], st.mean(joint)
        oracle = _argbest(test, test_higher)
        out["alpha_oracle"] = oracle
        if joint_val != base_val:
            grr = lambda a: (test[a] - base_val) / (joint_val - base_val)  # noqa: E731
            # alpha* may sit off the grid (it is a mean of argmins); score its nearest point.
            near = min(test, key=lambda x: abs(x - alpha))
            gv, go = grr(near), grr(oracle)
            out["grr_val"] = round(gv, 4)
            out["grr_oracle"] = round(go, 4)
            if go:
                out["honest_alpha_cost_pct"] = round(100 * (go - gv) / go, 1)
        if n:
            inv = 1.0 / n
            near_inv = min(test, key=lambda x: abs(x - inv))
            best = test[oracle]
            pen = (best - test[near_inv]) / best if test_higher else (test[near_inv] - best) / best
            out["alpha_one_over_n"] = round(inv, 4)
            out["penalty_one_over_n_pct"] = round(100 * pen, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--val_metric", default="forecast/mse")
    parser.add_argument("--test_metric", default="forecast/mse")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    groups: dict[str, list[Path]] = defaultdict(list)
    for run in args.run_dirs:
        groups[run.parent.name].append(run)

    rows = []
    for name, runs in sorted(groups.items()):
        result = analyse(runs, args.val_metric, args.test_metric)
        if result is None or "error" in result:
            log.warning("%s: %s", name, (result or {}).get("error", "no data"))
            continue
        rows.append({"group": name, **result})

    if not rows:
        raise SystemExit("nothing computable — check --val_metric and that the curves carry "
                         "validation columns (--pipeline_curve_include_val)")

    width = max(len(r["group"]) for r in rows)
    log.info("\n%-*s %3s %6s %8s %8s %8s %8s %8s",
             width, "group", "n", "vb%", "alpha*", "a*.n", "a_excl", "excl.n", "1/n pen")
    for r in rows:
        f = lambda k, s="—": (f"{r[k]:.3f}" if isinstance(r.get(k), (int, float)) else s)  # noqa: E731
        log.info("%-*s %3s %6s %8s %8s %8s %8s %7s%%",
                 width, r["group"], r.get("n_segments") or "?",
                 f("val_base_weight_pct"), f("alpha_star"), f("alpha_star_n"),
                 f("alpha_excl_val_base"), f("alpha_excl_n"), f("penalty_one_over_n_pct"))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "scale_summary.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
