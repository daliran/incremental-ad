"""P1/P2/P4 for adaptive-λ sequential fine-tuning — §1.39's script of record.

    python -m incremental_ad.analysis.adaptive_lambda_report --runs_root $WORK/runs \\
        --floors results_archive/audit/floors.csv --out $OUT/adaptive_lambda

⚠️ **This is not BECAME and is never called that** (§1.39, `C26`'s discipline). Only Eq. 20's
coefficient is used, inside this project's sequential chain; the published method's gradient
projection is absent. Every column here says `adaptive`.

Three of the four registered predictions can be read off finished runs; each gets its own file.

- **P1 — ACC.** `continual_summary/result.json` already carries ACC, BWT and the base-slice
  retention ratio, for the adaptive chain and for the plain chain it is paired against. Lower is
  better: these are loss-shaped, the opposite orientation from the classification literature's
  ACC, and `forgetting_report.py`'s docstring is the authority for all three definitions.
- **P2 — λ*_t vs 1/t.** Straight from `adaptive_lambdas.csv`, which records both.
- **P4 — chain displacement.** `‖θ*_T − θ₀‖ / mean_i ‖θ̂_i − θ₀‖`, from the two distance columns
  the pipeline writes at the point each is *defined*. ⚠️ "implied α·n" is a merging-frame
  quantity with no meaning in the chain and is not computed.

**Pairing comes from the recorded checkpoint, not from a name.** Each adaptive run was launched
with `--pipeline_baseline_checkpoint` pointing into the plain chain it is compared against, so
the control is identified by resolving that path back to its run directory. Matching on an
experiment prefix would pair across seeds, and GPU placement alone moves a baseline by up to
18.8% at fixed seed (§3.2) — a paired comparison that silently crosses runs measures the
scheduler. Runs whose control cannot be resolved are reported by name, never dropped silently.

**A difference inside the reproducibility floor is a tie, not a result.** The floor is per
(dataset, metric) from `floors.csv` — seed variability is not a property of the dataset alone
(§1.9).

Training-free: reads `result.json`, `adaptive_lambdas.csv` and `config.json` only.
"""

import argparse
import csv
import json
import logging
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("adaptive_lambda_report")

# Run's `dataset` argument -> the label `floors.csv` and EXPERIMENTS.md use. Explicit rather
# than derived from the class name: `EtthForecastDataset` is ETTh1 and nothing in the string
# says so. A dataset missing here is an error, not a blank floor.
DATASET_LABELS = {
    "EtthForecastDataset": "ETTh1",
    "Etth2ForecastDataset": "ETTh2",
    "Ettm2ForecastDataset": "ETTm2",
    "ExchangeRateForecastDataset": "exchange_rate",
    "Psm": "PSM",
    "Swat": "SWaT",
}

# The Fisher estimator's batch size is part of the cell's identity, not a detail. `E[F_hat] =
# g^2 + sigma^2/B` depends on B, so two runs of the same configuration at different B are
# different measurements and must never be pooled. Carried as a column and as part of every
# group key, which is what makes the pooling impossible rather than merely discouraged.
# `lambda_source` leads because it is the coarsest distinction: `became` derives lambda from the
# Fisher, `one_over_t` imposes it and computes no Fisher at all. Two runs of the same
# configuration under the two rules are different methods, not different settings, and pooling
# them would make the P3 control disappear into the thing it controls.
ESTIMATOR_FIELDS = ["lambda_source", "fixed_lambda", "fisher_batch_size", "fisher_batches",
                    "fisher_samples"]

ACC_FIELDS = ["dataset", "n_segments", "metric"] + ESTIMATOR_FIELDS + ["n_seeds", "floor_pct",
              "acc_adaptive", "acc_adaptive_sd", "acc_plain", "acc_plain_sd",
              "acc_delta_pct", "margin_ratio", "borderline", "verdict",
              "own_spread_pct", "margin_over_own_spread",
              "bwt_adaptive", "bwt_plain", "base_slice_adaptive", "base_slice_plain",
              "fisher_samples_source"]
SEED_FIELDS = ["dataset", "n_segments", "seed", "metric"] + ESTIMATOR_FIELDS + [
               "adaptive_run", "plain_run",
               "acc_adaptive", "acc_plain", "bwt_adaptive", "bwt_plain",
               "base_slice_adaptive", "base_slice_plain", "fisher_samples_source"]
STEP_FIELDS = ["dataset", "n_segments", "seed"] + ESTIMATOR_FIELDS + [
               "step", "lambda_star", "one_over_t",
               "lambda_over_one_over_t", "fisher_num", "fisher_den", "lambda_term",
               "lambda_over_fisher", "floor_is_t",
               "fisher_star_num", "asymmetry_hat_over_star", "d_norm", "step_norm",
               "merged_dist_from_base", "unconstrained_dist_from_base",
               "fisher_samples_source", "fisher_hat_samples", "fisher_star_samples",
               "fisher_seed_samples"]
TEST_FIELDS = ["dataset", "n_segments", "metric", "higher_is_better"] + ESTIMATOR_FIELDS + [
    "n_seeds", "floor_pct", "final_step",
    "adaptive", "adaptive_sd", "plain", "plain_sd",
    "delta_pct", "margin_ratio", "borderline", "verdict",
    "own_spread_pct", "margin_over_own_spread", "fisher_samples_source"]
CURVE_FIELDS = ["dataset", "n_segments", "metric", "n_points", "n_seeds", "floor_pct",
                "lambdas", "values", "best_lambda", "best_value", "value_at_lambda_1",
                "interior_gain_pct", "interior_gain_over_floor", "borderline", "shape",
                "best_lambda_over_adaptive"]
ASYMMETRY_FIELDS = ["scope", "n_cells", "pearson_r", "slope", "r_squared"]

# Metrics whose direction is "up is better". Same list as `results_audit`, restated rather than
# imported so this module does not depend on that one's loading order.
HIGHER_IS_BETTER = ("auroc", "auprc", "f1", "precision", "recall", "accuracy")


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower() for k in HIGHER_IS_BETTER)
DIST_FIELDS = ["dataset", "n_segments", "seed"] + ESTIMATOR_FIELDS + [
               "n_steps", "chain_dist_from_base",
               "mean_unconstrained_dist", "ratio_p4", "fisher_samples_source"]


def _label(dataset: str) -> str:
    if dataset not in DATASET_LABELS:
        raise KeyError(
            f"no floor label for dataset {dataset!r} — add it to DATASET_LABELS (and check "
            f"analysis_specs/floor_spec.csv has a floor for it) rather than letting the "
            f"comparison run without a floor")
    return DATASET_LABELS[dataset]


def load_floors(path: Path | None) -> dict[tuple[str, str], float]:
    """(dataset, metric) -> floor_pct, from the `role == floor` rows only.

    `floors.csv` also carries `role == comparison` rows — the spread of the models a verdict
    compares, which the published floor does not use (§1.9). Reading those by accident would
    silently change every verdict in this file.
    """
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {(r["dataset"], r["metric"]): float(r["floor_pct"])
                for r in csv.DictReader(fh) if r["role"] == "floor" and r["floor_pct"] != ""}


# A gap under 1.5x its floor is a *borderline* call, not a clean one — the vault's
# convention, applied here so a 1.43x row cannot be quoted next to a 20x row as though the
# two were the same kind of statement. Below 1.0 is a tie; the band between is flagged.
BORDERLINE_RATIO = 1.5


def margin_ratio(delta_pct: float, floor_pct: float | None) -> float | str:
    """The gap as a multiple of its own floor — the quantity a verdict is really a threshold on."""
    if floor_pct is None or floor_pct == 0:
        return ""
    return abs(delta_pct) / floor_pct


def verdict(delta_pct: float, floor_pct: float | None, up_is_better: bool = False) -> str:
    """`better` / `tie` / `worse` for a signed relative change of adaptive against plain.

    `delta_pct` is always the raw change in the metric, so its sign means different things for
    an error and for an AUROC. The orientation is passed in rather than inferred here, and it
    is written into the CSV beside every row: a table with two orientations in it and no column
    saying which is which is a cell waiting to be read backwards.
    """
    if floor_pct is None:
        return "no_floor"
    if abs(delta_pct) <= floor_pct:
        return "tie"
    improved = delta_pct > 0 if up_is_better else delta_pct < 0
    return "better" if improved else "worse"


def estimator(config: dict) -> dict:
    """The Fisher estimator's identity: batch size, batch count and their product.

    `pipeline_fisher_batch_size` is absent from runs that predate the flag; those fell back to
    the loader's batch size, which is what the fallback below reads. Getting this wrong pools
    two different estimators into one cell.
    """
    source = config.get("pipeline_lambda_source", "none")
    # `fixed` runs differ ONLY in lambda, so without it every point of the grid collapses into
    # one cell and the curve the grid exists to draw disappears. Blank for the derived and
    # scheduled rules, where there is no such constant.
    fixed = (config.get("pipeline_fixed_lambda") if source == "fixed" else "")
    if source != "became":
        # No Fisher is computed on the imposed-lambda paths, so a batch size would be a number
        # with no referent. Written as 0 rather than left blank: a blank reads as missing data.
        return {"lambda_source": source, "fixed_lambda": fixed, "fisher_batch_size": 0,
                "fisher_batches": 0, "fisher_samples": 0, "fisher_samples_source": "n/a"}
    batch_size = config.get("pipeline_fisher_batch_size") or config["loader_batch_size"]
    batches = config["pipeline_fisher_batches"]
    return {"lambda_source": source, "fixed_lambda": fixed,
            "fisher_batch_size": int(batch_size), "fisher_batches": int(batches),
            "fisher_samples": int(batch_size) * int(batches),
            # INFERRED from the flags, and labelled so. Runs whose pipeline recorded the count
            # it actually used are relabelled "recorded" in main(), where their per-estimate
            # counts are read; old rows are never backfilled.
            "fisher_samples_source": "flags"}


def curve_shape(lambdas: list[float], values: list[float], boundary: float,
                floor_pct: float | None) -> tuple[str, float, float]:
    """`(shape, gain over the boundary in %, that gain as a multiple of the floor)`.

    λ = 1 **is** the plain chain — `_pullback` at λ = 1 leaves the model at θ̂_t, and
    `verify_adaptive_lambda.py` gate 1 asserts it reproduces the plain chain bitwise. So the
    plain chain's own result is the curve's boundary point and does not need to be re-run.

    `interior` means the best swept λ beats that boundary by more than the floor: a useful model
    exists strictly between θ\*_{t−1} and θ̂_t, which is adaptive-λ's premise (`C41`, §1.39d).
    `boundary` means it does not, and then **no** coefficient of the form A/(A+B) can win,
    because that form is strictly inside (0, 1) — the failure is the premise, not the rule.

    ⚠️ A curve is called `interior` on the **margin**, not on the ordering. Five noisy points
    almost always have an argmin away from the edge; requiring the margin to clear the floor is
    what stops "the minimum is not at λ = 0.9" from being reported as a finding.
    """
    best = min(range(len(values)), key=lambda i: values[i])
    gain = 100.0 * (boundary - values[best]) / boundary
    ratio = gain / floor_pct if floor_pct else float("nan")
    if floor_pct is not None and gain > floor_pct:
        shape = "interior"
    elif floor_pct is not None and gain < -floor_pct:
        shape = "boundary (every swept lambda loses)"
    else:
        shape = "flat within the floor"
    return shape, gain, ratio


def log_fit(pairs: list[tuple[float, float]]) -> tuple[float, float, int]:
    """`(pearson r, slope, n)` of log y on log x — for `excess ~ asymmetry^slope`.

    A slope of 1 means the at-a-minimum asymmetry accounts for the excess one-for-one; the
    correlation alone would not say that, since any monotone relation gives a high r.
    """
    n = len(pairs)
    mx, my = st.fmean(x for x, _ in pairs), st.fmean(y for _, y in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    return sxy / math.sqrt(sxx * syy), sxy / sxx, n


def summary_metrics(run: Path) -> dict[str, float]:
    path = run / "continual_summary" / "result.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text()) or {}).get("metrics") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def final_test_metrics(run: Path) -> tuple[int, dict[str, float]]:
    """`(step index, metrics)` from the LAST `continual_*/test/result.json`.

    ⚠️ **This is a different quantity from the ACC table above, and the difference is the point.**
    ACC is the mean over regimes of the *loss* the chain minimises; §1.12 established that
    reconstruction loss is blind to detection quality, so on AD it cannot answer the only
    question AD asks. These are the task's real test metrics — `window_auroc` and the rest of
    the suite — produced by the configurator's own test evaluator.

    The pipeline evaluates the test set **after** the pullback, so the model scored here is
    theta*_t, the chain's model, not theta_hat_t. Scoring the unconstrained model instead would
    compare a different model to the control and read as a result.
    """
    steps = [(int(p.parent.parent.name.split("_")[1]), p)
             for p in run.glob("continual_*/test/result.json")
             if p.parent.parent.name.split("_")[1].isdigit()]
    if not steps:
        return -1, {}
    step, path = max(steps)
    try:
        return step, (json.loads(path.read_text()) or {}).get("metrics") or {}
    except (json.JSONDecodeError, OSError):
        return -1, {}


def control_run(config: dict) -> Path | None:
    """The plain chain an adaptive run is paired against, from its own baseline checkpoint.

    `<run>/baseline/checkpoints/best.pt` -> `<run>`. Returns None when the run trained its own
    baseline, in which case there is no paired control and the run is excluded by name.
    """
    checkpoint = config.get("pipeline_baseline_checkpoint")
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if path.name != "best.pt" or path.parent.name != "checkpoints":
        return None
    return path.parents[2]


def discover(runs_root: Path) -> list[tuple[Path, dict]]:
    """Adaptive runs, identified by the file only the adaptive path writes."""
    out = []
    for lambdas in sorted(runs_root.glob("*/*/continual_summary/adaptive_lambdas.csv")):
        run = lambdas.parents[1]
        try:
            config = json.loads((run / "config.json").read_text())["args"]
        except (json.JSONDecodeError, OSError, KeyError):
            log.warning("[skip] %s — unreadable config.json", run)
            continue
        if config.get("pipeline_lambda_source") not in ("became", "one_over_t", "fixed"):
            continue
        out.append((run, config))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path)
    parser.add_argument("--floors", type=Path,
                        default=Path("results_archive/audit/floors.csv"))
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="restrict the ACC table to these metrics (default: all present)")
    parser.add_argument("--test_metrics", nargs="+",
                        default=["window_auroc", "window_auprc", "forecast/mse"],
                        help="metrics for the final-step TEST table — the task's own metrics, "
                             "not the loss the chain minimises")
    parser.add_argument("--exclude_prefix", nargs="+", default=["gate_"],
                        help="experiment-name prefixes to leave out; the gate runs are "
                             "fixtures, not results")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if args.runs_root is None:
        parser.error("--runs_root is required (or pass --self-test)")

    floors = load_floors(args.floors)
    seed_rows, step_rows, dist_rows, test_seed_rows = [], [], [], []
    excluded: list[str] = []

    for run, config in discover(args.runs_root):
        experiment = run.parent.name
        if any(experiment.startswith(p) for p in args.exclude_prefix):
            continue
        dataset = _label(config["dataset"])
        n_segments = int(config["dataset_n_finetune_segments"])
        seed = int(config["seed"])
        estimator_id = estimator(config)
        with (run / "continual_summary" / "adaptive_lambdas.csv").open(encoding="utf-8") as fh:
            recorded = [r.get("fisher_hat_samples", "") for r in csv.DictReader(fh)]
        if estimator_id["lambda_source"] == "became" and recorded and all(recorded):
            # The counts differ per estimate (base seed vs each period), so there is no one
            # cell-level number; the step table carries them. Blank, never the flag product.
            estimator_id = {**estimator_id, "fisher_samples": "",
                            "fisher_samples_source": "recorded"}

        control = control_run(config)
        if control is None or not (control / "config.json").is_file():
            excluded.append(f"{experiment}/{run.name}: no resolvable paired plain chain")
            continue
        control_config = json.loads((control / "config.json").read_text())["args"]
        # The control must differ from the adaptive run in the pullback and nothing else. A
        # mismatch here means the comparison is measuring something other than the method.
        mismatch = [k for k in ("dataset", "model", "task", "seed",
                                "dataset_n_finetune_segments")
                    if control_config.get(k) != config.get(k)]
        if control_config.get("pipeline_lambda_source", "none") != "none":
            mismatch.append("pipeline_lambda_source")
        if mismatch:
            excluded.append(
                f"{experiment}/{run.name}: control {control.parent.name}/{control.name} "
                f"differs on {', '.join(mismatch)}")
            continue

        adaptive_metrics, plain_metrics = summary_metrics(run), summary_metrics(control)
        if not adaptive_metrics or not plain_metrics:
            excluded.append(f"{experiment}/{run.name}: missing continual_summary/result.json")
            continue

        metrics = sorted({m.rsplit("/", 1)[0] for m in adaptive_metrics if m.endswith("/ACC")})
        for metric in metrics:
            if args.metrics and metric not in args.metrics:
                continue
            if f"{metric}/ACC" not in plain_metrics:
                excluded.append(f"{experiment}/{run.name}: control has no {metric}/ACC")
                continue
            seed_rows.append({
                "dataset": dataset, "n_segments": n_segments, "seed": seed, "metric": metric,
                **estimator_id,
                "adaptive_run": f"{experiment}/{run.name}",
                "plain_run": f"{control.parent.name}/{control.name}",
                "acc_adaptive": adaptive_metrics[f"{metric}/ACC"],
                "acc_plain": plain_metrics[f"{metric}/ACC"],
                "bwt_adaptive": adaptive_metrics.get(f"{metric}/BWT"),
                "bwt_plain": plain_metrics.get(f"{metric}/BWT"),
                "base_slice_adaptive":
                    adaptive_metrics.get(f"{metric}/base_slice_ratio_final"),
                "base_slice_plain": plain_metrics.get(f"{metric}/base_slice_ratio_final"),
            })

        # --- the task's own test metrics at the end of the chain ---
        step, adaptive_test = final_test_metrics(run)
        control_step, plain_test = final_test_metrics(control)
        if adaptive_test and plain_test and step == control_step:
            for metric in args.test_metrics:
                if metric not in adaptive_test or metric not in plain_test:
                    continue
                test_seed_rows.append({
                    "dataset": dataset, "n_segments": n_segments, "seed": seed,
                    "metric": metric, **estimator_id, "final_step": step,
                    "adaptive": adaptive_test[metric], "plain": plain_test[metric]})
        elif adaptive_test and plain_test:
            excluded.append(
                f"{experiment}/{run.name}: final test step {step} != control's {control_step}")

        with (run / "continual_summary" / "adaptive_lambdas.csv").open(encoding="utf-8") as fh:
            steps = list(csv.DictReader(fh))
        for step in steps:
            num, den = float(step["fisher_num"]), float(step["fisher_den"])
            lam_term = den - num
            star = float(step["fisher_star_num"]) if step.get("fisher_star_num") else 0.0
            step_rows.append({
                "dataset": dataset, "n_segments": n_segments, "seed": seed, **estimator_id,
                "step": int(step["step"]),
                "lambda_star": float(step["lambda_star"]),
                "one_over_t": float(step["one_over_t"]),
                "lambda_over_one_over_t": float(step["lambda_star"]) / float(step["one_over_t"]),
                "fisher_num": num, "fisher_den": den, "lambda_term": lam_term,
                # Lambda_t is a sum of t Fishers, so t is this ratio's floor, not 1. Both are
                # carried so the excess over the floor is read rather than inferred.
                "lambda_over_fisher": lam_term / num if num else "",
                "floor_is_t": int(step["step"]),
                # d^T F_t(theta_hat_t) d / d^T F_t(theta*_t) d — same task, same data, same d,
                # only the evaluation point moves. This is the at-a-minimum vs not-at-a-minimum
                # asymmetry with everything else held constant, and it is the candidate
                # mechanism for whatever suppression of lambda the 1/B artifact leaves behind.
                # Blank on runs written before the pipeline emitted the column.
                "fisher_star_num": star, "asymmetry_hat_over_star": num / star if star else "",
                "d_norm": float(step["d_norm"]), "step_norm": float(step["step_norm"]),
                "merged_dist_from_base": float(step["merged_dist_from_base"]),
                "unconstrained_dist_from_base": float(step["unconstrained_dist_from_base"]),
                "fisher_hat_samples": step.get("fisher_hat_samples", ""),
                "fisher_star_samples": step.get("fisher_star_samples", ""),
                "fisher_seed_samples": step.get("fisher_seed_samples", ""),
            })
        unconstrained = [float(s["unconstrained_dist_from_base"]) for s in steps]
        chain = float(steps[-1]["merged_dist_from_base"])
        dist_rows.append({
            "dataset": dataset, "n_segments": n_segments, "seed": seed, **estimator_id,
            "n_steps": len(steps),
            "chain_dist_from_base": chain,
            "mean_unconstrained_dist": st.fmean(unconstrained),
            "ratio_p4": chain / st.fmean(unconstrained),
        })

    # --- P1: aggregate over seeds ---
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["n_segments"], row["metric"], row["lambda_source"],
                 str(row["fixed_lambda"]), row["fisher_batch_size"],
                 row["fisher_batches"], row["fisher_samples_source"])].append(row)
    acc_rows = []
    for (dataset, n_segments, metric, source, fixed, batch_size, batches, n_source), rows in sorted(
            grouped.items()):
        adaptive = [r["acc_adaptive"] for r in rows]
        plain = [r["acc_plain"] for r in rows]
        delta = 100.0 * (st.fmean(adaptive) - st.fmean(plain)) / st.fmean(plain)
        floor = floors.get((dataset, metric))
        # The same own-spread column the test table carries. A verdict near its floor can be
        # decisive by the published floor and inside the spread of the runs being compared, and
        # the reader should be able to see that on every near-floor cell, not only the ones
        # somebody thought to check.
        spread = (100.0 * math.sqrt((st.variance(adaptive) + st.variance(plain)) / 2)
                  / st.fmean(plain)) if len(rows) > 1 else 0.0
        acc_rows.append({
            "dataset": dataset, "n_segments": n_segments, "metric": metric,
            "lambda_source": source, "fixed_lambda": rows[0]["fixed_lambda"],
            "fisher_batch_size": batch_size,
            "fisher_batches": batches,
            "fisher_samples": batch_size * batches if n_source != "recorded" else "",
            "fisher_samples_source": n_source,
            "n_seeds": len(rows), "floor_pct": floor if floor is not None else "",
            "acc_adaptive": st.fmean(adaptive),
            "acc_adaptive_sd": st.stdev(adaptive) if len(adaptive) > 1 else "",
            "acc_plain": st.fmean(plain),
            "acc_plain_sd": st.stdev(plain) if len(plain) > 1 else "",
            "acc_delta_pct": delta, "margin_ratio": margin_ratio(delta, floor),
            "own_spread_pct": spread,
            "margin_over_own_spread": abs(delta) / spread if spread else "",
            "borderline": (floor is not None
                           and floor < abs(delta) < BORDERLINE_RATIO * floor),
            "verdict": verdict(delta, floor),
            "bwt_adaptive": st.fmean([r["bwt_adaptive"] for r in rows
                                      if r["bwt_adaptive"] is not None]),
            "bwt_plain": st.fmean([r["bwt_plain"] for r in rows
                                   if r["bwt_plain"] is not None]),
            "base_slice_adaptive": st.fmean([r["base_slice_adaptive"] for r in rows
                                             if r["base_slice_adaptive"] is not None]),
            "base_slice_plain": st.fmean([r["base_slice_plain"] for r in rows
                                          if r["base_slice_plain"] is not None]),
        })

    # --- The task's own metric at the end of the chain ---
    test_grouped = defaultdict(list)
    for row in test_seed_rows:
        test_grouped[(row["dataset"], row["n_segments"], row["metric"], row["lambda_source"],
                      str(row["fixed_lambda"]), row["fisher_batch_size"],
                      row["fisher_batches"], row["fisher_samples_source"])].append(row)
    test_rows = []
    for (dataset, n_segments, metric, source, fixed, batch_size, batches, n_source), rows in sorted(
            test_grouped.items()):
        adaptive = [r["adaptive"] for r in rows]
        plain = [r["plain"] for r in rows]
        delta = 100.0 * (st.fmean(adaptive) - st.fmean(plain)) / st.fmean(plain)
        floor = floors.get((dataset, metric))
        up = higher_is_better(metric)
        # Pooled sd of the two arms, as a percentage of the control's mean — the same units as
        # `delta_pct` and as `floor_pct`, so the three are directly comparable.
        spread = (100.0 * math.sqrt((st.variance(adaptive) + st.variance(plain)) / 2)
                  / st.fmean(plain)) if len(rows) > 1 else 0.0
        test_rows.append({
            "dataset": dataset, "n_segments": n_segments, "metric": metric,
            "higher_is_better": up, "lambda_source": source,
            "fixed_lambda": rows[0]["fixed_lambda"], "fisher_batch_size": batch_size,
            "fisher_batches": batches,
            "fisher_samples": batch_size * batches if n_source != "recorded" else "",
            "fisher_samples_source": n_source,
            "n_seeds": len(rows), "floor_pct": floor if floor is not None else "",
            "final_step": rows[0]["final_step"],
            "adaptive": st.fmean(adaptive),
            "adaptive_sd": st.stdev(adaptive) if len(adaptive) > 1 else "",
            "plain": st.fmean(plain),
            "plain_sd": st.stdev(plain) if len(plain) > 1 else "",
            "delta_pct": delta, "margin_ratio": margin_ratio(delta, floor),
            "borderline": (floor is not None
                           and floor < abs(delta) < BORDERLINE_RATIO * floor),
            "verdict": verdict(delta, floor, up),
            # §1.9's open question, as a number rather than a caveat: the published floor comes
            # from a dedicated base-model experiment, but run-to-run spread is a property of the
            # MODEL too, and the two do not move together. This is the spread of the two arms
            # actually being compared. A verdict whose margin is below 1x its own spread is
            # decisive only by the convention, and the reader should be able to see that.
            "own_spread_pct": spread, "margin_over_own_spread":
                abs(delta) / spread if spread else ""})

    for row in acc_rows:
        log.info("[P1] %-14s n=%d %-22s %-10s adaptive %.4f vs plain %.4f  %+.2f%% "
                 "(floor %s, %sx) -> %s%s", row["dataset"], row["n_segments"], row["metric"],
                 f'{row["lambda_source"][:6]}/{row["fisher_batch_size"]}',
                 row["acc_adaptive"], row["acc_plain"],
                 row["acc_delta_pct"], row["floor_pct"],
                 f'{row["margin_ratio"]:.2f}' if row["margin_ratio"] != "" else "-",
                 row["verdict"], "  BORDERLINE" if row["borderline"] else "")
    tally = defaultdict(int)
    for row in acc_rows:
        tally[row["verdict"]] += 1
    log.info("[P1] %s", "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    for row in test_rows:
        log.info("[TEST] %-14s n=%d %-14s %-10s %s  adaptive %.4f vs plain %.4f  %+.2f%% "
                 "(floor %s, %sx) -> %s%s",
                 row["dataset"], row["n_segments"], row["metric"],
                 f'{row["lambda_source"][:6]}/{row["fisher_batch_size"]}',
                 "up" if row["higher_is_better"] else "dn",
                 row["adaptive"], row["plain"], row["delta_pct"], row["floor_pct"],
                 f'{row["margin_ratio"]:.2f}' if row["margin_ratio"] != "" else "-",
                 row["verdict"], "  BORDERLINE" if row["borderline"] else "")
    for row in acc_rows + test_rows:
        if row["margin_over_own_spread"] != "" and row["margin_over_own_spread"] < 1.0 \
                and row["verdict"] in ("better", "worse"):
            log.warning("[TEST] ⚠️  %s n=%d %s: margin %.2f%% is %.2fx these runs' OWN spread "
                        "(%.2f%%) — decisive by the published floor (%.3f%%) only",
                        row["dataset"], row["n_segments"], row["metric"],
                        row.get("delta_pct", row.get("acc_delta_pct")),
                        row["margin_over_own_spread"], row["own_spread_pct"], row["floor_pct"])
    test_tally = defaultdict(int)
    for row in test_rows:
        test_tally[row["verdict"]] += 1
    log.info("[TEST] %s", "  ".join(f"{k}={v}" for k, v in sorted(test_tally.items())))
    for row in sorted(dist_rows, key=lambda r: (r["dataset"], r["n_segments"],
                                                r["fisher_batch_size"], r["seed"])):
        log.info("[P4] %-14s n=%d seed=%-4d B=%-4d ratio=%.3f", row["dataset"],
                 row["n_segments"], row["seed"], row["fisher_batch_size"], row["ratio_p4"])
    # --- Does the at-a-minimum asymmetry explain the excess over Lambda/F's floor of t? ---
    # **t = 1 is excluded, and must be**: Lambda_1 is Lambda_0 alone, taken at theta_0 on the
    # base shard, so it carries no merged-point term for the asymmetry to be about. Pooling it
    # in inflates the slope from 0.97 to 1.82 — a cell that cannot speak to the hypothesis
    # driving the number that tests it.
    # Estimator-of-record rows only. The full-pass robustness re-run (§1.39, recorded sample
    # counts) re-estimates the SAME exchange_rate chains; pooling it in would count those three
    # configurations twice and moved the published slope from 0.97 to 1.10 when it first landed.
    of_record = [r for r in step_rows if r.get("fisher_samples_source") != "recorded"]
    pairs = [(math.log(1.0 / float(r["asymmetry_hat_over_star"])),
              math.log(float(r["lambda_over_fisher"]) / r["step"]))
             for r in of_record
             if r["step"] > 1 and r["asymmetry_hat_over_star"] not in ("", None)
             and r["lambda_over_fisher"] not in ("", None)
             and float(r["lambda_over_fisher"]) > 0]
    # --- The fixed-lambda curve: C41's test (§1.39d) ---
    curve_rows = []
    by_config = defaultdict(dict)
    boundary = {}
    for row in test_rows:
        key = (row["dataset"], row["n_segments"], row["metric"])
        if row["lambda_source"] == "fixed" and row["fixed_lambda"] not in ("", None):
            by_config[key][float(row["fixed_lambda"])] = row["adaptive"]
            # Every row of a config carries the same control, so the boundary is read off
            # whichever row is present rather than requiring a separate lambda=1 run.
            boundary[key] = row["plain"]
        elif key not in boundary:
            boundary[key] = row["plain"]
    # Mean derived lambda over the periods after the first. t=1 is excluded because every rule
    # agrees there is nothing to brake against yet on the first period -- the accumulator is
    # theta_0 -- and including it hides the collapse that happens afterwards.
    derived = defaultdict(list)
    for row in step_rows:
        if (row["lambda_source"] == "became" and row["fisher_batch_size"] == 1 and row["step"] > 1
                and row.get("fisher_samples_source") != "recorded"):
            derived[(row["dataset"], row["n_segments"], "forecast/mse")].append(
                row["lambda_star"])
    for key, points in sorted(by_config.items()):
        dataset, n_segments, metric = key
        lambdas = sorted(points)
        values = [points[lam] for lam in lambdas]
        floor = floors.get((dataset, metric))
        shape, gain, ratio = curve_shape(lambdas, values, boundary[key], floor)
        best = min(range(len(values)), key=lambda i: values[i])
        curve_rows.append({
            "dataset": dataset, "n_segments": n_segments, "metric": metric,
            "n_points": len(lambdas),
            "n_seeds": min(r["n_seeds"] for r in test_rows
                           if (r["dataset"], r["n_segments"], r["metric"]) == key
                           and r["lambda_source"] == "fixed"),
            "floor_pct": floor if floor is not None else "",
            "lambdas": " ".join(f"{lam:g}" for lam in lambdas),
            "values": " ".join(f"{v:.4f}" for v in values),
            "best_lambda": lambdas[best], "best_value": values[best],
            "value_at_lambda_1": boundary[key],
            "interior_gain_pct": gain, "interior_gain_over_floor": ratio,
            # Same <1.5x band the verdict tables use: an interior optimum that clears its floor
            # by 1.35x is a different kind of statement from one that clears it by 3.85x.
            "borderline": (floor is not None
                           and floor < abs(gain) < BORDERLINE_RATIO * floor),
            "shape": shape,
            # How far the DERIVED coefficient sits from the curve's own optimum, at the periods
            # where the two can differ. This is the quantity that separates "the premise fails"
            # from "the coefficient fails": a large ratio with an interior optimum means the
            # point exists and Eq. 20 does not find it.
            "best_lambda_over_adaptive": (
                lambdas[best] / st.fmean(derived[key]) if derived.get(key) else "")})
    for row in curve_rows:
        log.info("[CURVE] %-14s n=%d %-13s lambda*=%.1f  %s  (best %.4f vs plain %.4f, "
                 "%+.2f%% = %.2fx floor) -> %s",
                 row["dataset"], row["n_segments"], row["metric"], row["best_lambda"],
                 row["values"], row["best_value"], row["value_at_lambda_1"],
                 row["interior_gain_pct"], row["interior_gain_over_floor"],
                 row["shape"] + ("  BORDERLINE" if row["borderline"] else "")
                 + (f'  [best lambda is {row["best_lambda_over_adaptive"]:.1f}x the derived one]'
                    if row["best_lambda_over_adaptive"] != "" else ""))

    asymmetry_rows = []
    if len(pairs) > 2:
        correlation, slope, n_pairs = log_fit(pairs)
        # Both the t>1 fit and the all-cells fit are emitted. The second exists so the
        # difference is on the record rather than in a commit message: pooling t=1 in moves the
        # slope from 0.97 to 1.82, and a reader who wonders why t=1 was dropped can see it.
        everything, slope_all, n_all = log_fit(
            pairs + [(math.log(1.0 / float(r["asymmetry_hat_over_star"])),
                      math.log(float(r["lambda_over_fisher"]) / r["step"]))
                     for r in of_record
                     if r["step"] == 1 and r["asymmetry_hat_over_star"] not in ("", None)
                     and r["lambda_over_fisher"] not in ("", None)
                     and float(r["lambda_over_fisher"]) > 0])
        asymmetry_rows = [
            {"scope": "t>1", "n_cells": n_pairs, "pearson_r": correlation, "slope": slope,
             "r_squared": correlation ** 2},
            {"scope": "all_t", "n_cells": n_all, "pearson_r": everything, "slope": slope_all,
             "r_squared": everything ** 2},
        ]
        log.info("[asymmetry] log(excess over floor) vs log(1/asymmetry), t>1, n=%d: "
                 "r=%+.3f  slope=%+.3f  (slope 1 = accounts for it one-for-one; "
                 "r^2=%.2f, so per-cell scatter is large)",
                 n_pairs, correlation, slope, correlation ** 2)
        log.info("[asymmetry] pooling t=1 in would give slope %+.3f over %d cells — t=1 carries "
                 "no merged-point term and is excluded", slope_all, n_all)

    if excluded:
        log.warning("[excluded] %d run(s):", len(excluded))
        for reason in excluded:
            log.warning("  %s", reason)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (
            ("adaptive_lambda_acc.csv", ACC_FIELDS, acc_rows),
            ("adaptive_lambda_per_seed.csv", SEED_FIELDS, seed_rows),
            ("adaptive_lambda_steps.csv", STEP_FIELDS, step_rows),
            ("adaptive_lambda_distance.csv", DIST_FIELDS, dist_rows),
            ("adaptive_lambda_test.csv", TEST_FIELDS, test_rows),
            ("adaptive_lambda_curve.csv", CURVE_FIELDS, curve_rows),
            ("adaptive_lambda_asymmetry_fit.csv", ASYMMETRY_FIELDS, asymmetry_rows),
        ):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            log.info("wrote %s (%d row(s))", args.out / name, len(rows))
        if excluded:
            (args.out / "adaptive_lambda_excluded.txt").write_text(
                "\n".join(excluded) + "\n", encoding="utf-8")


def _self_test() -> None:
    """Each rule below can fail — the point of the check is that it does."""
    assert verdict(-9.0, 14.107) == "tie", "inside the floor must be a tie, not a win"
    assert verdict(-20.0, 14.107) == "better", "loss-shaped: a negative delta is better"
    assert verdict(20.0, 14.107) == "worse"
    assert verdict(-20.0, None) == "no_floor", "a missing floor must not read as a win"

    # Orientation: the SAME signed delta must read opposite ways for an error and an AUROC.
    assert verdict(-20.0, 1.0, up_is_better=False) == "better"
    assert verdict(-20.0, 1.0, up_is_better=True) == "worse"
    assert verdict(+20.0, 1.0, up_is_better=True) == "better"
    assert verdict(0.5, 1.0, up_is_better=True) == "tie", "the floor applies either way round"
    assert higher_is_better("window_auroc") and higher_is_better("pa_f1")
    assert not higher_is_better("forecast/mse") and not higher_is_better("reconstruction/score_mean")

    assert abs(margin_ratio(12.51, 8.759) - 1.428) < 1e-3
    assert margin_ratio(1.0, None) == "", "no floor means no ratio, not a ratio of 1"
    # The band must be exclusive at both ends: a tie is not borderline, and a clean call is not.
    for delta, floor, expected in ((12.51, 8.759, True), (49.09, 8.759, False),
                                   (8.0, 8.759, False), (13.2, 8.759, False)):
        got = floor < abs(delta) < BORDERLINE_RATIO * floor
        assert got is expected, (delta, floor, got, expected)

    control = control_run({"pipeline_baseline_checkpoint":
                           "/w/runs/ettm2_continual_n3/80597/baseline/checkpoints/best.pt"})
    assert control == Path("/w/runs/ettm2_continual_n3/80597"), control
    assert control_run({}) is None, "a run that trained its own baseline has no control"
    assert control_run({"pipeline_baseline_checkpoint": "/w/runs/x/1/baseline/best.pt"}) is None

    assert estimator({"pipeline_lambda_source": "became", "pipeline_fisher_batches": 64,
                      "loader_batch_size": 128}) == {
        "lambda_source": "became", "fixed_lambda": "", "fisher_batch_size": 128,
        "fisher_batches": 64, "fisher_samples": 8192, "fisher_samples_source": "flags"}, \
        "a run predating --pipeline_fisher_batch_size used the loader's batch size"
    assert estimator({"pipeline_lambda_source": "became", "pipeline_fisher_batch_size": 1,
                      "pipeline_fisher_batches": 512,
                      "loader_batch_size": 128})["fisher_batch_size"] == 1
    # An imposed-lambda run computes no Fisher, so it must not inherit a batch size that would
    # let it group with a `became` run of the same configuration.
    imposed = estimator({"pipeline_lambda_source": "one_over_t", "pipeline_fisher_batches": 64,
                         "loader_batch_size": 128})
    assert imposed == {"lambda_source": "one_over_t", "fixed_lambda": "", "fisher_batch_size": 0,
                       "fisher_batches": 0, "fisher_samples": 0,
                       "fisher_samples_source": "n/a"}, imposed
    # Two points of a fixed-lambda grid must not be the same cell, or the curve collapses.
    grid = [estimator({"pipeline_lambda_source": "fixed", "pipeline_fixed_lambda": lam,
                       "loader_batch_size": 128}) for lam in (0.1, 0.9)]
    assert grid[0] != grid[1] and grid[0]["fixed_lambda"] == 0.1, grid
    assert estimator({"pipeline_lambda_source": "became", "pipeline_fixed_lambda": 1.0,
                      "pipeline_fisher_batches": 64,
                      "loader_batch_size": 128})["fixed_lambda"] == "", \
        "a derived-lambda run has no fixed lambda, whatever the inert flag says"

    lams = [0.1, 0.3, 0.5, 0.7, 0.9]
    # A genuine dip that clears the floor is interior; the same ordering inside the floor is not.
    dip = [1.0, 0.9, 0.80, 0.9, 0.95]
    assert curve_shape(lams, dip, 1.0, 5.0)[0] == "interior"
    assert curve_shape(lams, [1.0, 0.99, 0.98, 0.99, 0.995], 1.0, 5.0)[0] == \
        "flat within the floor", "an argmin away from the edge is not by itself a finding"
    # Monotone toward the boundary: every swept lambda loses to lambda = 1.
    assert curve_shape(lams, [1.6, 1.45, 1.3, 1.2, 1.1], 1.0, 5.0)[0] == \
        "boundary (every swept lambda loses)"
    assert curve_shape(lams, dip, 1.0, 5.0)[1] > 0, "a dip must report a positive gain"
    assert curve_shape(lams, [1.6, 1.45, 1.3, 1.2, 1.1], 1.0, 5.0)[1] < 0

    xs = [1.0, 2.0, 4.0, 8.0]
    exact = [(math.log(x), math.log(x)) for x in xs]
    r, slope, n = log_fit(exact)
    assert abs(r - 1.0) < 1e-12 and abs(slope - 1.0) < 1e-12 and n == 4
    half = [(math.log(x), 0.5 * math.log(x)) for x in xs]
    assert abs(log_fit(half)[1] - 0.5) < 1e-12, "the slope must be able to come back != 1"
    assert log_fit([(math.log(x), -math.log(x)) for x in xs])[0] < 0

    try:
        _label("NotADataset")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown dataset must raise, not silently lose its floor")
    log.info("self-test OK")


if __name__ == "__main__":
    main()
