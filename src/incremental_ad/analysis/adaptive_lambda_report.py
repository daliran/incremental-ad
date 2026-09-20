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
ESTIMATOR_FIELDS = ["fisher_batch_size", "fisher_batches", "fisher_samples"]

ACC_FIELDS = ["dataset", "n_segments", "metric"] + ESTIMATOR_FIELDS + ["n_seeds", "floor_pct",
              "acc_adaptive", "acc_adaptive_sd", "acc_plain", "acc_plain_sd",
              "acc_delta_pct", "margin_ratio", "borderline", "verdict",
              "bwt_adaptive", "bwt_plain", "base_slice_adaptive", "base_slice_plain"]
SEED_FIELDS = ["dataset", "n_segments", "seed", "metric"] + ESTIMATOR_FIELDS + [
               "adaptive_run", "plain_run",
               "acc_adaptive", "acc_plain", "bwt_adaptive", "bwt_plain",
               "base_slice_adaptive", "base_slice_plain"]
STEP_FIELDS = ["dataset", "n_segments", "seed"] + ESTIMATOR_FIELDS + [
               "step", "lambda_star", "one_over_t",
               "lambda_over_one_over_t", "fisher_num", "fisher_den", "lambda_term",
               "lambda_over_fisher", "floor_is_t",
               "fisher_star_num", "asymmetry_hat_over_star", "d_norm", "step_norm",
               "merged_dist_from_base", "unconstrained_dist_from_base"]
ASYMMETRY_FIELDS = ["scope", "n_cells", "pearson_r", "slope", "r_squared"]
DIST_FIELDS = ["dataset", "n_segments", "seed"] + ESTIMATOR_FIELDS + [
               "n_steps", "chain_dist_from_base",
               "mean_unconstrained_dist", "ratio_p4"]


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


def verdict(delta_pct: float, floor_pct: float | None) -> str:
    """`better` / `tie` / `worse`, where ACC is loss-shaped so a negative delta is better."""
    if floor_pct is None:
        return "no_floor"
    if abs(delta_pct) <= floor_pct:
        return "tie"
    return "better" if delta_pct < 0 else "worse"


def estimator(config: dict) -> dict:
    """The Fisher estimator's identity: batch size, batch count and their product.

    `pipeline_fisher_batch_size` is absent from runs that predate the flag; those fell back to
    the loader's batch size, which is what the fallback below reads. Getting this wrong pools
    two different estimators into one cell.
    """
    batch_size = config.get("pipeline_fisher_batch_size") or config["loader_batch_size"]
    batches = config["pipeline_fisher_batches"]
    return {"fisher_batch_size": int(batch_size), "fisher_batches": int(batches),
            "fisher_samples": int(batch_size) * int(batches)}


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
        if config.get("pipeline_lambda_source") != "became":
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
    seed_rows, step_rows, dist_rows = [], [], []
    excluded: list[str] = []

    for run, config in discover(args.runs_root):
        experiment = run.parent.name
        if any(experiment.startswith(p) for p in args.exclude_prefix):
            continue
        dataset = _label(config["dataset"])
        n_segments = int(config["dataset_n_finetune_segments"])
        seed = int(config["seed"])
        estimator_id = estimator(config)

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
        grouped[(row["dataset"], row["n_segments"], row["metric"],
                 row["fisher_batch_size"], row["fisher_batches"])].append(row)
    acc_rows = []
    for (dataset, n_segments, metric, batch_size, batches), rows in sorted(grouped.items()):
        adaptive = [r["acc_adaptive"] for r in rows]
        plain = [r["acc_plain"] for r in rows]
        delta = 100.0 * (st.fmean(adaptive) - st.fmean(plain)) / st.fmean(plain)
        floor = floors.get((dataset, metric))
        acc_rows.append({
            "dataset": dataset, "n_segments": n_segments, "metric": metric,
            "fisher_batch_size": batch_size, "fisher_batches": batches,
            "fisher_samples": batch_size * batches,
            "n_seeds": len(rows), "floor_pct": floor if floor is not None else "",
            "acc_adaptive": st.fmean(adaptive),
            "acc_adaptive_sd": st.stdev(adaptive) if len(adaptive) > 1 else "",
            "acc_plain": st.fmean(plain),
            "acc_plain_sd": st.stdev(plain) if len(plain) > 1 else "",
            "acc_delta_pct": delta, "margin_ratio": margin_ratio(delta, floor),
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

    for row in acc_rows:
        log.info("[P1] %-14s n=%d %-22s B=%-4d adaptive %.4f vs plain %.4f  %+.2f%% "
                 "(floor %s, %sx) -> %s%s", row["dataset"], row["n_segments"], row["metric"],
                 row["fisher_batch_size"], row["acc_adaptive"], row["acc_plain"],
                 row["acc_delta_pct"], row["floor_pct"],
                 f'{row["margin_ratio"]:.2f}' if row["margin_ratio"] != "" else "-",
                 row["verdict"], "  BORDERLINE" if row["borderline"] else "")
    tally = defaultdict(int)
    for row in acc_rows:
        tally[row["verdict"]] += 1
    log.info("[P1] %s", "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    for row in sorted(dist_rows, key=lambda r: (r["dataset"], r["n_segments"],
                                                r["fisher_batch_size"], r["seed"])):
        log.info("[P4] %-14s n=%d seed=%-4d B=%-4d ratio=%.3f", row["dataset"],
                 row["n_segments"], row["seed"], row["fisher_batch_size"], row["ratio_p4"])
    # --- Does the at-a-minimum asymmetry explain the excess over Lambda/F's floor of t? ---
    # **t = 1 is excluded, and must be**: Lambda_1 is Lambda_0 alone, taken at theta_0 on the
    # base shard, so it carries no merged-point term for the asymmetry to be about. Pooling it
    # in inflates the slope from 0.97 to 1.82 — a cell that cannot speak to the hypothesis
    # driving the number that tests it.
    pairs = [(math.log(1.0 / float(r["asymmetry_hat_over_star"])),
              math.log(float(r["lambda_over_fisher"]) / r["step"]))
             for r in step_rows
             if r["step"] > 1 and r["asymmetry_hat_over_star"] not in ("", None)
             and r["lambda_over_fisher"] not in ("", None)
             and float(r["lambda_over_fisher"]) > 0]
    asymmetry_rows = []
    if len(pairs) > 2:
        correlation, slope, n_pairs = log_fit(pairs)
        # Both the t>1 fit and the all-cells fit are emitted. The second exists so the
        # difference is on the record rather than in a commit message: pooling t=1 in moves the
        # slope from 0.97 to 1.82, and a reader who wonders why t=1 was dropped can see it.
        everything, slope_all, n_all = log_fit(
            pairs + [(math.log(1.0 / float(r["asymmetry_hat_over_star"])),
                      math.log(float(r["lambda_over_fisher"]) / r["step"]))
                     for r in step_rows
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

    assert estimator({"pipeline_fisher_batches": 64, "loader_batch_size": 128}) == {
        "fisher_batch_size": 128, "fisher_batches": 64, "fisher_samples": 8192}, \
        "a run predating --pipeline_fisher_batch_size used the loader's batch size"
    assert estimator({"pipeline_fisher_batch_size": 1, "pipeline_fisher_batches": 512,
                      "loader_batch_size": 128})["fisher_batch_size"] == 1

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
