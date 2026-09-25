"""How much of the λ collapse is the batch-mean Fisher estimator, and how much is real.

    python -m incremental_ad.analysis.fisher_scaling_report \\
        --scaling_root $WORK/fisher_scaling --out $OUT/fisher_scaling

Reads the CSVs `scripts/diagnose_fisher_batch_scaling.py` writes and turns the B-sweep into the
two numbers the question needs: a **scaling exponent** for each quadratic form, and a
**multiplicative decomposition** of λ's suppression into an estimator part and a residual.

**Why an exponent settles it.** `diagonal_fisher` squares the gradient of a batch-mean loss, so
`E[F_hat] = g^2 + sigma^2/B`. A form evaluated at a minimum of its own shard has g ~ 0 and must
scale as `B^-1`; one evaluated away from a minimum saturates at `g^2` and must scale as `B^0`.
λ*'s numerator is `F_t(theta_hat_t)`, taken at exactly such a minimum; Λ's terms after the first
are taken at merged points that are minima of nothing. So the prediction is not "λ moves with B"
— it is **two different exponents**, and the fit either shows them or it does not.

**B = 1 is not an extrapolation; it is the estimator being correct.** At B = 1 the batch mean is
the per-sample gradient and the quantity computed *is* the empirical Fisher. So the B = 1 column
is the corrected measurement, the run's own B is the published one, and the ratio between them
is the artifact — no model of the noise is needed to size it.

**The floor is t, not 1.** Λ_t sums t Fishers, so Λ/F ~ t even with a perfect estimator and
identical curvature everywhere. `excess_over_floor` is the quantity to read; the raw ratio is
carried beside it because the raw ratio is what the run logs.

**The sample count, checked (`--steps`).** The B-sweep holds N = 8192; the published B = 1 runs
use N = 512 (512 batches of 1). E[F_hat] does not depend on N, but λ* is a RATIO of two
estimates, so N can still move it. `fisher_sample_agreement.csv` compares the two at t = 1 —
the only step where both chains hold the same model (each fine-tunes the same base on the same
first period with the same seed; later steps diverge because the B-sweep follows the B = 128
chain). `d_norm` is emitted from both sides so "the same model" is checked, not assumed.

Pure CSV aggregation: safe on a login node, no checkpoints, no GPU.
"""

import argparse
import csv
import logging
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("fisher_scaling_report")

SCALING_FIELDS = ["dataset", "n_segments", "seed", "step", "n_points",
                  "exponent_numerator", "exponent_lambda_term",
                  "scaling_share", "scaling_share_pct", "flat_over_scaling",
                  "flat_terms_expected",
                  "num_at_bmin", "num_at_bmax", "lambda_term_at_bmin", "lambda_term_at_bmax",
                  "b_min", "b_max"]
AGREEMENT_FIELDS = ["dataset", "n_segments", "seed", "step", "batch_size", "samples_run",
                    "samples_sweep", "lambda_run", "lambda_sweep", "delta_pct", "d_norm_run",
                    "d_norm_sweep", "d_norm_rel_diff"]
DECOMP_FIELDS = ["dataset", "n_segments", "seed", "step", "b_published", "b_corrected",
                 "lambda_published", "lambda_corrected", "one_over_t",
                 "suppression_total", "suppression_estimator", "suppression_residual",
                 "estimator_share_log", "ratio_published", "ratio_corrected",
                 "excess_over_floor_published", "excess_over_floor_corrected"]


def log_log_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of log y against log x — the exponent p in `y ~ x^p`.

    Reported rather than a single end-to-end ratio because the prediction is about the *shape*
    of the curve: a pure-noise form gives -1 across the whole range, and a saturating one bends.
    Two endpoints cannot tell those apart.
    """
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = st.fmean(lx), st.fmean(ly)
    denominator = sum((x - mx) ** 2 for x in lx)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / denominator


def scaling_share(exponent: float, exponent_at_t1: float) -> float:
    """What fraction of Λ still scales with B, from the two exponents alone.

    Λ_t is a **sum**, not one Fisher: Λ₀, taken at θ₀ on the base shard — a minimum, so it
    scales — plus one term per completed period, taken at a merged point that is a minimum of
    nothing, so it saturates. Write `Λ(B) = A·B^p + C`, and

        d log Λ / d log B = p · A·B^p / (A·B^p + C) = p · (scaling share)

    so the share is the observed exponent divided by the scaling component's own exponent. No
    fit is needed and none is done: `p` is measured at t = 1, where Λ **is** Λ₀ and nothing
    else, and the share is then read off the ratio. It is a share averaged over the swept range
    of B, because the numerator is a least-squares slope over that range.

    The model makes a second, sharper prediction it cannot dodge: each later period adds one
    more flat term, so the share must **fall** with t. That is a direction the two exponents
    were not fitted to produce.
    """
    return exponent / exponent_at_t1


def load(scaling_root: Path) -> list[dict]:
    rows = []
    # Both layouts: one directory per seed as the sweep jobs write it, and flat as the archive
    # stores it. `dict.fromkeys` keeps order and drops the duplicate when a path matches twice.
    candidates = dict.fromkeys(sorted(scaling_root.glob("fisher_scaling_*.csv"))
                               + sorted(scaling_root.glob("*/fisher_scaling_*.csv")))
    for path in candidates:
        # `fisher_scaling_<experiment>.csv`, and the experiment name carries dataset/n/seed —
        # but the run's own config is the authority for those, so they are parsed back out of
        # the file name only for grouping and are printed for the reader to check.
        experiment = path.stem[len("fisher_scaling_"):]
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # This report's own output can land under `scaling_root` and matches the same glob.
            # Identify the inputs by the column only a sweep has, rather than by excluding an
            # output directory by name — the check then holds wherever the output is written.
            if "batch_size" not in (reader.fieldnames or []):
                log.info("[skip] %s — no batch_size column; not a sweep output", path)
                continue
            for row in reader:
                row["experiment"] = experiment
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scaling_root", type=Path)
    parser.add_argument("--dataset", default="ETTm2")
    parser.add_argument("--n_segments", type=int, default=3)
    parser.add_argument("--b_published", type=int, default=128,
                        help="the Fisher batch size the runs under analysis actually used")
    parser.add_argument("--steps", type=Path,
                        help="adaptive_lambda_steps.csv; adds the N = 512 vs N = 8192 check")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if args.scaling_root is None:
        parser.error("--scaling_root is required (or pass --self-test)")

    rows = load(args.scaling_root)
    if not rows:
        raise SystemExit(f"no fisher_scaling_*.csv under {args.scaling_root}")

    grouped = defaultdict(list)
    for row in rows:
        seed = row["experiment"].rsplit("_s", 1)[1]
        grouped[(seed, int(row["step"]))].append(row)

    scaling_rows, decomp_rows = [], []
    seed_scaling_exponent: dict[str, float] = {}
    for (seed, step), group in sorted(grouped.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        group.sort(key=lambda r: int(r["batch_size"]))
        batch_sizes = [float(r["batch_size"]) for r in group]
        numerators = [float(r["fisher_num"]) for r in group]
        lambda_terms = [float(r["lambda_term"]) for r in group]
        common = {"dataset": args.dataset, "n_segments": args.n_segments,
                  "seed": int(seed), "step": step}
        lambda_exponent = log_log_slope(batch_sizes, lambda_terms)
        # The scaling component's own exponent is measured at t=1, where Lambda IS Lambda_0 and
        # nothing else — which is the second thing the t=1 row buys beyond the negative control.
        seed_scaling_exponent.setdefault(seed, lambda_exponent)
        share = scaling_share(lambda_exponent, seed_scaling_exponent[seed])
        scaling_rows.append({
            **common, "n_points": len(group),
            "exponent_numerator": log_log_slope(batch_sizes, numerators),
            "exponent_lambda_term": lambda_exponent,
            "scaling_share": share, "scaling_share_pct": 100 * share,
            "flat_over_scaling": (1 - share) / share if share else "",
            "flat_terms_expected": step - 1,
            "num_at_bmin": numerators[0], "num_at_bmax": numerators[-1],
            "lambda_term_at_bmin": lambda_terms[0], "lambda_term_at_bmax": lambda_terms[-1],
            "b_min": int(batch_sizes[0]), "b_max": int(batch_sizes[-1]),
        })

        by_b = {int(r["batch_size"]): r for r in group}
        published = by_b.get(args.b_published)
        corrected = by_b[min(by_b)]
        if published is None:
            log.warning("[skip decomposition] seed %s step %d: no B=%d point",
                        seed, step, args.b_published)
            continue
        # The published lambda is the one the chain actually applied, read from the run's own
        # CSV rather than re-derived: the replay reproduces it to within loader sampling noise,
        # and using the replayed value instead would hide any divergence between the two.
        lambda_published = float(published["recorded_lambda"])
        lambda_corrected = float(corrected["implied_lambda"])
        one_over_t = 1.0 / (step + 1)      # the base model is task 1 (§1.39, notation)
        total = one_over_t / lambda_published
        estimator = lambda_corrected / lambda_published
        decomp_rows.append({
            **common,
            "b_published": args.b_published, "b_corrected": min(by_b),
            "lambda_published": lambda_published, "lambda_corrected": lambda_corrected,
            "one_over_t": one_over_t,
            "suppression_total": total,
            "suppression_estimator": estimator,
            "suppression_residual": one_over_t / lambda_corrected,
            # Shares are taken in log space because the two factors multiply; a linear share of
            # a 1000x suppression would read as ~100% estimator for any estimator factor at all.
            "estimator_share_log": math.log(estimator) / math.log(total) if total > 1 else "",
            "ratio_published": float(published["ratio_lambda_over_f"]),
            "ratio_corrected": float(corrected["ratio_lambda_over_f"]),
            "excess_over_floor_published": float(published["ratio_lambda_over_f"]) / step,
            "excess_over_floor_corrected": float(corrected["ratio_lambda_over_f"]) / step,
        })

    log.info("Exponent p in `form ~ B^p`  (prediction: -1 at a minimum, 0 away from one)")
    log.info("%-6s %-4s %10s %10s", "seed", "t", "F(theta^)", "Lambda")
    for row in scaling_rows:
        log.info("%-6d %-4d %10.3f %10.3f", row["seed"], row["step"],
                 row["exponent_numerator"], row["exponent_lambda_term"])
    log.info("Lambda as a mixture of Lambda_0 (scales) and merged-point terms (flat)")
    log.info("%-6s %-4s %9s %9s %14s", "seed", "t", "scaling", "flat", "flat/scaling")
    for row in scaling_rows:
        log.info("%-6d %-4d %8.1f%% %8.1f%% %10.2f  (%d flat term(s))",
                 row["seed"], row["step"], 100 * row["scaling_share"],
                 100 * (1 - row["scaling_share"]),
                 row["flat_over_scaling"] if row["flat_over_scaling"] != "" else float("nan"),
                 row["flat_terms_expected"])
    for step in sorted({r["step"] for r in scaling_rows if r["step"] > 1}):
        values = [r["scaling_share"] for r in scaling_rows if r["step"] == step]
        log.info("  mean scaling share t=%d: %.1f%%", step, 100 * st.fmean(values))
    for label, key in (("numerator F(theta_hat)", "exponent_numerator"),
                       ("Lambda term", "exponent_lambda_term")):
        for step in sorted({r["step"] for r in scaling_rows}):
            values = [r[key] for r in scaling_rows if r["step"] == step]
            log.info("  mean exponent, %-22s t=%d: %+.3f", label, step, st.fmean(values))

    log.info("Decomposition of lambda's suppression below 1/t (multiplicative)")
    log.info("%-6s %-4s %10s %10s %10s %10s %10s", "seed", "t", "lambda@B%d" % args.b_published,
             "lambda@B=1", "total", "estimator", "residual")
    for row in decomp_rows:
        log.info("%-6d %-4d %10.5f %10.5f %9.1fx %9.1fx %9.1fx", row["seed"], row["step"],
                 row["lambda_published"], row["lambda_corrected"], row["suppression_total"],
                 row["suppression_estimator"], row["suppression_residual"])
    log.info("Lambda/F excess over its floor of t")
    for row in decomp_rows:
        log.info("  seed %-4d t=%d: %8.1fx -> %6.1fx", row["seed"], row["step"],
                 row["excess_over_floor_published"], row["excess_over_floor_corrected"])

    agreement_rows = sample_agreement(rows, args.steps, args.dataset, args.n_segments) \
        if args.steps else []
    for row in agreement_rows:
        log.info("[samples] seed %-4s t=1  lambda N=%s %.4f vs N=%s %.4f  (%+.2f%%; d_norm "
                 "rel diff %.1e)", row["seed"], row["samples_run"], row["lambda_run"],
                 row["samples_sweep"], row["lambda_sweep"], row["delta_pct"],
                 row["d_norm_rel_diff"])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        outputs = [("fisher_scaling_exponents.csv", SCALING_FIELDS, scaling_rows),
                   ("fisher_scaling_decomposition.csv", DECOMP_FIELDS, decomp_rows)]
        if agreement_rows:
            outputs.append(("fisher_sample_agreement.csv", AGREEMENT_FIELDS, agreement_rows))
        for name, fields, out_rows in outputs:
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(out_rows)
            log.info("wrote %s (%d row(s))", args.out / name, len(out_rows))


def sample_agreement(rows: list[dict], steps_path: Path, dataset: str,
                     n_segments: int) -> list[dict]:
    """λ* at B = 1 from the published runs (N = their own) against the sweep's (N = 8192), t = 1."""
    with steps_path.open(encoding="utf-8") as fh:
        steps = {r["seed"]: r for r in csv.DictReader(fh)
                 if r["dataset"] == dataset and r["n_segments"] == str(n_segments)
                 and r["lambda_source"] == "became" and r["fisher_batch_size"] == "1"
                 and r["step"] == "1"}
    out = []
    for row in rows:
        if row["step"] != "1" or row["batch_size"] != "1":
            continue
        seed = row["experiment"].rsplit("_s", 1)[-1]
        run = steps.get(seed)
        if run is None:
            continue
        a, b = float(run["lambda_star"]), float(row["implied_lambda"])
        da, db = float(run["d_norm"]), float(row["d_norm"])
        out.append({"dataset": dataset, "n_segments": n_segments, "seed": seed, "step": 1,
                    "batch_size": 1, "samples_run": run["fisher_samples"],
                    "samples_sweep": row["samples"], "lambda_run": a, "lambda_sweep": b,
                    "delta_pct": 100.0 * (a - b) / b, "d_norm_run": da, "d_norm_sweep": db,
                    "d_norm_rel_diff": abs(da - db) / db})
    return sorted(out, key=lambda r: int(r["seed"]))


def _self_test() -> None:
    xs = [1.0, 8.0, 64.0, 128.0]
    assert abs(log_log_slope(xs, [1 / x for x in xs]) + 1.0) < 1e-12, "pure 1/B must fit p = -1"
    assert abs(log_log_slope(xs, [1.0 for _ in xs])) < 1e-12, "a saturated form must fit p = 0"
    assert abs(log_log_slope(xs, [x ** -0.5 for x in xs]) + 0.5) < 1e-12
    # The fit must be able to come back NOT -1, or it cannot distinguish the two cases it exists
    # to distinguish.
    assert log_log_slope(xs, [1 / x for x in xs]) < log_log_slope(xs, [1.0 for _ in xs])

    # A curve that is all Lambda_0 must read as 100% scaling, and a flat one as 0% — and the
    # share must fall as a flat component is added, which is the model's own prediction.
    assert abs(scaling_share(-0.68, -0.68) - 1.0) < 1e-12
    assert abs(scaling_share(0.0, -0.68)) < 1e-12
    mixed = [0.4 * (x ** -0.68) + 0.6 for x in xs]
    assert scaling_share(log_log_slope(xs, mixed), -0.68) < 1.0
    assert scaling_share(-0.29, -0.68) > scaling_share(-0.26, -0.68), (
        "a flatter Lambda must read as a smaller scaling share")

    total, estimator = 1000.0, 10.0
    share = math.log(estimator) / math.log(total)
    assert abs(share - 1 / 3) < 1e-12, "a 10x factor inside a 1000x suppression is one third"
    assert share < estimator / total * 100, "log share must not read as a linear share"
    log.info("self-test OK")


if __name__ == "__main__":
    main()
