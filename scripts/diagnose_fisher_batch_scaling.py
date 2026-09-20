"""Does the Fisher numerator scale as 1/B? The measurement that separates artifact from finding.

    python scripts/diagnose_fisher_batch_scaling.py --run_dir <adaptive run> \\
        --batch_sizes 1 8 64 128 --samples 8192 --out results_archive/audit/fisher_scaling

`diagonal_fisher` squares the gradient of a **batch-mean** loss, so for batch size B it estimates

    E[F_hat] = g^2 + sigma^2 / B

At a converged minimum g ~ 0 and the estimate reads sigma^2/B — a property of the dataloader, not
of the model. lambda*'s numerator is F_t(theta_hat_t), taken exactly at such a minimum, while
Lambda's terms are taken at merged points that are NOT minima of their own shard and so keep
g^2 > 0. That asymmetry suppresses lambda by an amount that depends on B.

**The sample count is not the variable.** K does not appear in the expectation above: more samples
shrink the *variance* of the estimate, not its mean. So N is held fixed here and only B moves; an
earlier version of this check varied the two together and would have measured nothing extra.

**The expected floor is t, not 1.** Lambda_t is a sum of t Fishers, so even with a perfect
estimator and identical curvature everywhere Lambda/F ~ t. A residual above that floor once the
1/B effect is removed is a real property of the geometry -- the accumulator sitting far from
every shard's minimum while each fresh fine-tune sits at its own -- and is reportable as such.
The outcome to plan for is "artifact of size X, finding of size Y", not one or the other.

Training-free: it reloads the run's saved checkpoints and replays the accumulator chain from the
**recorded** lambdas, so every B sees the identical trajectory and the estimator is the only thing
that varies. theta_hat_t is checkpointed; theta*_t is not, but it is exactly determined by
theta*_t = (1-lambda_t) theta*_(t-1) + lambda_t theta_hat_t from theta_0 and those lambdas.
Cost is Fisher passes only.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("fisher_scaling")


def _float_keys(state):
    return [k for k, v in state.items() if v.is_floating_point()]


def _quadratic(fisher, displacement):
    """d^T F d, over the parameters F actually covers (it omits buffers)."""
    return sum(float((fisher[k] * displacement[k] ** 2).sum())
               for k in displacement if k in fisher)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 64, 128])
    parser.add_argument("--samples", type=int, default=8192,
                        help="held FIXED across batch sizes; only B varies")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.merging.became import accumulate_precision, diagonal_fisher

    run = args.run_dir
    recorded = list(csv.DictReader(
        (run / "continual_summary" / "adaptive_lambdas.csv").open(encoding="utf-8")))
    config = json.loads((run / "config.json").read_text())["args"]
    seed_from_base = bool(config.get("pipeline_lambda_seed_from_base", False))
    data_weighted = config.get("pipeline_lambda_fisher_weighting") == "data"

    dataset, model, _configurator, runner, _cfg = _load_run(run)
    model.to(runner.device)

    theta_zero = load_model_state(Path(config["pipeline_baseline_checkpoint"]))
    checkpoints = sorted(
        (p for p in run.glob("continual_*/checkpoints/best.pt")
         if p.parent.parent.name.split("_")[1].isdigit()),
        key=lambda p: int(p.parent.parent.name.split("_")[1]))
    segments = dataset.get_incremental_segments()
    assert len(checkpoints) == len(recorded) == len(segments), (
        f"{len(checkpoints)} checkpoint(s), {len(recorded)} recorded lambda(s), "
        f"{len(segments)} segment(s) — these must line up one-to-one")
    keys = _float_keys(theta_zero)

    rows = []
    for batch_size in args.batch_sizes:
        n_batches = max(1, args.samples // batch_size)
        log.info("[scaling] B=%d x %d batches = %d samples", batch_size, n_batches,
                 n_batches * batch_size)

        accumulator = {k: theta_zero[k].clone() for k in keys}
        precision = None
        if seed_from_base:
            base = dataset.get_baseline()
            model.load_state_dict(theta_zero)
            weight = float(len(base.train)) if data_weighted else 1.0
            precision = accumulate_precision(
                None,
                diagonal_fisher(
                    model,
                    runner.loader_config.make_loader(
                        base.train, shuffle=True, batch_size=batch_size),
                    runner.device, max_batches=n_batches),
                weight)

        for record, checkpoint, segment in zip(recorded, checkpoints, segments):
            step = int(record["step"])
            lam = float(record["lambda_star"])
            theta_hat = load_model_state(checkpoint)
            displacement = {k: theta_hat[k].double().cpu() - accumulator[k].double().cpu()
                            for k in keys}

            model.load_state_dict(theta_hat)
            fisher_hat = diagonal_fisher(
                model,
                runner.loader_config.make_loader(
                    segment.train, shuffle=True, batch_size=batch_size),
                runner.device, max_batches=n_batches)
            numerator = _quadratic(fisher_hat, displacement)
            lambda_term = _quadratic(precision, displacement) if precision else 0.0
            denominator = numerator + lambda_term
            rows.append({
                "batch_size": batch_size,
                "n_batches": n_batches,
                "samples": n_batches * batch_size,
                "step": step,
                "fisher_num": numerator,
                "lambda_term": lambda_term,
                "fisher_den": denominator,
                "ratio_lambda_over_f": lambda_term / numerator if numerator else "",
                "floor_is_t": step,
                "implied_lambda": numerator / denominator if denominator else "",
                "recorded_lambda": lam,
                "recorded_fisher_num": float(record["fisher_num"]),
                "recorded_fisher_den": float(record["fisher_den"]),
                "d_norm": float(record["d_norm"]),
                "fisher_star_num": "",
                "asymmetry_hat_over_star": "",
            })
            log.info("  step %d  num=%.4e  Lambda-term=%.4e  ratio=%.1fx (floor %d)  "
                     "implied lambda=%.4f  (recorded %.4f)",
                     step, numerator, lambda_term,
                     lambda_term / numerator if numerator else float("nan"), step,
                     numerator / denominator if denominator else float("nan"), lam)

            # Advance with the RECORDED lambda: the trajectory is held fixed so that the only
            # thing differing between batch sizes is the estimator.
            merged = {k: v.clone() for k, v in theta_hat.items()}
            for key in keys:
                merged[key] = ((1.0 - lam) * accumulator[key].double()
                               + lam * theta_hat[key].double()).to(accumulator[key].dtype)
            model.load_state_dict(merged)
            weight = float(len(segment.train)) if data_weighted else 1.0
            fisher_star = diagonal_fisher(
                model,
                runner.loader_config.make_loader(
                    segment.train, shuffle=True, batch_size=batch_size),
                runner.device, max_batches=n_batches)
            # The same quadratic form as the numerator, on the same task, the same data and the
            # same d -- only the evaluation point moves. F_t(theta_hat_t)/F_t(theta*_t) is the
            # at-a-minimum vs not-at-a-minimum asymmetry with everything else held constant,
            # and it is the candidate mechanism for the residual the 1/B artifact leaves behind.
            rows[-1]["fisher_star_num"] = _quadratic(fisher_star, displacement)
            rows[-1]["asymmetry_hat_over_star"] = (
                numerator / rows[-1]["fisher_star_num"] if rows[-1]["fisher_star_num"] else "")
            log.info("    F_t(theta_hat)/F_t(theta*) = %.1fx",
                     numerator / rows[-1]["fisher_star_num"]
                     if rows[-1]["fisher_star_num"] else float("nan"))
            precision = accumulate_precision(precision, fisher_star, weight)
            accumulator = {k: merged[k].clone() for k in keys}

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"fisher_scaling_{run.parent.name}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s (%d row(s))", path, len(rows))


if __name__ == "__main__":
    main()
