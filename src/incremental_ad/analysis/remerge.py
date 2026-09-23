"""Re-merge a finished run under different rules, without retraining anything.

    python -m incremental_ad.analysis.remerge --run_dir $RUNS_ROOT/<exp>/<id> \\
        --coefficient_source became --fisher_batches 256 --out $OUT/remerge

A finished `IncrementalTaskArithmeticPipeline` run already contains everything a merge needs:
`baseline/checkpoints/best.pt` and one `finetune_i/checkpoints/best.pt` per shard. Changing the
merge *rule* or the *coefficient* therefore costs a merge and an evaluation pass — not a
retrain. That is what makes §1.32's question answerable at all: sweeping the Fisher sample size
over four values and two draws is 8 re-merges per run, and 8 retrains would not be affordable.

**Never writes into the run it reads.** Output goes to `--out/<experiment>__<run_id>/<tag>/`.
A run directory is evidence; an analysis that mutates its own input destroys the ability to
re-run it, and this project has already lost numbers that way (§1.7, §1.8, §1.10's seed 42).

**Configuration is read back from the run's own `config.json`**, the same discipline
`diagnose.py` uses, so the dataset and model cannot drift from what produced the checkpoints.

**Self-check, run by default.** Before computing anything, the stored
`merged/checkpoints/best.pt` is rebuilt **the way the source run built it** — plain sum at the
committed α for a `sum`+`scale` run (`merge_scale/selected` first, `config.json` second), or the
recorded λ\* for a BECAME run — and must match **bitwise**. If it does not, the re-merge path is
not the pipeline's path and nothing computed here is comparable to that run.

The first version of this guard always assumed a `sum`+`scale` source and so failed on all 18
BECAME runs it was pointed at: it refused to emit numbers, which is the wanted behaviour, but for
the wrong reason. Reconstructing what the source *actually* did is both correct and a stronger
check, since for a BECAME source it exercises the exact fold this script then uses.
`--skip_self_check` is for a source with no `merged/` at all, not a way past a failure.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from incremental_ad.analysis.remerge_provenance import write_result

log = logging.getLogger("remerge")


def committed_alpha(run: Path, args: dict) -> tuple[float, str]:
    """The α the stored merge was built at. `merge_scale/selected` wins over `config.json`."""
    result = run / "merged" / "val" / "result.json"
    if result.is_file():
        try:
            metrics = (json.loads(result.read_text()) or {}).get("metrics") or {}
            if "merge_scale/selected" in metrics:
                return float(metrics["merge_scale/selected"]), "merge_scale/selected"
        except (json.JSONDecodeError, OSError):
            pass
    return float(args.get("pipeline_merge_scale", 1.0)), "config.json"


def evaluate(model, dataset, configurator, runner, seed,
             test_only: bool = False) -> dict[str, float]:
    """Merged-val and test metrics for whatever `model` currently holds."""
    metrics: dict[str, float] = {}
    splits = [("val", dataset.get_merged_val_eval_dataset), ("test", dataset.get_test_dataset)]
    for split, loader_fn in (splits[1:] if test_only else splits):
        evaluator = (configurator.create_val_evaluator() if split == "val"
                     else configurator.create_test_evaluator())
        try:
            scored = runner.run(model, evaluator, loader_fn(), seed=seed)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("  %s evaluation failed: %s", split, exc)
            continue
        metrics.update({f"{split}/{k}": v for k, v in scored.items()})
    return metrics


def run_baseline_grid(args, run, run_args, base_state, taus, alpha, alpha_source,
                      model, dataset, configurator, runner) -> None:
    """Build one rule's Delta once, then evaluate theta_0 + a * Delta for every a in the grid.

    Also records ``||Delta||`` and TA's ``||sum tau||``, so the distance each merge travels from
    the base — ``a * ||Delta||`` — can be read against task arithmetic's at its own alpha. The
    rules put their deltas on different scales (module docstring of `interference.py`), and
    §1.37/§1.38 showed that comparing coefficients across a transform, rather than distances,
    can overstate a difference about twofold.
    """
    import torch

    from incremental_ad.framework.merging.interference import DELTAS
    from incremental_ad.framework.merging.task_vectors import apply_task_vectors

    seed = run_args.get("seed")
    rule = args.baseline_rule
    builder = DELTAS[rule]
    # DARE's mask is seeded by the TRAINING seed, so the three seeds of a configuration draw
    # three independent masks and the mask's own variance lands in the reported seed spread.
    delta = builder(taus, seed=int(seed or 0)) if rule == "dare" else builder(taus)

    def norm(state) -> float:
        return float(torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in state.values())))

    delta_norm = norm(delta)
    ta_norm = norm(DELTAS["ta"](taus))
    log.info("[%s] ||Delta||=%.4f  ||sum tau||=%.4f  ratio %.4f  — %d alphas",
             rule, delta_norm, ta_norm, delta_norm / ta_norm if ta_norm else float("nan"),
             len(args.alpha_grid))

    points = [(value, f"{rule}_a{value:.2f}") for value in args.alpha_grid]
    if args.distance_match_alpha is not None:
        matched = args.distance_match_alpha * ta_norm / delta_norm
        points.append((matched, f"{rule}_dm"))
        log.info("[%s] distance-matched to TA at alpha=%.2f: alpha=%.6f travels %.6f",
                 rule, args.distance_match_alpha, matched, matched * delta_norm)

    for value, tag in points:
        model.load_state_dict(apply_task_vectors(base_state, [delta], value))
        metrics = evaluate(model, dataset, configurator, runner, seed, args.test_only)
        out_dir = args.out / f"{run.parent.name}__{run.name}" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_run": str(run), "merge_rule": f"baseline:{rule}",
            "baseline_rule": rule, "alpha": value, "alpha_grid": list(args.alpha_grid),
            "committed_alpha": alpha, "committed_alpha_source": alpha_source,
            "delta_norm": round(delta_norm, 6), "ta_sum_norm": round(ta_norm, 6),
            "distance_from_base": round(value * delta_norm, 6),
            "distance_matched_to_ta_alpha": (args.distance_match_alpha
                                             if tag.endswith("_dm") else None),
            "dare_seed": int(seed or 0) if rule == "dare" else None,
            "seed": seed, "n_shards": len(taus), "metrics": metrics,
        }
        log.info("[%s] alpha=%.2f  %s", rule, value,
                 {k: round(v, 5) for k, v in metrics.items()
                  if isinstance(v, float) and k.split("/", 1)[1] in
                  ("forecast/mse", "window_auroc", "reconstruction/score_mean")})
        write_result(out_dir, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--merge_rule",
                        choices=["sum", "opcm", "opcm_paper", "opcm_paper_committed",
                                 "opcm_paper_distance"],
                        default="sum",
                        help="'opcm' is the SIMPLIFIED operator of §1.31/§1.35 (residual against "
                             "the flattened predecessors); 'opcm_paper' is Tang et al. 2025 "
                             "Algorithm 1 — two-sided projection out of the top-alpha singular "
                             "subspace of the accumulated merged matrix, plus the norm-stabilising "
                             "lambda. They are different rules and are never reported as one. "
                             "'opcm_paper_committed' is the paper's PROJECTION with its Thm-5.2 "
                             "rescale replaced by the run's committed alpha (§1.37, C34) — not "
                             "the paper's method either, and labelled 'the paper's projection at "
                             "a chosen strength'. 'opcm_paper_distance' is the same projection "
                             "scaled so the merge travels EXACTLY as far from the base as plain "
                             "summation at the committed alpha (§1.38, P5) — the control that "
                             "coefficient-matching only approximates once a transform shrinks "
                             "the task vectors.")
    parser.add_argument("--coefficient_source",
                        choices=["scale", "became", "became_rescaled"], default="scale")
    parser.add_argument("--reverse_order", action="store_true",
                        help="feed the periods NEWEST-FIRST. Order is inert for plain summation "
                             "and decisive for OPCM, which projects each incoming vector out of "
                             "its predecessors' span: reversed, the newest shard is never "
                             "projected and the oldest is projected most. This is the "
                             "falsification test for the recency-filter hypothesis (§1.36).")
    parser.add_argument("--merge_scale", type=float, default=None,
                        help="for --coefficient_source scale; defaults to the run's committed α")
    parser.add_argument("--opcm_threshold", type=float, default=0.5)
    parser.add_argument("--fisher_batches", type=int, default=64)
    parser.add_argument("--fisher_seed", type=int, default=0,
                        help="shuffling seed for the Fisher loader. Two draws at one sample size "
                             "measure the estimator's own variance, which is the thing §1.32 "
                             "needs to separate from real shard differences.")
    parser.add_argument("--baseline_rule", choices=["ta", "dare", "ties", "iso_c", "tsv"],
                        default=None,
                        help="the supervisor's interference-reducing rules "
                             "(framework/merging/interference.py). Each is theta_0 + alpha * "
                             "Delta with Delta independent of alpha, so Delta is built ONCE and "
                             "every alpha in --alpha_grid is evaluated from it; one result.json "
                             "per alpha. 'ta' is the same-protocol control. Selection of alpha "
                             "happens downstream, on the val metrics written here — this script "
                             "measures, it does not choose.")
    parser.add_argument("--alpha_grid", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--distance_match_alpha", type=float, default=None,
                        help="--baseline_rule only: ALSO evaluate the rule at the alpha where "
                             "||alpha * Delta|| equals task arithmetic's ||A * sum tau||, i.e. "
                             "where it travels exactly as far from the base as TA does at alpha "
                             "A. Written under tag '<rule>_dm'. This is §1.38's control: the "
                             "rules put Delta on different scales, so a shared alpha compares "
                             "magnitude and direction at once, and a shared DISTANCE isolates "
                             "direction.")
    parser.add_argument("--test_only", action="store_true",
                        help="--baseline_rule only: skip the merged-val pass. For AD, where val "
                             "selection is refused (§1.12) and the protocol reads a fixed alpha, "
                             "that pass is never used — and on SWaT each one costs minutes.")
    parser.add_argument("--tag", default=None, help="output subdirectory name")
    parser.add_argument("--skip_self_check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.merging.became import became_weights, diagonal_fisher
    from incremental_ad.framework.merging.task_vectors import (
        merge_sequential, merge_task_arithmetic, opcm_residual, task_vector,
    )
    from incremental_ad.framework.pipelines.incremental_task_arithmetic_pipeline import (
        became_uniformity, became_weights_per_vector,
    )

    run = args.run_dir
    config = json.loads((run / "config.json").read_text())
    run_args = config.get("args") or {}
    alpha, alpha_source = committed_alpha(run, run_args)
    if args.merge_scale is not None:
        alpha, alpha_source = args.merge_scale, "--merge_scale"

    baseline = run / "baseline" / "checkpoints" / "best.pt"
    finetunes = sorted(run.glob("finetune_*/checkpoints/best.pt"),
                       key=lambda p: int(p.parent.parent.name.split("_")[1]))
    if not (baseline.is_file() and finetunes):
        raise SystemExit(f"{run} has no baseline + finetune checkpoints to merge")

    dataset, model, configurator, runner, _cfg = _load_run(run)
    model.to(runner.device)
    base_state = load_model_state(baseline)
    ft_states = [load_model_state(p) for p in finetunes]
    taus = [task_vector(base_state, ft) for ft in ft_states]
    log.info("[remerge] %s — %d shards, committed alpha=%s (from %s)",
             run, len(taus), alpha, alpha_source)

    # --- self-check: rebuild the stored merge the way the source run built it ------------------
    # The check must reconstruct what the *source run* did, not what this invocation is asking
    # for. A first version always tried plain sum at the committed alpha, which is correct only
    # for a `sum`+`scale` source; against the BECAME runs it failed all 18 times — the guard
    # firing on itself rather than on a real defect. It refused to emit numbers, which is the
    # behaviour wanted, but for the wrong reason.
    stored = run / "merged" / "checkpoints" / "best.pt"
    source_rule = run_args.get("pipeline_merge_rule", "sum")
    source_coefficient = run_args.get("pipeline_coefficient_source", "scale")
    if not args.skip_self_check and stored.is_file():
        recorded = run / "merged" / "became_lambdas.csv"
        if source_coefficient == "became" and recorded.is_file():
            # lambda* is not recomputable without the original Fishers, but it was *recorded*.
            # Rebuilding from the recorded values is the stronger check anyway: it exercises the
            # exact fold this script will use, on this run's own coefficients.
            with recorded.open() as fh:
                lambdas_ref = [float(r["lambda_star"]) for r in csv.DictReader(fh)]
            reference_weights = [(1.0 - lam, lam) for lam in lambdas_ref]
            how = f"recorded lambda* {[round(x, 4) for x in lambdas_ref]}"
        elif source_coefficient == "became":
            log.warning("[remerge] self-check skipped: source used BECAME and has no recorded "
                        "lambdas, so its merge cannot be rebuilt from checkpoints")
            reference_weights = None
            how = ""
        else:
            reference_weights = [(1.0, alpha)] * len(taus)
            how = f"plain sum at alpha={alpha}"

        if reference_weights is not None:
            reference_transform = (opcm_residual(run_args.get("pipeline_opcm_threshold", 0.5))
                                   if source_rule == "opcm" else None)
            rebuilt = merge_sequential(base_state, taus, reference_weights,
                                       transform=reference_transform)
            reference = load_model_state(stored)
            mismatched = [k for k, v in reference.items()
                          if k not in rebuilt or not torch.equal(v, rebuilt[k])]
            if mismatched:
                raise SystemExit(
                    f"self-check FAILED: rebuilding the source merge ({source_rule} + "
                    f"{source_coefficient}, {how}) does not reproduce {stored} "
                    f"({len(mismatched)} tensor(s) differ). The re-merge path is not the "
                    f"pipeline's path, so nothing computed here is comparable to this run."
                )
            log.info("[remerge] self-check ok — %s + %s via %s is bitwise identical to the "
                     "stored merge", source_rule, source_coefficient, how)

    # --- the supervisor's rules: one Delta, a grid of alphas --------------------------------
    # Entirely separate from the path below, which produces every published re-merge; nothing in
    # it is touched. The self-check above has already run, so a run that fails to rebuild its own
    # stored merge never reaches this point.
    if args.baseline_rule is not None:
        run_baseline_grid(args, run, run_args, base_state, taus, alpha, alpha_source,
                          model, dataset, configurator, runner)
        return

    # --- the requested merge -------------------------------------------------------------------
    lambdas: list[float] = []
    from incremental_ad.framework.merging.opcm import merge_opcm_paper

    transform = opcm_residual(args.opcm_threshold) if args.merge_rule == "opcm" else None
    if args.merge_rule.startswith("opcm_paper") and args.coefficient_source != "scale":
        raise SystemExit("--merge_rule opcm_paper sets its own coefficients (Algorithm 1 line 14); "
                         "it cannot be combined with a coefficient source")

    # Order reversal happens here, *after* the self-check has validated the forward-order
    # reconstruction, so a reversed run still proves it can rebuild the original merge first.
    if args.reverse_order:
        taus = list(reversed(taus))
        ft_states = list(reversed(ft_states))
        log.info("[remerge] periods reversed — newest first, so the newest shard is never "
                 "projected and the oldest is projected most")

    if args.coefficient_source in ("became", "became_rescaled"):
        segments = dataset.get_incremental_segments()
        if len(segments) != len(ft_states):
            raise SystemExit(f"{len(segments)} segments but {len(ft_states)} finetunes — the "
                             f"dataset read back from config.json does not match the run")
        fishers = []
        saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for index, (segment, ft_state) in enumerate(zip(segments, ft_states)):
            model.load_state_dict(ft_state)
            torch.manual_seed(args.fisher_seed)
            loader = runner.loader_config.make_loader(segment.train, shuffle=True)
            log.info("[fisher] shard %d, %d batches, draw seed %d",
                     index, args.fisher_batches, args.fisher_seed)
            fishers.append(diagonal_fisher(model, loader, runner.device,
                                           max_batches=args.fisher_batches))
        model.load_state_dict(saved)
        weights, lambdas = became_weights(base_state, taus, fishers)
        if args.coefficient_source == "became_rescaled":
            # BECAME's *relative* weighting, at a chosen total strength.
            #
            # The convex fold pins the per-period weights to sum to 1, i.e. alpha*n = 1.0, whatever
            # the Fishers say (§1.35). That conflates two questions: is the Fisher *weighting*
            # useful, and is the *magnitude* right? This separates them — keep the ratios, rescale
            # the total to the source run's committed alpha*n — and must be labelled
            # "BECAME's weighting at a chosen strength", never "BECAME".
            per_vector = became_weights_per_vector(lambdas)
            total = sum(per_vector)
            target = alpha * len(taus)
            if total <= 0:
                raise SystemExit("BECAME weights sum to zero — cannot rescale")
            scaled = [w * target / total for w in per_vector]
            # decay 1 with explicit per-vector coefficients: the fold then lands on
            # theta_0 + sum(scaled_i * tau_i) rather than on a convex combination.
            weights = [(1.0, w) for w in scaled]
            log.info("[remerge] became_rescaled — relative weights %s rescaled to alpha*n=%.4f "
                     "(committed alpha %.4f x %d shards)",
                     [round(w / total, 4) for w in per_vector], target, alpha, len(taus))
    else:
        weights = [(1.0, alpha)] * len(taus)

    _uses_fisher = args.coefficient_source in ("became", "became_rescaled")
    if args.merge_rule.startswith("opcm_paper"):
        # The paper's OPCM sets its own magnitude (lambda^(t) pins the merged model at the mean
        # task-vector norm from the base), so no merge scale applies to it. The committed alpha is
        # still read and self-checked above — that is what the PLAIN-SUM comparison runs at — but
        # it does not enter this merge, and pretending otherwise would not be the paper's method.
        # §1.37: the matched-magnitude variant merges at the strength the run committed to,
        # so its comparison against plain summation at that same alpha isolates the projection.
        committed = alpha if args.merge_rule == "opcm_paper_committed" else None
        at_distance = alpha if args.merge_rule == "opcm_paper_distance" else None
        merged, opcm_info = merge_opcm_paper(base_state, taus, args.opcm_threshold,
                                             scale_to_alpha=committed,
                                             match_distance_at_alpha=at_distance)
        log.info("[opcm_paper] alpha_threshold=%.2f  lambda^(T)=%.4f  mean||tau||=%.4f  "
                 "||merged-base||=%.4f  norm_ratio=%.6f  implied alpha*n=%.4f  "
                 "(%d matrices projected, %d tensors passed through)",
                 args.opcm_threshold, opcm_info["lambda_final"],
                 opcm_info["mean_task_vector_norm"], opcm_info["merged_norm"],
                 opcm_info["norm_ratio"], opcm_info["implied_alpha_times_n"],
                 int(opcm_info["projected_matrices"]), int(opcm_info["passthrough_tensors"]))
        if at_distance is not None:
            # The control IS the identity: if the merge did not travel exactly as far as plain
            # summation at this alpha, the cell compares direction AND magnitude, and P5 cannot
            # separate them. Abort rather than write a row that looks like evidence.
            if abs(opcm_info["distance_ratio"] - 1.0) > 1e-6:
                raise SystemExit(f"distance_ratio {opcm_info['distance_ratio']:.9f} != 1.0 — the "
                                 f"merge is not distance-matched to plain summation")
            target = at_distance * len(taus)
            if abs(opcm_info["implied_alpha_times_n"] - target) > 1e-6:
                raise SystemExit(f"implied alpha*n {opcm_info['implied_alpha_times_n']} != "
                                 f"{target}")
            log.info("[opcm_distance] distance-matched to plain sum at alpha=%.4f: "
                     "||merged-base||=%.4f, ratio %.9f, alpha*n %.4f; the paper's own rule would "
                     "have travelled %.4f", at_distance, opcm_info["merged_norm"],
                     opcm_info["distance_ratio"], target,
                     opcm_info["mean_task_vector_norm"])
        elif committed is None:
            # Theorem 5.2 is the paper's guarantee and must hold on every real merge.
            if abs(opcm_info["norm_ratio"] - 1.0) > 1e-6:
                raise SystemExit(f"OPCM norm_ratio {opcm_info['norm_ratio']:.9f} != 1.0 — Theorem "
                                 f"5.2's rescaling did not hold, so this is not the paper's "
                                 f"operator")
        else:
            # Under the matched-magnitude rule alpha*n is an identity, not an estimate: every
            # projected vector enters with coefficient alpha. If it is not hit exactly the
            # comparison is no longer at matched magnitude and the row cannot close C34.
            target = committed * len(taus)
            if abs(opcm_info["implied_alpha_times_n"] - target) > 1e-9:
                raise SystemExit(f"implied alpha*n {opcm_info['implied_alpha_times_n']} != "
                                 f"target {target} — the merge is not at the committed magnitude")
            # The paper's rule enters every projected vector at 1/lambda^(T), so its own
            # alpha*n would have been n/lambda. Logging both makes the overshoot §1.36 measured
            # visible per run rather than only in the aggregate.
            log.info("[opcm_committed] merged at committed alpha=%.4f x %d shards -> alpha*n "
                     "%.4f; the paper's rule would have used %.4f",
                     committed, len(taus), target,
                     len(taus) / opcm_info["lambda_final"])
    else:
        opcm_info = {}
        merged = merge_sequential(base_state, taus, weights, transform=transform)
    model.load_state_dict(merged)

    # --- evaluate ------------------------------------------------------------------------------
    metrics: dict[str, float] = {}
    for split, loader_fn in (("val", dataset.get_merged_val_eval_dataset),
                             ("test", dataset.get_test_dataset)):
        evaluator = (configurator.create_val_evaluator() if split == "val"
                     else configurator.create_test_evaluator())
        try:
            scored = runner.run(model, evaluator, loader_fn(), seed=run_args.get("seed"))
        except Exception as exc:                                    # noqa: BLE001
            log.warning("  %s evaluation failed: %s", split, exc)
            continue
        metrics.update({f"{split}/{k}": v for k, v in scored.items()})

    tag = args.tag or (f"{args.merge_rule}_{args.coefficient_source}"
                       f"_fb{args.fisher_batches}_fs{args.fisher_seed}")
    out_dir = args.out / f"{run.parent.name}__{run.name}" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run": str(run), "merge_rule": args.merge_rule,
        "coefficient_source": args.coefficient_source,
        "reverse_order": args.reverse_order,
        "alpha": alpha, "alpha_source": alpha_source,
        "opcm_threshold": args.opcm_threshold,
        "fisher_batches": args.fisher_batches if _uses_fisher else None,
        "fisher_seed": args.fisher_seed if _uses_fisher else None,
        "implied_alpha_times_n": (round(opcm_info["implied_alpha_times_n"], 6) if opcm_info
                                  else round(sum(c for _d, c in weights), 6)),
        **{f"opcm_{k}": round(v, 6) for k, v in opcm_info.items()},
        "seed": run_args.get("seed"), "n_shards": len(taus), "metrics": metrics,
    }
    if lambdas:
        # For a rescaled run the realised per-vector weights are the fold's coefficients, not the
        # convex ones — recording the convex values would make implied_alpha_times_n read 1.0 and
        # hide the very thing this mode changes.
        per_vector = ([coefficient for _decay, coefficient in weights]
                      if args.coefficient_source == "became_rescaled"
                      else became_weights_per_vector(lambdas))
        deviation = became_uniformity(lambdas)
        with (out_dir / "became_lambdas.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "lambda_star", "one_over_t", "weight", "one_over_n",
                             "max_dev_from_uniform", "fisher_batches", "fisher_seed", "seed"])
            for step, lam in enumerate(lambdas):
                writer.writerow([step, lam, 1.0 / (step + 1), per_vector[step],
                                 1.0 / len(lambdas), deviation, args.fisher_batches,
                                 args.fisher_seed, run_args.get("seed")])
        log.info("[remerge] lambda*=%s  weights=%s  max departure from uniform %.1f%%",
                 [round(x, 4) for x in lambdas], [round(w, 4) for w in per_vector],
                 100 * deviation)
    log.info("[remerge] %s", {k: round(v, 6) for k, v in metrics.items()
                              if isinstance(v, float)})
    # Written LAST and atomically: `result.json` is this run's commit marker, so a job killed at
    # any earlier point leaves no file rather than a file a collector would happily read.
    log.info("wrote %s", write_result(out_dir, payload))


if __name__ == "__main__":
    main()
