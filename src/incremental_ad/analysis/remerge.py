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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--merge_rule", choices=["sum", "opcm", "opcm_paper"], default="sum",
                        help="'opcm' is the SIMPLIFIED operator of §1.31/§1.35 (residual against "
                             "the flattened predecessors); 'opcm_paper' is Tang et al. 2025 "
                             "Algorithm 1 — two-sided projection out of the top-alpha singular "
                             "subspace of the accumulated merged matrix, plus the norm-stabilising "
                             "lambda. They are different rules and are never reported as one.")
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

    # --- the requested merge -------------------------------------------------------------------
    lambdas: list[float] = []
    from incremental_ad.framework.merging.opcm import merge_opcm_paper

    transform = opcm_residual(args.opcm_threshold) if args.merge_rule == "opcm" else None
    if args.merge_rule == "opcm_paper" and args.coefficient_source != "scale":
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
    if args.merge_rule == "opcm_paper":
        # The paper's OPCM sets its own magnitude (lambda^(t) pins the merged model at the mean
        # task-vector norm from the base), so no merge scale applies to it. The committed alpha is
        # still read and self-checked above — that is what the PLAIN-SUM comparison runs at — but
        # it does not enter this merge, and pretending otherwise would not be the paper's method.
        merged, opcm_info = merge_opcm_paper(base_state, taus, args.opcm_threshold)
        log.info("[opcm_paper] alpha_threshold=%.2f  lambda^(T)=%.4f  mean||tau||=%.4f  "
                 "||merged-base||=%.4f  norm_ratio=%.6f  implied alpha*n=%.4f  "
                 "(%d matrices projected, %d tensors passed through)",
                 args.opcm_threshold, opcm_info["lambda_final"],
                 opcm_info["mean_task_vector_norm"], opcm_info["merged_norm"],
                 opcm_info["norm_ratio"], opcm_info["implied_alpha_times_n"],
                 int(opcm_info["projected_matrices"]), int(opcm_info["passthrough_tensors"]))
        if abs(opcm_info["norm_ratio"] - 1.0) > 1e-6:
            raise SystemExit(f"OPCM norm_ratio {opcm_info['norm_ratio']:.9f} != 1.0 — Theorem "
                             f"5.2's rescaling did not hold, so this is not the paper's operator")
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
