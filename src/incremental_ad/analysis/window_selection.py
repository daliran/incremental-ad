"""Choose the window budget W on validation, not on test — the one asymmetry in §1.26.

    python -m incremental_ad.analysis.window_selection --runs_root results_archive/runs \\
        --spec analysis_specs/method_comparison_spec.csv --out $OUT/window_selection

§1.26's `window` column is **the best of W ∈ {1,2,3} on test**. Every other column in that table
is either a single method (`joint`, `sequential`) or val-selected (`merge`, since §1.29). So the
window retrain — the method that wins six of the twelve decisive forecasting rows — is the only
one handed an oracle, and it is handed one over the exact axis that defines it.

⚠️ **The obvious way to remove it does not work, and that is the first result here.** Each
`window_<dataset>_W<k>` run has its own validation tail (`finetune_0/val`), and selecting on that
is invalid: the runs use different `baseline_fraction` (0.9 / 0.8 / 0.7), so each one's fine-tune
segment — and therefore its val tail — is a **different slice of the series**. W=1's tail is the
most recent and easiest, so it wins on val almost regardless of merit. Measured: the val-picked
budget is W=1 in 33 of 54 cells while the test-best is W=3 in 48 of 54, giving a nonsense ~40%
"penalty". Those numbers describe the selection artefact, not the method, and `--mode own_val`
exists only to reproduce them.

**`--mode common_val` is the valid form.** Every window checkpoint is scored on the *same* held-out
data — the merged-val union, which is what merge-scale selection already uses — so the budgets are
compared on one slice. That needs a forward pass per (dataset, W, seed), which is why this mode
needs a GPU while `own_val` is pure file reading.

Reported alongside `window_best` rather than replacing it, because the two answer different
questions — "how good is a window retrain if you choose W perfectly?" and "how good is it if you
choose W the way you would have to?" — and §1.21's retention argument is about the first while
§1.26's method comparison should use the second.

Training-free: it reads `result.json` files that already exist, so it runs from the archive.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("window_selection")

FIELDS = ["dataset", "n", "metric", "seed", "selected_W", "test_best_W", "agrees",
          "window_val", "window_best", "penalty_pct"]
SUMMARY_FIELDS = ["dataset", "n", "metric", "n_seeds", "window_val", "window_val_sd",
                  "window_best", "window_best_sd", "penalty_pct", "agreement"]


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower()
               for k in ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))


def metric_of(run: Path, block: str, metric: str) -> float | None:
    path = run / block / "result.json"
    if not path.is_file():
        return None
    try:
        return (json.loads(path.read_text()) or {}).get("metrics", {}).get(metric)
    except (json.JSONDecodeError, OSError):
        return None


def by_seed(runs_root: Path, experiment: str, block: str, metric: str) -> dict[int, float]:
    """seed -> value, for one experiment and block."""
    out: dict[int, float] = {}
    group = runs_root / experiment
    if not group.is_dir():
        return out
    for run in sorted(p for p in group.iterdir() if p.is_dir()):
        config = run / "config.json"
        if not config.is_file():
            continue
        try:
            seed = (json.loads(config.read_text()) or {}).get("args", {}).get("seed")
        except (json.JSONDecodeError, OSError):
            continue
        value = metric_of(run, block, metric)
        if seed is not None and value is not None:
            out[seed] = value
    return out


def common_val_scores(runs_root: Path, experiment: str, metric: str) -> dict[int, float]:
    """seed -> score of this budget's model on the **merged-val union**.

    The union is the same held-out set merge-scale selection uses, so every budget is judged on
    identical data and the comparison is meaningful. It is not the slice any single window run
    trained against, which is the point: a selection signal has to be common to the things being
    selected between.

    Loads the run's own dataset from its `config.json`, so the evaluated windows cannot drift
    from what the run was trained on. Needs a GPU pass per (experiment, seed).
    """
    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state

    out: dict[int, float] = {}
    group = runs_root / experiment
    if not group.is_dir():
        return out
    for run in sorted(p for p in group.iterdir() if p.is_dir()):
        checkpoint = run / "finetune_0" / "checkpoints" / "best.pt"
        config = run / "config.json"
        if not (checkpoint.is_file() and config.is_file()):
            continue
        try:
            seed = (json.loads(config.read_text()) or {}).get("args", {}).get("seed")
            dataset, model, configurator, runner, _cfg = _load_run(run)
            model.to(runner.device)
            model.load_state_dict(load_model_state(checkpoint))
            metrics = runner.run(model, configurator.create_val_evaluator(),
                                 dataset.get_merged_val_eval_dataset(), seed=seed)
        except Exception as exc:                                   # noqa: BLE001
            log.warning("  %s/%s — skipped (%s)", experiment, run.name, exc)
            continue
        if seed is not None and metric in metrics:
            out[seed] = metrics[metric]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=["own_val", "common_val"], default="common_val",
                        help="own_val reproduces the invalid per-run selection (see the module "
                             "docstring); common_val scores every budget on the merged-val "
                             "union, which is the comparable signal.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    rows: list[dict] = []
    summary: list[dict] = []
    for entry in spec:
        metric = entry["metric"]
        higher = higher_is_better(metric)
        pick = max if higher else min
        budgets = {}
        for k in (1, 2, 3):
            experiment = entry.get(f"window_W{k}_experiment")
            if not experiment:
                continue
            val = (by_seed(args.runs_root, experiment, "finetune_0/val", metric)
                   if args.mode == "own_val"
                   else common_val_scores(args.runs_root, experiment, metric))
            test = by_seed(args.runs_root, experiment, "finetune_0/test", metric)
            if val and test:
                budgets[k] = (val, test)
        if len(budgets) < 2:
            continue

        seeds = sorted(set.intersection(*[set(v) for v, _ in budgets.values()]))
        per_seed_val, per_seed_best, agree = [], [], []
        for seed in seeds:
            chosen = pick(budgets, key=lambda k: budgets[k][0][seed])
            best = pick(budgets, key=lambda k: budgets[k][1][seed])
            chosen_test = budgets[chosen][1][seed]
            best_test = budgets[best][1][seed]
            per_seed_val.append(chosen_test)
            per_seed_best.append(best_test)
            agree.append(chosen == best)
            penalty = (100 * (best_test - chosen_test) / best_test if higher
                       else 100 * (chosen_test - best_test) / best_test)
            rows.append({
                "dataset": entry["dataset"], "n": entry["n"], "metric": metric, "seed": seed,
                "selected_W": chosen, "test_best_W": best, "agrees": int(chosen == best),
                "window_val": round(chosen_test, 6), "window_best": round(best_test, 6),
                "penalty_pct": round(penalty, 3),
            })
        if not per_seed_val:
            continue
        mean_val, mean_best = st.mean(per_seed_val), st.mean(per_seed_best)
        summary.append({
            "dataset": entry["dataset"], "n": entry["n"], "metric": metric,
            "n_seeds": len(seeds),
            "window_val": round(mean_val, 6),
            "window_val_sd": round(st.stdev(per_seed_val), 6) if len(per_seed_val) > 1 else 0.0,
            "window_best": round(mean_best, 6),
            "window_best_sd": round(st.stdev(per_seed_best), 6) if len(per_seed_best) > 1 else 0.0,
            "penalty_pct": round(100 * (mean_val - mean_best) / mean_best
                                 * (-1 if higher else 1), 3),
            "agreement": round(sum(agree) / len(agree), 3),
        })
        log.info("  %-14s n=%s  val-picked W agrees with test-best in %d/%d seeds, "
                 "penalty %.2f%%", entry["dataset"], entry["n"], sum(agree), len(agree),
                 summary[-1]["penalty_pct"])

    if not summary:
        raise SystemExit("no window experiments resolved — check --runs_root against the spec")
    agreements = [r["agreement"] for r in summary]
    log.info("\nval-selected W equals test-best W in %.0f%% of (dataset, n, seed) cells; "
             "mean penalty %.2f%%", 100 * st.mean(agreements),
             st.mean(r["penalty_pct"] for r in summary))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, data, fields in (("window_selection_per_seed.csv", rows, FIELDS),
                                   ("window_selection.csv", summary, SUMMARY_FIELDS)):
            with (args.out / name).open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            log.info("wrote %s", args.out / name)


if __name__ == "__main__":
    main()
