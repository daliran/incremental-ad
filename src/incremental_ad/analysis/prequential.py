"""Prequential scoring of the forecasting strategies: update at period k, score on period k + 1.

    python -m incremental_ad.analysis.prequential --run_dir $RUNS_ROOT/<merge_exp>/<id> \\
        --chain_dir $RUNS_ROOT/<seq_exp>/<id> --out $WORK/prequential

§1.43. Training-free: every model scored here already exists as a checkpoint. After the update
at fine-tune period k (0-indexed, k = 0 … n-2) three strategies have a model, and each is scored
on period k + 1's held-out validation slice, which none of them has seen:

- **merge_k** — task arithmetic over the first k + 1 task vectors, θ₀ + α·Σ_{i≤k} τ_i. α is
  chosen by the pipeline's own rule restricted to what exists at that time: minimum
  `forecast/mse` on the union of the base's and periods 0…k's validation slices, over §1.40's
  grid, ties to the smaller α. The run's committed α was chosen for all n periods and would use
  data from the future.
- **chain_k** — the sequential chain after step k (`continual_k` of the paired chain run).
- **specialist_k** — the newest specialist, θ₀ + τ_k (`finetune_k`).

The base θ₀ is scored as a reference. **Joint training and the window strategy are NOT scored**:
joint training has one model trained on every period at once, so there is no "after period k"
model; the window strategy trains a new model per budget W and only its final model exists.
Approximating either would be a different experiment.

Self-checks, each a hard failure:
- the chain run must share the merge run's dataset partition (every `dataset_*` argument);
- the chain score recomputed here must equal the chain run's own `backward_transfer.csv` entry
  (forecasting scoring is deterministic), which binds "period k + 1's slice" to the same data the
  pipeline evaluated;
- at k = n - 1 the prefix union has the same size as the pipeline's merged-val union.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("prequential")

GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
METRIC = "forecast/mse"
CHAIN_TOL = 1e-5
# Dataset flags added after some runs were made. A run that predates a flag used its default, so
# an ABSENT key is read as that default — named one by one, never inferred, and the chain
# rescoring check below still verifies the data is the same.
PREDATES_FLAG = {"dataset_series_fraction": 1.0}


def score(model, state, data, configurator, runner, seed) -> float:
    model.load_state_dict(state)
    evaluator = configurator.create_val_evaluator()
    return float(runner.run(model, evaluator, data, seed=seed)[METRIC])


def chain_csv_value(chain_dir: Path, after_step: int, column: str) -> float | None:
    path = chain_dir / "continual_summary" / "backward_transfer.csv"
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (int(row["after_step"]) == after_step and row["column"] == column
                    and row["metric"] == METRIC):
                return float(row["value"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True, help="merge run")
    parser.add_argument("--chain_dir", type=Path, required=True, help="paired sequential run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    from torch.utils.data import ConcatDataset

    from incremental_ad.analysis.remerge import eval_seed_of
    from incremental_ad.analysis.remerge_provenance import write_result
    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.merging.task_vectors import apply_task_vectors, task_vector

    run, chain = args.run_dir, args.chain_dir
    run_args = json.loads((run / "config.json").read_text())["args"]
    chain_args = json.loads((chain / "config.json").read_text())["args"]
    differ = sorted(k for k in set(run_args) | set(chain_args)
                    if k.startswith("dataset_")
                    and run_args.get(k, PREDATES_FLAG.get(k)) != chain_args.get(k, PREDATES_FLAG.get(k)))
    if differ or run_args.get("dataset") != chain_args.get("dataset") \
            or run_args.get("seed") != chain_args.get("seed"):
        raise SystemExit(f"chain {chain} is not paired with {run}: differs on "
                         f"{differ or 'dataset/seed'}")

    dataset, model, configurator, runner, _cfg = _load_run(run)
    model.to(runner.device)
    seed = eval_seed_of(run_args)
    base = load_model_state(run / "baseline" / "checkpoints" / "best.pt")
    finetunes = sorted(run.glob("finetune_*/checkpoints/best.pt"),
                       key=lambda p: int(p.parent.parent.name.split("_")[1]))
    specialists = [load_model_state(p) for p in finetunes]
    taus = [task_vector(base, ft) for ft in specialists]
    n = len(taus)
    chain_base = chain_args.get("pipeline_baseline_checkpoint")
    shared_base = bool(chain_base) and Path(chain_base).resolve() == \
        (run / "baseline" / "checkpoints" / "best.pt").resolve()

    base_val = dataset.get_baseline_val_eval_dataset()
    period_val = [dataset.get_finetune_val_eval_dataset(i) for i in range(n)]
    full_union = dataset.get_merged_val_eval_dataset()
    prefix_full = ConcatDataset([base_val, *period_val])
    if len(prefix_full) != len(full_union):
        raise SystemExit(f"prefix union at k = n-1 has {len(prefix_full)} windows but the "
                         f"pipeline's merged-val union has {len(full_union)}")

    steps = []
    for k in range(n - 1):
        target = period_val[k + 1]
        union = ConcatDataset([base_val, *period_val[:k + 1]])
        prefix = [taus[i] for i in range(k + 1)]
        curve = []
        for alpha in GRID:
            state = apply_task_vectors(base, prefix, alpha)
            curve.append((alpha, score(model, state, union, configurator, runner, seed),
                          score(model, state, target, configurator, runner, seed)))
        chosen = min(curve, key=lambda c: (c[1], c[0]))
        chain_state = load_model_state(chain / f"continual_{k}" / "checkpoints" / "best.pt")
        chain_score = score(model, chain_state, target, configurator, runner, seed)
        recorded = chain_csv_value(chain, k + 1, f"val_{k + 1}")
        if recorded is None or abs(chain_score - recorded) > CHAIN_TOL * max(1.0, recorded):
            raise SystemExit(f"k={k}: chain rescored {chain_score} but its own CSV says "
                             f"{recorded} — period {k + 1}'s slice is not the pipeline's")
        row = {
            "k": k, "target_period": k + 1, "n_target_windows": len(target),
            "n_union_windows": len(union),
            "base": score(model, base, target, configurator, runner, seed),
            "specialist": score(model, specialists[k], target, configurator, runner, seed),
            "chain": chain_score, "chain_csv": recorded,
            "merge": chosen[2], "merge_alpha": chosen[0],
            "merge_curve": [{"alpha": a, "union_mse": v, "target_mse": t} for a, v, t in curve],
        }
        log.info("[k=%d -> %d] base %.4f  merge %.4f (a=%g)  chain %.4f  specialist %.4f",
                 k, k + 1, row["base"], row["merge"], row["merge_alpha"], row["chain"],
                 row["specialist"])
        steps.append(row)

    out = args.out / f"{run.parent.name}__{run.name}"
    out.mkdir(parents=True, exist_ok=True)
    write_result(out, {
        "source_run": str(run), "chain_run": str(chain), "n_shards": n,
        "seed": run_args.get("seed"), "eval_seed": seed, "metric": METRIC,
        "alpha_grid": list(GRID), "shared_base": shared_base,
        "not_scored": {"joint": "no model exists after period k (trained once on all periods)",
                       "window": "only the final model per budget W exists"},
        "steps": steps, "metrics": {},
    })


if __name__ == "__main__":
    main()
