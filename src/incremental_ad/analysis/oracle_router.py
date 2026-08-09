"""The best a router could do on the test set: per-window minimum over the specialists.

    python -m incremental_ad.analysis.oracle_router --run_dir $RUNS_ROOT/<exp>/<id> \\
        --out $OUT/oracle_router

For every test window, score each specialist theta_0 + tau_i and keep the **smallest** error,
then aggregate over windows the usual way. That is a true upper bound on routing: no router can
beat picking the best model per window, because that is what it would have to choose. Reported
in the same units as joint / merge / sequential / window, so it sits on the same row rather than
as a percentage against a different baseline.

**Forecasting only, and that is a scope limit rather than a gap.** MSE and MAE are means over
per-window errors, so a per-window minimum is well defined and aggregates correctly. AUROC and
AUPRC are *ranking* metrics over the whole test set: there is no per-window value to minimise,
and mixing anomaly scores from different specialists into one ranking makes the ranking
incoherent -- a high score from one model is not comparable to a high score from another. The
script refuses on anomaly detection rather than emitting a number that looks usable.

This is a different quantity from the routing headroom in EXPERIMENTS.md 1.16, which asks how
far the merged model sits from the best *per-regime* model on that regime's own validation
slice. Both are called "routing"; they answer different questions and must not be pooled. This
one is an oracle on test, so it is unobtainable in practice -- it bounds the prize, it is not a
method anyone can deploy.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("oracle_router")

FIELDS = ["run", "dataset", "n_segments", "seed", "metric", "base", "best_single",
          "oracle_router", "oracle_vs_best_single_pct"]


def _per_window_errors(model, loader, device):
    """(mse, mae) per test window, as 1-D tensors."""
    import torch
    from incremental_ad.framework.core.device import move_to_device

    mse, mae = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds, targets = model.predict_step(move_to_device(batch, device))
            diff = (preds - targets).detach().cpu()
            dims = tuple(range(1, diff.ndim))
            mse.append((diff ** 2).mean(dim=dims))
            mae.append(diff.abs().mean(dim=dims))
    return torch.cat(mse), torch.cat(mae)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state

    config = json.loads((args.run_dir / "config.json").read_text())
    task = config["args"].get("task", "")
    if "forecast" not in task.lower():
        raise SystemExit(
            f"task is {task!r}: an oracle router is only defined for forecasting. AUROC/AUPRC "
            f"rank the whole test set, so there is no per-window minimum to take and scores "
            f"from different specialists are not comparable. See the module docstring."
        )

    dataset, model, _configurator, runner, _cfg = _load_run(args.run_dir)
    device = runner.device
    model.to(device)
    loader = runner.loader_config.make_loader(dataset.get_test_dataset(), shuffle=False)

    specialists = sorted(args.run_dir.glob("finetune_*/checkpoints/best.pt"),
                         key=lambda p: int(p.parent.parent.name.split("_")[1]))
    if not specialists:
        raise SystemExit(f"no finetune_*/checkpoints/best.pt under {args.run_dir}")

    per_model = {}
    for path in specialists:
        name = path.parent.parent.name
        model.load_state_dict(load_model_state(path))
        per_model[name] = _per_window_errors(model, loader, device)
        log.info("  %-12s mse=%.6f", name, per_model[name][0].mean().item())

    base_path = args.run_dir / "baseline" / "checkpoints" / "best.pt"
    base = None
    if base_path.exists():
        model.load_state_dict(load_model_state(base_path))
        base = _per_window_errors(model, loader, device)
        log.info("  %-12s mse=%.6f", "base", base[0].mean().item())

    rows = []
    for index, metric in enumerate(("forecast/mse", "forecast/mae")):
        stacked = torch.stack([v[index] for v in per_model.values()])
        oracle = stacked.min(dim=0).values.mean().item()
        best_single = min(v[index].mean().item() for v in per_model.values())
        rows.append({
            "run": args.run_dir.name,
            "dataset": config.get("dataset"),
            "n_segments": config["args"].get("dataset_n_finetune_segments"),
            "seed": config["args"].get("seed"),
            "metric": metric,
            "base": round(base[index].mean().item(), 6) if base is not None else "",
            "best_single": round(best_single, 6),
            "oracle_router": round(oracle, 6),
            # How much the per-window choice buys over always using the single best specialist.
            "oracle_vs_best_single_pct": round(100.0 * (best_single - oracle) / best_single, 3),
        })
        log.info("%s: oracle %.6f vs best single specialist %.6f (%.1f%% better)",
                 metric, oracle, best_single, rows[-1]["oracle_vs_best_single_pct"])

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"oracle_router_{args.run_dir.parent.name}_{args.run_dir.name}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
