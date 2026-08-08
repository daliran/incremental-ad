"""Does the signal used to *stop training* track the metric we actually care about?

Loads every per-epoch checkpoint of a finished run, scores each on the test set, and puts the
early-stopping signal next to the test metric so the two optima can be compared.

    # 1. train with periodic checkpoints
    python -m incremental_ad.main ... --trainer_checkpoint_interval 1
    # 2. probe them
    python -m incremental_ad.analysis.selection_probe --run_dir $RUNS_ROOT/<exp>/<id> \\
        --step baseline --out /tmp/probe

**Why this file exists.** EXPERIMENTS.md §1.12 shows that on anomaly detection the validation
signal and test AUROC disagree about the *merge coefficient* — validation says α ≈ 0.25 while
AUROC keeps improving to 1.5. But α is not the only thing chosen on that signal. Every AD
checkpoint in this project is selected by `StandardTrainer`, which early-stops on
`model.compute_loss` over the validation slice — the masked-reconstruction objective. So every
base model, every specialist, the sequential chain and the joint reference were all chosen by
minimising reconstruction error, which §1.12 shows can be *actively opposed* to detection
quality along one axis.

The implicit assumption is that the epoch axis is more forgiving than the α axis: training
improves reconstruction and detection together, whereas §1.12 measured them in opposition. That
assumption has never been tested. This script tests it.

- **Optima coincide** → the selection problem is confined to α. That narrows §1.12 and
  strengthens it.
- **Optima diverge** → every AD number in the project carries an unmeasured selection cost, and
  that belongs in the structural limits rather than a footnote.

Run it on **PSM**, not SWaT: SWaT's base model sits within 1.1% of joint training, so every
checkpoint is near the ceiling and nothing is distinguishable from noise.

The early-stopping signal is read from each checkpoint's own metadata (`val_loss`), which is
exactly the number `StandardTrainer` compared — not a recomputation that might differ.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("selection_probe")

FIELDS = ["epoch", "val_loss", "train_loss", "is_early_stop_choice"]


def _load_run(run_dir: Path):
    """Rebuild the dataset, model and evaluator this run used, from its own config."""
    import incremental_ad.framework.evaluators  # noqa: F401
    import incremental_ad.framework.pipelines  # noqa: F401
    import incremental_ad.framework.trainers  # noqa: F401
    import incremental_ad.project.datasets  # noqa: F401
    import incremental_ad.project.models  # noqa: F401
    from incremental_ad.framework.contracts.dataset import DataLoaderConfig, Dataset
    from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
    from incremental_ad.framework.evaluators.evaluation_runner import EvaluationRunner
    from incremental_ad.main import build_parser, parse_components

    config = json.loads((run_dir / "config.json").read_text())
    args = config["args"]

    # Re-parse the run's own arguments so every component is constructed exactly as it was.
    argv: list[str] = []
    known = parse_components(
        ["--model", args["model"], "--dataset", args["dataset"],
         "--task", args["task"], "--pipeline", args["pipeline"]]
    )
    parser = build_parser(known)
    accepted = {a.dest: a for a in parser._actions if a.dest != "help"}
    for dest, value in args.items():
        action = accepted.get(dest)
        if action is None or value is None:
            continue
        if action.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            if bool(value) != bool(action.default):
                argv.append(action.option_strings[0])
            continue
        flag = action.option_strings[0] if action.option_strings else None
        if flag is None:
            continue
        if isinstance(value, (list, tuple)):
            if value:
                argv += [flag, *[str(v) for v in value]]
        else:
            argv += [flag, str(value)]
    cfg = parser.parse_args(argv)

    dataset = Dataset._registry[args["dataset"]].from_config(cfg)
    model = Model._registry[args["model"]].from_config(cfg)
    configurator = TaskModelConfigurator.lookup(args["task"], type(model)).from_config(cfg)
    configurator.configure(model, dataset)
    runner = EvaluationRunner(DataLoaderConfig.from_config(cfg), device="auto")
    return dataset, model, configurator, runner, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--step", default="baseline",
                        help="which step's checkpoints to probe (baseline, finetune_0, train, …)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ckpt_dir = args.run_dir / args.step / "checkpoints"
    epochs = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not epochs:
        raise SystemExit(
            f"no epoch_*.pt under {ckpt_dir} — the run must be trained with "
            f"--trainer_checkpoint_interval 1 (or --finetune_trainer_checkpoint_interval)"
        )

    from incremental_ad.framework.core.checkpoints import (
        load_checkpoint_metadata,
        load_model_state,
    )

    dataset, model, configurator, runner, cfg = _load_run(args.run_dir)
    evaluator = configurator.create_test_evaluator()
    test_data = dataset.get_test_dataset()
    reference = (
        dataset.get_reference_dataset() if hasattr(dataset, "get_reference_dataset") else None
    )
    eval_seed = json.loads((args.run_dir / "config.json").read_text()).get("eval_seed")

    rows = []
    for path in epochs:
        model.load_state_dict(load_model_state(path))
        payload = load_checkpoint_metadata(path)
        metrics = runner.run(model, evaluator, test_data,
                             reference_dataset=reference, seed=eval_seed)
        epoch = int(re.search(r"epoch_(\d+)", path.name).group(1))
        row = {"epoch": epoch,
               "val_loss": payload.get("val_loss"),
               "train_loss": payload.get("train_loss"),
               "is_early_stop_choice": False,
               **{k: round(v, 6) for k, v in metrics.items()}}
        rows.append(row)
        log.info("epoch %3d  val_loss %s  %s", epoch,
                 f"{row['val_loss']:.6f}" if row["val_loss"] is not None else "—",
                 "  ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:3]))

    scored = [r for r in rows if r["val_loss"] is not None]
    if scored:
        chosen = min(scored, key=lambda r: r["val_loss"])
        chosen["is_early_stop_choice"] = True
        log.info("\nearly stopping would pick epoch %d (lowest val_loss)", chosen["epoch"])
        # Metric names carry a slash on forecasting (`forecast/mse`) but not on AD
        # (`window_auroc`), so filtering on "/" silently reported nothing for the very task
        # this script exists to probe. Select by exclusion instead.
        bookkeeping = set(FIELDS)
        for metric in sorted(k for k in rows[0] if k not in bookkeeping):
            higher = any(t in metric.lower() for t in
                         ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))
            best = (max if higher else min)(rows, key=lambda r: r[metric])
            gap = (100 * (best[metric] - chosen[metric]) / abs(best[metric])
                   if best[metric] else 0.0)
            verdict = "COINCIDE" if best["epoch"] == chosen["epoch"] else "DIVERGE"
            log.info("  %-22s best at epoch %3d (%.4f) vs chosen %.4f  -> %s, %+.2f%% left on "
                     "the table", metric, best["epoch"], best[metric], chosen[metric],
                     verdict, abs(gap))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "selection_probe.csv"
        fields = FIELDS + sorted(k for k in rows[0] if k not in FIELDS)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("\nwrote %s", path)


if __name__ == "__main__":
    main()
