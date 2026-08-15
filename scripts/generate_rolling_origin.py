"""Generate the rolling-origin and baseline-fraction sweeps from existing run configs.

    python scripts/generate_rolling_origin.py --runs_root $RUNS_ROOT --out $WORK/sweeps

**Why this exists.** Every forecasting result in the project sits on one train/test cut: the
last 20% of the series. That is the deployment-realistic split, but it gives no error bar for
the *choice of cut*, and the distance from a training period to the test block varies about 2x
within a single dataset (EXPERIMENTS.md) -- so a conclusion could plausibly turn on where the
cut landed. A rolling origin re-runs the same comparison at several cuts and reports the spread.

Origins are produced with `--dataset_series_fraction`, which truncates the series *before* the
split. f=1.0 tests on [0.80, 1.00] of the series, f=0.875 on [0.70, 0.875], f=0.75 on
[0.60, 0.75]. Data past the cut is discarded rather than held out, so no origin ever trains on
its own future.

The second sweep varies `--dataset_baseline_fraction` (how much of the stream the base model
sees before the incremental steps begin). It is fixed at 0.5 everywhere, which is a choice
nobody has tested: a smaller base leaves more for the segments to change and should raise both
the headroom and the noise floor.

Commands are built by **reading an existing run's `config.json`** and overriding only the swept
arguments, so hyperparameters cannot drift from the published runs. Every command is validated
against the real argument parser before it is written -- a typo becomes an error here rather
than a job that dies twenty minutes into a queue.
"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("generate_rolling_origin")

# (label, reference experiment) per dataset — the runs the published numbers come from.
REFERENCES = {
    "etth1": {"merge": "noisefloor_etth", "sequential": "continual_etth",
              "joint": "noisefloor_std_etth", "window": "window_etth1_W3"},
    "exchange": {"merge": "exch_incremental", "sequential": "exch_continual",
                 "joint": "exch_gate_standard", "window": "window_exchange_W3"},
}
ORIGINS = (0.75, 0.875, 1.0)
BASELINE_FRACTIONS = (0.3, 0.7)          # 0.5 is the existing published setting
SEEDS = (7, 42, 123)
SKIP = {"experiment_name", "seed", "dataset_series_fraction", "dataset_baseline_fraction"}


HEAD_KEYS = ("model", "dataset", "task", "pipeline")


def _actions_for(head: list[str]) -> dict:
    """The argument actions this component combination accepts, keyed by dest."""
    import incremental_ad.framework.evaluators  # noqa: F401
    import incremental_ad.framework.pipelines  # noqa: F401
    import incremental_ad.framework.trainers  # noqa: F401
    import incremental_ad.project.datasets  # noqa: F401
    import incremental_ad.project.models  # noqa: F401
    from incremental_ad.main import build_parser, parse_components

    parser = build_parser(parse_components(head))
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def command(args: dict, overrides: dict) -> list[str]:
    """A `python -m incremental_ad.main` command from a config, with overrides applied.

    Parser-aware, because a boolean in `config.json` means two different things: a store-true
    flag like `--configurator_debug` must be *omitted* when false, while an explicit-valued
    flag like `--mae_tx_patch_norm` is `required=True` and must be passed as `--flag false`.
    Treating both the same way produced a command the parser rejected -- caught here rather
    than after submitting a hundred jobs.
    """
    head: list[str] = []
    for key in HEAD_KEYS:
        head += [f"--{key}", str(args[key])]
    actions = _actions_for(head)

    merged = {k: v for k, v in args.items() if k not in SKIP}
    merged.update(overrides)
    out = ["python", "-m", "incremental_ad.main", *head]
    for key in sorted(merged):
        if key in HEAD_KEYS:
            continue
        value = merged[key]
        action = actions.get(key)
        if value is None or action is None:
            continue
        if action.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            if bool(value) != bool(action.default):
                out.append(f"--{key}")
            continue
        if isinstance(value, (list, tuple)):
            if value:
                out += [f"--{key}", *[str(v) for v in value]]
        else:
            out += [f"--{key}", str(value)]
    return out


def validate(cmd: list[str]) -> None:
    """Parse the command with the real parser so a bad flag fails here, not in the queue."""
    import incremental_ad.framework.evaluators  # noqa: F401
    import incremental_ad.framework.pipelines  # noqa: F401
    import incremental_ad.framework.trainers  # noqa: F401
    import incremental_ad.project.datasets  # noqa: F401
    import incremental_ad.project.models  # noqa: F401
    from incremental_ad.main import build_parser, parse_components

    argv = cmd[3:]
    known = parse_components(argv)
    build_parser(known).parse_args(argv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []

    for dataset, refs in REFERENCES.items():
        for role, experiment in refs.items():
            configs = sorted((args.runs_root / experiment).glob("*/config.json"))
            if not configs:
                log.warning("%s: no runs under %s — skipped", role, experiment)
                continue
            base = json.loads(configs[0].read_text())["args"]

            for origin in ORIGINS:
                if origin == 1.0 and role in refs:
                    pass  # still re-run: the published runs predate series_fraction
                for seed in SEEDS:
                    name = f"origin_{dataset}_{role}_f{str(origin).replace('.', '')}"
                    cmd = command(base, {"experiment_name": name, "seed": seed,
                                         "dataset_series_fraction": origin,
                                         "dataset_baseline_fraction":
                                             base.get("dataset_baseline_fraction", 0.5)})
                    validate(cmd)
                    commands.append(" ".join(cmd))

        # Baseline-fraction sweep: merge pipeline only, at the published origin.
        merge_configs = sorted((args.runs_root / refs["merge"]).glob("*/config.json"))
        if merge_configs:
            base = json.loads(merge_configs[0].read_text())["args"]
            for fraction in BASELINE_FRACTIONS:
                for seed in SEEDS:
                    name = f"basefrac_{dataset}_{str(fraction).replace('.', '')}"
                    cmd = command(base, {"experiment_name": name, "seed": seed,
                                         "dataset_series_fraction": 1.0,
                                         "dataset_baseline_fraction": fraction})
                    validate(cmd)
                    commands.append(" ".join(cmd))

    path = args.out / "rolling_origin_commands.sh"
    path.write_text("\n".join(commands) + "\n")
    log.info("wrote %s (%d commands, all parser-validated)", path, len(commands))


if __name__ == "__main__":
    main()
