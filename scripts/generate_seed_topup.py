"""Generate the runs needed to bring every experiment up to a full set of seeds.

Reads each finished run's own `config.json` and re-emits it verbatim with a different
`--seed`, so a top-up run differs from the originals in exactly one argument. Retyping the
command by hand is how configurations silently drift apart between seeds.

    python scripts/generate_seed_topup.py --runs_root $RUNS_ROOT --out $WORK/seed_topup
    python scripts/generate_seed_topup.py --runs_root $RUNS_ROOT --out $WORK/seed_topup \\
        --experiment segsweep_swat_merge_n2 --experiment noisefloor_psm

Writes one shell file per missing (experiment, seed) plus a `submit_all.sh`. Nothing is
submitted from here — inspect the commands, then run the submit script.

**Why every experiment needs three seeds.** The reproducibility floor is the threshold every
"this difference is real" claim in EXPERIMENTS.md is measured against, and it is a standard
deviation: it cannot be computed from one run, and from two it is barely an estimate. The AD
experiments were mostly run at a single seed, so their floors were borrowed from a different
experiment group — and §1.9 shows independent estimates of the *same* configuration differ by
up to 3.3×. Until each group has its own floor, an AD difference cannot be called real or not.

Seeds default to the project's standard triple (7, 42, 123), which is what the forecasting
experiments already use, so topped-up groups become directly comparable to them.

Files are written only after **every** command has been built and validated. An earlier
generator in this project wrote each file as it went and left a directory of empty commands
behind when one failed halfway.
"""

import argparse
import contextlib
import io
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

STANDARD_SEEDS = (7, 42, 123)

# Keys in config.json["args"] that are positional/structural rather than namespaced options.
HEAD_KEYS = ("model", "dataset", "task", "pipeline")

# Never re-emitted: set per run, or derived.
SKIP_KEYS = {"seed", "experiment_name", "eval_seed", *HEAD_KEYS}

# Experiment name prefixes that are hyperparameter searches, not published measurements —
# a sweep trial does not need replicating, only the configuration that won it.
SWEEP_PREFIXES = ("slurm_grid_", "etth_vf15_regsweep")

# Diagnostics runs are derived from a source run rather than trained, so they cannot be
# re-emitted as a `main.py` invocation. They are regenerated from the topped-up source runs
# afterwards, with scripts/sbatch_merge_diagnostics.sh.
DERIVED_SUFFIXES = ("_diagnostics",)


def _actions_for(head: list[str]) -> dict:
    """dest -> argparse action, for the parser this (model, dataset, task, pipeline) builds.

    Needed because a boolean argument may be either a `store_true` flag, which must be emitted
    bare or not at all, or an ordinary option taking `True`/`False`. Guessing from the value's
    type gets one of the two wrong and argparse rejects the command.
    """
    from incremental_ad.main import build_parser, parse_components

    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        parser = build_parser(parse_components(head))
    return {a.dest: a for a in parser._actions}


def build_command(args: dict, experiment: str, seed: int) -> str:
    """Rebuild the `python -m incremental_ad.main ...` invocation for one seed."""
    head: list[str] = []
    for key in HEAD_KEYS:
        if args.get(key) is None:
            raise ValueError(f"{experiment}: config.json is missing '{key}'")
        head += [f"--{key}", str(args[key])]
    actions = _actions_for(head)

    parts = ["python", "-m", "incremental_ad.main", *head,
             "--experiment_name", experiment, "--seed", str(seed)]
    for key in sorted(args):
        if key in SKIP_KEYS:
            continue
        value = args[key]
        if value is None:
            continue
        action = actions.get(key)
        if action is None:
            # Recorded in config.json but not accepted by this parser build — emitting it
            # would abort the run, so drop it and let the default stand.
            continue
        if action.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            if bool(value) != bool(action.default):
                parts.append(f"--{key}")
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            parts += [f"--{key}", *[str(v) for v in value]]
        else:
            parts += [f"--{key}", str(value)]
    return " ".join(shlex.quote(p) if " " in p else p for p in parts)


def validate(command: str) -> None:
    """Parse the command with the project's own parser; raise if it would be rejected."""
    from incremental_ad.main import build_parser, parse_components

    argv = shlex.split(command)[3:]
    with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
        try:
            build_parser(parse_components(argv)).parse_args(argv)
        except SystemExit as exc:
            raise ValueError(err.getvalue().strip().splitlines()[-1]) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD_SEEDS))
    parser.add_argument("--experiment", action="append", dest="only",
                        help="restrict to these experiments (repeatable)")
    parser.add_argument("--include-sweeps", action="store_true",
                        help="also top up hyperparameter-sweep experiments")
    args = parser.parse_args()

    # Populate the component registries so the parser can be built for any run.
    sys.argv = sys.argv[:1]
    import incremental_ad.framework.evaluators  # noqa: F401
    import incremental_ad.framework.pipelines  # noqa: F401
    import incremental_ad.framework.trainers  # noqa: F401
    import incremental_ad.project.datasets  # noqa: F401
    import incremental_ad.project.models  # noqa: F401

    # One representative config per (experiment, seed).
    configs: dict[str, dict[int, dict]] = defaultdict(dict)
    for cfg_path in sorted(args.runs_root.glob("*/*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        run_args = cfg.get("args") or {}
        seed = run_args.get("seed")
        if seed is None:
            continue
        configs[cfg_path.parent.parent.name].setdefault(seed, run_args)

    wanted = set(args.seeds)
    commands: list[tuple[str, int, str]] = []
    skipped_sweeps = []
    for experiment, by_seed in sorted(configs.items()):
        if args.only and experiment not in args.only:
            continue
        if experiment.endswith(DERIVED_SUFFIXES):
            continue
        if not args.include_sweeps and experiment.startswith(SWEEP_PREFIXES):
            skipped_sweeps.append(experiment)
            continue
        missing = sorted(wanted - set(by_seed))
        if not missing:
            continue
        template = by_seed[sorted(by_seed)[0]]
        for seed in missing:
            # Build AND parse-check every command before anything is written.
            command = build_command(template, experiment, seed)
            validate(command)
            commands.append((experiment, seed, command))

    if not commands:
        print("every experiment already has the requested seeds — nothing to do")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    submit_lines = ["#!/bin/bash", "set -euo pipefail",
                    "# Generated by scripts/generate_seed_topup.py — review before running.",
                    'HERE="$(cd "$(dirname "$0")" && pwd)"', ""]
    for experiment, seed, command in commands:
        path = args.out / f"{experiment}_s{seed}.sh"
        path.write_text(command + "\n")
        submit_lines.append(
            f'CMD="$HERE/{path.name}" sbatch --job-name={experiment[:20]}_s{seed} '
            f"scripts/sbatch_run_command.sh"
        )
    submit = args.out / "submit_all.sh"
    submit.write_text("\n".join(submit_lines) + "\n")
    submit.chmod(0o755)

    by_experiment = defaultdict(list)
    for experiment, seed, _ in commands:
        by_experiment[experiment].append(seed)
    print(f"{len(commands)} run(s) needed across {len(by_experiment)} experiment(s):")
    for experiment, seeds in sorted(by_experiment.items()):
        print(f"   {experiment:38} missing seeds {seeds}")
    if skipped_sweeps:
        print(f"\nskipped {len(skipped_sweeps)} hyperparameter-sweep experiment(s) "
              f"(pass --include-sweeps to override)")
    print(f"\nwrote {submit}")


if __name__ == "__main__":
    main()
