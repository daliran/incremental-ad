"""Run MergeDiagnosticsPipeline against a finished run, reusing that run's own arguments.

The diagnostics pipeline demands the model and dataset args of the run it analyses -- some
sixty flags -- and hard-errors on any difference, since mismatched args would evaluate the
wrong regions or score the checkpoints differently. Retyping them is both tedious and the
most likely way to get a rejected job, or worse an accepted one that is subtly wrong.

Every run already records its complete argument namespace in config.json, so this reads
them back:

    python -m incremental_ad.analysis.diagnose --source_run_dir runs/<exp>/<run_id>
    python -m incremental_ad.analysis.diagnose --source_run_dir <dir> --dry-run
    python -m incremental_ad.analysis.diagnose --source_run_dir <dir> \
        --standard_run_dir <dir> --merge_scales 0.0 0.5 1.0 -- --pipeline_eval_seeds 43 44

Which args to carry over is decided by the target invocation's *own* parser rather than a
hardcoded list: anything the diagnostics pipeline does not declare (every trainer_* flag,
pipeline_merge_scale, ...) is dropped automatically, so the list cannot rot as components
change. Args are re-emitted as command-line tokens and parsed normally, which is what
converts `"causal_mask"` back into a TrainingMode rather than leaving it a bare string.
"""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

from incremental_ad.framework.experiment import Experiment
from incremental_ad.main import build_parser, parse_components

TARGET_PIPELINE = "MergeDiagnosticsPipeline"


def load_run_args(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{run_dir}: not a run directory (no config.json)")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _tokens_for(action: argparse.Action, value: object) -> list[str]:
    """Render one recorded arg value back into command-line tokens."""
    flag = action.option_strings[0]
    if value is None:  # e.g. eval_seed: null -- let the target's default apply
        return []
    if action.nargs == 0:  # store_true, e.g. --configurator_debug
        return [flag] if value else []
    if isinstance(value, bool):  # type=_str_to_bool expects the word, not "True"
        return [flag, str(value).lower()]
    if isinstance(value, list):
        return [flag, *(str(v) for v in value)]
    return [flag, str(value)]


def build_argv(source_config: dict, overrides: dict[str, str], passthrough: list[str]) -> list[str]:
    args = source_config["args"]
    known = Namespace(
        model=args["model"],
        dataset=args["dataset"],
        task=args["task"],
        pipeline=TARGET_PIPELINE,
    )
    parser = build_parser(known)

    # Anything the target invocation does not declare is silently dropped -- that is how
    # trainer_*/finetune_trainer_*/pipeline_merge_scale disappear without a whitelist.
    accepted = {a.dest: a for a in parser._actions if a.dest != "help"}
    argv = ["--model", known.model, "--dataset", known.dataset,
            "--task", known.task, "--pipeline", TARGET_PIPELINE]
    for dest, value in args.items():
        if dest in ("model", "dataset", "task", "pipeline") or dest not in accepted:
            continue
        if dest in overrides:
            continue
        argv += _tokens_for(accepted[dest], value)

    for dest, value in overrides.items():
        if dest in accepted:
            argv += [accepted[dest].option_strings[0], *str(value).split()]

    # Last-wins in argparse, so anything after `--` overrides everything above.
    return argv + passthrough


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_run_dir", type=Path, required=True,
                        help="a completed IncrementalTaskArithmeticPipeline run directory")
    parser.add_argument("--standard_run_dir", type=Path, default=None,
                        help="optional StandardPipeline run, added as the GRR reference row")
    parser.add_argument("--experiment_name", default=None,
                        help="default: <source experiment_name>_diagnostics")
    parser.add_argument("--merge_scales", nargs="*", default=None,
                        help="scales for the merge-scale curve; omitted skips the curve")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the composed command line and exit")
    parser.add_argument("passthrough", nargs="*", default=[],
                        help="extra args after `--`, passed through verbatim (last wins)")
    args = parser.parse_args()

    source_config = load_run_args(args.source_run_dir)
    overrides: dict[str, str] = {
        "experiment_name": args.experiment_name
        or f"{source_config.get('experiment_name', 'run')}_diagnostics",
        "pipeline_source_run_dir": str(args.source_run_dir),
    }
    if args.standard_run_dir is not None:
        overrides["pipeline_standard_run_dir"] = str(args.standard_run_dir)
    if args.merge_scales is not None:
        overrides["pipeline_merge_scales"] = " ".join(args.merge_scales)

    argv = build_argv(source_config, overrides, args.passthrough)

    if args.dry_run:
        lines: list[list[str]] = []
        for token in argv:
            if token.startswith("--") or not lines:
                lines.append([token])
            else:
                lines[-1].append(token)
        print("python -m incremental_ad.main \\")
        print(" \\\n".join("    " + " ".join(line) for line in lines))
        return

    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    known = parse_components(argv)
    Experiment.from_config(build_parser(known).parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
