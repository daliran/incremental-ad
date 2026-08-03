import argparse
import logging
import sys

# Pre-load pyarrow/pandas before torch to avoid Windows CUDA DLL conflict
import pyarrow  # noqa: F401
import pandas  # noqa: F401

# These imports trigger registry population via __init_subclass__.
# To add a new model / dataset / pipeline etc., update the package __init__.py —
# no changes to main.py required.
import incremental_ad.project.datasets  # noqa: F401
import incremental_ad.project.models  # noqa: F401
import incremental_ad.framework.trainers  # noqa: F401
import incremental_ad.framework.pipelines  # noqa: F401

from incremental_ad.framework.contracts.dataset import Dataset
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.framework.contracts.pipeline import Pipeline
from incremental_ad.framework.experiment import Experiment


def parse_components(argv: list[str] | None = None) -> argparse.Namespace:
    """Resolve and validate --model/--dataset/--task/--pipeline before anything else.

    Which args exist at all depends on these four, so they are parsed first and the rest
    of the parser is built from them.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", required=True)
    pre.add_argument("--dataset", required=True)
    pre.add_argument(
        "--task", required=True, choices=TaskModelConfigurator.registered_tasks()
    )
    pre.add_argument("--pipeline", required=True)
    known, _ = pre.parse_known_args(argv)

    _validate_component("model", known.model, Model._registry)
    _validate_component("dataset", known.dataset, Dataset._registry)
    _validate_component("pipeline", known.pipeline, Pipeline._registry)
    return known


def build_parser(known: argparse.Namespace) -> argparse.ArgumentParser:
    """The full parser for one (model, dataset, task, pipeline) combination.

    Split out from main() so other entry points can ask what a given invocation accepts,
    and reuse argparse's own type conversion, rather than reimplementing either.
    """
    parser = argparse.ArgumentParser()

    Experiment.add_args(parser)

    Model._registry[known.model].add_args(parser)
    Dataset._registry[known.dataset].add_args(parser)

    # Trainer args are registered by the pipeline (each pipeline owns its trainers).
    Pipeline._registry[known.pipeline].add_args(parser)

    model_cls = Model._registry[known.model]
    configurator_cls = TaskModelConfigurator.lookup(known.task, model_cls)

    if configurator_cls is None:
        print(
            f"Error: no configurator registered for (task={known.task}, model={known.model}).",
            file=sys.stderr,
        )
        sys.exit(1)

    configurator_cls.add_args(parser)
    return parser


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    known = parse_components()
    parser = build_parser(known)

    args = parser.parse_args()

    experiment = Experiment.from_config(args)

    experiment.run()


def _validate_component(label: str, name: str, registry: dict) -> None:
    if name not in registry:
        available = ", ".join(sorted(registry)) or "none"
        print(
            f"Error: unknown {label} '{name}'. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
