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
from incremental_ad.framework.contracts.task import Task
from incremental_ad.framework.experiment import Experiment


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pre arguments parsing for validation
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", required=True)
    pre.add_argument("--dataset", required=True)
    pre.add_argument("--task", required=True, choices=[t.value for t in Task])
    pre.add_argument("--pipeline", required=True)
    known, _ = pre.parse_known_args()

    _validate_component("model", known.model, Model._registry)
    _validate_component("dataset", known.dataset, Dataset._registry)
    _validate_component("pipeline", known.pipeline, Pipeline._registry)

    # Actual arguments registration
    parser = argparse.ArgumentParser()

    Experiment.add_args(parser)

    Model._registry[known.model].add_args(parser)
    Dataset._registry[known.dataset].add_args(parser)

    # Trainer args are registered by the pipeline (each pipeline owns its trainers).
    Pipeline._registry[known.pipeline].add_args(parser)

    task = Task(known.task)
    model_cls = Model._registry[known.model]
    configurator_cls = TaskModelConfigurator._registry.get((task, model_cls))

    if configurator_cls is None:
        print(
            f"Error: no configurator registered for (task={known.task}, model={known.model}).",
            file=sys.stderr,
        )
        sys.exit(1)

    configurator_cls.add_args(parser)

    # Actual arguments parsing
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
