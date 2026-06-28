from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from incremental_ad.framework.contracts.evaluator import Evaluator


class Debugger(ABC):
    @abstractmethod
    def run(
        self,
        model: Any,
        dataset: Any,
        evaluator: Evaluator[Any],
        step_dir: Path,
    ) -> None:
        """Produce debug artifacts (plots, CSVs) for one evaluation step.

        The debugger is responsible for extracting whatever it needs from the evaluator.
        """
        ...
