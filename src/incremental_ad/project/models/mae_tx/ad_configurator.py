from argparse import ArgumentParser, Namespace
from typing import Any, Self

from torch import Tensor

from incremental_ad.framework.contracts.dataset import Dataset, DatasetCapability, TimeSeriesDataset
from incremental_ad.framework.contracts.debugger import Debugger
from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.project.task import Task
from incremental_ad.framework.evaluators.ad_val_evaluator import AdValEvaluator
from incremental_ad.framework.evaluators.ad_test_evaluator import AdTestEvaluator
from incremental_ad.project.models.mae_tx.mae import InferenceMode, MaeTx


@TaskModelConfigurator.register(Task.AD, MaeTx)
class MaeTxAdConfigurator(TaskModelConfigurator):

    def __init__(
        self,
        threshold_strategy: str = "oracle",
        threshold_percentile: float = 95.0,
        debug: bool = False,
    ) -> None:
        self.threshold_strategy = threshold_strategy
        self.threshold_percentile = threshold_percentile
        self.debug = debug

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(
            f"--{p}_threshold_strategy",
            choices=["oracle", "percentile"],
            default="oracle",
        )
        parser.add_argument(
            f"--{p}_threshold_percentile",
            type=float,
            default=95.0,
        )
        parser.add_argument(
            f"--{p}_debug",
            action="store_true",
            default=False,
            help="Produce debug visualizations (score timeline, reconstructions, etc.)",
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            threshold_strategy=getattr(cfg, f"{p}_threshold_strategy"),
            threshold_percentile=getattr(cfg, f"{p}_threshold_percentile"),
            debug=getattr(cfg, f"{p}_debug", False),
        )

    def _configure(self, model: Model, dataset: Dataset) -> None:
        assert isinstance(model, MaeTx), f"expected MaeTx, got {type(model).__name__}"
        assert isinstance(
            dataset, TimeSeriesDataset
        ), f"expected TimeSeriesDataset, got {type(dataset).__name__}"
        # AD scores windows against ground-truth anomaly labels at test time.
        assert DatasetCapability.TEST_LABELS in dataset.capabilities, (
            f"{type(dataset).__name__} must provide test labels (DatasetCapability.TEST_LABELS) "
            "to be used for anomaly detection."
        )
        
        model.n_features = dataset.n_features
        model.seq_len = dataset.window_len
        model.inference_mode = InferenceMode.AD

    def create_val_evaluator(self) -> Evaluator[tuple[Tensor, Any]]:
        return AdValEvaluator()

    def create_test_evaluator(self) -> Evaluator[tuple[Tensor, Tensor]]:
        return AdTestEvaluator(
            threshold_strategy=self.threshold_strategy,
            threshold_percentile=self.threshold_percentile,
        )

    def create_debugger(self) -> Debugger | None:
        if not self.debug:
            return None
        from incremental_ad.project.models.mae_tx.ad_debugger import MaeTxAdDebugger

        return MaeTxAdDebugger()
