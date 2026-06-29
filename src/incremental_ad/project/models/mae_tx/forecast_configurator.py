from argparse import ArgumentParser, Namespace
from dataclasses import replace
from typing import Self

from incremental_ad.framework.contracts.dataset import Dataset, ForecastDataset, TimeSeriesDataset
from incremental_ad.framework.contracts.debugger import Debugger
from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.project.task import Task
from incremental_ad.framework.evaluators.forecasting_evaluator import ForecastingEvaluator
from incremental_ad.project.models.mae_tx.mae import InferenceMode, MaeTx, TrainingMode


@TaskModelConfigurator.register(Task.FORECAST, MaeTx)
class MaeTxForecastingConfigurator(TaskModelConfigurator):
    """Switches MaeTx into forecasting mode.

    The configurator forces CAUSAL_MASK training and derives forecast_patches
    automatically from dataset.forecast_len // model.config.patch_len. The only CLI arg
    is --configurator_debug, which enables sample predicted-vs-actual visualizations.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(
            f"--{p}_debug",
            action="store_true",
            default=False,
            help="Produce forecast debug visualizations (sample predicted vs actual future).",
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(debug=getattr(cfg, f"{p}_debug", False))

    def _configure(self, model: Model, dataset: Dataset) -> None:
        assert isinstance(model, MaeTx), f"expected MaeTx, got {type(model).__name__}"
        
        assert isinstance(dataset, TimeSeriesDataset), (
            f"expected TimeSeriesDataset, got {type(dataset).__name__}"
        )

        assert isinstance(dataset, ForecastDataset), (
            f"{type(dataset).__name__} must expose a 'forecast_len' property "
            "(implement the ForecastDataset protocol) for use with MaeTxForecastingConfigurator."
        )

        if dataset.forecast_len % model.config.patch_len != 0:
            raise ValueError(
                f"dataset.forecast_len ({dataset.forecast_len}) must be divisible by "
                f"mae_tx_patch_len ({model.config.patch_len})."
            )
        
        if model.config.patch_norm:
            raise ValueError(
                "patch_norm=True is not supported for forecasting: the decoder outputs patches "
                "in per-patch-normalized space, but the evaluator compares against raw future values. "
                "Set --mae_tx_patch_norm False."
            )
        forecast_patches = dataset.forecast_len // model.config.patch_len

        model.n_features = dataset.n_features
        model.seq_len = dataset.window_len          # context_len + forecast_len
        model.forecast_patches = forecast_patches

        # Force causal mask — imposed by the configurator, not a CLI arg
        model.config = replace(model.config, training_mode=TrainingMode.CAUSAL_MASK)
        model.inference_mode = InferenceMode.FORECAST

    def create_val_evaluator(self) -> Evaluator:
        return ForecastingEvaluator()

    def create_test_evaluator(self) -> Evaluator:
        return ForecastingEvaluator()

    def create_debugger(self) -> Debugger | None:
        if not self.debug:
            return None
        from incremental_ad.project.models.mae_tx.forecast_debugger import (
            MaeTxForecastDebugger,
        )

        return MaeTxForecastDebugger()
