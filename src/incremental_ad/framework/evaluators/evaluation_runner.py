from argparse import ArgumentParser, Namespace
from typing import Self

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from incremental_ad.framework.contracts.dataset import DataLoaderConfig
from incremental_ad.framework.contracts.evaluator import Evaluator, ReferenceScoredEvaluator
from incremental_ad.framework.contracts.model import Model
from incremental_ad.framework.core.device import move_to_device
from incremental_ad.framework.core.seed import set_seed


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )


class EvaluationRunner:
    """Counterpart to StandardTrainer: orchestrates one full evaluation pass.

    Owns device management for inference so pipelines and models stay device-agnostic.
    The caller creates the evaluator, passes it in, and can inspect its state after
    run() returns (e.g. for wandb chart logging or debugger visualization).
    """

    ARG_PREFIX = "runner"

    def __init__(
        self,
        loader_config: DataLoaderConfig,
        device: str | torch.device = "auto",
    ) -> None:
        self.loader_config = loader_config
        self.device = _resolve_device(device)

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_device", type=str, default="auto")

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(
            loader_config=DataLoaderConfig.from_config(cfg),
            device=getattr(cfg, f"{p}_device", "auto"),
        )

    def run(
        self,
        model: Model,
        evaluator: Evaluator,
        dataset: TorchDataset,
        *,
        reference_dataset: TorchDataset | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Run one full evaluation pass and return the computed metrics.

        Moves the model to device, resets the evaluator, optionally collects reference
        scores for threshold calibration, then iterates the dataset. The evaluator retains
        its state after return so callers can log wandb charts or run a debugger.
        """
        if seed is not None:
            set_seed(seed)

        model.to(self.device)

        evaluator.reset()

        if reference_dataset is not None and isinstance(evaluator, ReferenceScoredEvaluator):
            if evaluator.needs_reference_scores():
                evaluator.set_reference_scores(self.collect_scores(model, reference_dataset))

        loader = self.loader_config.make_loader(dataset, shuffle=False)

        model.eval()

        with torch.no_grad():
            for batch in loader:
                evaluator.update(model.predict_step(move_to_device(batch, self.device)))

        return evaluator.compute()

    def collect_scores(self, model: Model, dataset: TorchDataset) -> np.ndarray:
        """Collect per-sample anomaly scores from an unlabeled single-tensor dataset.

        Used to populate a ReferenceScoredEvaluator before test evaluation.
        Expects the dataset to yield plain tensors (no labels), e.g. SlidingWindowDataset.
        """
        loader = self.loader_config.make_loader(dataset, shuffle=False)
        model.to(self.device)
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch in loader:
                scores = model.score(batch.to(self.device))
                all_scores.append(scores.detach().cpu().numpy())

        return np.concatenate(all_scores, axis=0)
