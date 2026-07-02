import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from torch.utils.data import DataLoader

from incremental_ad.framework.contracts.configurable import Configurable
from incremental_ad.framework.contracts.dataset import Segment
from incremental_ad.framework.contracts.model import Model


@dataclass
class TrainSummary:
    final_train_loss: float  # train loss at the last epoch run
    best_train_loss: float  # lowest train loss seen across all epochs
    final_val_loss: float | None  # val loss at the last epoch run; None if no val split
    best_val_loss: float | None  # lowest val loss seen; None if no val split
    best_epoch: int | None  # epoch where best_val_loss occurred; None if no val split
    epochs_trained: (
        int  # total epochs actually run (may be < n_epochs if early stopped)
    )
    checkpoint_path: (
        Path | None
    )  # path to saved best.pt, or None if no checkpoint_dir given
    secondary_val_losses: dict[str, float] = field(
        default_factory=dict
    )  # final loss per named secondary loader


class Trainer(Configurable, ABC):
    ARG_PREFIX = "trainer"
    _registry: ClassVar[dict[str, type["Trainer"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            Trainer._registry[cls.__name__] = cls

    @abstractmethod
    def fit(
        self,
        model: Model,
        segment: Segment,
        checkpoint_dir: Path | None = None,
        secondary_loaders: dict[str, DataLoader] | None = None,
        step_name: str = "",
        step_offset: int = 0,
        reference_state: dict[str, Any] | None = None,
    ) -> TrainSummary:
        """Train for one phase on the given segment.

        checkpoint_dir: if provided, writes last.pt every epoch and best.pt on val improvement.
        secondary_loaders: named extra val loaders evaluated each epoch for monitoring (no effect on early stopping).
        step_name: label used for log prefixes and wandb metric keys (e.g. "baseline", "finetune_0").
        step_offset: added to the local epoch number when logging to wandb, so multiple phases don't share x-axis values.
        reference_state: optional fixed parameter snapshot (e.g. a baseline's state_dict) to
            regularize toward — implementations that support this add a penalty term to the
            loss; implementations/configs that don't use it must ignore it (default: no-op).
        """
        ...
