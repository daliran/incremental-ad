from abc import ABC, abstractmethod

import torch


class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    def debug_step(self, batch: torch.Tensor) -> dict[str, object] | None:
        """Per-sample debug info for one window (batch_size=1).

        Return None if the model has nothing to contribute.
        Models that support debug output should override this and return a dict
        with whichever of the following keys they can provide:

          original:       Tensor (B, T, F)        raw input signal
          reconstruction: Tensor (B, T, F)        averaged decoder output
          token_errors:   Tensor (B, n_patches)   per-patch MSE

        RunDebugger renders whichever keys are present; absent keys are silently skipped.
        """
        return None