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