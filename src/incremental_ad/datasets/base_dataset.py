from abc import ABC
from typing import Literal

import torch
from torch.utils.data import Dataset

Split = Literal["val", "test"]


class BaseDataset(Dataset, ABC):
    def __init__(self, labels: torch.Tensor | None = None) -> None:
        super().__init__()
        self.labels = labels
