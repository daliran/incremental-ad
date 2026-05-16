from abc import ABC

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, labels: torch.Tensor | None = None) -> None:
        super().__init__()
        self.labels = labels
