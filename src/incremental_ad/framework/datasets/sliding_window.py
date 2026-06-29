import torch
from torch import Tensor


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Sliding-window view over a contiguous [T, F] tensor.

    Each sample is a window of shape [window_len, F].
    When labels are provided, each sample is (window, labels[start:start+window_len]).
    Labels can be any per-timestep tensor whose first dimension aligns with the time
    axis (e.g. class/anomaly labels [T], feature targets [T, F]).
    """

    def __init__(
        self,
        data: Tensor,
        window_len: int,
        stride: int,
        labels: Tensor | None = None,
    ) -> None:
        self.data = data
        self.window_len = window_len
        self.stride = stride
        self.labels = labels

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, Tensor]:
        start = idx * self.stride
        window = self.data[start : start + self.window_len]
        if self.labels is not None:
            return window, self.labels[start : start + self.window_len]
        return window
