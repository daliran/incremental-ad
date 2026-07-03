from torch import Tensor
from torch.utils.data import Dataset as TorchDataset


class ForecastWindowDataset(TorchDataset):
    """Yields (full_window [W, F], future [forecast_len, F]) 2-tuples.

    full_window spans context_len + forecast_len timesteps; future is the last
    forecast_len steps of full_window (the target). Dataset-agnostic — any
    contiguous [T, F] tensor works, promoted here from project/datasets/etth.py so
    other single-series forecast datasets (weather, traffic, exchange_rate, ...)
    can share it instead of redefining it.
    """

    def __init__(
        self, data: Tensor, window_len: int, forecast_len: int, stride: int
    ) -> None:
        self.data = data
        self.window_len = window_len
        self.context_len = window_len - forecast_len
        self.forecast_len = forecast_len
        self.stride = stride

    def __len__(self) -> int:
        if len(self.data) < self.window_len:
            return 0
        return (len(self.data) - self.window_len) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.stride
        full_window = self.data[start : start + self.window_len]
        return full_window, full_window[self.context_len :]
