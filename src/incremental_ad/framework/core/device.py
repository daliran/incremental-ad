from typing import Any

import torch
from torch import Tensor


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to device, preserving nested tuple/list structure."""
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        moved = tuple(move_to_device(b, device) for b in batch)
        return type(batch)(moved) if isinstance(batch, tuple) else list(moved)
    return batch
