import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    """Capture the current RNG state across Python, NumPy, and PyTorch.
    This is useful when continuining a training from a checkpoint.
    Restoring the RNG state allows to continue from the exact conditions when the training stopped.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore the capture RNG state"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
