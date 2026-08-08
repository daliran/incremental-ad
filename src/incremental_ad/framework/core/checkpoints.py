"""The on-disk checkpoint format: one place that defines how model weights are written
and read back, so producers (trainer, merge) and consumers (eval, analysis) cannot drift.
"""

from pathlib import Path

import torch
from torch import Tensor


def save_model_state(
    path: Path, state_dict: dict[str, Tensor], **metadata: object
) -> None:
    """Write model weights, plus whatever metadata the caller wants alongside them.

    Metadata is optional and free-form — the trainer records epoch and losses, a merged
    model has none — so readers must not assume any key beyond ``model_state_dict``.
    Creates the parent directory if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict, **metadata}, path)


def load_checkpoint_metadata(path: Path | str) -> dict:
    """The non-weight fields a checkpoint carries (``epoch``, ``train_loss``, ``val_loss``).

    Returns ``{}`` for a bare state dict or a merged checkpoint, which carry no metadata —
    readers must not assume any key beyond ``model_state_dict`` exists. Kept here so that
    `torch.load` stays confined to this module; `analysis/selection_probe.py` needs the
    trainer's recorded ``val_loss`` to compare the early-stopping signal against a test metric,
    and reading it anywhere else would fork checkpoint I/O.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return {}
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}


def load_model_state(path: Path | str) -> dict[str, Tensor]:
    """Load a checkpoint's model weights onto the CPU.

    Accepts both layouts written here: the trainer's
    ``{"model_state_dict", "epoch", "train_loss", "val_loss"}`` and the merged model's
    ``{"model_state_dict"}``. A bare state dict is accepted too, so checkpoints from
    elsewhere load without conversion.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model_state_dict", checkpoint)
