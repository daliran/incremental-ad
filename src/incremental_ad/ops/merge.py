import logging
from pathlib import Path
from typing import Any

import torch

from incremental_ad.core import checkpoint

log = logging.getLogger(__name__)


def _merge_task_vectors(
    base_state: dict[str, torch.Tensor],
    ft_states: list[dict[str, torch.Tensor]],
    scale: float,
) -> dict[str, torch.Tensor]:
    """Task arithmetic: theta_merged = theta_base + scale * sum_i(theta_ft_i - theta_base).

    Non-floating tensors (e.g. integer buffers) are copied from the base unchanged.
    """
    merged: dict[str, torch.Tensor] = {}

    for key, base_param in base_state.items():

        if not torch.is_floating_point(base_param):
            merged[key] = base_param.clone()
            continue

        delta = torch.zeros_like(base_param)

        for ft_state in ft_states:
            delta += ft_state[key] - base_param

        merged[key] = base_param + scale * delta

    return merged


def run_merge(
    *,
    base_checkpoint_path: Path,
    ft_checkpoint_paths: list[Path],
    scale: float,
    out_path: Path,
) -> dict[str, Any]:
    """Merge fine-tuned checkpoints into the base via task arithmetic.

    Clones the base checkpoint (so it keeps the same configs/architecture) and
    swaps in the merged weights, marking op="merge". Saves it to out_path and
    returns the merged checkpoint dict (ready to hand to the eval op).
    """
    base_ckpt = checkpoint.load_checkpoint(base_checkpoint_path)

    ft_states = [
        checkpoint.load_checkpoint(p)["model_state"] for p in ft_checkpoint_paths
    ]

    merged_state = _merge_task_vectors(base_ckpt["model_state"], ft_states, scale)

    merged_ckpt = {**base_ckpt, "model_state": merged_state, "op": "merge"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(merged_ckpt, out_path)

    log.info(
        f"Merged {len(ft_states)} fine-tunings into the base "
        f"(scale={scale}) -> {out_path}"
    )

    return merged_ckpt