"""Task-vector construction and merging over model state dicts.

A *task vector* is the difference a fine-tune made to a base model,
``tau_i = theta_ft_i - theta_base``. Merging combines several of them back into the
base. Everything here operates on plain ``dict[str, Tensor]`` state dicts — no model,
task, or dataset knowledge — so the same functions serve both the training pipeline
that produces a merge and the post-hoc diagnostics that analyse one.

Non-floating-point tensors (integer buffers) are never merged: they are copied from
the base unchanged, since arithmetic on them is not meaningful.
"""

import torch
from torch import Tensor

StateDict = dict[str, Tensor]


def merge_task_arithmetic(
    base_state: StateDict,
    ft_states: list[StateDict],
    scale: float,
) -> StateDict:
    """``theta_base + scale * sum_i(theta_ft_i - theta_base)``.

    Note this is a *sum* of task vectors, not an average — with N fine-tunes at
    ``scale=1.0`` the merged model sits N task vectors away from the base. Pass
    ``scale=1/N`` for averaging.
    """
    merged = {}
    for key, base_tensor in base_state.items():
        if base_tensor.is_floating_point():
            task_vectors = [
                ft[key].to(base_tensor.device) - base_tensor for ft in ft_states
            ]
            merged[key] = base_tensor + scale * torch.stack(task_vectors).sum(dim=0)
        else:
            merged[key] = base_tensor.clone()
    return merged
