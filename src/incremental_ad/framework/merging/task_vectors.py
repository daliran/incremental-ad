"""Task-vector construction and merging over model state dicts.

A *task vector* is the difference a fine-tune made to a base model,
``tau_i = theta_ft_i - theta_base``. Merging combines several of them back into the
base. Everything here operates on plain ``dict[str, Tensor]`` state dicts — no model,
task, or dataset knowledge — so the same functions serve both the training pipeline
that produces a merge and the post-hoc diagnostics that analyse one.

Non-floating-point tensors (integer buffers) are never merged: they are copied from
the base unchanged, since arithmetic on them is not meaningful.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

StateDict = dict[str, Tensor]


def float_keys(state: StateDict) -> list[str]:
    """Sorted names of the floating-point tensors — the canonical flattening order.

    Callers that flatten several state dicts must pass the *same* key list to each so
    the resulting vectors are index-aligned.
    """
    return sorted(k for k, v in state.items() if v.is_floating_point())


def task_vector(base_state: StateDict, ft_state: StateDict) -> StateDict:
    """``theta_ft - theta_base``, over floating-point tensors only.

    Iterating the *base* keys means a key present only in ``ft_state`` would be dropped in
    silence rather than raising, so the key sets are checked. A missing key already fails
    loudly with KeyError; this covers the other direction.
    """
    extra = set(ft_state) - set(base_state)
    assert not extra, (
        f"fine-tuned state has {len(extra)} key(s) absent from the base state, which would "
        f"be silently dropped from the task vector: {sorted(extra)[:5]}"
    )
    return {
        key: ft_state[key].to(base_tensor.device) - base_tensor
        for key, base_tensor in base_state.items()
        if base_tensor.is_floating_point()
    }


def flatten_state(state: StateDict, keys: Sequence[str]) -> Tensor:
    """Concatenate the named tensors into one 1-D float64 vector, in ``keys`` order.

    float64 because task vectors are small differences of float32 weights, and inner
    products accumulated over ~10^6 parameters lose meaningful precision in float32.
    """
    return torch.cat([state[k].reshape(-1).to(torch.float64) for k in keys])


def apply_task_vectors(
    base_state: StateDict,
    task_vectors: Sequence[StateDict],
    scale: float,
) -> StateDict:
    """``theta_base + scale * sum_i(tau_i)``.

    Split out from :func:`merge_task_arithmetic` so a caller sweeping the scale can
    build the task vectors once and re-apply them at every value.

    Note this is a *sum*, not an average — with N task vectors at ``scale=1.0`` the
    result sits N task vectors away from the base. Pass ``scale=1/N`` for averaging.
    """
    merged = {}
    for key, base_tensor in base_state.items():
        if base_tensor.is_floating_point():
            stacked = torch.stack([tau[key] for tau in task_vectors])
            merged[key] = base_tensor + scale * stacked.sum(dim=0)
        else:
            merged[key] = base_tensor.clone()
    return merged


def merge_task_arithmetic(
    base_state: StateDict,
    ft_states: Sequence[StateDict],
    scale: float,
) -> StateDict:
    """``theta_base + scale * sum_i(theta_ft_i - theta_base)`` — plain task arithmetic."""
    return apply_task_vectors(
        base_state, [task_vector(base_state, ft) for ft in ft_states], scale
    )
