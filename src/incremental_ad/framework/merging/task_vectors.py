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


# --- Sequential merging: one fold, two independent knobs ----------------------------------
#
# Plain task arithmetic, OPCM and BECAME are all the same accumulation with different knobs
# turned, and writing them as one fold is what makes the 2x2 ablation compose instead of
# needing three parallel code paths that each have to be kept honest separately:
#
#     A_0 = 0
#     A_t = decay_t * A_{t-1} + coefficient_t * transform(tau_t, tau_1..t-1)
#     theta_merged = theta_base + A_n
#
# - **transform** is the *merge rule* — what part of the incoming vector is kept. Identity
#   for plain summation; the component orthogonal to the accumulated subspace for OPCM.
# - **(decay, coefficient)** is the *coefficient source* — how much is kept. A fixed alpha
#   with decay 1 recovers `theta_base + alpha * sum(tau)` **exactly**, which is the property
#   that makes the baseline cell of the ablation the same number the rest of the project
#   already published rather than a re-derivation of it.
#
# BECAME's convex update theta*_t = (1 - lam_t) theta*_{t-1} + lam_t theta_hat_t becomes
# decay_t = 1 - lam_t, coefficient_t = lam_t: substituting theta_hat_t = theta_base + tau_t
# cancels the base term and leaves exactly this fold on the task vectors.


def merge_sequential(
    base_state: StateDict,
    task_vectors_list: Sequence[StateDict],
    weights: Sequence[tuple[float, float]],
    transform=None,
) -> StateDict:
    """Fold task vectors into the base one at a time.

    ``weights[t]`` is ``(decay, coefficient)`` for step t. ``transform(tau, history)``
    returns the part of ``tau`` to accumulate and defaults to identity; ``history`` is the
    list of task vectors already folded, in order, so a rule can be defined against what
    has been seen rather than against the running accumulator.

    Equivalence to :func:`apply_task_vectors` at ``weights = [(1.0, alpha)] * n`` and no
    transform is asserted by a unit check in `scripts/verify_merge_rules.py`, because the
    ablation is only interpretable if its baseline cell reproduces the published one.
    """
    assert len(weights) == len(task_vectors_list), (
        f"{len(weights)} weight pair(s) for {len(task_vectors_list)} task vector(s) — the "
        f"coefficient source must emit exactly one (decay, coefficient) per merge step"
    )
    # When every step has decay 1 and the same coefficient, this fold *is* plain task
    # arithmetic — so delegate to it rather than recompute it. Not an optimisation: summing
    # first and scaling once is a different floating-point expression from scaling each term
    # and adding, and the two differ by ~1e-7 in float32. That is far below any floor here,
    # but it would make the ablation's baseline cell a numerically-equivalent *re-derivation*
    # of the published merge instead of the published merge, and every comparison in the 2x2
    # is read against that cell.
    decays = {round(d, 12) for d, _ in weights}
    coefficients = {round(c, 12) for _, c in weights}
    if transform is None and decays == {1.0} and len(coefficients) == 1:
        return apply_task_vectors(base_state, task_vectors_list, next(iter(coefficients)))

    keys = [k for k, v in base_state.items() if v.is_floating_point()]
    accumulator: StateDict = {k: torch.zeros_like(base_state[k]) for k in keys}
    history: list[StateDict] = []

    for tau, (decay, coefficient) in zip(task_vectors_list, weights):
        contribution = tau if transform is None else transform(tau, history)
        for key in keys:
            accumulator[key] = (
                decay * accumulator[key] + coefficient * contribution[key].to(base_state[key].device)
            )
        history.append(tau)

    merged = {}
    for key, base_tensor in base_state.items():
        merged[key] = (
            base_tensor + accumulator[key] if base_tensor.is_floating_point() else base_tensor.clone()
        )
    return merged


def opcm_residual(threshold: float = 0.5):
    """Merge rule for OPCM: keep the component of each task vector *outside* the span of
    its predecessors.

    This is the exact complement of `geometry.sequential_overlap`, which reports
    ``rho = ||P(tau)||^2 / ||tau||^2`` — the fraction already inside the accumulated
    subspace. That function measures what OPCM throws away; this one throws it away.
    Sharing the projection means the diagnostic and the method cannot disagree about what
    the subspace is.

    Per-tensor, because the projection needs a matrix: 2-D weights are projected against
    the stack of previous vectors' matching tensors; 1-D tensors (biases, norm scales) have
    no meaningful row space at this size and are passed through unchanged, which is what
    the paper does. ``threshold`` is the fraction of squared singular values retained when
    truncating the subspace — the paper reports a stable optimum in 0.4-0.6.
    """

    def transform(tau: StateDict, history: list[StateDict]) -> StateDict:
        if not history:
            return tau
        out: StateDict = {}
        for key, tensor in tau.items():
            if tensor.ndim != 2 or min(tensor.shape) < 2:
                out[key] = tensor
                continue
            previous = torch.stack([h[key].reshape(-1) for h in history]).to(torch.float64)
            _, singular, right = torch.linalg.svd(previous, full_matrices=False)
            squared = singular**2
            total = squared.sum()
            if total <= 0:
                out[key] = tensor
                continue
            cumulative = torch.cumsum(squared / total, dim=0)
            rank = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype))) + 1
            rank = min(rank, right.shape[0])
            flat = tensor.reshape(-1).to(torch.float64)
            # residual = tau - P(tau), P projecting onto the leading `rank` directions
            projected = right[:rank].T @ (right[:rank] @ flat)
            out[key] = (flat - projected).reshape(tensor.shape).to(tensor.dtype)
        return out

    return transform
