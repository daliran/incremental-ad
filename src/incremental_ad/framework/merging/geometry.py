"""Parameter-space geometry of a set of task vectors.

Measures where the task vectors point relative to each other, using nothing but the
state dicts — no data, no model forward pass, no GPU. This is the cheap proxy: it says
whether the updates *overlap in parameter space*, which is not the same question as
whether they *interfere functionally*. Two task vectors can be geometrically orthogonal
and still conflict, because orthogonal weight updates may still modify the same
features. Treat these as predictors to be checked against a functional measurement, not
as evidence of disentanglement on their own.

All computation is float64 on the CPU; returns are plain numpy/Python so the results
serialise directly.

A task vector that is exactly zero (a fine-tune that moved nothing) has no direction, so
cosines involving it are ``nan`` rather than an arbitrary value.
"""

from collections.abc import Sequence

import numpy as np
import torch

from incremental_ad.framework.merging.task_vectors import (
    StateDict,
    flatten_state,
    float_keys,
    task_vector,
)


def _stack(taus: Sequence[StateDict], keys: Sequence[str]) -> torch.Tensor:
    """[n, D] matrix of flattened task vectors, index-aligned by ``keys``."""
    return torch.stack([flatten_state(t, keys) for t in taus])


def _cosine_from_stack(flat: torch.Tensor) -> np.ndarray:
    """Pairwise cosine of the rows of [n, D]; nan wherever a row has zero norm."""
    norms = flat.norm(dim=1)
    denom = norms[:, None] * norms[None, :]
    cosine = torch.where(denom > 0, (flat @ flat.T) / denom, torch.nan)
    # A vector is exactly parallel to itself; guard against float drift off 1.0.
    diagonal = torch.where(norms > 0, torch.ones_like(norms), torch.nan)
    cosine.diagonal().copy_(diagonal)
    return cosine.numpy()


def cosine_matrix(taus: Sequence[StateDict], keys: Sequence[str]) -> np.ndarray:
    """[n, n] pairwise cosine between whole flattened task vectors.

    High off-diagonal values mean the fine-tunes pushed the weights in the same
    direction (redundant or conflicting); values near zero mean they edited largely
    independent directions.
    """
    return _cosine_from_stack(_stack(taus, keys))


def per_tensor_cosine(
    taus: Sequence[StateDict], keys: Sequence[str]
) -> dict[str, np.ndarray]:
    """``{parameter_name: [n, n] cosine}`` — the same measurement per weight tensor.

    Reveals whether overlap concentrates in particular blocks (attention vs. MLP vs.
    embedding vs. decoder head) rather than being spread evenly.
    """
    return {
        key: _cosine_from_stack(
            torch.stack([t[key].reshape(-1).to(torch.float64) for t in taus])
        )
        for key in keys
    }


def delta_norms(
    taus: Sequence[StateDict], base_state: StateDict, keys: Sequence[str]
) -> dict[str, list[float] | float]:
    """How far each fine-tune actually moved, absolutely and relative to the base.

    ``tau_over_base`` near zero is the signature of degenerate fine-tuning: no merge
    strategy can recover signal from task vectors that carry none.
    """
    base_norm = float(flatten_state(base_state, keys).norm())
    tau_norms = [float(flatten_state(t, keys).norm()) for t in taus]
    return {
        "base_norm": base_norm,
        "tau_norm": tau_norms,
        "tau_over_base": [n / base_norm if base_norm > 0 else float("nan") for n in tau_norms],
    }


def effective_rank(taus: Sequence[StateDict], keys: Sequence[str]) -> dict:
    """``exp(H(p))`` where p normalises the squared singular values of the stacked
    task-vector matrix.

    Bounded in [1, n], so read it as "effective directions out of n available". A value
    near 1 means every fine-tune made essentially the same edit; a value near n means
    they span independent directions.
    """
    singular_values = torch.linalg.svdvals(_stack(taus, keys))
    energy = singular_values**2
    total = energy.sum()
    if total <= 0:
        return {
            "effective_rank": float("nan"),
            "entropy": float("nan"),
            "singular_values": singular_values.tolist(),
            "p": [float("nan")] * len(singular_values),
        }

    p = energy / total
    nonzero = p > 0  # 0 log 0 := 0
    entropy = float(-(p[nonzero] * p[nonzero].log()).sum())
    return {
        "effective_rank": float(np.exp(entropy)),
        "entropy": entropy,
        "singular_values": singular_values.tolist(),
        "p": p.tolist(),
    }


def cosine_vs_distance(cosine: np.ndarray) -> dict[int, dict[str, float | list[float]]]:
    """Cosine grouped by temporal separation ``|i - j|`` between segments.

    The load-bearing plot of the geometry set. If similarity does not decay as segments
    grow further apart in time, temporal distribution shift is not what differentiates
    the task vectors, whatever else the results show.
    """
    n = cosine.shape[0]
    grouped = {}
    for distance in range(1, n):
        values = [float(cosine[i, i + distance]) for i in range(n - distance)]
        finite = [v for v in values if np.isfinite(v)]
        grouped[distance] = {
            "values": values,
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "std": float(np.std(finite)) if finite else float("nan"),
            "n": len(finite),
        }
    return grouped


def sequential_overlap(
    taus: Sequence[StateDict], keys: Sequence[str], *, energy: float = 0.90
) -> dict:
    """How much of each incoming task vector already lies in the span of its predecessors.

    At merge step k, ``rho_k = ||P(tau_k)||^2 / ||tau_k||^2`` where P projects onto the
    leading right-singular subspace of the matrix stacking tau_0..tau_{k-1}, truncated at
    the rank covering ``energy`` of its squared singular values. rho near 1 means the new
    shard is re-editing directions already claimed; near 0 means it is opening new ones.

    This is the per-merge diagnostic the model-materialisation question needs: a rho that
    stays high says continuing to merge adds little, which is a candidate trigger for
    starting a new model instead.

    Note the accumulated update is treated as the *matrix* of previous task vectors, not
    their sum -- summing first collapses the subspace to a single direction and makes the
    rank truncation vacuous. A consequence worth checking: at k=1 the subspace is
    necessarily 1-D, so rho_1 equals the squared cosine between tau_0 and tau_1 exactly.
    """
    flat = _stack(taus, keys)
    rho: list[float] = []
    ranks: list[int] = []

    for k in range(1, len(taus)):
        previous = flat[:k]
        _, singular_values, right = torch.linalg.svd(previous, full_matrices=False)
        squared = singular_values**2
        total = squared.sum()
        if total <= 0:
            rho.append(float("nan"))
            ranks.append(0)
            continue

        cumulative = torch.cumsum(squared / total, dim=0)
        rank = int(torch.searchsorted(cumulative, torch.tensor(energy, dtype=cumulative.dtype))) + 1
        rank = min(rank, right.shape[0])

        incoming = flat[k]
        norm_squared = incoming.dot(incoming)
        projected = right[:rank] @ incoming
        rho.append(float(projected.dot(projected) / norm_squared) if norm_squared > 0 else float("nan"))
        ranks.append(rank)

    return {"rho": rho, "rank_used": ranks, "energy": energy}


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Principal angles in radians between two column-orthonormal bases, ascending.

    Zero means the subspaces share a direction exactly; pi/2 means that direction of one
    is orthogonal to all of the other.
    """
    singular_values = torch.linalg.svdvals(a.T @ b)
    return np.arccos(np.clip(singular_values.numpy(), -1.0, 1.0))


def subspace_principal_angles(
    taus: Sequence[StateDict], keys: Sequence[str], *, rank: int = 8
) -> dict[str, dict]:
    """Principal angles between the leading column subspaces of each 2-D weight delta.

    Flattening a weight matrix into one long vector throws away its structure, so two
    updates spanning the same subspace can still look uncorrelated by cosine. This is the
    quantity subspace-projection merging methods actually operate on, so it is the one to
    have measured before comparing against them.

    2-D tensors are selected structurally (``ndim == 2``) rather than by name: that picks
    every linear weight including fused attention projections, and excludes 1-D norm
    parameters and 3-D positional encodings, without hardcoding any model's naming.
    """
    report: dict[str, dict] = {}
    for key in keys:
        if taus[0][key].ndim != 2:
            continue

        bases = []
        for tau in taus:
            delta = tau[key].to(torch.float64)
            left, _, _ = torch.linalg.svd(delta, full_matrices=False)
            bases.append(left[:, : min(rank, *delta.shape)])

        n = len(bases)
        angles = [[None] * n for _ in range(n)]
        mean_angle = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                pair = principal_angles(bases[i], bases[j])
                angles[i][j] = pair.tolist()
                mean_angle[i][j] = float(pair.mean())

        report[key] = {
            "rank": bases[0].shape[1],
            "shape": list(taus[0][key].shape),
            "angles_rad": angles,
            "mean_angle_rad": mean_angle,
        }
    return report


def geometry_report(
    base_state: StateDict,
    ft_states: Sequence[StateDict],
    *,
    svd_rank: int = 8,
    energy: float = 0.90,
) -> dict:
    """Every measurement above for one set of fine-tunes, as one serialisable dict.

    ``ft_states`` must be in temporal order — ``cosine_vs_distance`` and
    ``sequential_overlap`` both read the index as the segment's position on the time axis.
    """
    keys = float_keys(base_state)
    taus = [task_vector(base_state, ft) for ft in ft_states]

    cosine = cosine_matrix(taus, keys)
    return {
        "n_segments": len(taus),
        "n_parameters": int(sum(base_state[k].numel() for k in keys)),
        "n_tensors": len(keys),
        "cosine": cosine.tolist(),
        "per_tensor_cosine": {k: v.tolist() for k, v in per_tensor_cosine(taus, keys).items()},
        "norms": delta_norms(taus, base_state, keys),
        "effective_rank": effective_rank(taus, keys),
        "cosine_vs_distance": cosine_vs_distance(cosine),
        "sequential_overlap": sequential_overlap(taus, keys, energy=energy),
        "subspace_principal_angles": subspace_principal_angles(taus, keys, rank=svd_rank),
    }
