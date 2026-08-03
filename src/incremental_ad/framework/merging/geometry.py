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


def geometry_report(base_state: StateDict, ft_states: Sequence[StateDict]) -> dict:
    """Every measurement above for one set of fine-tunes, as one serialisable dict.

    ``ft_states`` must be in temporal order — ``cosine_vs_distance`` reads the index as
    the segment's position on the time axis.
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
    }
