"""OPCM as published — Tang et al., *Merging on the Fly Without Retraining*, NeurIPS 2025.

This is the paper's operator, implemented from the paper. It is a **separate** function from
`opcm_residual` in `task_vectors.py`, which is a simplification (residual against the *flattened*
predecessors) used in EXPERIMENTS.md §1.31 and §1.35. Neither is described as the other anywhere:
the simplified one is never called the paper's OPCM, and this one never inherits its results.

Algorithm 1 of the paper, with every step named:

    theta_merged^(1) = theta^(1)          (line 1: the first expert is taken whole, lambda^(1) = 1)
    n^(1)            = ||dtheta^(1)||_2   (line 1: running average of task-vector norms)

    for t = 2..T:
        dW_proj^(t)  = P_alpha^(t-1)( dW^(t) ; dW_merged^(t-1) )   (line 6,  per 2-D weight matrix)
        dp^(t)       = p^(t) - p^(0)                               (line 9,  everything else)
        n^(t)        = (t-1)/t * n^(t-1) + 1/t * ||dtheta^(t)||_2  (line 13)
        lambda^(t)   = || lambda^(t-1) dtheta_merged^(t-1) + dtheta_proj^(t) ||_2 / n^(t)   (line 14)
        theta_merged^(t) = theta^(0)
                         + (lambda^(t-1) dtheta_merged^(t-1) + dtheta_proj^(t)) / lambda^(t)  (15)

**The projection** (paper §4, the equation defining `P_alpha^(t-1)`). Full SVD of the accumulated
merged task matrix, ``dW_merged^(t-1) = U Sigma V^T``, and

    P_alpha(dW) = sum_{i,j >= r_alpha, i != j} <dW, u_i v_j^T>_F  u_i v_j^T

with ``r_alpha`` the minimal rank such that ``sum_{i<=r_alpha} sigma_i >= alpha * sum_i sigma_i``.
Written as one matrix expression — which is how it is computed here — that is
``U (M * (U^T dW V)) V^T`` with ``M`` the 0/1 mask of the retained (i, j).

Three things this operator does that the simplified one does not:

1. It projects out the top singular subspace of the **accumulated merged** matrix, not the span of
   the individual predecessors, and it does so **on both sides** (rows *and* columns are cut at
   ``r_alpha``).
2. It also drops the diagonal ``i == j``. Theorem 5.1's orthogonality result
   ``<P(dW^(t)), dW_merged^(t-1)>_F = 0`` (Eq. 8) rests on that exclusion **alone** —
   ``<u_i v_j^T, U Sigma V^T>_F = sigma_i delta_ij`` — so the ``r_alpha`` cut is a knowledge-
   retention knob, not what makes the result orthogonal. Both are asserted in
   `scripts/verify_merge_rules.py`.
3. It carries the norm-stabilising ``lambda^(t)``, which is not a tunable strength: by
   construction ``|| theta_merged - theta^(0) ||_2 == n^(T)``, the **mean task-vector norm**
   (the paper's Avg(||dtheta^(i)||_2), and the point of Theorem 5.2). So OPCM chooses its own
   magnitude and no external merge scale applies to it. That matters here, because this project's
   central quantity is exactly that magnitude: see EXPERIMENTS.md §1.18 (alpha*.n) and §1.35,
   where BECAME was found pinned at alpha.n = 1.0 by *its* update rule. `merge_opcm_paper`
   therefore returns the realised strength alongside the merge rather than accepting one.

**Indexing note, stated because it is a real ambiguity.** The paper writes the sum as starting at
``i, j = r_alpha``, which in 1-based mathematical notation would *retain* the ``r_alpha``-th
direction even though it is inside the top-alpha energy. The intent is plainly to remove the
directions carrying that energy, and a 0-based implementation (``[r_alpha:]``) does exactly that,
so that is what is implemented here. The two readings differ by one singular direction out of
min(m, n) = 64-256 for this backbone, and Theorem 5.1 holds under either.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from .task_vectors import StateDict, float_keys


def _retained_rank(singular_values: Tensor, threshold: float) -> int:
    """``r_alpha``: how many leading directions carry ``threshold`` of the spectral mass.

    Returned as a count, i.e. the 0-based index of the first *retained* direction. A degenerate
    matrix (all-zero task vector) has no mass to cut, and returns 0 so the projection becomes the
    identity rather than silently annihilating the update.
    """
    total = float(singular_values.sum())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(singular_values, dim=0)
    # Minimal r with cumulative[r-1] >= threshold * total; +1 converts the count of values
    # strictly below the target into that rank.
    return int((cumulative < threshold * total).sum().item()) + 1


def project_orthogonal(
    incoming: Tensor, accumulated: Tensor, threshold: float
) -> Tensor:
    """``P_alpha(incoming; accumulated)`` for one 2-D weight matrix.

    ``accumulated`` is the merged task matrix so far, whose SVD defines the subspace. Computed in
    float64: the coefficients are inner products over ~10^5 entries of small weight differences,
    and the orthogonality this is supposed to guarantee is checked to 1e-9 downstream.
    """
    assert incoming.ndim == 2 and accumulated.ndim == 2, "2-D weight matrices only"
    if not torch.any(accumulated):
        # Nothing has been merged into this matrix yet, so there is no subspace to project out
        # of. Returning the identity here (rather than zero) is what makes step t = 2 fold in the
        # second expert whole when the first contributed nothing to this tensor.
        return incoming
    left, singular_values, right_h = torch.linalg.svd(
        accumulated.to(torch.float64), full_matrices=True
    )
    rank = _retained_rank(singular_values, threshold)
    coefficients = left.T @ incoming.to(torch.float64) @ right_h.T     # <dW, u_i v_j^T>_F
    mask = torch.ones_like(coefficients)
    mask[:rank, :] = 0.0                       # cut the top-r_alpha rows ...
    mask[:, :rank] = 0.0                       # ... and columns: the projection is two-sided
    diagonal = min(coefficients.shape)
    mask[range(diagonal), range(diagonal)] = 0.0   # i == j, which is what Theorem 5.1 needs
    return (left @ (mask * coefficients) @ right_h).to(incoming.dtype)


def merge_opcm_paper(
    base_state: StateDict,
    task_vectors_list: Sequence[StateDict],
    threshold: float = 0.5,
    scale_to_alpha: float | None = None,
) -> tuple[StateDict, dict[str, float]]:
    """Merge task vectors with the paper's OPCM. Returns ``(merged_state, diagnostics)``.

    ``threshold`` is the paper's projection threshold alpha (its Figure 6b optimum is ~0.4-0.6).
    It is **not** a merge scale: OPCM sets its own magnitude, and the returned diagnostics report
    what that magnitude came out to, so it can be read against this project's alpha*.n.

    Diagnostics:
      ``lambda_final``          the paper's lambda^(T).
      ``mean_task_vector_norm`` n^(T) = Avg(||dtheta^(i)||_2).
      ``merged_norm``           ||theta_merged - theta^(0)||_2, which **must** equal n^(T).
      ``norm_ratio``            merged_norm / n^(T); 1.0 by construction under the paper's rule
                                (Theorem 5.2's point), and deliberately NOT 1.0 once
                                ``scale_to_alpha`` replaces it.
      ``implied_alpha_times_n`` n * merged_norm / ||sum_i dtheta^(i)||_2 — the uniform alpha.n that
                                plain summation would need to travel the same distance. This is the
                                only scalar here comparable with a committed merge scale.
      ``projected_matrices``    how many tensors took the SVD path.
      ``passthrough_tensors``   how many did not (1-D and anything not exactly 2-D).

    ``scale_to_alpha`` replaces the paper's final rescale and **nothing else** (EXPERIMENTS.md
    §1.37, `C34`). Eq. 7 enters every projected vector with the same coefficient ``1 / lambda^(T)``;
    passing alpha sets that coefficient to alpha instead, so the per-vector weights sum to
    ``alpha * n`` exactly and the merge becomes plain task arithmetic *on the projected task
    vectors*. The projection is untouched — it is computed from the accumulator, whose SVD basis
    is invariant to positive scaling, so no step upstream of the final line moves. With it set,
    the result is **not the paper's method** and must never be reported as such (`C26`); it is
    "the paper's projection at a chosen strength", exactly as rescaled BECAME is "BECAME's
    weighting at a chosen strength".

    Default ``None`` keeps Algorithm 1 exactly, so §1.36's numbers are unaffected.
    """
    assert task_vectors_list, "no task vectors to merge"
    assert 0.0 < threshold <= 1.0, f"projection threshold must be in (0, 1], got {threshold}"
    keys = float_keys(base_state)

    def norm(state: StateDict) -> float:
        return float(torch.sqrt(sum((state[k].to(torch.float64) ** 2).sum() for k in keys)))

    # Algorithm 1, line 1: theta_merged^(1) = theta^(1), so the accumulator starts as the first
    # task vector in full -- never projected, and never scaled.
    accumulated: StateDict = {k: task_vectors_list[0][k].clone() for k in keys}
    mean_norm = norm(task_vectors_list[0])
    lambda_t = 1.0
    projected, passthrough = 0, 0

    for step, incoming in enumerate(task_vectors_list[1:], start=2):
        contribution: StateDict = {}
        for key in keys:
            tensor = incoming[key]
            if tensor.ndim == 2:
                contribution[key] = project_orthogonal(tensor, accumulated[key], threshold)
                projected += 1
            else:
                # Paper §4: "For biases in linear layers and other parameters, we simply set
                # P_alpha^(t-1) in Eq.(6) as the identity mapping."
                contribution[key] = tensor
                passthrough += 1
        # lambda^(t-1) * dtheta_merged^(t-1) == accumulated, because dtheta_merged^(t-1) is itself
        # accumulated / lambda^(t-1) (Eq. 7). So the fold is a plain addition and lambda never
        # compounds -- which is also why the projection subspace is scale-invariant.
        for key in keys:
            accumulated[key] = accumulated[key] + contribution[key]
        mean_norm = (step - 1) / step * mean_norm + norm(incoming) / step   # line 13
        lambda_t = norm(accumulated) / mean_norm                            # line 14

    # The ONLY line that differs between the paper's rule and the matched-magnitude variant.
    coefficient = (1.0 / lambda_t) if scale_to_alpha is None else scale_to_alpha
    merged = {
        key: (value + coefficient * accumulated[key] if key in accumulated else value.clone())
        for key, value in base_state.items()
    }
    merged_norm = norm(accumulated) * coefficient
    total = {k: sum(tau[k] for tau in task_vectors_list) for k in keys}
    plain_sum_norm = norm(total)
    # With an explicit alpha the implied total strength is the coefficient sum, alpha * n, by
    # construction -- an exact identity the caller asserts. Under the paper's rule there is no
    # such coefficient, so the only comparable quantity is the norm ratio against plain summation.
    implied = (len(task_vectors_list) * scale_to_alpha if scale_to_alpha is not None
               else (len(task_vectors_list) * merged_norm / plain_sum_norm
                     if plain_sum_norm else float("nan")))
    return merged, {
        "scale_to_alpha": float("nan") if scale_to_alpha is None else scale_to_alpha,
        "lambda_final": lambda_t,
        "mean_task_vector_norm": mean_norm,
        "merged_norm": merged_norm,
        "norm_ratio": merged_norm / mean_norm if mean_norm else float("nan"),
        "implied_alpha_times_n": implied,
        "projected_matrices": float(projected),
        "passthrough_tensors": float(passthrough),
    }
