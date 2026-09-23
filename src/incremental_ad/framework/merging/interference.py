"""Interference-reducing merge rules: DARE, TIES, Iso-C and TSV-Merge.

Ported from the supervisor's reference implementation (`other/merging.py`,
`other/ties_merging_utility.py`, left untouched), checked line by line against each method's
**official** code, with four defects fixed and every remaining deviation from the published
method stated below rather than left for a reader to discover.

**Shape of every function here: an α-independent delta.** Each rule returns Δ such that the
merged model is ``theta_base + alpha * Delta`` — which is exactly how the reference code applies
alpha (it multiplies at the very end, in every method). So a scale sweep builds Δ once and
re-applies it with `apply_task_vectors(base, [Delta], alpha)`, the same split task arithmetic
already uses, and the SVDs in Iso-C and TSV are paid once per run rather than once per alpha.

**The alphas are NOT on a common scale, and that is inherited, not introduced.** Task
arithmetic and DARE sum the task vectors, so their natural alpha is ~1/n. TIES merges by a
(disjoint) mean, so its natural alpha is ~1. Iso-C and TSV put matrices at *sum* scale and every
other tensor at *mean* scale — which is what their official implementations do. A single fixed
alpha therefore favours whichever method it happens to suit; the only fair protocol is to select
alpha per method under the same rule, and the analysis does exactly that.

Defects fixed (each reproduced numerically on the reference code by
`scripts/verify_merge_baselines.py` before being changed):

1. **TIES divided the matrix delta by n.** The reference computes
   ``(alpha / num_tasks) * disjoint_mean`` for matrices but ``alpha * mean`` for everything
   else, so for one alpha the matrices sat at 1/n of the scale of the biases — with three
   identical task vectors, every surviving matrix entry came back at 0.333 of its value. The
   disjoint mean is already a mean; the official TIES code applies ``ptm + lamda * merged_tv``
   with no division. A global alpha cannot undo this, because it scales both parts equally.
2. **Iso-C flattened the MEAN task matrix instead of the SUM.** The official `iso_c` averages,
   then multiplies matrices back by ``len(tvs)`` before the SVD; the reference omits that line,
   so its flattened spectrum was 1/n of the paper's (3.000x smaller at n = 3, measured) while
   non-matrix tensors were unaffected. Same class of defect as (1), same reason alpha cannot
   absorb it.
3. **DARE was unseeded.** Its drop mask came from the global RNG, so two runs of the same merge
   differed. It is now drawn from an explicit per-task generator: the same seed gives the same
   merge, and the three training seeds of a run give three independent masks — the mask's
   variance is part of the method and must show up in the seed spread, not be hidden by it.
4. **TSV silently zeroed any matrix with rank < n.** ``int(rank / n)`` is 0 there, nothing is
   written into the concatenated factors, and the layer's delta comes back as exactly zero with
   no warning — a (4, 10) matrix at n = 5 was dropped entirely. The official code has the same
   arithmetic. It now raises instead. **No matrix in this project's models has a dimension
   below 5**, so this never fires on real runs; it is a guard, not a behaviour change.

Deviations from the published methods, **kept as the reference wrote them** (they are design
choices, not bugs, and changing them would stop this being the supervisor's method):

- **TIES trims per matrix, and only matrices.** The official code trims the whole flattened
  model to its top-k% at once; here each 2-D tensor is trimmed to its own top 20%, and every
  non-matrix tensor is merged by plain mean with no trim or sign election. Per-tensor trimming
  is also what several widely used implementations do.
- **DARE drops only from 2-D tensors.** The official MergeLM masks every parameter; here
  biases, norms and the 3-D positional tensors are summed unmasked.
- **Non-matrix tensors in TIES, Iso-C and TSV are merged by mean** — which matches the official
  Iso-C and TSV code exactly, and is part of why their alphas are not on task arithmetic's scale.

"Matrix" means a 2-D tensor throughout, as in the reference. This project's models also carry
3-D positional tensors of shape (1, N, d); like every other non-2-D tensor they take the
non-matrix path.
"""

import torch
from torch import Tensor

from incremental_ad.framework.merging.task_vectors import StateDict

# Hyperparameters exactly as the reference defaults them.
TIES_DENSITY = 0.2      # keep the top 20% of each matrix by magnitude (TIES paper's k = 20)
DARE_DROP_RATE = 0.7    # the reference's p; each 2-D entry survives with probability 1 - p


def _is_matrix(tensor: Tensor) -> bool:
    return tensor.dim() == 2


def _keys(taus: list[StateDict]) -> list[str]:
    if not taus:
        raise ValueError("no task vectors to merge")
    keys = list(taus[0])
    for tau in taus[1:]:
        if set(tau) != set(keys):
            raise ValueError("task vectors do not share one key set")
    return keys


def delta_task_arithmetic(taus: list[StateDict]) -> StateDict:
    """``sum_i tau_i`` — task arithmetic's delta. Included as the same-protocol control."""
    return {k: torch.stack([t[k] for t in taus]).sum(dim=0) for k in _keys(taus)}


def delta_dare(taus: list[StateDict], drop_rate: float = DARE_DROP_RATE,
               seed: int = 0) -> StateDict:
    """DARE (Yu et al. 2024) on top of task arithmetic: drop, rescale, then sum.

    Each 2-D entry of each task vector is kept with probability ``1 - drop_rate`` and the kept
    ones are multiplied by ``1 / (1 - drop_rate)``, so the expectation is unchanged. Non-2-D
    tensors are summed unmasked (the reference's choice; see the module docstring).

    The mask comes from ``torch.Generator().manual_seed(seed * 1000 + task_index)`` on the CPU,
    so it is identical across devices and across reruns, and independent between tasks.
    """
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    keys = _keys(taus)
    out = {k: torch.zeros_like(taus[0][k]) for k in keys}
    keep = 1.0 - drop_rate
    for index, tau in enumerate(taus):
        generator = torch.Generator().manual_seed(seed * 1000 + index)
        for k in keys:
            value = tau[k]
            if not _is_matrix(value) or drop_rate == 0.0:
                out[k] += value
                continue
            mask = torch.bernoulli(torch.full(value.shape, keep, dtype=torch.float32),
                                   generator=generator).to(value.device, value.dtype)
            out[k] += value * mask / keep
    return out


def _ties_matrix(stacked: Tensor, density: float) -> Tensor:
    """TIES on one (n_tasks, d) block: trim, elect sign, disjoint mean.

    A faithful transcription of the reference's `topk_values_mask` / `resolve_sign` /
    `disjoint_merge("mean")` — including its tie behaviour (entries equal to the threshold are
    kept, so a row can keep slightly more than ``density``) and its zero-sign rule (an elected
    sign of 0 takes the majority sign over the whole block).
    """
    n, d = stacked.shape
    if density < 1.0:
        k = d - int(d * density)                     # the k-th smallest is the cut-off
        if d == 1:
            threshold = stacked.abs()
        else:
            threshold, _ = stacked.abs().kthvalue(k, dim=1, keepdim=True)
        trimmed = stacked * (stacked.abs() >= threshold)
    else:
        trimmed = stacked
    sign = torch.sign(trimmed.sum(dim=0))
    sign[sign == 0] = torch.sign(sign.sum())
    agree = torch.where(sign.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
    selected = trimmed * agree
    count = (selected != 0).sum(dim=0).to(selected.dtype)
    return selected.sum(dim=0) / count.clamp(min=1)


def delta_ties(taus: list[StateDict], density: float = TIES_DENSITY) -> StateDict:
    """TIES-Merging (Yadav et al. 2023), per matrix; mean elsewhere.

    ⚠️ Fixed: the reference scaled the matrix delta by an extra 1/n (module docstring, defect 1).
    """
    keys = _keys(taus)
    out = {}
    for k in keys:
        values = [t[k] for t in taus]
        if _is_matrix(values[0]):
            stacked = torch.stack([v.reshape(-1) for v in values])
            out[k] = _ties_matrix(stacked, density).reshape(values[0].shape)
        else:
            out[k] = torch.stack(values).mean(dim=0)
    return out


def delta_iso_c(taus: list[StateDict]) -> StateDict:
    """Iso-C (Marczak et al. 2025): flatten the summed task matrix's spectrum to its mean.

    For each matrix, ``M = sum_i tau_i = U S V^T`` and the delta is ``mean(S) * U V^T`` — every
    direction the tasks jointly span gets the same weight. Non-matrix tensors take the mean.

    ⚠️ Fixed: the reference flattened the MEAN matrix, 1/n of the official sum (defect 2). The
    SVD is done in float64 and cast back, so the result does not depend on how well-conditioned
    the float32 sum happens to be.
    """
    keys = _keys(taus)
    n = len(taus)
    out = {}
    for k in keys:
        values = [t[k] for t in taus]
        mean = torch.stack(values).mean(dim=0)
        if _is_matrix(values[0]):
            summed = (mean * n).to(torch.float64)          # the official code's `*= len(tvs)`
            u, s, vh = torch.linalg.svd(summed, full_matrices=False)
            flat = torch.full_like(s, s.mean().item())
            out[k] = torch.linalg.multi_dot((u, torch.diag(flat), vh)).to(values[0].dtype)
        else:
            out[k] = mean
    return out


def _tsv_matrix(values: list[Tensor]) -> Tensor:
    """TSV-Merge on one layer — the reference's `get_tsv_delta_w` with unit task weights.

    Keep each task's top ``rank / n`` singular triplets, concatenate the U blocks, the V blocks
    and the singular values, then replace the concatenated U and V by their orthogonal polar
    factors (Procrustes: X = P S Q^T -> P Q^T), which is what removes the interference.
    """
    n = len(values)
    blocks_u, blocks_s, blocks_v = [], [], []
    per_task = None
    for value in values:
        u, s, vh = torch.linalg.svd(value.to(torch.float64), full_matrices=False)
        if per_task is None:
            per_task = int(s.shape[0] / n)
            if per_task == 0:
                # ⚠️ Fixed (defect 4): the reference writes nothing here and returns zeros.
                raise ValueError(
                    f"TSV cannot merge a matrix of rank {s.shape[0]} across {n} tasks: "
                    f"rank/n rounds to 0, which would silently zero this layer's delta")
            sum_u = torch.zeros_like(u)
            sum_s = torch.zeros_like(s)
            sum_v = torch.zeros_like(vh)
        blocks_u.append(u[:, :per_task])
        blocks_s.append(s[:per_task])
        blocks_v.append(vh[:per_task, :])
    width = per_task * n
    sum_u[:, :width] = torch.cat(blocks_u, dim=1)
    sum_s[:width] = torch.cat(blocks_s)
    sum_v[:width, :] = torch.cat(blocks_v, dim=0)
    u_u, _s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    merged = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    return merged.to(values[0].dtype)


def delta_tsv(taus: list[StateDict]) -> StateDict:
    """TSV-Merge (Gargiulo et al. 2025) per matrix; mean elsewhere, as the official code does."""
    keys = _keys(taus)
    out = {}
    for k in keys:
        values = [t[k] for t in taus]
        out[k] = (_tsv_matrix(values) if _is_matrix(values[0])
                  else torch.stack(values).mean(dim=0))
    return out


DELTAS = {
    "ta": delta_task_arithmetic,
    "dare": delta_dare,
    "ties": delta_ties,
    "iso_c": delta_iso_c,
    "tsv": delta_tsv,
}
