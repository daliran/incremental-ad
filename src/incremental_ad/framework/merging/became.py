"""BECAME's adaptive merge coefficient, and the diagonal Fisher it needs.

The closed form, for merge step t:

    lambda*_t = (d^T F_t d) / (d^T (F_t + sum_{i<t} F_i) d),    d = theta_hat_t - theta*_{t-1}
    theta*_t  = (1 - lambda*_t) theta*_{t-1} + lambda*_t theta_hat_t

Read it as a signal-to-total ratio in the metric the Fisher induces: the numerator is how
much the *incoming* task cares about the direction being moved, the denominator how much
every task seen so far cares. When all tasks care equally the ratio tends to 1/t, so the
derivation reproduces the 1/n rule this project measures empirically (EXPERIMENTS.md
§1.18). That is a prediction of the method, stated here before it is run.

**Scope: the coefficient only.** BECAME's first stage — theta_GP via gradient projection
(GPM/NSCL) — is deliberately not implemented here.

⚠️ The reason previously given in this docstring was **wrong** and is corrected (2026-09-20).
It said GPM was "class-incremental-classification machinery built around a growing label
space". It is not: GPM constrains each update to the orthogonal complement of the subspace
spanned by previous tasks' **layer input activations**, obtained by SVD of those activations
(paper, Appendix A). It needs no label space and is architecture-agnostic, so nothing about
this project's lack of classes rules it out.

The real reasons are two, and only the second is an argument: (1) it is out of frame here —
this module serves *merging*, where every shard is fine-tuned from the frozen theta_0 and
there is no sequential trajectory to project along; and (2) it is expected to *saturate* on
consecutive windows of one series, because the periods share input channels and most of
their activation subspace, so the projection would remove nearly the whole gradient. That
second reason is **a hypothesis, and it was not measured**: no activation-overlap or
retained-gradient measurement exists in this repository. An earlier version of this docstring
cited a `scripts/diagnose_activation_overlap.py` that was never written. Gradient projection
was not built and not measured (EXPERIMENTS.md §1.39); nothing here should be read as evidence
that it would saturate.

This is therefore *BECAME's coefficient applied to this project's merges*, not a
reimplementation of BECAME, and it must not be described as one.

**Frame mismatch, stated because it changes what the coefficient means.** BECAME assumes
theta_hat_t was fine-tuned *from the accumulator* theta*_{t-1}, so d is a genuine one-step
displacement. Every fine-tune here starts from the frozen theta_0, because that is what
makes tau_i = theta_i - theta_0 a task vector in one shared coordinate frame — the
assumption the whole project rests on. Applying the coefficient to frozen-base checkpoints
keeps that frame and deviates from the paper's; d is then the displacement from the
accumulator to a model trained elsewhere, which is the quantity actually available in a
zero-retention setting. Re-training each shard from the accumulator would satisfy the paper
and destroy the frame, so it is not done.

The Fisher is diagonal and empirical: the mean of squared per-parameter gradients of the
training loss over a sample of a shard's own data. Diagonal because the full Fisher is
p x p, and p is ~700k here.

⚠️ **On anomaly detection the Fisher inherits the objective, not just the tuning.** It is
computed from the reconstruction loss, and §1.12 shows that loss is blind to detection
quality. So lambda* removes the *tuning* problem — no validation sweep — without touching
the *objective* problem. A coefficient derived from a signal that cannot see AUROC is not
an honest alpha for AD merely because it was not tuned.
"""

import logging

import torch
from torch import Tensor

from incremental_ad.framework.merging.task_vectors import StateDict

log = logging.getLogger(__name__)


def diagonal_fisher(
    model,
    loader,
    device,
    max_batches: int | None = None,
    counts: dict | None = None,
) -> dict[str, Tensor]:
    """Mean squared gradient of the training loss, per parameter, over `loader`.

    Needs a model, a loader and a loss — which is why this module is the exception to
    `merging/`'s state-dicts-only rule, and why it is imported by the pipeline rather than
    by the geometry tooling. Gradients come from `model.compute_loss`, the same objective the
    trainer minimises, so the Fisher describes the loss actually optimised rather than a
    proxy for it.

    `max_batches` bounds the cost: the Fisher is an expectation, and a few hundred batches
    estimate it well enough for a ratio of two quadratic forms. Batches are taken in loader
    order, so pass a shuffled loader if the shard is not homogeneous.

    `counts`, if given, is filled with the batches and SAMPLES actually used. The sample count
    cannot be read off the flags: a short shard ends the loader before `max_batches`, and a last
    partial batch holds fewer than `batch_size` samples. Callers that report a sample count
    should report this one.
    """
    from incremental_ad.framework.core.device import move_to_device

    fisher = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.eval()          # no dropout: the Fisher should describe the deployed function
    batches = samples = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = model.compute_loss(move_to_device(batch, device))
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in fisher:
                fisher[name] += (parameter.grad.detach().double() ** 2).cpu()
        batches += 1
        samples += _batch_len(batch)
    model.zero_grad(set_to_none=True)

    if batches == 0:
        raise ValueError("diagonal_fisher got an empty loader — cannot estimate a Fisher")
    for name in fisher:
        fisher[name] /= batches
    log.info("[fisher] estimated over %d batch(es), %d sample(s); mean diagonal %.3e",
             batches, samples, float(torch.cat([v.reshape(-1) for v in fisher.values()]).mean()))
    if counts is not None:
        counts.update({"batches": batches, "samples": samples})
    return fisher


def _batch_len(batch) -> int:
    """Leading dimension of the first tensor in a (possibly nested) batch."""
    if isinstance(batch, Tensor):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        batch = list(batch.values())
    for item in batch:
        found = _batch_len(item) if isinstance(item, (Tensor, list, tuple, dict)) else None
        if found:
            return found
    raise ValueError("cannot infer the sample count of a batch with no tensor in it")


def became_lambda(
    displacement: StateDict,
    fisher_new: dict[str, Tensor],
    fisher_previous: list[dict[str, Tensor]],
) -> float:
    """`lambda*` for one merge step: (d^T F_t d) / (d^T (F_t + sum F_i) d).

    Both quadratic forms are diagonal, so each is a sum of ``F[k] * d[k]**2``. Only keys
    present in *both* the displacement and the Fisher contribute: the Fisher covers trainable
    parameters while a state dict also carries buffers, and silently treating a missing
    Fisher entry as zero would be the same thing as excluding it — but excluding it visibly
    is what lets the count be logged and checked.

    Returns 1.0 for the first step (no previous Fisher, so the ratio is d^T F d over itself)
    and falls back to 1.0 with a warning if the denominator vanishes, which happens only when
    the displacement lies entirely in directions no task has any curvature in.
    """
    shared = [k for k in displacement if k in fisher_new]
    numerator = 0.0
    denominator = 0.0
    for key in shared:
        squared = displacement[key].detach().double().cpu() ** 2
        own = float((fisher_new[key] * squared).sum())
        others = sum(float((f[key] * squared).sum()) for f in fisher_previous if key in f)
        numerator += own
        denominator += own + others
    if denominator <= 0:
        log.warning("[became] zero denominator over %d shared tensor(s) — falling back to "
                    "lambda=1.0; the displacement has no curvature under any task's Fisher",
                    len(shared))
        return 1.0
    value = numerator / denominator
    log.info("[became] lambda* = %.6f  (numerator %.4e / denominator %.4e over %d tensors, "
             "%d previous Fisher(s))", value, numerator, denominator, len(shared),
             len(fisher_previous))
    return value


def became_weights(
    base_state: StateDict,
    task_vectors_list: list[StateDict],
    fishers: list[dict[str, Tensor]],
) -> tuple[list[tuple[float, float]], list[float]]:
    """`(decay, coefficient)` per step for `merge_sequential`, plus the raw lambdas.

    The displacement `d_t = theta_hat_t - theta*_{t-1}` is computed against the *running*
    accumulator, so it depends on every lambda chosen before it — the coefficients cannot be
    derived in one pass from the task vectors alone. That sequential dependence is the reason
    this returns weights rather than a single scalar, and the reason the accumulator is
    rebuilt here instead of being read back out of `merge_sequential`.
    """
    assert len(fishers) == len(task_vectors_list), (
        f"{len(fishers)} Fisher(s) for {len(task_vectors_list)} task vector(s) — one per "
        f"shard is required, since lambda*_t compares shard t against all earlier shards"
    )
    keys = [k for k, v in base_state.items() if v.is_floating_point()]
    accumulated = {k: torch.zeros_like(base_state[k]) for k in keys}   # theta*_t - theta_0
    weights: list[tuple[float, float]] = []
    lambdas: list[float] = []

    for step, (tau, fisher) in enumerate(zip(task_vectors_list, fishers)):
        # theta_hat_t - theta*_{t-1} = (theta_0 + tau_t) - (theta_0 + accumulated)
        displacement = {k: tau[k].to(accumulated[k].device) - accumulated[k] for k in keys}
        lam = became_lambda(displacement, fisher, fishers[:step])
        lambdas.append(lam)
        weights.append((1.0 - lam, lam))
        for key in keys:
            accumulated[key] = (1.0 - lam) * accumulated[key] + lam * tau[key].to(
                accumulated[key].device
            )
    return weights, lambdas


def accumulate_precision(
    total: dict[str, Tensor] | None,
    fisher: dict[str, Tensor],
    weight: float = 1.0,
) -> dict[str, Tensor]:
    """``Lambda_t = Lambda_{t-1} + weight * F_t`` — Algorithm 1 line 9, one term at a time.

    Used by the *chain* frame (EXPERIMENTS.md §1.39), not by `became_weights`, which keeps the
    full list because the merging frame needs each shard's Fisher separately for its own
    bookkeeping. Here only the running sum is ever read, so holding n Fishers of one float per
    parameter would be pure waste — at 0.7M parameters and n = 5 that is five copies of the
    model to carry for a quantity that is always consumed as a sum.

    Passing the result to `became_lambda` as a ONE-ELEMENT list is exactly equivalent to passing
    the full list: that function computes ``sum_i F_i[k] * d[k]**2`` over the list, and summing
    the Fishers first factors the same arithmetic. `verify_adaptive_lambda.py` asserts the
    equivalence rather than relying on the reader to believe it.

    ``weight`` implements the `data` Fisher weighting of the brief's §4.4. `diagonal_fisher`
    returns a **mean** over batches, so every Fisher arrives on the same per-sample scale and
    line 9's plain sum treats a period holding 10% of the stream as equal to the base model
    holding 50% of it. Weighting by sample count is arguably closer to the Laplace derivation,
    where a posterior precision after N samples scales with N. The paper can gloss the choice
    because its tasks are equal-sized; ours are not, so it is a flag and an ablation, not a
    default.

    Returns a new dict; neither argument is mutated, because the caller keeps `fisher` alive to
    write its diagnostics and an in-place add here would silently corrupt them.
    """
    if total is None:
        return {k: (v.double() * weight).clone() for k, v in fisher.items()}
    out = {k: v.clone() for k, v in total.items()}
    for key, value in fisher.items():
        contribution = value.double() * weight
        out[key] = out[key] + contribution if key in out else contribution
    return out
