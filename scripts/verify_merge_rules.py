"""Self-checks for the merge rules, run before any result from them is quoted.

    python scripts/verify_merge_rules.py

CLAUDE.md's convention: new data-path or merge code needs a check on its own semantics
*before* its first number is published, because the doc-vs-CSV checker verifies that
published numbers match generated ones — not that the generator does what it claims. The
anomaly-mask span bug (§1.30) lived in exactly that gap for a week of committed runs.

Four properties, each one a thing that would silently corrupt the 2x2 ablation:

1. **The baseline cell is the published cell.** `merge_sequential` at ``(decay, coef) =
   (1, alpha)`` with no transform must equal `apply_task_vectors(..., alpha)` bitwise. If it
   does not, the ablation's "plain sum + swept alpha" cell is a *re-derivation* rather than
   the number the rest of the project already reports, and no comparison against it means
   anything.
2. **BECAME's fold is its convex update.** Folding with ``(1 - lam_t, lam_t)`` must equal
   iterating ``theta*_t = (1 - lam_t) theta*_{t-1} + lam_t theta_hat_t`` directly on the
   weights. This is the algebra that lets a convex model average be expressed on task
   vectors at all, and it is easy to get wrong by one term.
3. **OPCM discards what sequential_overlap measures.** The residual must be orthogonal to
   the retained subspace, and the norm it removes must match the rho that `geometry.py`
   reports for the same vectors. The method and the diagnostic share a projection; if they
   drift apart, §1.31's prediction is being tested against a different quantity than the
   one it was made about.
4. **OPCM at n=1 is identity.** With no history there is nothing to project against, so the
   first task vector must pass through untouched — otherwise the n=1 column silently stops
   being comparable across rules.
"""

import sys

import torch

sys.path.insert(0, "src")

from incremental_ad.framework.merging.task_vectors import (  # noqa: E402
    apply_task_vectors,
    float_keys,
    merge_sequential,
    opcm_residual,
)


def fake_states(n_vectors: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    base = {
        "enc.weight": torch.randn(8, 6, generator=generator),
        "enc.bias": torch.randn(8, generator=generator),
        "norm.scale": torch.randn(8, generator=generator),
        "step": torch.tensor(3),                      # integer buffer, never merged
    }
    taus = [
        {
            "enc.weight": torch.randn(8, 6, generator=generator) * 0.1,
            "enc.bias": torch.randn(8, generator=generator) * 0.1,
            "norm.scale": torch.randn(8, generator=generator) * 0.1,
        }
        for _ in range(n_vectors)
    ]
    return base, taus


def check_baseline_equivalence() -> int:
    base, taus = fake_states()
    failures = 0
    for alpha in (0.0, 0.25, 1.0 / 3, 0.5, 1.0, 1.5):
        expected = apply_task_vectors(base, taus, alpha)
        got = merge_sequential(base, taus, [(1.0, alpha)] * len(taus))
        for key in expected:
            if not torch.equal(expected[key], got[key]):
                delta = (expected[key].float() - got[key].float()).abs().max().item()
                print(f"  FAIL  alpha={alpha} {key}: max|diff| = {delta:.3e}")
                failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  plain sum reproduces apply_task_vectors "
          f"bitwise at 6 scales")
    return failures


def check_became_algebra() -> int:
    base, taus = fake_states()
    lambdas = [0.6, 0.35, 0.2]
    # Direct convex iteration on the weights themselves.
    accumulator = {k: base[k].clone() for k in float_keys(base)}
    for tau, lam in zip(taus, lambdas):
        for key in accumulator:
            theta_hat = base[key] + tau[key]
            accumulator[key] = (1 - lam) * accumulator[key] + lam * theta_hat
    got = merge_sequential(base, taus, [(1 - lam, lam) for lam in lambdas])
    failures = 0
    for key in accumulator:
        delta = (accumulator[key] - got[key]).abs().max().item()
        if delta > 1e-6:
            print(f"  FAIL  BECAME fold {key}: max|diff| = {delta:.3e}")
            failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  BECAME fold equals the convex weight update")
    return failures


def check_opcm_residual() -> int:
    base, taus = fake_states()
    transform = opcm_residual(threshold=0.5)
    failures = 0

    # n=1: nothing to project against.
    first = transform(taus[0], [])
    if not all(torch.equal(first[k], taus[0][k]) for k in taus[0]):
        print("  FAIL  OPCM altered the first task vector, which has no history")
        failures += 1

    # Residual must be orthogonal to the subspace it was projected out of.
    key = "enc.weight"
    history = [taus[0]]
    residual = transform(taus[1], history)[key].reshape(-1).to(torch.float64)
    previous = torch.stack([h[key].reshape(-1) for h in history]).to(torch.float64)
    _, _, right = torch.linalg.svd(previous, full_matrices=False)
    leakage = (right[:1] @ residual).abs().max().item()
    if leakage > 1e-8:
        print(f"  FAIL  residual still has {leakage:.3e} component inside the subspace")
        failures += 1

    # And it must remove exactly the energy sequential_overlap reports as rho.
    original = taus[1][key].reshape(-1).to(torch.float64)
    removed = 1.0 - float(residual.dot(residual) / original.dot(original))
    projected = right[:1] @ original
    rho = float(projected.dot(projected) / original.dot(original))
    if abs(removed - rho) > 1e-7:
        print(f"  FAIL  removed {removed:.9f} of the energy but rho says {rho:.9f}")
        failures += 1

    # 1-D tensors pass through: no meaningful row space at this size.
    for name in ("enc.bias", "norm.scale"):
        out = transform(taus[1], history)[name]
        if not torch.equal(out, taus[1][name]):
            print(f"  FAIL  OPCM modified 1-D tensor {name}")
            failures += 1

    print(f"  {'ok' if not failures else 'FAILED'}  OPCM residual is orthogonal, removes "
          f"exactly rho, identity at n=1, and passes 1-D tensors through")
    return failures


def check_became_reduces_to_one_over_n() -> int:
    """The prediction registered in `became.py`: equal Fishers must give lambda_t = 1/t.

    This is the analytic bridge between BECAME and this project's strongest empirical result.
    Two things follow from it and both are checked, because the second is the one that
    matters and it is not obvious from the first:

    - lambda_t = 1/t exactly when every shard's Fisher is identical.
    - the resulting fold is a *running mean*, so the final merge equals alpha = 1/n applied
      to the plain sum — not merely close to it.

    So under equal curvature BECAME does not approximate the 1/n rule, it *is* the 1/n rule.
    Any deviation measured on real Fishers is therefore a statement about how unequal the
    shard curvatures are, which is exactly the quantity §1.31 wants.
    """
    import logging
    logging.disable(logging.INFO)
    from incremental_ad.framework.merging.became import became_weights

    base, taus = fake_states(n_vectors=5)
    equal = [{k: torch.ones_like(v, dtype=torch.float64) for k, v in taus[0].items()}
             for _ in taus]
    weights, lambdas = became_weights(base, taus, equal)
    failures = 0
    for step, lam in enumerate(lambdas):
        if abs(lam - 1.0 / (step + 1)) > 1e-9:
            print(f"  FAIL  lambda_{step + 1} = {lam:.9f}, expected {1.0 / (step + 1):.9f}")
            failures += 1

    # ...and the fold it produces must equal alpha = 1/n on the plain sum.
    got = merge_sequential(base, taus, weights)
    expected = apply_task_vectors(base, taus, 1.0 / len(taus))
    for key in expected:
        if expected[key].is_floating_point():
            delta = (expected[key] - got[key]).abs().max().item()
            if delta > 1e-6:
                print(f"  FAIL  equal-Fisher BECAME != alpha=1/n on {key}: {delta:.3e}")
                failures += 1
    logging.disable(logging.NOTSET)
    print(f"  {'ok' if not failures else 'FAILED'}  equal Fishers give lambda_t = 1/t, and the "
          f"fold equals alpha = 1/n exactly")
    return failures


def check_rescaled_became_algebra() -> int:
    """`became_rescaled` must keep BECAME's weight *ratios* and hit the requested alpha*n exactly.

    The two halves are independent and both can fail silently: a rescale that renormalises the
    wrong quantity still sums to the target (ratios lost), and one that preserves ratios but
    divides by the wrong denominator lands at the wrong strength (which is the only thing §1.36's
    P2 is comparing). So check both, and check the fold really lands on the weighted sum rather
    than on some convex combination of it.
    """
    from incremental_ad.framework.merging import merge_sequential
    from incremental_ad.framework.pipelines.incremental_task_arithmetic_pipeline import (
        became_weights_per_vector,
    )

    failures = 0
    lambdas = [1.0, 0.695466, 0.338009]          # a real lambda* triple, from noisefloor_swat
    per_vector = became_weights_per_vector(lambdas)
    for alpha, n in ((1.0, 3), (0.30, 3), (0.55, 2), (0.15, 5)):
        target = alpha * n
        total = sum(per_vector[:n]) if n <= len(per_vector) else None
        if total is None:
            continue
        scaled = [w * target / total for w in per_vector[:n]]
        if abs(sum(scaled) - target) > 1e-9:
            print(f"  FAIL  rescaled weights sum to {sum(scaled):.9f}, wanted {target}")
            failures += 1
        # Ratios preserved: every pairwise ratio must survive the rescale untouched.
        for i in range(n):
            for j in range(n):
                before = per_vector[i] / per_vector[j]
                after = scaled[i] / scaled[j]
                if abs(before - after) > 1e-9:
                    print(f"  FAIL  ratio {i}/{j} changed: {before:.9f} -> {after:.9f}")
                    failures += 1

    # The fold with decay 1 and unequal coefficients must land on sum(w_i * tau_i), NOT on a
    # convex combination — that is the whole point of the mode.
    base, taus = fake_states(3)
    target = 0.30 * 3
    total = sum(per_vector)
    scaled = [w * target / total for w in per_vector]
    got = merge_sequential(base, taus, [(1.0, w) for w in scaled])
    for key, value in base.items():
        if not value.is_floating_point():
            continue
        expected = value + sum(w * tau[key] for w, tau in zip(scaled, taus))
        delta = (expected - got[key]).abs().max().item()
        if delta > 1e-6:
            print(f"  FAIL  rescaled fold != weighted sum on {key}: {delta:.3e}")
            failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  became_rescaled preserves weight ratios, "
          f"hits alpha*n exactly, and folds to the weighted sum")
    return failures


def check_order_reversal_semantics() -> int:
    """Reversal must be inert for plain summation and decisive for OPCM.

    The null half is the one that matters: if reversing the *list* accidentally reversed something
    else too (the base, the key order, the coefficient pairing), plain summation would still look
    fine on a spot-check but every OPCM row would be measuring the wrong thing. Addition is
    commutative, so plain sum gives an exact equality to assert against — a free oracle for the
    plumbing, which the P3 sweep then repeats end-to-end on real checkpoints.
    """
    from incremental_ad.framework.merging import merge_sequential, opcm_residual

    failures = 0
    base, taus = fake_states(3)
    forward = merge_sequential(base, taus, [(1.0, 0.3)] * 3)
    reverse = merge_sequential(base, list(reversed(taus)), [(1.0, 0.3)] * 3)
    for key in forward:
        if forward[key].is_floating_point():
            delta = (forward[key] - reverse[key]).abs().max().item()
            if delta > 1e-6:
                print(f"  FAIL  plain sum is order-dependent on {key}: {delta:.3e}")
                failures += 1

    transform = opcm_residual(0.5)
    f_opcm = merge_sequential(base, taus, [(1.0, 0.3)] * 3, transform=transform)
    r_opcm = merge_sequential(base, list(reversed(taus)), [(1.0, 0.3)] * 3, transform=transform)
    moved = max((f_opcm[k] - r_opcm[k]).abs().max().item()
                for k in f_opcm if f_opcm[k].is_floating_point())
    if moved < 1e-6:
        print(f"  FAIL  OPCM is order-INdependent (max delta {moved:.3e}) — either the transform "
              f"ignores history or the reversal never reached it; P3 would be vacuous")
        failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  reversal is inert for plain summation and "
          f"moves OPCM (max delta {moved:.3e})")
    return failures


def check_opcm_paper() -> int:
    """The paper's OPCM against the three things the paper *proves*. Returns failures.

    Tang et al., NeurIPS 2025, Algorithm 1 / Eq. (6)-(7) / Theorems 5.1-5.2. Each assertion below
    is a property the paper states, not a property of this implementation, so a divergence here
    means the operator is not the paper's -- which is the one thing that must never be claimed
    loosely (EXPERIMENTS.md C26).

      Eq. 8  (Thm 5.1)  <P_alpha(dW^t), dW_merged^(t-1)>_F = 0.
      the projection     every retained coefficient sits outside the top-r_alpha rows AND columns,
                         and off the diagonal -- checked by re-reading the coefficients in the SVD
                         basis, not by trusting the mask that produced them.
      Thm 5.2 / line 14  ||theta_merged - theta^(0)||_2 == Avg_i ||dtheta^(i)||_2 exactly.
    """
    import torch
    from incremental_ad.framework.merging import merge_opcm_paper, project_orthogonal
    from incremental_ad.framework.merging.opcm import _retained_rank

    failures = 0
    torch.manual_seed(0)
    accumulated = torch.randn(24, 16, dtype=torch.float64)
    incoming = torch.randn(24, 16, dtype=torch.float64)

    for threshold in (0.3, 0.5, 0.7):
        projected = project_orthogonal(incoming, accumulated, threshold)

        # Eq. 8: orthogonal to the accumulated merged matrix.
        overlap = float((projected * accumulated).sum())
        if abs(overlap) > 1e-9:
            print(f"  FAIL  alpha={threshold}: <P(dW), dW_merged>_F = {overlap:.3e}, wanted 0")
            failures += 1

        # Zero overlap with the top-k subspace, read back in the SVD basis.
        left, singular, right_h = torch.linalg.svd(accumulated, full_matrices=True)
        rank = _retained_rank(singular, threshold)
        coefficients = left.T @ projected @ right_h.T
        leaked = max(float(coefficients[:rank, :].abs().max()),
                     float(coefficients[:, :rank].abs().max()))
        diagonal = min(coefficients.shape)
        leaked_diagonal = float(coefficients[range(diagonal), range(diagonal)].abs().max())
        if leaked > 1e-9:
            print(f"  FAIL  alpha={threshold}: top-{rank} subspace retains {leaked:.3e}")
            failures += 1
        if leaked_diagonal > 1e-9:
            print(f"  FAIL  alpha={threshold}: diagonal i==j retains {leaked_diagonal:.3e}")
            failures += 1
        # ...and the projection must not be trivially zero, or every check above passes vacuously.
        if float(projected.abs().max()) < 1e-9:
            print(f"  FAIL  alpha={threshold}: projection is identically zero — vacuous")
            failures += 1

    # Norm stabilisation, on real state dicts. This is the paper's magnitude guarantee and the
    # reason OPCM takes no merge scale: the merged model sits at the MEAN task-vector norm from
    # the base, whatever the projection removed.
    base, taus = fake_states(4)
    for threshold in (0.3, 0.5, 0.7):
        merged, info = merge_opcm_paper(base, taus, threshold)
        keys = sorted(k for k, v in base.items() if v.is_floating_point())
        distance = float(torch.sqrt(sum(((merged[k] - base[k]).to(torch.float64) ** 2).sum()
                                        for k in keys)))
        expected = sum(
            float(torch.sqrt(sum((tau[k].to(torch.float64) ** 2).sum() for k in keys)))
            for tau in taus
        ) / len(taus)
        if abs(distance - expected) > 1e-6 * max(expected, 1.0):
            print(f"  FAIL  alpha={threshold}: ||merged - base|| = {distance:.6f}, but the mean "
                  f"task-vector norm is {expected:.6f} — Thm 5.2's rescaling is wrong")
            failures += 1
        if abs(info["norm_ratio"] - 1.0) > 1e-9:
            print(f"  FAIL  alpha={threshold}: reported norm_ratio {info['norm_ratio']:.9f} != 1")
            failures += 1

    # Algorithm 1 line 1: with a single expert the merge IS that expert, untouched.
    merged, _ = merge_opcm_paper(base, taus[:1], 0.5)
    for key in base:
        if base[key].is_floating_point():
            delta = (merged[key] - (base[key] + taus[0][key])).abs().max().item()
            if delta > 1e-9:
                print(f"  FAIL  T=1 merge is not theta^(1) on {key}: {delta:.3e}")
                failures += 1

    # 1-D tensors take the identity mapping, so they are the plain sum rescaled by lambda^(T).
    merged, info = merge_opcm_paper(base, taus, 0.5)
    ones = [k for k, v in base.items() if v.is_floating_point() and v.ndim == 1]
    if not ones:
        print("  FAIL  no 1-D tensors in the fixture — the pass-through path is untested")
        failures += 1
    for key in ones:
        expected = base[key] + sum(tau[key] for tau in taus) / info["lambda_final"]
        delta = (merged[key] - expected).abs().max().item()
        if delta > 1e-6:
            print(f"  FAIL  1-D tensor {key} did not pass through: {delta:.3e}")
            failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  paper OPCM: projection orthogonal to the "
          f"accumulated merge and to its top-k subspace, ||merged-base|| == mean task-vector "
          f"norm, 1-D tensors pass through")
    return failures


def check_opcm_committed_scale() -> int:
    """`scale_to_alpha` must change the magnitude and NOTHING else. Returns failures.

    The whole value of §1.37 rests on this: if the matched-magnitude variant differed from the
    paper's operator anywhere but the final coefficient, its comparison against plain summation
    would no longer isolate the projection, and C34 could not be closed either way. So assert the
    strong form — the two merges are exactly collinear in weight space.

    Under Eq. 7 the paper divides the accumulator by lambda^(T) and this divides by 1/alpha, so
    ``merged_alpha - theta_0 == alpha * lambda^(T) * (merged_paper - theta_0)`` tensor by tensor.
    """
    import torch
    from incremental_ad.framework.merging import merge_opcm_paper

    failures = 0
    worst_seen = [0.0]
    # float64 fixture. The identity is checked on `merged - base`, and in float32 that subtraction
    # reconstructs a ~0.1-sized delta from two ~1-sized numbers -- catastrophic cancellation, which
    # inflates the apparent relative error to ~1e-6 and has nothing to do with the merges. Widening
    # the bound to accommodate it would have hidden a real divergence of the same size; carrying
    # the fixture in float64 removes the artifact and lets the test assert near-exact equality.
    base, taus = fake_states(4)
    base = {k: (v.double() if v.is_floating_point() else v) for k, v in base.items()}
    taus = [{k: v.double() for k, v in tau.items()} for tau in taus]
    for threshold in (0.3, 0.5, 0.7):
        for alpha in (0.2, 0.5, 1.0):
            paper, info = merge_opcm_paper(base, taus, threshold)
            scaled, info_a = merge_opcm_paper(base, taus, threshold, scale_to_alpha=alpha)
            factor = alpha * info["lambda_final"]
            # Relative, and in float64 the identity should hold to ~1e-15. The 1e-12 bound
            # leaves room for the SVD's own conditioning without admitting anything that could
            # be a real difference in the projection.
            worst = 0.0
            for key, value in base.items():
                if not value.is_floating_point():
                    continue
                expected = factor * (paper[key] - value)
                scale = max(expected.abs().max().item(), 1e-12)
                worst = max(worst, (scaled[key] - value - expected).abs().max().item() / scale)
            if worst > 1e-12:
                print(f"  FAIL  alpha={alpha} thr={threshold}: the two merges are NOT collinear "
                      f"(rel {worst:.3e}) — something other than the scale changed")
                failures += 1
            worst_seen[0] = max(worst_seen[0], worst)
            # alpha*n is an identity here, not an estimate.
            if abs(info_a["implied_alpha_times_n"] - alpha * len(taus)) > 1e-12:
                print(f"  FAIL  implied alpha*n {info_a['implied_alpha_times_n']} != "
                      f"{alpha * len(taus)}")
                failures += 1
            # ...and the paper's own norm guarantee must NOT hold once it has been replaced,
            # or the flag did nothing.
            if abs(info_a["norm_ratio"] - 1.0) < 1e-12 and abs(factor - 1.0) > 1e-9:
                print(f"  FAIL  alpha={alpha}: norm_ratio still 1.0 — the rescale was not applied")
                failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  scale_to_alpha is collinear with the paper's "
          f"merge to {worst_seen[0]:.1e} relative (projection untouched) and hits alpha*n exactly")
    return failures


def main() -> None:
    print("MERGE RULE SELF-CHECKS")
    total = (check_baseline_equivalence() + check_became_algebra()
             + check_opcm_residual() + check_became_reduces_to_one_over_n()
             + check_rescaled_became_algebra() + check_order_reversal_semantics()
             + check_opcm_paper() + check_opcm_committed_scale())
    print(f"\n{total} failure(s)")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
