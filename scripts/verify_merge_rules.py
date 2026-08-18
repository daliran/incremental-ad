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


def main() -> None:
    print("MERGE RULE SELF-CHECKS")
    total = (check_baseline_equivalence() + check_became_algebra()
             + check_opcm_residual() + check_became_reduces_to_one_over_n())
    print(f"\n{total} failure(s)")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
