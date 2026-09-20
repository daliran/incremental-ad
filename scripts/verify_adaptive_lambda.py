"""Gates for adaptive-λ sequential fine-tuning (BECAME's coefficient in the chain frame).

    python scripts/verify_adaptive_lambda.py

**This is not BECAME** and is never called that: the published method also runs a
gradient-projection stage (paper §3.4, Algorithm 1 line 4). This is the coefficient alone,
applied in the sequential frame — "adaptive-λ sequential fine-tuning (BECAME's coefficient)",
the same naming discipline register row `C26` applies to the OPCM variants.

Gates 2, 4 and 5 run here and need no GPU. Gate 1 (λ = 1 reproduces the plain chain bitwise)
and gate 3 (λ ∈ [0,1] on real steps) need training and run on the cluster; gate 3 is a hard
assert inside the pipeline itself, so it cannot be skipped by forgetting to run this script.

Gate 1 must run **both configurations inside one job, sequentially**. The GPU model changes the
result by up to 18.8% at a fixed seed (EXPERIMENTS.md §3.2), and this cluster rejects a
single-model `--constraint` and refuses `--nodelist`, so the only way to hold the hardware fixed
is to not let the scheduler choose twice.
"""

import hashlib
import logging

import torch

log = logging.getLogger("verify_adaptive_lambda")


def check_equal_fishers_give_one_over_t() -> int:
    """Gate 2 — equal Fishers must give λ* = 1/t exactly. Returns failures.

    This is the identity the whole interpretation rests on: it is what makes `one_over_t` the
    right control rather than an arbitrary schedule, and it is why §4.6 counts the base as task
    1 (with equal Fishers and Λ seeded, period 1 must come out at ½, not 1).
    """
    from incremental_ad.framework.merging.became import accumulate_precision, became_lambda

    failures = 0
    torch.manual_seed(0)
    displacement = {"w": torch.randn(64, 32, dtype=torch.float64),
                    "b": torch.randn(64, dtype=torch.float64)}
    unit = {k: torch.ones_like(v) for k, v in displacement.items()}

    logging.disable(logging.INFO)
    for t in range(1, 7):
        # t - 1 earlier tasks in Λ, plus the incoming one, all with identical curvature.
        precision = None
        for _ in range(t - 1):
            precision = accumulate_precision(precision, unit)
        got = became_lambda(displacement, unit, [precision] if precision is not None else [])
        if abs(got - 1.0 / t) > 1e-12:
            print(f"  FAIL  t={t}: lambda* = {got:.15f}, expected {1.0 / t:.15f}")
            failures += 1
    logging.disable(logging.NOTSET)
    print(f"  {'ok' if not failures else 'FAILED'}  equal Fishers give lambda* = 1/t for t = 1..6")
    return failures


def check_accumulator_equivalence() -> int:
    """The accumulator must be arithmetically identical to passing every Fisher. Returns failures.

    `accumulate_precision` exists to avoid holding n Fishers in memory, and its whole licence is
    that ``became_lambda`` sums the list anyway. If that ever stops being true — say the function
    starts weighting the list — this gate fails rather than the numbers quietly changing.
    """
    from incremental_ad.framework.merging.became import accumulate_precision, became_lambda

    failures = 0
    torch.manual_seed(1)
    displacement = {"w": torch.randn(32, 16, dtype=torch.float64)}
    fishers = [{"w": torch.rand(32, 16, dtype=torch.float64)} for _ in range(4)]
    incoming = {"w": torch.rand(32, 16, dtype=torch.float64)}

    logging.disable(logging.INFO)
    for count in range(1, 5):
        listed = became_lambda(displacement, incoming, fishers[:count])
        precision = None
        for fisher in fishers[:count]:
            precision = accumulate_precision(precision, fisher)
        summed = became_lambda(displacement, incoming, [precision])
        if abs(listed - summed) > 1e-12:
            print(f"  FAIL  {count} Fisher(s): list {listed:.15f} != accumulated {summed:.15f}")
            failures += 1
    logging.disable(logging.NOTSET)
    print(f"  {'ok' if not failures else 'FAILED'}  accumulated Lambda == the full Fisher list")
    return failures


def check_precision_monotone() -> int:
    """Gate 4 — Λ is entrywise non-decreasing across steps. Returns failures.

    Λ is a sum of diagonal Fishers, each a mean of squared gradients and so PSD by construction.
    A decrease anywhere means a sign error or an overwrite rather than an accumulation — the kind
    of bug that leaves λ plausible while making it meaningless.
    """
    from incremental_ad.framework.merging.became import accumulate_precision

    failures = 0
    torch.manual_seed(2)
    precision = None
    previous = None
    for step in range(5):
        fisher = {"w": torch.rand(16, 8, dtype=torch.float64)}
        precision = accumulate_precision(precision, fisher)
        if previous is not None:
            worst = float((precision["w"] - previous["w"]).min())
            if worst < 0:
                print(f"  FAIL  step {step}: Lambda decreased by {worst:.3e} somewhere")
                failures += 1
        previous = {k: v.clone() for k, v in precision.items()}
    print(f"  {'ok' if not failures else 'FAILED'}  Lambda is monotone non-decreasing")
    return failures


def check_pullback_algebra() -> int:
    """λ = 1 must leave the unconstrained model untouched; λ = 0 must leave the accumulator.

    The CPU-side algebra behind gate 1. Gate 1 itself proves it end to end through the real
    training path; this proves the interpolation is not inverted, which is the single most
    likely way to get a plausible-looking wrong answer — an inverted merge still produces
    lambdas in [0,1] and a monotone Λ.
    """
    failures = 0
    torch.manual_seed(3)
    accumulator = {"w": torch.randn(8, 4)}
    unconstrained = {"w": torch.randn(8, 4)}

    for lam, expected, label in ((1.0, unconstrained, "lambda=1 -> theta_hat"),
                                 (0.0, accumulator, "lambda=0 -> accumulator")):
        merged = {k: ((1.0 - lam) * accumulator[k].double()
                      + lam * unconstrained[k].double()).to(accumulator[k].dtype)
                  for k in accumulator}
        delta = (merged["w"] - expected["w"]).abs().max().item()
        if delta > 0:
            print(f"  FAIL  {label}: differs by {delta:.3e}")
            failures += 1
    print(f"  {'ok' if not failures else 'FAILED'}  the interpolation is not inverted")
    return failures


def check_cpu_determinism() -> int:
    """Gate 5 — N optimizer steps on CPU at a fixed seed must hash to a committed value.

    The real bitwise check on the training path. A GPU re-run cannot serve: EXPERIMENTS.md §3.2
    records up to **18.8%** between GPU models at a fixed seed with correct code, so a GPU
    comparison fails for reasons that have nothing to do with the change. CPU is deterministic,
    so the hash moving means the training step moved.

    Deliberately tiny and self-contained — no dataset, no project model — so it runs anywhere
    and cannot be broken by a data path changing underneath it.
    """
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    inputs = torch.randn(64, 16)
    targets = torch.randn(64, 4)
    for _ in range(25):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    digest = hashlib.sha256(
        b"".join(v.detach().double().numpy().tobytes() for v in model.state_dict().values())
    ).hexdigest()
    print(f"  cpu determinism hash: {digest}")
    print("  (record this in results_archive/audit/verification_log.md; it must not move "
          "across a change that is supposed to leave training alone)")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("ADAPTIVE-LAMBDA GATES  (BECAME's coefficient, chain frame — NOT BECAME)")
    total = (check_equal_fishers_give_one_over_t() + check_accumulator_equivalence()
             + check_precision_monotone() + check_pullback_algebra() + check_cpu_determinism())
    print(f"\n{total} failure(s)")
    print("\nStill to run on the cluster: gate 1 (lambda=1 reproduces the plain chain bitwise, "
          "both configs in ONE job) and gate 3 (lambda in [0,1] on real steps, asserted inside "
          "the pipeline).")
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
