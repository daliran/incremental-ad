"""Gates for `framework/merging/interference.py` — run before any result from it is quoted.

    python scripts/verify_merge_baselines.py

The strongest check available is not a property test but **agreement with the supervisor's
reference code**, which is imported read-only from `other/` and never modified. Each port must
reproduce the reference exactly wherever nothing was changed, and differ from it by exactly the
documented defect where something was — so a fix cannot quietly change anything else:

- TA and TSV: identical to the reference.
- TIES: identical on every non-matrix tensor, and **exactly n times** the reference on every
  matrix (defect 1 was a stray 1/n and nothing else).
- Iso-C: identical on every non-matrix tensor, and **exactly n times** the reference on every
  matrix (defect 2 was flattening the mean instead of the sum, and SVD commutes with scaling).
- DARE: the reference draws its mask from the unseeded global RNG, so it cannot be matched
  sample for sample; it is matched on its support (every kept entry is exactly v / (1 - p)) and
  on its unbiasedness, and the seeded port is checked for determinism.

Then properties of the published methods, independent of anybody's code.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "other"))         # the reference, imported read-only

from incremental_ad.framework.merging.interference import (  # noqa: E402
    DARE_DROP_RATE, delta_dare, delta_iso_c, delta_task_arithmetic, delta_ties, delta_tsv,
)

FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synthetic(n: int, seed: int) -> list[dict]:
    """Shapes chosen to exercise every path: square and tall matrices, a bias, a 3-D tensor."""
    g = torch.Generator().manual_seed(seed)
    shapes = {"enc.w": (12, 8), "enc.b": (8,), "head.w": (6, 12), "pos": (1, 5, 8)}
    return [{k: torch.randn(*s, generator=g) for k, s in shapes.items()} for _ in range(n)]


def reference(cls, taus: list[dict]) -> dict:
    merger = cls(torch.device("cpu"), alpha=1.0)
    for tau in taus:
        merger.add({k: v.clone() for k, v in tau.items()})
    return {k: v.clone() for k, v in merger.merge().items()}


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


def main() -> None:
    import merging as ref                       # other/merging.py

    print("AGREEMENT WITH THE SUPERVISOR'S REFERENCE (other/, read-only)")
    for n in (2, 3, 5):
        taus = synthetic(n, seed=n)
        mats = [k for k, v in taus[0].items() if v.dim() == 2]
        rest = [k for k, v in taus[0].items() if v.dim() != 2]

        mine, theirs = delta_task_arithmetic(taus), reference(ref.TaskArithmetic, taus)
        gate(f"n={n} TA identical", all(torch.allclose(mine[k], theirs[k], atol=1e-6)
                                        for k in mine))

        mine, theirs = delta_tsv(taus), reference(ref.TSV, taus)
        worst = max(max_rel(mine[k], theirs[k]) for k in mine)
        gate(f"n={n} TSV identical", worst < 1e-6, f"max rel diff {worst:.1e}")

        mine, theirs = delta_ties(taus), reference(ref.TIES, taus)
        gate(f"n={n} TIES non-matrix identical",
             all(torch.allclose(mine[k], theirs[k], atol=1e-6) for k in rest))
        worst = max(max_rel(mine[k], n * theirs[k]) for k in mats)
        gate(f"n={n} TIES matrices = n x reference (defect 1 and nothing else)",
             worst < 1e-6, f"max rel diff {worst:.1e}")

        mine, theirs = delta_iso_c(taus), reference(ref.ISO, taus)
        gate(f"n={n} Iso-C non-matrix identical",
             all(torch.allclose(mine[k], theirs[k], atol=1e-6) for k in rest))
        worst = max(max_rel(mine[k], n * theirs[k]) for k in mats)
        gate(f"n={n} Iso-C matrices = n x reference (defect 2 and nothing else)",
             worst < 1e-5, f"max rel diff {worst:.1e}")

        # DARE: the reference's mask is unseeded, so match its SUPPORT, not its samples.
        theirs = reference(ref.DARE, taus[:1])
        for k in mats:
            v = taus[0][k]
            kept = theirs[k] != 0
            ok = torch.allclose(theirs[k][kept], (v / (1 - DARE_DROP_RATE))[kept], atol=1e-5)
            gate(f"n={n} DARE reference support on {k} is {{0, v/(1-p)}}", ok)
        mine = delta_dare(taus[:1], seed=0)
        for k in mats:
            v = taus[0][k]
            kept = mine[k] != 0
            gate(f"n={n} DARE port support on {k} is {{0, v/(1-p)}}",
                 torch.allclose(mine[k][kept], (v / (1 - DARE_DROP_RATE))[kept], atol=1e-5))
        gate(f"n={n} DARE non-matrix identical to reference (summed unmasked)",
             all(torch.allclose(mine[k], theirs[k], atol=1e-6) for k in rest))

    print("\nDEFECTS REPRODUCED ON THE REFERENCE (so each fix answers a real failure)")
    taus = [synthetic(1, seed=7)[0]] * 3
    theirs = reference(ref.TIES, taus)
    kept = theirs["enc.w"] != 0
    ratio = float((theirs["enc.w"][kept] / taus[0]["enc.w"][kept]).mean())
    gate("reference TIES returns identical tasks' surviving matrix entries at 1/n",
         abs(ratio - 1 / 3) < 1e-6, f"ratio {ratio:.4f}")
    theirs = reference(ref.ISO, taus)
    ratio = float(torch.linalg.svdvals(3 * taus[0]["enc.w"]).mean()
                  / torch.linalg.svdvals(theirs["enc.w"]).mean())
    gate("reference Iso flattens 1/n of the official spectrum", abs(ratio - 3) < 1e-4,
         f"official/reference {ratio:.4f}")
    small = [{"w": torch.randn(4, 10), "b": torch.randn(3)} for _ in range(5)]
    theirs = reference(ref.TSV, small)
    gate("reference TSV zeroes a rank-4 matrix at n=5", float(theirs["w"].abs().sum()) == 0.0)
    try:
        delta_tsv(small)
        gate("port TSV raises on rank < n instead of zeroing", False)
    except ValueError:
        gate("port TSV raises on rank < n instead of zeroing", True)
    a, b = reference(ref.DARE, taus[:1]), reference(ref.DARE, taus[:1])
    gate("reference DARE is nondeterministic", not torch.equal(a["enc.w"], b["enc.w"]))

    print("\nPROPERTIES OF THE PUBLISHED METHODS")
    taus = synthetic(3, seed=11)
    gate("DARE seed-deterministic", torch.equal(delta_dare(taus, seed=4)["enc.w"],
                                                delta_dare(taus, seed=4)["enc.w"]))
    gate("DARE differs across seeds", not torch.equal(delta_dare(taus, seed=4)["enc.w"],
                                                      delta_dare(taus, seed=5)["enc.w"]))
    gate("DARE with p=0 is task arithmetic",
         all(torch.allclose(delta_dare(taus, drop_rate=0.0)[k],
                            delta_task_arithmetic(taus)[k]) for k in taus[0]))
    draws = torch.stack([delta_dare(taus, seed=s)["enc.w"] for s in range(4000)]).mean(dim=0)
    err = max_rel(draws, delta_task_arithmetic(taus)["enc.w"])
    gate("DARE is unbiased: E[DARE] = task arithmetic", err < 0.05,
         f"max rel error over 4000 masks {err:.3f}")

    iso = delta_iso_c(taus)
    for k in ("enc.w", "head.w"):
        s = torch.linalg.svdvals(iso[k].to(torch.float64))
        gate(f"Iso-C {k} spectrum is flat", float(s.max() - s.min()) < 1e-5 * float(s.max()),
             f"spread {float(s.max() - s.min()):.1e}")
        summed = delta_task_arithmetic(taus)[k].to(torch.float64)
        gate(f"Iso-C {k} keeps the sum's mean singular value",
             abs(float(s.mean() - torch.linalg.svdvals(summed).mean())) < 1e-5)

    one = synthetic(1, seed=13)
    gate("TSV with one task reconstructs it",
         all(torch.allclose(delta_tsv(one)[k], one[0][k], atol=1e-5) for k in one[0]))
    same = [one[0]] * 3
    gate("TIES with density 1 on identical tasks returns the task",
         all(torch.allclose(delta_ties(same, density=1.0)[k], one[0][k], atol=1e-6)
             for k in one[0]))
    trimmed = delta_ties(same)["enc.w"]
    frac = float((trimmed != 0).float().mean())
    gate("TIES keeps ~20% of each matrix", 0.19 <= frac <= 0.25, f"kept {frac:.3f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} gate(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("all gates pass")


if __name__ == "__main__":
    main()
