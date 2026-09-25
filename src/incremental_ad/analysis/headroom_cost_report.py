"""Does OPCM's cost track how much there was to lose? `C31`'s measurement.

    python -m incremental_ad.analysis.headroom_cost_report --audit_dir results_archive/audit \\
        --out $OUT/headroom_cost

§1.35's `C20` predicted OPCM's cost would order by the accumulated-subspace overlap rho, and the
sweep **refuted** it. `C31` is the surviving alternative: the cost tracks **base-to-joint
headroom** — how much a joint model improves on the base at all — rather than how much the
projection removes. This joins the two quantities that were already in the archive and measures
it. **No new runs**, which is what CLAUDE.md's freeze requires of anything touching this chapter.

**The join.** `derived.csv:headroom_pct` against `remerge_closeout.csv` `P1_paper_opcm` rows, on
(dataset, n_segments). Headroom is averaged over **every** experiment in `derived.csv` that
carries it for that cell, not taken from the experiment of record. That choice is not free and is
not hidden: `--of_record` recomputes from `analysis_specs/experiment_of_record.csv` instead, and
both go into the output so the sensitivity is on the record rather than in a commit message.

**Two exclusions, both principled:**

- **SWaT-forecast** — its headroom is *negative* at n = 3 and n = 5 (−99.87%, −93.64%): the joint
  model is worse than the base. "How much there was to lose" is not defined for a cell where
  joint training loses, so the cell cannot speak to the claim either way.
- **PSM and SWaT (AD, `window_auroc`)** — a different metric, and their headroom is 1-3%. At that
  headroom *every* merge scores nearly alike, so |delta| is small **by construction**. Including
  them would add points that support the correlation for a reason that is arithmetic rather than
  mechanistic, which is the most flattering kind of wrong.

⚠️ **15 cells are not 15 independent points.** Headroom is close to a dataset-level property, so
the 15 (dataset, n) cells carry **5** distinct x values. The primary inference here is therefore
the **5-dataset collapse** with a permutation test that shuffles headroom across datasets — the
cluster the dependence actually lives at. The per-cell correlation is reported beside it because
it is the number the claim was first stated in, not because it is the stronger evidence.
"""

import argparse
import csv
import itertools
import logging
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("headroom_cost_report")

# `derived.csv` keys datasets by class name; every other audit CSV uses the published label.
LABELS = {"EtthForecastDataset": "ETTh1", "Etth2ForecastDataset": "ETTh2",
          "Ettm2ForecastDataset": "ETTm2", "ExchangeRateForecastDataset": "exchange",
          "PsmForecastDataset": "PSM-forecast", "SwatForecastDataset": "SWaT-forecast",
          "Psm": "PSM", "Swat": "SWaT"}
EXCLUDED = {"SWaT-forecast": "headroom is negative at n=3,5 — the joint loses to the base",
            "PSM": "AD metric; headroom 1-3% makes every delta small by construction",
            "SWaT": "AD metric; headroom 1-3% makes every delta small by construction"}

CELL_FIELDS = ["dataset", "n_segments", "threshold", "headroom_pct", "n_headroom_experiments",
               "delta_pct", "verdict"]
FIT_FIELDS = ["threshold", "scope", "n_points", "pearson", "spearman", "headroom_source",
              "permutation_p", "permutations"]


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.fmean(xs), st.fmean(ys)
    numerator = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return numerator / denominator if denominator else float("nan")


def ranks(values: list[float]) -> list[float]:
    """Midranks, so ties do not silently become an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def permutation_p(xs: list[float], ys: list[float]) -> tuple[float, int]:
    """Exact two-sided p over **every** assignment of the x values to the y values.

    With 5 datasets there are 120 permutations, so the test is exact and enumerable rather than
    sampled — and the smallest attainable p is 2/120 = 0.017. A result near that value is real
    and is also near the resolution limit of five datasets, which is worth stating whenever it
    is quoted.
    """
    observed = abs(pearson(xs, ys))
    total = extreme = 0
    for permuted in itertools.permutations(xs):
        total += 1
        if abs(pearson(list(permuted), ys)) >= observed - 1e-12:
            extreme += 1
    return extreme / total, total


NOT_A_MEASUREMENT = ("gate_", "fullfisher_")


def load_headroom(audit: Path, of_record: Path | None) -> dict[tuple[str, str], list[float]]:
    """(dataset label, n) -> every headroom value the archive holds for that cell."""
    rows = list(csv.DictReader((audit / "derived.csv").open(encoding="utf-8")))
    wanted = None
    if of_record is not None:
        wanted = {r["experiment"] for r in csv.DictReader(of_record.open(encoding="utf-8"))
                  if r["role"] == "merge"}
    out = defaultdict(list)
    for row in rows:
        if not row.get("headroom_pct") or not row.get("n_segments"):
            continue
        if wanted is not None and row["experiment"] not in wanted:
            continue
        # `gate_` runs are fixtures, and `fullfisher_` is a robustness re-run of the `fisherfix_`
        # chains on the SAME bases (§1.39). "Every experiment" means every distinct measurement;
        # counting those rows would re-weight exchange_rate n=3 toward three bases already in.
        if row["experiment"].startswith(NOT_A_MEASUREMENT):
            continue
        out[(LABELS.get(row["dataset"], row["dataset"]), row["n_segments"])].append(
            float(row["headroom_pct"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit_dir", type=Path)
    parser.add_argument("--closeout", type=Path,
                        help="the P1 source, ABSOLUTE. Defaults to "
                             "<audit_dir>/remerge_closeout/remerge_closeout.csv, but the "
                             "regeneration script must pass it explicitly: that tree is GPU-"
                             "produced and is carried into the output dir only at the END of "
                             "the run, so reading it from there would silently find nothing.")
    parser.add_argument("--of_record", type=Path,
                        default=Path("analysis_specs/experiment_of_record.csv"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if args.audit_dir is None:
        parser.error("--audit_dir is required (or pass --self-test)")

    pooled = load_headroom(args.audit_dir, None)
    of_record = load_headroom(args.audit_dir, args.of_record)
    closeout_path = args.closeout or (args.audit_dir / "remerge_closeout"
                                      / "remerge_closeout.csv")
    closeout = [r for r in csv.DictReader(closeout_path.open(encoding="utf-8"))
                if r["test"] == "P1_paper_opcm" and r["threshold"]]

    cells, fits = [], []
    for threshold in sorted({r["threshold"] for r in closeout}):
        rows = [r for r in closeout
                if r["threshold"] == threshold and r["dataset"] not in EXCLUDED]
        per_cell = []
        for row in rows:
            key = (row["dataset"], row["n_segments"])
            if key not in pooled:
                continue
            per_cell.append((st.fmean(pooled[key]), float(row["delta_pct"]), row["dataset"]))
            cells.append({"dataset": row["dataset"], "n_segments": row["n_segments"],
                          "threshold": threshold, "headroom_pct": st.fmean(pooled[key]),
                          "n_headroom_experiments": len(pooled[key]),
                          "delta_pct": float(row["delta_pct"]), "verdict": row["verdict"]})
        if len(per_cell) < 3:
            continue
        xs = [c[0] for c in per_cell]
        ys = [c[1] for c in per_cell]
        fits.append({"threshold": threshold, "scope": "per_cell", "n_points": len(xs),
                     "pearson": pearson(xs, ys), "spearman": spearman(xs, ys),
                     "headroom_source": "mean over every experiment", "permutation_p": "",
                     "permutations": ""})

        # The same thing under the other join rule, so the sensitivity is a row, not a footnote.
        alt = [(st.fmean(of_record[(r["dataset"], r["n_segments"])]), float(r["delta_pct"]))
               for r in rows if (r["dataset"], r["n_segments"]) in of_record]
        if len(alt) >= 3:
            fits.append({"threshold": threshold, "scope": "per_cell", "n_points": len(alt),
                         "pearson": pearson([a[0] for a in alt], [a[1] for a in alt]),
                         "spearman": spearman([a[0] for a in alt], [a[1] for a in alt]),
                         "headroom_source": "experiment of record", "permutation_p": "",
                         "permutations": ""})

        # Primary: collapse to datasets, which is where the dependence actually lives.
        grouped = defaultdict(list)
        for headroom, delta, dataset in per_cell:
            grouped[dataset].append((headroom, delta))
        names = sorted(grouped)
        dx = [st.fmean([h for h, _ in grouped[d]]) for d in names]
        dy = [st.fmean([v for _, v in grouped[d]]) for d in names]
        p_value, permutations = permutation_p(dx, dy)
        fits.append({"threshold": threshold, "scope": "per_dataset", "n_points": len(dx),
                     "pearson": pearson(dx, dy), "spearman": spearman(dx, dy),
                     "headroom_source": "mean over every experiment",
                     "permutation_p": p_value, "permutations": permutations})
        log.info("thr %s: per-cell r=%+.3f rho=%+.3f (n=%d)  |  per-dataset r=%+.3f rho=%+.3f "
                 "(n=%d, exact permutation p=%.3f over %d)",
                 threshold, fits[-3]["pearson"], fits[-3]["spearman"], len(xs),
                 fits[-1]["pearson"], fits[-1]["spearman"], len(dx), p_value, permutations)
        if threshold == min({r["threshold"] for r in closeout}):
            for i, name in enumerate(names):
                held_x = [v for j, v in enumerate(dx) if j != i]
                held_y = [v for j, v in enumerate(dy) if j != i]
                log.info("    without %-14s r = %+.3f", name, pearson(held_x, held_y))
    for dataset, why in sorted(EXCLUDED.items()):
        log.info("[excluded] %s — %s", dataset, why)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("headroom_cost_cells.csv", CELL_FIELDS, cells),
                                   ("headroom_cost_fit.csv", FIT_FIELDS, fits)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            log.info("wrote %s (%d row(s))", args.out / name, len(rows))


def _self_test() -> None:
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert abs(pearson([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-12
    # Midranks, or a tie silently becomes an ordering.
    assert ranks([5.0, 5.0, 1.0]) == [2.5, 2.5, 1.0]
    assert abs(spearman([1, 2, 3, 4], [1, 4, 9, 16]) - 1.0) < 1e-12, "monotone, not linear"
    assert abs(spearman([1, 2, 3, 4], [1, 4, 9, 16])) > abs(pearson([1, 2, 3, 4], [1, 4, 9, 16]))

    # The permutation test must be able to come back non-significant, and must bottom out at
    # 2/n! rather than 0 — a p of 0 from an exact enumeration would be a bug.
    p_strong, total = permutation_p([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert total == 120 and abs(p_strong - 2 / 120) < 1e-12, (p_strong, total)
    p_weak, _ = permutation_p([1, 2, 3, 4, 5], [3, 1, 4, 1, 5])
    assert p_weak > 0.1, p_weak
    log.info("self-test OK")


if __name__ == "__main__":
    main()
