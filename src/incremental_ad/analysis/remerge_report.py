"""Collect the OPCM / BECAME re-merge sweep into one table, against each dataset's own floor.

    python -m incremental_ad.analysis.remerge_report --runs_root $RUNS_ROOT \\
        --remerge_dir $OUT/remerge_sweep --floors $OUT/floors.csv \\
        --geometry results_archive/audit/geometry/geometry_by_dataset.csv \\
        --out results_archive/audit/remerge_sweep

Every row compares a re-merge against **the plain-sum merge of the same run at the same
strength** — the number already published in §1.26 — so the only thing that differs is the merge
rule or the coefficient source. Nothing here is tuned: `remerge.py` reads the committed strength
from `merge_scale/selected`, falling back to `config.json`.

**Differences smaller than the dataset-and-metric floor are reported as ties**, in the `verdict`
column, and the raw delta is kept beside it so a reader can see what was set aside rather than
having to trust the label. The floors come from `floors.csv`, the §1.9 definition, not from
anything recomputed here.

Two extra columns exist to make specific claims checkable:

- **`implied_alpha_times_n`** for BECAME. The fold is convex —
  ``accumulated = (1 - lam) * accumulated + lam * tau`` — so the per-period weights sum to
  **exactly 1** and the implied total strength is α·n = 1.0 *by construction*, whatever the
  Fishers say. Emitting it means the structural claim is verified per run rather than asserted;
  if it ever came out ≠ 1.0, the fold is not what its docstring says.
- **`rho`**, the accumulated-subspace overlap per dataset, so OPCM's cost can be tabulated
  against the quantity that mechanically determines how much it discards.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("remerge_report")

FIELDS = ["dataset", "n_segments", "metric", "rule", "coefficient_source", "threshold",
          "n_seeds", "floor_pct", "rho",
          "plain", "plain_sd", "remerged", "remerged_sd", "delta_pct", "verdict",
          "implied_alpha_times_n", "committed_alpha", "source_experiment"]

# rho is read **per merge experiment**, not per dataset label.
#
# A first version aliased "PSM-forecast" -> "PSM" and "SWaT-forecast" -> "SWaT" because they share
# raw data. That is wrong and it mattered: PSM-forecast's rho is 0.034, PSM's is 0.216 — a factor
# of six — because they are different *tasks*, and rho is a property of the task vectors a task
# produces, not of the series they were computed from. The alias would have put the correlation's
# most informative point at the wrong x-coordinate.
#
# Keyed by experiment, so a dataset that shares a name with another cannot inherit its geometry.
GEOMETRY_ALIAS: dict[str, str] = {"exchange": "Exchange"}


def higher_is_better(metric: str) -> bool:
    return any(k in metric.lower()
               for k in ("auroc", "auprc", "f1", "precision", "recall", "accuracy"))


def load_floors(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("role", "floor") == "floor":
                try:
                    out[(row["dataset"], row["metric"])] = float(row["floor_pct"])
                except (TypeError, ValueError):
                    continue
    return out


def load_rho(by_dataset: Path, per_experiment: list[Path]) -> tuple[dict, dict]:
    """(dataset -> rho, experiment -> rho).

    The per-experiment map wins wherever it has an entry, because it is measured on the exact
    task vectors the re-merge used. The per-dataset map is the fallback for datasets whose
    geometry was only ever aggregated.
    """
    datasets: dict = {}
    if by_dataset.is_file():
        with by_dataset.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    datasets[row["dataset"]] = float(row["mean_sequential_overlap"])
                except (TypeError, ValueError, KeyError):
                    continue
    experiments: dict = defaultdict(list)
    for path in per_experiment:
        if not path.is_file():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                value = row.get("mean_sequential_overlap")
                if value not in ("", None):
                    try:
                        experiments[row["experiment_name"]].append(float(value))
                    except (TypeError, ValueError):
                        continue
    return datasets, {k: st.mean(v) for k, v in experiments.items()}


def _rho_for(experiment: str, dataset: str, by_dataset: dict, by_experiment: dict) -> str:
    """rho for this row: the experiment's own measurement first, the dataset aggregate second."""
    if experiment in by_experiment:
        return round(by_experiment[experiment], 4)
    key = GEOMETRY_ALIAS.get(dataset, dataset)
    return round(by_dataset[key], 4) if key in by_dataset else ""


def plain_metric(run: Path, metric: str) -> float | None:
    """The run's own merged/test value — the published plain-sum merge at the committed α."""
    path = run / "merged" / "test" / "result.json"
    if not path.is_file():
        return None
    try:
        return (json.loads(path.read_text()) or {}).get("metrics", {}).get(metric)
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--remerge_dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--floors", type=Path, required=True)
    parser.add_argument("--geometry", type=Path,
                        default=Path("results_archive/audit/geometry/geometry_by_dataset.csv"))
    parser.add_argument("--geometry_summary", type=Path, nargs="*", default=[],
                        help="geometry_summary.csv files keyed by experiment_name; these win "
                             "over the per-dataset aggregate because they are measured on the "
                             "exact task vectors the re-merge used")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    floors = load_floors(args.floors)
    rhos, rhos_by_experiment = load_rho(args.geometry, args.geometry_summary)
    with args.spec.open() as fh:
        spec = list(csv.DictReader(fh))

    rows: list[dict] = []
    for entry in spec:
        dataset, n, metric = entry["dataset"], entry["n"], entry["metric"]
        experiment = entry["merge_experiment"]
        group = args.runs_root / experiment
        if not group.is_dir():
            log.warning("  %s n=%s: %s absent — reported missing, not retrained",
                        dataset, n, experiment)
            continue

        # tag -> per-seed (plain, remerged, alpha_n, committed_alpha)
        collected: dict[str, list[tuple]] = defaultdict(list)
        for run in sorted(p for p in group.iterdir() if p.is_dir()):
            plain = plain_metric(run, metric)
            if plain is None:
                continue
            out_dir = args.remerge_dir / f"{experiment}__{run.name}"
            if not out_dir.is_dir():
                continue
            for tag_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
                result = tag_dir / "result.json"
                if not result.is_file():
                    continue
                try:
                    payload = json.loads(result.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                value = (payload.get("metrics") or {}).get(f"test/{metric}")
                if value is None:
                    continue
                alpha_n = ""
                lambdas = tag_dir / "became_lambdas.csv"
                if lambdas.is_file():
                    with lambdas.open() as fh:
                        weights = [float(r["weight"]) for r in csv.DictReader(fh)]
                    # Convex fold: the weights are the per-period strengths, and their sum is the
                    # total strength a uniform alpha would need — i.e. alpha * n.
                    alpha_n = round(sum(weights), 6)
                collected[tag_dir.name].append(
                    (plain, value, alpha_n, payload.get("alpha"),
                     payload.get("merge_rule"), payload.get("coefficient_source"),
                     payload.get("opcm_threshold"))
                )

        # floors.csv keys exchange_rate by its dataset-class name while the spec says
        # "exchange", so a floors-only lookup silently drops that dataset — and exchange is the
        # one where OPCM *helps*, i.e. exactly the row worth not losing. `method_comparison.py`
        # already falls back to the spec's own floor_pct column; do the same, so the two tables
        # cannot disagree about which floor applies.
        floor = floors.get((dataset, metric))
        if floor is None:
            try:
                floor = float(entry["floor_pct"])
            except (TypeError, ValueError, KeyError):
                floor = None
        for tag, items in sorted(collected.items()):
            plains = [i[0] for i in items]
            merged = [i[1] for i in items]
            alpha_ns = [i[2] for i in items if i[2] != ""]
            rule, source = items[0][4], items[0][5]
            threshold = items[0][6] if rule == "opcm" else ""
            base, new = st.mean(plains), st.mean(merged)
            # Signed so that positive always means "the re-merge is worse".
            delta = (100.0 * (base - new) / base if higher_is_better(metric)
                     else 100.0 * (new - base) / base)
            verdict = ("tie (inside floor)" if floor is not None and abs(delta) < floor
                       else ("worse" if delta > 0 else "better"))
            rows.append({
                "dataset": dataset, "n_segments": n, "metric": metric,
                "rule": rule, "coefficient_source": source, "threshold": threshold,
                "n_seeds": len(items),
                "floor_pct": round(floor, 3) if floor is not None else "",
                "rho": _rho_for(experiment, dataset, rhos, rhos_by_experiment),
                "plain": round(base, 6),
                "plain_sd": round(st.stdev(plains), 6) if len(plains) > 1 else 0.0,
                "remerged": round(new, 6),
                "remerged_sd": round(st.stdev(merged), 6) if len(merged) > 1 else 0.0,
                "delta_pct": round(delta, 3),
                "verdict": verdict,
                "implied_alpha_times_n": round(st.mean(alpha_ns), 6) if alpha_ns else "",
                "committed_alpha": items[0][3],
                "source_experiment": experiment,
            })

    if not rows:
        raise SystemExit("no re-merge outputs found — check --remerge_dir")

    log.info("%-15s %-3s %-9s %-7s %10s %10s %9s  %s", "dataset", "n", "rule", "thr",
             "plain", "remerged", "delta", "verdict")
    for row in rows:
        log.info("%-15s %-3s %-9s %-7s %10.4f %10.4f %+8.2f%%  %s",
                 row["dataset"], row["n_segments"],
                 row["rule"] if row["coefficient_source"] == "scale" else "became",
                 str(row["threshold"]), row["plain"], row["remerged"], row["delta_pct"],
                 row["verdict"])

    ties = sum(1 for r in rows if r["verdict"].startswith("tie"))
    log.info("\n%d rows · %d inside their dataset's floor and reported as ties", len(rows), ties)
    bad_alpha = [r for r in rows if r["implied_alpha_times_n"] != ""
                 and abs(float(r["implied_alpha_times_n"]) - 1.0) > 1e-6]
    if bad_alpha:
        log.warning("⚠️  %d BECAME row(s) have implied alpha*n != 1.0 — the convex-fold claim is "
                    "WRONG and this is the headline finding: %s", len(bad_alpha),
                    [(r["dataset"], r["n_segments"], r["implied_alpha_times_n"])
                     for r in bad_alpha[:5]])
    elif any(r["implied_alpha_times_n"] != "" for r in rows):
        log.info("BECAME rows: implied alpha*n = 1.0 on every one, as the convex fold requires")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "remerge_sweep.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
