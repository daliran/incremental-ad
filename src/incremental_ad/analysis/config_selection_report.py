"""How the shared configurations of record sit inside the grid searches that produced them.

    python -m incremental_ad.analysis.config_selection_report \\
        --grid_dir results_archive/audit/grid_search --floors results_archive/audit/floors.csv \\
        --out $OUT/config_selection

Two questions an examiner asks of any comparison, answered from the archived grid-search CSVs
(no new runs):

1. **Was the configuration picked on the test set?** `slurm_grid_search/report.py` can only rank
   trials on test columns, and the grid record was removed from EXPERIMENTS.md in August 2026, so
   the selection criterion cannot be read off the repo. What CAN be measured is where each run of
   record ranks under every criterion the grid recorded — validation and test alike. A
   configuration chosen by maximising a test metric would rank 1st on it.
2. **How much would a differently tuned joint-training reference move?** All three pipelines
   share ONE configuration (§2), a controlled comparison — but joint training is the denominator
   of GRR and of every "merging vs joint" claim, and it was not tuned for itself. The grid's
   joint-training sweep measures how far other configurations would move that reference.

Grid trials are single-seed (seed 42), so both answers are about **sensitivity**, not level: the
joint references GRR divides by are separate three-seed runs of the shared configuration.

⚠️ On AD, validation carries no labels, so "the validation-best joint" means best
*reconstruction error* — which §1.12 shows is blind to detection. It is reported because it is
the only label-free choice available, not because it would have been a better choice.
"""

import argparse
import csv
import logging
from pathlib import Path

log = logging.getLogger("config_selection_report")

# The runs of record, from EXPERIMENTS.md §2.1-2.3 (the three datasets whose configuration is a
# grid-search trial). Each maps to its grid file stem and its task family.
RECORDS = {
    "SWaT": ("swat", "58941", "58930", "window_auroc", "reconstruction/score_mean"),
    "PSM": ("psm", "59101", "59090", "window_auroc", "reconstruction/score_mean"),
    "ETTh1": ("etth_forecast", "59077", "59062", "forecast/mse", "forecast/mse"),
}
MIN_VISIBLE_PATCHES = 4        # mirrors MaeTx._assert_pretext_non_degenerate

RANK_FIELDS = ["dataset", "sweep", "record_run_id", "criterion", "split", "rank", "n_trials"]
JOINT_FIELDS = ["dataset", "metric", "floor_pct", "record_run_id", "record_value",
                "val_best_run_id", "val_best_value", "gap_to_val_best_pct",
                "gap_to_val_best_over_floor", "test_best_run_id", "test_best_value",
                "gap_to_test_best_pct", "test_best_degenerate", "valid_test_best_run_id",
                "valid_test_best_value", "gap_to_valid_test_best_pct",
                "gap_to_valid_test_best_over_floor", "n_trials"]


def higher_is_better(metric: str) -> bool:
    return "auroc" in metric or "auprc" in metric or "f1" in metric


def degenerate(row: dict) -> bool:
    """True if MaeTx would now refuse this trial: fewer than 4 visible patches (random mask)."""
    try:
        patch = int(row["mae_tx_patch_len"])
        window = int(row["dataset_window_len"])
        ratio = float(row["mae_tx_mask_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    if row.get("mae_tx_training_mode", "random_mask") != "random_mask":
        return False
    n_patches = window // patch
    return n_patches - int(n_patches * ratio) < MIN_VISIBLE_PATCHES


def rank_of(rows, run_id, column, up):
    usable = [r for r in rows if r.get(column) not in ("", None)]
    ordered = sorted(usable, key=lambda r: float(r[column]), reverse=up)
    position = next((i for i, r in enumerate(ordered, 1) if r["run_id"] == run_id), None)
    return position, len(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid_dir", type=Path)
    parser.add_argument("--floors", type=Path, default=Path("results_archive/audit/floors.csv"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true", dest="self_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        _self_test()
        return
    if args.grid_dir is None or not args.grid_dir.is_dir():
        raise SystemExit(f"--grid_dir {args.grid_dir} does not exist")

    floors = {(r["dataset"], r["metric"]): float(r["floor_pct"])
              for r in csv.DictReader(args.floors.open(encoding="utf-8"))
              if r["role"] == "floor" and r["floor_pct"]}
    ranks, joint = [], []
    for dataset, (stem, inc_id, std_id, test_metric, val_metric) in RECORDS.items():
        up = higher_is_better(test_metric)
        for sweep, run_id, prefix in (("train_incremental", inc_id, "merged"),
                                      ("train_standard", std_id, "train")):
            rows = list(csv.DictReader((args.grid_dir / f"{stem}_{sweep}_results.csv")
                                       .open(encoding="utf-8")))
            # Every criterion the grid recorded that could have picked a winner.
            criteria = [(f"{prefix}_val_{val_metric}", "val", False),
                        (f"{prefix}_test_{test_metric}", "test", up)]
            if not up:
                pass
            else:
                criteria += [(f"{prefix}_test_{m}", "test", True)
                             for m in ("window_auprc", "window_f1", "pa_f1")]
            if prefix == "merged":
                criteria.append((f"baseline_val_{val_metric}", "val", False))
            for column, split, better_up in criteria:
                position, total = rank_of(rows, run_id, column, better_up)
                if position is None:
                    continue
                ranks.append({"dataset": dataset, "sweep": sweep, "record_run_id": run_id,
                              "criterion": column, "split": split, "rank": position,
                              "n_trials": total})

        rows = list(csv.DictReader((args.grid_dir / f"{stem}_train_standard_results.csv")
                                   .open(encoding="utf-8")))
        tcol, vcol = f"train_test_{test_metric}", f"train_val_{val_metric}"
        usable = [r for r in rows if r.get(tcol) not in ("", None)]
        record = next(r for r in usable if r["run_id"] == std_id)
        pick = max if up else min
        test_best = pick(usable, key=lambda r: float(r[tcol]))
        val_best = min((r for r in usable if r.get(vcol)), key=lambda r: float(r[vcol]))
        valid = [r for r in usable if not degenerate(r)]
        valid_best = pick(valid, key=lambda r: float(r[tcol]))
        floor = floors.get((dataset, test_metric))

        def gap(other):
            # Positive = the other configuration is BETTER than the record, in % of the record.
            a, b = float(record[tcol]), float(other[tcol])
            return 100.0 * ((b - a) if up else (a - b)) / a

        joint.append({
            "dataset": dataset, "metric": test_metric, "floor_pct": floor,
            "record_run_id": std_id, "record_value": float(record[tcol]),
            "val_best_run_id": val_best["run_id"], "val_best_value": float(val_best[tcol]),
            "gap_to_val_best_pct": gap(val_best),
            "gap_to_val_best_over_floor": gap(val_best) / floor if floor else "",
            "test_best_run_id": test_best["run_id"], "test_best_value": float(test_best[tcol]),
            "gap_to_test_best_pct": gap(test_best),
            "test_best_degenerate": degenerate(test_best),
            "valid_test_best_run_id": valid_best["run_id"],
            "valid_test_best_value": float(valid_best[tcol]),
            "gap_to_valid_test_best_pct": gap(valid_best),
            "gap_to_valid_test_best_over_floor": gap(valid_best) / floor if floor else "",
            "n_trials": len(usable)})

    for row in ranks:
        log.info("[rank] %-6s %-18s %s  %-4s %-45s %2d / %d", row["dataset"], row["sweep"],
                 row["record_run_id"], row["split"], row["criterion"], row["rank"],
                 row["n_trials"])
    firsts = [r for r in ranks if r["rank"] == 1]
    log.info("[rank] record ranks FIRST on %d of %d criteria", len(firsts), len(ranks))
    for row in joint:
        log.info("[joint] %-6s record %.4f | val-best %+.2f%% (%.1fx floor) | best valid config "
                 "%+.2f%% (%.1fx floor)%s", row["dataset"], row["record_value"],
                 row["gap_to_val_best_pct"], row["gap_to_val_best_over_floor"] or 0,
                 row["gap_to_valid_test_best_pct"],
                 row["gap_to_valid_test_best_over_floor"] or 0,
                 "  (the raw test-best is degenerate and now refused)"
                 if row["test_best_degenerate"] else "")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("config_selection_ranks.csv", RANK_FIELDS, ranks),
                                   ("joint_sensitivity.csv", JOINT_FIELDS, joint)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            log.info("wrote %s (%d row(s))", args.out / name, len(rows))


def _self_test() -> None:
    rows = [{"run_id": "a", "x": "0.9"}, {"run_id": "b", "x": "0.8"}, {"run_id": "c", "x": ""}]
    assert rank_of(rows, "a", "x", True) == (1, 2)
    assert rank_of(rows, "a", "x", False) == (2, 2), "lower-is-better must reverse the order"
    # PSM's patch_len=25 winner: window 100 -> 4 patches, mask 0.8 -> 0 visible. Refused.
    assert degenerate({"mae_tx_patch_len": "25", "dataset_window_len": "100",
                       "mae_tx_mask_ratio": "0.8"})
    assert not degenerate({"mae_tx_patch_len": "5", "dataset_window_len": "100",
                           "mae_tx_mask_ratio": "0.8"}), "the proven recipe sits at exactly 4"
    log.info("self-test OK")


if __name__ == "__main__":
    main()
