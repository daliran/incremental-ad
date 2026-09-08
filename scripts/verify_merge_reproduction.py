"""Recompute every archived merge from its baseline + finetunes and require bitwise equality.

    srun ... python scripts/verify_merge_reproduction.py --runs_root $RUNS_ROOT \\
        --out results_archive/audit/merge_reproduction.csv

CLAUDE.md calls this the project's strongest regression test — "stronger than any unit test you
could write" — and cites 87/87. It had **no committed script**: it was run ad-hoc and the number
was carried in prose, which is the same gap that let §1.2's α\\* column and §1.10's seed-42 row
become unreproducible. This is that check, as a script of record.

**Reads `merge_scale/selected` first, `config.json` second.** With
`--pipeline_select_merge_scale_on_val`, `pipeline_merge_scale` records the value that was
*asked for* while the checkpoint was built at the one that *won*. Recomputing from the config
makes every selecting run look broken — it once produced 12 spurious mismatches, all of them
exactly the runs where the two differ.

**Merge rule and coefficient source matter too** (§1.31). A run merged with `opcm` or `became`
is not reproducible by plain task arithmetic, so those are reconstructed through the same
`merge_sequential` path the pipeline used — except for BECAME, whose λ\\* depends on Fishers that
are not stored. Those runs are reported as `skipped_became` rather than counted as failures:
claiming a pass on a check that was not performed is worse than reporting the gap.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

log = logging.getLogger("verify_merge_reproduction")

FIELDS = ["experiment", "run_id", "merge_rule", "coefficient_source", "alpha_source",
          "alpha", "n_finetunes", "status", "n_tensors", "max_abs_diff"]


def committed_alpha(run: Path, args: dict) -> tuple[float | None, str]:
    """The scale the checkpoint was actually built at, and where that was read from."""
    result = run / "merged" / "val" / "result.json"
    if result.is_file():
        try:
            metrics = (json.loads(result.read_text()) or {}).get("metrics") or {}
            if "merge_scale/selected" in metrics:
                return float(metrics["merge_scale/selected"]), "merge_scale/selected"
        except (json.JSONDecodeError, OSError):
            pass
    value = args.get("pipeline_merge_scale")
    return (float(value), "config.json") if value is not None else (None, "absent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    from incremental_ad.framework.core.checkpoints import load_model_state
    from incremental_ad.framework.merging.task_vectors import (
        merge_sequential, merge_task_arithmetic, opcm_residual,
    )

    rows: list[dict] = []
    for config_path in sorted(args.runs_root.glob("*/*/config.json")):
        run = config_path.parent
        merged = run / "merged" / "checkpoints" / "best.pt"
        baseline = run / "baseline" / "checkpoints" / "best.pt"
        finetunes = sorted(run.glob("finetune_*/checkpoints/best.pt"),
                           key=lambda p: int(p.parent.parent.name.split("_")[1]))
        if not (merged.is_file() and baseline.is_file() and finetunes):
            continue
        try:
            run_args = (json.loads(config_path.read_text()) or {}).get("args") or {}
        except (json.JSONDecodeError, OSError):
            continue

        rule = run_args.get("pipeline_merge_rule", "sum")
        source = run_args.get("pipeline_coefficient_source", "scale")
        alpha, alpha_source = committed_alpha(run, run_args)
        row = {"experiment": run.parent.name, "run_id": run.name, "merge_rule": rule,
               "coefficient_source": source, "alpha_source": alpha_source, "alpha": alpha,
               "n_finetunes": len(finetunes), "n_tensors": "", "max_abs_diff": ""}

        if source == "became":
            # lambda* is derived from per-shard Fishers that are not stored with the run, so the
            # merge cannot be rebuilt from checkpoints alone. Reported, never silently passed.
            row["status"] = "skipped_became"
            rows.append(row)
            continue
        if alpha is None:
            row["status"] = "skipped_no_alpha"
            rows.append(row)
            continue

        base_state = load_model_state(baseline)
        ft_states = [load_model_state(p) for p in finetunes]
        if rule == "opcm":
            from incremental_ad.framework.merging.task_vectors import task_vector
            taus = [task_vector(base_state, ft) for ft in ft_states]
            rebuilt = merge_sequential(
                base_state, taus, [(1.0, alpha)] * len(taus),
                transform=opcm_residual(run_args.get("pipeline_opcm_threshold", 0.5)),
            )
        else:
            rebuilt = merge_task_arithmetic(base_state, ft_states, alpha)

        stored = load_model_state(merged)
        worst = 0.0
        identical = True
        for key, tensor in stored.items():
            if key not in rebuilt:
                identical = False
                break
            other = rebuilt[key]
            if not torch.equal(tensor, other):
                identical = False
                if tensor.is_floating_point():
                    worst = max(worst, float((tensor - other).abs().max()))
        row["status"] = "bitwise" if identical else "MISMATCH"
        row["n_tensors"] = len(stored)
        row["max_abs_diff"] = round(worst, 12)
        rows.append(row)
        log.info("  %-34s %-8s %-6s alpha=%-6s %s", row["experiment"], row["run_id"],
                 rule, alpha, row["status"])

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    checked = counts.get("bitwise", 0) + counts.get("MISMATCH", 0)
    log.info("\n%d/%d merges reproduce bitwise from baseline + finetunes at the committed alpha",
             counts.get("bitwise", 0), checked)
    for status, count in sorted(counts.items()):
        if status not in ("bitwise",):
            log.info("  %s: %d", status, count)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("wrote %s", args.out)
    raise SystemExit(1 if counts.get("MISMATCH") else 0)


if __name__ == "__main__":
    main()
