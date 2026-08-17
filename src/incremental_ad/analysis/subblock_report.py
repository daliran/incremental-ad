"""Does the method ranking depend on *which part* of the test block you score?

    python -m incremental_ad.analysis.subblock_report --runs_root $RUNS_ROOT \\
        --spec analysis_specs/method_comparison_spec.csv --out $OUT/subblocks

§1.27 re-ran each comparison at three train/test cuts and found the winner moves. But an origin
built with `--dataset_series_fraction` changes **three** things at once — how much training data
there is, where the test block sits, and how large it is — and §1.28 shows the training-size
component alone moves ETTh1's merged MSE from 1.19 to 0.49. So §1.27 cannot attribute its
instability to the test block, which is the thing it was run to test.

This isolates it at **zero training cost**. Every run's test set is already scored window by
window; the windows are contiguous and time-ordered, so partitioning them into K equal
position-ordered sub-blocks and aggregating each separately asks exactly one question — *within
one fixed training set and one fixed test block, does the ranking depend on which quarter you
look at?* Nothing is retrained; this is a regrouping of an evaluation pass.

Reading it:

- **Ranking is stable across sub-blocks** → §1.27's instability is about moving the *training*
  set, and the caveat on every forecasting conclusion narrows to "these rankings assume this
  much training data", which is far weaker than "these rankings assume this cut".
- **Ranking is unstable within one block** → a stronger result than §1.27, obtained for free:
  a single test block is not one sample but several, and no single-cut comparison in this file
  can be trusted at the resolution it is quoted at.

**Rankings, not raw means, are the comparable quantity across sub-blocks.** Different quarters of
a series have different difficulty, so absolute MSE is not comparable between them — the last
quarter of exchange_rate is genuinely harder than the first, and averaging across quarters just
recovers the whole-block number. Within a sub-block every method sees identical windows, so the
ordering is meaningful. Means are reported per sub-block for inspection and must not be pooled
across them.

Forecasting only: it needs a per-window error, which a ranking metric like AUROC does not have.
"""

import argparse
import csv
import json
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("subblock_report")

FIELDS = ["dataset", "n_segments", "method", "experiment", "subblock", "n_subblocks",
          "n_windows", "n_seeds", "mean", "sd", "rank"]

# Which checkpoint is "the model this method deploys", per pipeline shape.
# `sequential` and `window` depend on the run, so they are resolved by globbing.
CHECKPOINT = {
    "merge": "merged/checkpoints/best.pt",
    "joint": "train/checkpoints/best.pt",
    "window": "finetune_0/checkpoints/best.pt",
    "base": "baseline/checkpoints/best.pt",
}


def _sequential_checkpoint(run: Path) -> Path | None:
    """The last link of the continual chain — the model that method actually ends with."""
    steps = sorted(run.glob("continual_*/checkpoints/best.pt"),
                   key=lambda p: int(p.parent.parent.name.split("_")[1]))
    return steps[-1] if steps else None


def resolve(run: Path, method: str) -> Path | None:
    if method == "sequential":
        return _sequential_checkpoint(run)
    path = run / CHECKPOINT[method]
    return path if path.exists() else None


def subblock_means(errors, n_blocks: int) -> list[float]:
    """Mean error within each of `n_blocks` equal, contiguous, time-ordered spans.

    The loader runs with shuffle=False over a sliding-window dataset, so window i precedes
    window i+1 in the series and a contiguous slice of indices is a contiguous slice of time.
    A trailing remainder is folded into the last block rather than dropped, so every window is
    scored exactly once and the blocks differ in size by at most n_blocks-1 windows.
    """
    total = len(errors)
    edge = total // n_blocks
    out = []
    for b in range(n_blocks):
        start = b * edge
        stop = total if b == n_blocks - 1 else (b + 1) * edge
        out.append(float(errors[start:stop].mean()))
    return out


def write_rows(out: Path, rows: list[dict]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "subblock_summary.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarise(rows: list[dict], spec: list[dict], n_blocks: int) -> None:
    """Per configuration: does any sub-block *decisively* contradict the others?

    Ranking alone overcounts instability. Rank 1 is whichever method is numerically lowest, so a
    0.0002 gap between two methods — exchange_rate's third sub-block, where joint and window are
    indistinguishable — reads as a "winner change" and inflates the headline. A sub-block only
    counts as contradicting when its winner beats the runner-up by more than the combined
    seed spread of those two models, the same pairwise rule §1.9a uses everywhere else.
    """
    import math

    log.info("\n%-14s %-3s %-38s %s", "dataset", "n", "decisive winner per sub-block", "verdict")
    contradicted = 0
    counted = 0
    for entry in spec:
        key = (entry["dataset"], entry["n"])
        per_block = []
        for b in range(1, n_blocks + 1):
            cell = {r["method"]: (r["mean"], r["sd"]) for r in rows
                    if (r["dataset"], r["n_segments"]) == key and r["subblock"] == b}
            compete = {m: v for m, v in cell.items() if m != "base"}
            if len(compete) < 2:
                continue
            order = sorted(compete, key=lambda m: compete[m][0])
            best, runner = order[0], order[1]
            gap = compete[runner][0] - compete[best][0]
            threshold = math.sqrt(compete[best][1] ** 2 + compete[runner][1] ** 2)
            per_block.append(best if gap > threshold else None)
        if not per_block:
            continue
        counted += 1
        decisive = {m for m in per_block if m}
        moves = len(decisive) > 1
        contradicted += moves
        log.info("%-14s %-3s %-38s %s", key[0], key[1],
                 " → ".join(m or "tie" for m in per_block),
                 "*** CONTRADICTS" if moves else
                 ("stable" if decisive else "no decisive block"))
    if counted:
        log.info("\n%d of %d configurations have a sub-block that decisively contradicts "
                 "another; %d are consistent across all %d sub-blocks",
                 contradicted, counted, counted - contradicted, n_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True,
                        help="method_comparison_spec.csv — the same experiments of record §1.26 "
                             "reads, so this cannot drift from that table")
    parser.add_argument("--n_subblocks", type=int, default=4)
    parser.add_argument("--metric", default="forecast/mse", choices=["forecast/mse",
                                                                     "forecast/mae"])
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="restrict to these spec datasets (default: every forecasting one)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summarise_only", action="store_true",
                        help="re-derive the verdict from an existing subblock_summary.csv in "
                             "--out. Pure CSV work: the scoring pass needs a GPU, re-reading "
                             "its own output must not.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with args.spec.open() as fh:
        spec_all = [r for r in csv.DictReader(fh) if "forecast/" in r["metric"]]
    if args.datasets:
        spec_all = [r for r in spec_all if r["dataset"] in args.datasets]
    if args.summarise_only:
        with (args.out / "subblock_summary.csv").open() as fh:
            cached = [{**r, "mean": float(r["mean"]), "sd": float(r["sd"]),
                       "subblock": int(r["subblock"])} for r in csv.DictReader(fh)]
        summarise(cached, spec_all, args.n_subblocks)
        return

    from incremental_ad.analysis.oracle_router import _per_window_errors
    from incremental_ad.analysis.selection_probe import _load_run
    from incremental_ad.framework.core.checkpoints import load_model_state

    metric_index = 0 if args.metric == "forecast/mse" else 1

    with args.spec.open() as fh:
        spec = [r for r in csv.DictReader(fh) if "forecast/" in r["metric"]]
    if args.datasets:
        spec = [r for r in spec if r["dataset"] in args.datasets]
    if not spec:
        raise SystemExit("no forecasting rows in the spec — this is forecasting-only by design")

    rows: list[dict] = []
    for entry in spec:
        dataset, n = entry["dataset"], entry["n"]
        methods = {"joint": entry["joint_experiment"], "merge": entry["merge_experiment"],
                   "sequential": entry["seq_experiment"],
                   "window": entry["window_W3_experiment"], "base": entry["merge_experiment"]}
        log.info("\n%s n=%s", dataset, n)
        per_method: dict[str, list[list[float]]] = defaultdict(list)
        windows_seen = None
        for method, experiment in methods.items():
            for run in sorted(p for p in (args.runs_root / experiment).glob("*") if p.is_dir()):
                checkpoint = resolve(run, method)
                if checkpoint is None:
                    continue
                try:
                    data, model, _cfg, runner, _c = _load_run(run)
                    model.to(runner.device)
                    model.load_state_dict(load_model_state(checkpoint))
                    loader = runner.loader_config.make_loader(data.get_test_dataset(),
                                                              shuffle=False)
                    errors = _per_window_errors(model, loader, runner.device)[metric_index]
                except Exception as exc:                       # noqa: BLE001
                    log.warning("  %s/%s %s — skipped (%s)", experiment, run.name, method, exc)
                    continue
                # Every method must score the identical window set, or the sub-blocks are not
                # the same spans and the ranking within one is meaningless.
                if windows_seen is None:
                    windows_seen = len(errors)
                elif len(errors) != windows_seen:
                    log.warning("  %s %s: %d windows against %d — excluded, the sub-blocks "
                                "would not line up", experiment, method, len(errors),
                                windows_seen)
                    continue
                per_method[method].append(subblock_means(errors, args.n_subblocks))

        if not per_method:
            log.warning("  no scorable runs — skipped")
            continue

        for b in range(args.n_subblocks):
            block_means = {m: st.mean(v[b] for v in seeds)
                           for m, seeds in per_method.items() if seeds}
            # base is the frozen starting point, not a competitor — it does not take a rank.
            ranked = sorted((m for m in block_means if m != "base"), key=lambda m: block_means[m])
            order = {m: i + 1 for i, m in enumerate(ranked)}
            for method, seeds in per_method.items():
                values = [v[b] for v in seeds]
                rows.append({
                    "dataset": dataset, "n_segments": n, "method": method,
                    "experiment": methods[method], "subblock": b + 1,
                    "n_subblocks": args.n_subblocks, "n_windows": windows_seen,
                    "n_seeds": len(seeds), "mean": round(st.mean(values), 6),
                    "sd": round(st.stdev(values), 6) if len(values) > 1 else 0.0,
                    "rank": order.get(method, ""),
                })
            log.info("  block %d: %s", b + 1,
                     "  ".join(f"{m}={block_means[m]:.4f}" for m in ranked))
        # Flush after every configuration. Scoring cost scales with test-window count, which
        # spans 3.4k to 390k across these datasets, so the run order puts hours of work behind
        # minutes of it — and a single write at the end throws away the finished configs when
        # the wall clock runs out.
        if args.out:
            write_rows(args.out, rows)

    if not rows:
        raise SystemExit("nothing scored — check --runs_root against the spec's experiment names")

    summarise(rows, spec, args.n_subblocks)

    if args.out:
        log.info("wrote %s", write_rows(args.out, rows))


if __name__ == "__main__":
    main()
