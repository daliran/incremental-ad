"""§1.43's script of record: prequential ranking vs the final-test-block ranking.

    python -m incremental_ad.analysis.prequential_report --prequential_dir $WORK/prequential \\
        --runs_root $RUNS_ROOT --floors results_archive/audit/floors.csv --out $OUT/prequential

Reads `prequential.py`'s results (one per merge run) and applies the aggregation registered in
EXPERIMENTS.md §1.43 before the runs:

- **Prequential ranking** of {merge, chain, specialist}: per seed and per k, rank the three by
  `forecast/mse` on period k + 1; average the ranks over k, then over seeds. Mean MSE over k is
  reported beside it, because period scales differ (exchange_rate ~10×) and a raw mean would be
  a statement about the hardest period.
- **Final-test ranking** of the same three models: `merged/test`, `finetune_{n−1}/test` (merge
  run) and `continual_{n−1}/test` (paired chain run), mean over seeds.
- **Agreement** is counted over the three pairwise orderings. A pairwise margin is also judged
  against the dataset's floor, so an ordering inside the noise is labelled as such.
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

METHODS = ("merge", "chain", "specialist")
PAIRS = (("merge", "chain"), ("merge", "specialist"), ("chain", "specialist"))
EXCLUDED = {"SWaT-forecast"}
FLOOR_LABEL = {"exchange": "exchange_rate"}

FIELDS = ["dataset", "n", "n_seeds", "n_steps",
          "preq_rank_merge", "preq_rank_chain", "preq_rank_specialist",
          "preq_mse_merge", "preq_mse_chain", "preq_mse_specialist", "preq_mse_base",
          "final_mse_merge", "final_mse_chain", "final_mse_specialist",
          "preq_order", "final_order", "pairs_agree", "pairs_disagree",
          "chain_better_prequentially", "floor_pct", "preq_decisive_pairs",
          "final_decisive_pairs", "merge_alphas"]
PRED_FIELDS = ["prediction", "statement", "observed", "outcome"]


def test_mse(run: Path, block: str) -> float:
    payload = json.loads((run / block / "result.json").read_text())
    return float(payload.get("metrics", payload)["forecast/mse"])


def order(values: dict[str, float]) -> list[str]:
    return sorted(METHODS, key=lambda m: (values[m], METHODS.index(m)))


def decisive(values: dict[str, float], floor: float | None) -> int:
    if floor is None:
        return 0
    count = 0
    for a, b in PAIRS:
        lo, hi = sorted((values[a], values[b]))
        count += 100.0 * (hi - lo) / lo > floor
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prequential_dir", type=Path, required=True)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--floors", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.prequential_dir.is_dir():
        raise SystemExit(f"{args.prequential_dir} does not exist")

    floors = {(r["dataset"], r["metric"]): float(r["floor_pct"])
              for r in csv.DictReader(args.floors.open(encoding="utf-8"))
              if r["role"] == "floor" and r["floor_pct"]}
    spec = {r["merge_experiment"]: r for r in csv.DictReader(args.spec.open(encoding="utf-8"))
            if r["metric"] == "forecast/mse"}
    groups = defaultdict(list)
    for result in sorted(args.prequential_dir.glob("*/result.json")):
        payload = json.loads(result.read_text())
        run = Path(payload["source_run"])
        entry = spec.get(run.parent.name)
        if entry is None:
            continue
        groups[(entry["dataset"], int(entry["n"]))].append(payload)

    rows = []
    for (dataset, n), payloads in sorted(groups.items()):
        preq_rank = defaultdict(list)
        preq_mse = defaultdict(list)
        final = defaultdict(list)
        alphas = []
        for payload in payloads:
            steps = payload["steps"]
            ranks = defaultdict(list)
            for step in steps:
                values = {m: step[m] for m in METHODS}
                for position, method in enumerate(order(values), 1):
                    ranks[method].append(position)
                alphas.append(step["merge_alpha"])
            for m in METHODS:
                preq_rank[m].append(st.fmean(ranks[m]))
                preq_mse[m].append(st.fmean(s[m] for s in steps))
            preq_mse["base"].append(st.fmean(s["base"] for s in steps))
            # Rebased on --runs_root, never the absolute path the job recorded: the archive's own
            # runs/ tree holds every result.json, so the report runs with no $WORK mounted.
            run = args.runs_root / Path(payload["source_run"]).parent.name / \
                Path(payload["source_run"]).name
            chain = args.runs_root / Path(payload["chain_run"]).parent.name / \
                Path(payload["chain_run"]).name
            final["merge"].append(test_mse(run, "merged/test"))
            final["specialist"].append(test_mse(run, f"finetune_{n - 1}/test"))
            final["chain"].append(test_mse(chain, f"continual_{n - 1}/test"))
        rank = {m: st.fmean(preq_rank[m]) for m in METHODS}
        mse = {m: st.fmean(preq_mse[m]) for m in METHODS}
        fin = {m: st.fmean(final[m]) for m in METHODS}
        # Prequential order: mean rank, ties broken by mean MSE.
        p_order = sorted(METHODS, key=lambda m: (rank[m], mse[m]))
        f_order = order(fin)
        agree = [(a, b) for a, b in PAIRS
                 if (p_order.index(a) < p_order.index(b)) == (f_order.index(a) < f_order.index(b))]
        disagree = [pair for pair in PAIRS if pair not in agree]
        chain_up = sum(1 for a, b in disagree if "chain" in (a, b)
                       and p_order.index("chain") < p_order.index(b if a == "chain" else a))
        floor = floors.get((FLOOR_LABEL.get(dataset, dataset), "forecast/mse"))
        rows.append({
            "dataset": dataset, "n": n, "n_seeds": len(payloads),
            "n_steps": len(payloads[0]["steps"]),
            **{f"preq_rank_{m}": round(rank[m], 4) for m in METHODS},
            **{f"preq_mse_{m}": mse[m] for m in METHODS},
            "preq_mse_base": st.fmean(preq_mse["base"]),
            **{f"final_mse_{m}": fin[m] for m in METHODS},
            "preq_order": " < ".join(p_order), "final_order": " < ".join(f_order),
            "pairs_agree": len(agree), "pairs_disagree": len(disagree),
            "chain_better_prequentially": chain_up,
            "floor_pct": floor if floor is not None else "",
            "preq_decisive_pairs": decisive(mse, floor),
            "final_decisive_pairs": decisive(fin, floor),
            "merge_alphas": " ".join(f"{a:g}" for a in alphas)})

    scored = [r for r in rows if r["dataset"] not in EXCLUDED]
    ok1 = sum(r["pairs_agree"] >= 2 for r in scored)
    disagree_chain = up = 0
    for r in scored:
        p, f = r["preq_order"].split(" < "), r["final_order"].split(" < ")
        for a, b in PAIRS:
            if "chain" not in (a, b):
                continue
            if (p.index(a) < p.index(b)) != (f.index(a) < f.index(b)):
                disagree_chain += 1
                other = b if a == "chain" else a
                up += p.index("chain") < p.index(other)
    predictions = [
        {"prediction": "P1",
         "statement": ">= 2 of 3 pairwise orderings agree in >= 8 of 15 configurations",
         "observed": f"{ok1} of {len(scored)}",
         "outcome": "confirmed" if ok1 >= 8 else "refuted"},
        {"prediction": "P2",
         "statement": "among disagreeing pairwise orderings involving the chain, the chain ranks "
                      "higher prequentially in the majority",
         "observed": f"{up} of {disagree_chain}",
         "outcome": ("confirmed" if disagree_chain and up > disagree_chain / 2
                     else "not testable" if not disagree_chain else "refuted")},
    ]
    for r in rows:
        print(f"{r['dataset']:13s} n={r['n']}  preq {r['preq_order']:32s} final {r['final_order']:32s}"
              f" agree {r['pairs_agree']}/3")
    for p in predictions:
        print(f"[{p['prediction']}] {p['observed']}: {p['outcome'].upper()}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fields, data in (("prequential.csv", FIELDS, rows),
                                   ("prequential_predictions.csv", PRED_FIELDS, predictions)):
            with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            print(f"wrote {args.out / name} ({len(data)} rows)")


if __name__ == "__main__":
    main()
