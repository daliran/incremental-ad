"""Render the archived results as a standalone HTML report.

    python scripts/build_results_report.py --archive results_archive \\
        --out results_report.html --commit $(git rev-parse --short HEAD)

**Note.** The execution brief referred to this script as "provided separately"; it was not
attached, so this is a from-scratch implementation of what the brief describes — regenerated
with every archive refresh, committed alongside it. If the intended version turns up, prefer it
and delete this one rather than maintaining both.

**What it is for.** `EXPERIMENTS.md` is the record and is machine-checked cell by cell, but it is
9,000 lines and assumes the reader already knows the vocabulary. This is the other audience: one
page that opens with the headline comparison and the reproducibility floors, so a supervisor or
examiner can see the state of the work without reading the source of record. It is *generated*,
so it cannot disagree with the archive — every number is read from a CSV under `--archive`, none
is transcribed.

Deliberately not a substitute for the artifact page: that one is written prose with a designed
argument. This is a mechanical dump of the tables that matter, stamped with the commit and the
archive's own file count so a stale copy is visible on sight.
"""

import argparse
import csv
import html
from datetime import datetime, timezone
from pathlib import Path

STYLE = """
:root { --bg:#fbfbfa; --ink:#16191c; --muted:#5d666e; --line:#dcdfe3; --card:#fff;
        --accent:#12615c; --warn:#8a5b12; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#101416; --ink:#e6ebec; --muted:#96a1a6; --line:#252c30; --card:#171d20;
  --accent:#57cfc4; --warn:#d6ab5e; } }
:root[data-theme="dark"] { --bg:#101416; --ink:#e6ebec; --muted:#96a1a6; --line:#252c30;
  --card:#171d20; --accent:#57cfc4; --warn:#d6ab5e; }
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.55;
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:70rem; margin:0 auto; padding:2.5rem 1.25rem 5rem }
h1 { font-size:1.9rem; margin:0 0 .3rem; letter-spacing:-.01em }
h2 { font-size:1.15rem; margin:2.5rem 0 .6rem; padding-bottom:.3rem;
     border-bottom:2px solid var(--ink) }
.meta { color:var(--muted); font-size:.83rem; font-family:ui-monospace,monospace;
        display:flex; gap:1.2rem; flex-wrap:wrap; margin-bottom:.4rem }
.note { color:var(--muted); font-size:.87rem; max-width:44rem }
.scroll { overflow-x:auto; border:1px solid var(--line); background:var(--card);
          border-radius:3px; margin:.5rem 0 }
table { border-collapse:collapse; width:100%; font-size:.83rem;
        font-variant-numeric:tabular-nums; font-family:ui-monospace,monospace }
th,td { padding:.4rem .6rem; text-align:right; white-space:nowrap;
        border-bottom:1px solid var(--line) }
th { color:var(--muted); font-weight:600; text-align:right; font-size:.76rem }
td:first-child, th:first-child { text-align:left }
tbody tr:last-child td { border-bottom:none }
.warn { color:var(--warn) }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
         color:var(--muted); font-size:.82rem }
"""


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def table(rows: list[dict], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return '<p class="note">absent from this archive.</p>'
    shown = rows[:limit] if limit else rows
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        for r in shown
    )
    more = (f'<p class="note">{len(rows) - len(shown)} further rows in the CSV.</p>'
            if limit and len(rows) > limit else "")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead>' \
           f"<tbody>{body}</tbody></table></div>{more}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, default=Path("results_archive"))
    parser.add_argument("--out", type=Path, default=Path("results_report.html"))
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--generated_at", default=None,
                        help="ISO timestamp; defaults to now. Pass a fixed value to make the "
                             "output byte-stable for a reproducibility check.")
    args = parser.parse_args()

    audit = args.archive / "audit"
    stamp = args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_files = sum(1 for p in args.archive.rglob("*") if p.is_file())

    methods = read(audit / "method_comparison.csv")
    floors = [r for r in read(audit / "floors.csv") if r.get("role") == "floor"]
    scale = read(audit / "scale_forecast/scale_summary.csv") + \
        read(audit / "scale_psm_forecast/scale_summary.csv")
    routing = read(audit / "routing_forecast/routing_summary.csv") + \
        read(audit / "routing_psm_forecast/routing_summary.csv")
    subblocks = read(audit / "subblocks/subblock_summary.csv")
    concentration = read(audit / "concentration/error_concentration_ETTh1.csv")

    sections = [
        ("All strategies on one footing",
         "One row per (dataset, n). <code>decisive</code> marks a margin exceeding the combined "
         "seed spread of the two models compared, not a fixed floor.",
         table(methods, ["dataset", "n", "metric", "base", "joint", "merge", "sequential",
                         "window_best", "best", "decisive"])),
        ("Reproducibility floors",
         "Sample sd ÷ mean of the baseline-stage test metric, within one experiment. Not "
         "comparable across experiments.",
         table(floors, ["dataset", "metric", "experiment", "n_seeds", "mean", "sd",
                        "floor_pct"])),
        ("Merge scale",
         "α* is the mean over seeds of each seed's argmin on the window-weighted validation "
         "union. Both the pooled and per-seed GRR aggregations are shown; they do not coincide.",
         table(scale, ["group", "n_segments", "n_seeds", "alpha_star", "alpha_star_n",
                       "grr_val", "grr_oracle", "grr_val_per_seed", "grr_val_per_seed_sd",
                       "honest_alpha_cost_pct", "penalty_one_over_n_pct"])),
        ("Routing headroom",
         "How far the merged model sits above the per-regime optimum, and how far the newest "
         "specialist does. An upper bound on routing, not a deployable method.",
         table(routing, ["group", "n_segments", "oracle", "merged", "newest",
                         "merged_vs_oracle_pct", "newest_vs_oracle_pct"])),
        ("Test block, split in quarters",
         "Same training set, same test block, scored in four contiguous time-ordered spans. "
         "Isolates test-block position from training size.",
         table(subblocks, ["dataset", "n_segments", "method", "subblock", "n_seeds", "mean",
                           "sd", "rank"], limit=60)),
        ("Where the forecasting floor comes from",
         "Heavy tail and seed-divergence both refuted: trimming the worst windows does not move "
         "the floor, and seeds agree on which windows are hard.",
         table(concentration, ["label", "n_seeds", "n_windows", "floor_pct", "top1pct_share",
                               "trimmed1_floor_pct", "trimmed5_floor_pct",
                               "cross_seed_pearson"])),
    ]

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Incremental learning by model merging — archived results</title>",
        f"<style>{STYLE}</style></head><body><div class='wrap'>",
        "<h1>Incremental learning by model merging</h1>",
        f"<div class='meta'><span>commit {html.escape(args.commit)}</span>"
        f"<span>generated {html.escape(stamp)}</span>"
        f"<span>{n_files} archived files</span></div>",
        "<p class='note'>Generated from <code>results_archive/</code>. Every value is read from "
        "a CSV in the archive — none is transcribed — so this page cannot disagree with the "
        "evidence. <code>EXPERIMENTS.md</code> remains the source of record and carries the "
        "reasoning, caveats and withdrawals that these tables do not.</p>",
    ]
    for title, note, body in sections:
        parts += [f"<h2>{html.escape(title)}</h2>", f"<p class='note'>{note}</p>", body]
    parts += [
        "<footer>Rebuilt by <code>scripts/build_results_report.py</code> as part of "
        "<code>scripts/regenerate_analysis.sh</code>. If the file count above disagrees with "
        "<code>results_archive/MANIFEST.csv</code>, this copy is stale.</footer>",
        "</div></body></html>",
    ]
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, "
          f"{len(sections)} sections, {n_files} archived files)")


if __name__ == "__main__":
    main()
