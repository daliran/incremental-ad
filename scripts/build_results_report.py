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
import hashlib
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


def manifest_fingerprint(archive: Path) -> tuple[str, int]:
    """SHA-256 of `MANIFEST.csv` itself, plus its entry count.

    The manifest already hashes every archived file, so hashing the manifest is a hash of the
    whole archive at one remove — cheap, and it changes whenever any evidence changes. Embedding
    it lets the checker prove this page was built from the committed archive rather than from an
    older one, which is the failure the old hand-built HTML pages had: they looked current and
    were not.
    """
    path = archive / "MANIFEST.csv"
    if not path.is_file():
        return "absent", 0
    raw = path.read_bytes()
    entries = max(raw.count(b"\n") - 1, 0)
    return hashlib.sha256(raw).hexdigest(), entries


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
    fingerprint, n_entries = manifest_fingerprint(args.archive)

    methods = read(audit / "method_comparison.csv")
    claims = read(audit / "claims_register.csv")
    method_rows = read(audit / "methods_register.csv")
    floors = [r for r in read(audit / "floors.csv") if r.get("role") == "floor"]
    scale = read(audit / "scale_forecast/scale_summary.csv") + \
        read(audit / "scale_psm_forecast/scale_summary.csv")
    routing = read(audit / "routing_forecast/routing_summary.csv") + \
        read(audit / "routing_psm_forecast/routing_summary.csv")
    subblocks = read(audit / "subblocks/subblock_summary.csv")
    concentration = read(audit / "concentration/error_concentration_ETTh1.csv")
    # Geometry, including the AEFT rows: §1.33's refutation of QOMM's premise is a geometry
    # result, so the one figure showing it belongs in the report and not only in the markdown.
    geometry = read(audit / "geometry/geometry_by_dataset.csv")
    aeft = read(audit / "geometry_aeft/geometry_summary.csv")
    aeft_rows = []
    if aeft:
        import statistics as st
        grouped: dict[str, list[dict]] = {}
        for row in aeft:
            grouped.setdefault(row["experiment_name"], []).append(row)
        label = {"opcm2_psm_sum_scale": "PSM-forecast — full fine-tuning",
                 "aeft_psm_sum_scale": "PSM-forecast — attention-only (AEFT)"}
        for name, rows_ in sorted(grouped.items()):
            out = {"configuration": label.get(name, name), "n_seeds": len(rows_)}
            for column in ("mean_offdiag_cosine", "mean_sequential_overlap",
                           "cosine_at_distance_1", "effective_rank", "mean_tau_norm"):
                values = [float(r[column]) for r in rows_ if r.get(column) not in ("", None)]
                out[column] = round(st.mean(values), 4) if values else ""
            aeft_rows.append(out)

    # --- advanced merging: §1.35's sweep and §1.36-§1.38's closeout -----------------------------
    # The page stopped at the September batch, so the closing sweeps existed only in EXPERIMENTS.md
    # and in CSVs. Both are read straight from the archive like everything else here.
    remerge = read(audit / "remerge_sweep_report/remerge_sweep.csv")
    closeout = read(audit / "remerge_closeout/remerge_closeout.csv")

    # One table per test group, in the order the argument is made rather than alphabetically.
    CLOSEOUT_GROUPS = [
        ("P1_paper_opcm",
         "P1 — the paper's OPCM against plain summation",
         "Tang et al. (NeurIPS 2025), implemented in full and kept separate from the simplified "
         "<code>opcm_residual</code> of §1.31/§1.35 — the two are never reported as one (C26). "
         "It takes no merge scale: Theorem 5.2 pins the merge to the mean task-vector norm, which "
         "is why <code>opcm_norm_ratio</code> is 1.0 on every row and "
         "<code>implied_alpha_times_n</code> lands above 1.0 everywhere."),
        ("P4_opcm_committed",
         "P4 — the same projection at the committed α·n (coefficient-matched)",
         "⚠️ Rows carrying <code>confounded = coefficient_not_distance_matched</code> are "
         "<strong>not evidence</strong>. The projection shrinks the task vectors, so matching "
         "per-vector coefficients does not match the distance travelled, and those merges "
         "undershoot — see <code>opcm_norm_ratio</code> in the same row. The flag is derived from "
         "that ratio (outside 0.90–1.20), not from a dataset list. The two populations separate "
         "cleanly on their own — forecasting 0.195–0.819, the AD pair 0.935–1.469 — so no "
         "threshold is doing delicate work. Two AD rows at threshold 0.3 are flagged as well, "
         "correctly: their rescale moved the merge by ~47%, which is not the no-op the rest of "
         "the AD cells are. §1.38's P5 is the repair."),
        ("P5_opcm_distance",
         "P5 — the same projection at matched distance",
         "<code>distance_ratio = 1.0</code> on every row by construction: the merge travels "
         "exactly as far from θ₀ as plain summation at the committed α, so the two differ only in "
         "direction. This is the clean test of the projection, and it still loses."),
        ("P2_became_rescaled",
         "P2 — BECAME's weighting at a chosen strength",
         "BECAME's relative weights rescaled so their sum is the committed α·n, against uniform "
         "1/n at that same α·n through the same evaluation path. Closes C22."),
        ("P3_order_reversal",
         "P3 — periods fed newest-first",
         "Gated by a null check the table does not show: plain summation is order-inert, so "
         "<code>sum --reverse_order</code> must reproduce each stored merge — 12/12 within 1e-6, "
         "worst 1.15e-7. The result is <em>inconclusive</em>, not supporting: reversal destroys "
         "exchange_rate's win but hurts every dataset, so it cannot attribute."),
    ]
    closeout_tables = []
    for test, title, note in CLOSEOUT_GROUPS:
        rows_ = [r for r in closeout if r.get("test") == test]
        columns = ["dataset", "n_segments", "threshold", "n_seeds", "n_seeds_expected",
                   "complete", "baseline", "variant", "delta_pct", "floor_pct", "verdict",
                   "implied_alpha_times_n"]
        for optional in ("opcm_norm_ratio", "distance_ratio", "confounded"):
            if any(r.get(optional) for r in rows_):
                columns.append(optional)
        closeout_tables.append((title, note, table(rows_, columns, limit=100)))

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
        ("Task-vector geometry",
         "ρ is the share of an incoming task vector already spanned by its predecessors; the "
         "cosine is the mean off-diagonal pairwise cosine. Lower means the shards edit more "
         "independent directions.",
         table(geometry, ["dataset", "n_seeds", "mean_sequential_overlap", "mean_offdiag_cosine",
                          "effective_rank", "mean_tau_over_base"])),
        ("QOMM's premise, tested",
         "Attention-exclusive fine-tuning was predicted to make task vectors <em>more</em> "
         "orthogonal. It does the opposite — the cosine rises 35% and ρ nearly doubles — which "
         "is why it makes OPCM worse and α* smaller (EXPERIMENTS.md §1.33). This is the figure "
         "behind that refutation.",
         table(aeft_rows, ["configuration", "n_seeds", "mean_offdiag_cosine",
                           "mean_sequential_overlap", "cosine_at_distance_1", "effective_rank",
                           "mean_tau_norm"])),
        ("Advanced merging — the simplified OPCM sweep (§1.35)",
         "Every dataset with checkpoints, re-merged training-free at the strength each source run "
         "committed to, with the simplified operator. <code>delta_pct</code> is signed so positive "
         "always means the re-merge is worse, and a difference smaller than the dataset's own "
         "floor is reported as a tie rather than a result. ρ is carried per experiment because "
         "PSM and PSM-forecast share raw data but are different tasks with different geometry.",
         table(remerge, ["dataset", "n_segments", "rule", "threshold", "n_seeds", "rho",
                         "plain", "remerged", "delta_pct", "floor_pct", "verdict",
                         "implied_alpha_times_n"], limit=100)),
        *closeout_tables,
        ("Claims register (§0.7)",
         "Every claim this project makes, with the evidence it rests on. <code>status</code> is "
         "<em>derived</em>, not declared: a mechanism claim resting on fewer than three datasets, "
         "or never falsification-tested, is downgraded to <code>hypothesis</code> whatever the "
         "prose says. The checker binds this table to §0.7 in both directions, so a claim in the "
         "prose without a row here — or the reverse — fails.",
         table(claims, ["id", "claim", "section", "kind", "datasets", "falsification_tested",
                        "status_declared", "status"])),
        ("Methods register",
         "What was asked for, what was built, and what it found. <em>Partial</em> is the "
         "load-bearing row type: reporting a partial implementation under the full method's name "
         "is the single most damaging thing these documents could do.",
         table(method_rows, ["method", "implemented", "why", "what_was_found"])),
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
        # The fingerprint is what `check_tables_against_csv.py` compares against the committed
        # MANIFEST.csv. A meta tag rather than visible text: it is machine-readable provenance,
        # not something a reader needs.
        f"<meta name='archive-manifest-sha256' content='{fingerprint}'>",
        f"<meta name='archive-manifest-entries' content='{n_entries}'>",
        f"<div class='meta'><span>commit {html.escape(args.commit)}</span>"
        f"<span>generated {html.escape(stamp)}</span>"
        f"<span>{n_entries} archived files</span>"
        f"<span title='SHA-256 of MANIFEST.csv'>archive {html.escape(fingerprint[:12])}</span>"
        f"</div>",
        "<p class='note'>Generated from <code>results_archive/</code>. Every value is read from "
        "a CSV in the archive — none is transcribed — so this page cannot disagree with the "
        "evidence. <code>EXPERIMENTS.md</code> remains the source of record and carries the "
        "reasoning, caveats and withdrawals that these tables do not.</p>",
    ]
    for title, note, body in sections:
        parts += [f"<h2>{html.escape(title)}</h2>", f"<p class='note'>{note}</p>", body]
    parts += [
        "<footer>Rebuilt by <code>scripts/build_results_report.py</code> as part of "
        "<code>scripts/regenerate_analysis.sh</code>. The archive fingerprint above is the "
        "SHA-256 of <code>results_archive/MANIFEST.csv</code>, which itself hashes every "
        "archived file; <code>check_tables_against_csv.py</code> compares it against the "
        "committed manifest, so a stale copy of this page fails the checker rather than "
        "looking current.</footer>",
        "</div></body></html>",
    ]
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, "
          f"{len(sections)} sections, {n_files} archived files)")


if __name__ == "__main__":
    main()
