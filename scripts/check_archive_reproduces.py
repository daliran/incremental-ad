"""Does `regenerate_analysis.sh` actually reproduce the archive it claims to?

    bash scripts/regenerate_analysis.sh $WORK/audit_fresh $RUNS_ROOT
    python scripts/check_archive_reproduces.py --fresh $WORK/audit_fresh \\
        --archive results_archive/audit --strict

**The gap this closes.** `check_tables_against_csv.py` verifies that numbers in the documents
match the CSVs. **Nothing verified that the CSVs themselves regenerate.** Those are different
properties, and the difference is not academic: `remerge_closeout.csv` was *correct* for months
while the pipeline that claims to produce it emitted **24 of its 241 rows**, because
`--remerge_dir` pointed at the directory the report writes to rather than the per-run tree it
reads. Every check passed throughout, because the 24 rows it did produce were a correct subset.

That failure mode — a right file with a broken pipeline — is invisible to every other check in
this repo, and it has now been hit three times in one day in three different reports. The fix
each time was a comment at the call site. **Comments have failed twice**, so this is a check.

**What it asserts.** Every file the archive holds that regeneration is supposed to produce must
be byte-identical to what a fresh run just produced. Files the archive holds that regeneration
does NOT produce are reported separately and must be *declared*: they are GPU-only trees carried
forward, and an undeclared one is a file with no generator, which is how the archive silently
stopped being reproducible before.

`--checked_only` narrows the failure set to the files `check_tables_against_csv.py` actually
reads, which is the set a wrong number could hide in.
"""

import argparse
import filecmp
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("check_archive_reproduces")

# Trees that are GPU-produced and carried forward rather than regenerated. This list must match
# the carry loop in regenerate_analysis.sh; anything outside it is expected to reproduce.
CARRIED = {
    "oracle_router", "concentration", "novelty_swat", "selection_probe", "drift", "geometry",
    "novelty", "alignment", "subblocks", "mask_span", "window_selection", "remerge",
    "remerge_sweep", "remerge_closeout_runs", "geometry_gap", "geometry_aeft",
    "fisher_scaling_sweep", "subblocks_origin", "merge_baselines_runs", "merge_sensitivity_runs",
    "merge_calibration_runs", "prequential_runs",
    "grid_search",
}
# Individual files with no generator in the regeneration script, each produced by a separate
# deliberate step (a GPU job, or a verification run). Named so "no generator" is a declaration
# rather than an accident.
CARRIED_FILES = {
    "checkpoint_verification.csv": "scripts/verify_merge_reproduction.py (loads checkpoints)",
    "merge_reproduction.csv": "scripts/verify_merge_reproduction.py (loads checkpoints)",
    "outcomes.csv": "analysis/novelty_report.py outcomes (checkpoint reader)",
    "verification_log.md": "written by hand from the verification runs",
}
# Files INSIDE a carried directory that the regeneration script rebuilds from it — compared, not
# counted as carried. Both had no generator until 2026-09-25.
REGENERATED_INSIDE_CARRIED = {"oracle_router/oracle_router_summary.csv",
                              "remerge/fisher_sweep_summary.csv"}
# Files whose content legitimately changes on every run.
VOLATILE = {"unscoped_universals.csv"}


FLOAT_RTOL = 1e-9     # GPU reductions differ in the last digits between runs (1.7e-13 measured)


def numerically_equal(a: Path, b: Path) -> bool:
    """Same CSV shape and text cells, numbers equal within FLOAT_RTOL. Rows compared in order,
    then as a sorted multiset (a regeneration may list the same runs in a different order)."""
    import csv
    import math

    def rows(path):
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.reader(fh))

    def same(x, y):
        if len(x) != len(y):
            return False
        for u, v in zip(x, y):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return False
            if not math.isclose(fu, fv, rel_tol=FLOAT_RTOL, abs_tol=1e-15):
                return False
        return True

    ra, rb = rows(a), rows(b)
    if len(ra) != len(rb) or (ra and ra[0] != rb[0]):
        return False
    if all(same(x, y) for x, y in zip(ra, rb)):
        return True
    return all(same(x, y) for x, y in zip(sorted(ra[1:]), sorted(rb[1:])))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def checked_relatives(checker: Path) -> set[str]:
    """The audit-relative CSV paths `check_tables_against_csv.py` actually reads.

    Imported rather than grepped. A first version matched every `"*.csv"` string literal in the
    file and flagged five paths that are not audit files at all — per-run artefacts read from
    `runs_root` (`transfer_matrix.csv`, `merge_scale_curve.csv`), a spec (`analysis_specs/…`)
    and the archive-root manifest. Reading the CHECKS list gives exactly the `rel` values the
    checker resolves against `--audit_dir`, and nothing else.
    """
    spec = importlib.util.spec_from_file_location("_checker", checker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # builds CHECKS; main() runs only under __main__
    return {rel for _label, _pattern, rel, *_rest in module.CHECKS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fresh", type=Path, required=True,
                        help="the directory regenerate_analysis.sh just wrote")
    parser.add_argument("--archive", type=Path, default=Path("results_archive/audit"))
    parser.add_argument("--checker", type=Path,
                        default=Path("scripts/check_tables_against_csv.py"))
    parser.add_argument("--checked_only", action="store_true",
                        help="fail only on files the checker reads")
    parser.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    read_by_checker = checked_relatives(args.checker)
    # Which carried-type directories the regeneration actually COPIED from the archive. Anything
    # else under a carried-type directory was produced by this run (a GPU regeneration) and is
    # compared rather than skipped. Absent marker = the old behaviour (treat all as carried).
    marker = Path(str(args.fresh).rstrip("/") + ".carried")
    copied = set(marker.read_text().split()) if marker.is_file() else None
    archived = {p.relative_to(args.archive).as_posix()
                for p in args.archive.rglob("*") if p.is_file()}

    reproduced, differ, missing, carried, volatile, within_tol = [], [], [], [], [], []
    for rel in sorted(archived):
        top = rel.split("/")[0]
        # Decided by the top-level entry, not the file: a file missing from a directory this run
        # DID produce is a generator that stopped emitting it -- NOT PRODUCED, never "carried".
        produced_here = copied is not None and top not in copied and (args.fresh / top).exists()
        if ((top in CARRIED or rel in CARRIED_FILES) and rel not in REGENERATED_INSIDE_CARRIED
                and not produced_here):
            carried.append(rel)
            continue
        if Path(rel).name in VOLATILE:
            volatile.append(rel)
            continue
        fresh = args.fresh / rel
        if not fresh.is_file():
            missing.append(rel)
        elif filecmp.cmp(fresh, args.archive / rel, shallow=False):
            reproduced.append(rel)
        elif rel.endswith(".csv") and numerically_equal(fresh, args.archive / rel):
            within_tol.append(rel)
        else:
            differ.append(rel)

    log.info("REGENERATION vs ARCHIVE — %d archived file(s)", len(archived))
    log.info("  reproduced byte-for-byte : %d", len(reproduced))
    log.info("  reproduced within %.0e   : %d  (GPU float noise; same rows, same text)",
             FLOAT_RTOL, len(within_tol))
    log.info("  carried (declared)       : %d", len(carried))
    log.info("  volatile (declared)      : %d", len(volatile))
    log.info("  DIFFER                   : %d", len(differ))
    log.info("  NOT PRODUCED             : %d", len(missing))

    failures = 0
    for rel in differ:
        flag = "  <-- READ BY THE CHECKER" if rel in read_by_checker else ""
        log.warning("  DIFFERS      %s (archive %s, fresh %s)%s", rel,
                    digest(args.archive / rel), digest(args.fresh / rel), flag)
        if rel in read_by_checker or not args.checked_only:
            failures += 1
    for rel in missing:
        flag = "  <-- READ BY THE CHECKER" if rel in read_by_checker else ""
        log.warning("  NOT PRODUCED %s%s", rel, flag)
        if rel in read_by_checker or not args.checked_only:
            failures += 1

    # ⚠️ The honest limit of this guard. A carried file is one this run did not regenerate, so
    # its pipeline was never exercised — which is precisely the state `remerge_closeout.csv` was
    # in when it silently produced 24 of 241 rows. Carrying is legitimate (these need a GPU), but
    # "declared" is not the same as "checked", and a file that backs a published number while
    # its generator goes unrun is worth seeing every time rather than being bucketed as fine.
    unexercised = sorted(rel for rel in carried if rel in read_by_checker)
    if unexercised:
        log.warning("\n⚠️  %d of the %d CSVs the checker reads were CARRIED, not regenerated — "
                    "their generators are not exercised by this run:",
                    len(unexercised), len(read_by_checker))
        for rel in unexercised:
            log.warning("       %s", rel)
        log.warning("    Regenerate them with WITH_GEOMETRY=1 on a GPU node to cover these too.")

    # The inverse direction: a file the checker reads that is not in the archive at all means the
    # archive-only invariant cannot hold, whatever regeneration does.
    for rel in sorted(read_by_checker - archived):
        log.warning("  CHECKER READS a path the archive does not hold: %s", rel)
        failures += 1

    if failures:
        log.warning("\n%d failure(s). A file that is CORRECT but does not regenerate is the "
                    "case this check exists for — every other check in the repo passes on it.",
                    failures)
    else:
        if within_tol:
            for rel in within_tol:
                log.info("  WITHIN %.0e %s", FLOAT_RTOL, rel)
            log.info("\nEvery regenerable archived file reproduces: %d byte-for-byte, %d within "
                     "%.0e (GPU float noise).", len(reproduced), len(within_tol), FLOAT_RTOL)
        else:
            log.info("\nEvery regenerable archived file reproduces byte-for-byte.")
    if args.strict and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
