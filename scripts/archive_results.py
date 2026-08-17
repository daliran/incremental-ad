"""Copy the run artefacts every published number depends on into the repo.

    python scripts/archive_results.py --runs_root $RUNS_ROOT \\
        --audit_dir $OUT --geometry_root $OUT/geometry

**Why this exists.** Every number in EXPERIMENTS.md is checked against a CSV, but those CSVs
live under `$WORK`, which is scratch: not backed up, and purged on a schedule nobody controls.
If it goes, the documents become a set of assertions again — the exact state this project spent
four audit rounds climbing out of. The inputs are small (a few MB of CSV); the models are not
(11 GB), so this archives the *evidence*, not the artefacts that produced it.

What is copied:

- **audit CSVs** — `derived.csv`, `run_metrics.csv`, and every summary written by the analysis
  entry points (`scale_summary.csv`, `routing_summary.csv`, `method_comparison.csv`, …).
- **geometry** — `geometry_summary.csv` plus the small per-run files that back EXPERIMENTS.md
  1.7 and 1.8 (`sequential_overlap`, `norms`, `cosine_vs_distance`, `effective_rank`,
  `cosine_matrix`).
- **per-run diagnostics** — `transfer_matrix.csv` and `merge_scale_curve.csv` for every
  `*_diagnostics` run, which is what 1.3 and 1.4 are checked against cell by cell.

What is NOT copied, and why: checkpoints (`*.pt`) and `wandb/` are gigabytes and reproducible
from the run config; `principal_angles.csv`, `per_tensor_cosine.csv` and `geometry.json` are
33 MB combined and no published number reads them; raw datasets come from HuggingFace.

The archive is laid out so the checker runs against it directly, with no `$WORK` at all:

    python scripts/check_tables_against_csv.py \\
        --audit_dir results_archive/audit \\
        --runs_root results_archive/run_diagnostics --strict

`MANIFEST.csv` records every file with its size and SHA-256, so a later reader can tell whether
the archive is intact and when each part was captured.
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

# Per-run geometry files worth keeping: small, and each backs a documented table.
GEOMETRY_FILES = ("sequential_overlap.csv", "norms.csv", "cosine_vs_distance.csv",
                  "effective_rank.csv", "cosine_matrix.csv")
# Per-run diagnostics files the transfer-matrix and merge-scale-curve checks read.
DIAGNOSTIC_FILES = ("transfer_matrix.csv", "merge_scale_curve.csv", "source.json")
# A cap that catches "someone pointed this at the checkpoints directory" rather than a real
# limit; the archive is expected to land around 4 MB.
SIZE_WARN_MB = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(src: Path, dst: Path, names: tuple[str, ...] | None) -> list[Path]:
    """Copy `src` into `dst`, keeping only `names` (all files when None). Returns what landed."""
    written = []
    if not src.exists():
        return written
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if names is not None and path.name not in names:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True,
                        help="directory holding derived.csv / run_metrics.csv and the "
                             "per-tool summary subdirectories")
    parser.add_argument("--geometry_root", type=Path,
                        help="geometry_report output root (geometry_summary.csv + per-run dirs)")
    parser.add_argument("--out", type=Path, default=Path("results_archive"))
    args = parser.parse_args()

    written: list[Path] = []

    # 1. Everything the analysis entry points emit. These are already summaries, so take them
    #    wholesale rather than filtering by name — a new tool's output should land here without
    #    this script needing to know about it.
    written += copy_tree(args.audit_dir, args.out / "audit", None)

    # 2. Geometry: the summary plus the small per-run files.
    if args.geometry_root:
        summary = args.geometry_root / "geometry_summary.csv"
        if summary.exists():
            target = args.out / "geometry" / "geometry_summary.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary, target)
            written.append(target)
        written += copy_tree(args.geometry_root, args.out / "geometry", GEOMETRY_FILES)

    # 3. Per-run diagnostics, mirroring <group>/<run>/merge_diagnostics/ so that
    #    --runs_root can point straight at the archive.
    for group in sorted(args.runs_root.glob("*_diagnostics")):
        for run in sorted(p for p in group.iterdir() if p.is_dir()):
            source = run / "merge_diagnostics"
            if not source.exists():
                continue
            written += copy_tree(source, args.out / "run_diagnostics" / group.name / run.name
                                 / "merge_diagnostics", DIAGNOSTIC_FILES)

    if not written:
        raise SystemExit("nothing copied — check --runs_root and --audit_dir")

    total = sum(p.stat().st_size for p in written)
    # The manifest describes the **whole archive**, not just what this invocation copied.
    # It used to list `written` only, so an incremental re-archive — say, one that regenerates
    # the audit CSVs but is pointed at a geometry root holding only the two summary files —
    # silently shrank the integrity record from 1,443 entries to 373 while leaving 1,101 real
    # files on disk unlisted. An integrity record that quietly stops covering most of the
    # archive is worse than none, because it still reports "0 problems".
    everything = sorted(p for p in args.out.rglob("*")
                        if p.is_file() and p.name != "MANIFEST.csv")
    manifest = args.out / "MANIFEST.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "bytes", "sha256"])
        for path in everything:
            writer.writerow([path.relative_to(args.out).as_posix(), path.stat().st_size,
                             _sha256(path)])
    carried = len(everything) - len(written)

    print(f"archived {len(written)} files, {total / 1e6:.1f} MB -> {args.out}")
    for section in ("audit", "geometry", "run_diagnostics"):
        files = [p for p in written if (args.out / section) in p.parents]
        if files:
            size = sum(p.stat().st_size for p in files)
            print(f"  {section:16s} {len(files):>4d} files  {size / 1e6:>6.2f} MB")
    print(f"  MANIFEST.csv     {len(everything)} entries with SHA-256 "
          f"({len(written)} written now, {carried} already present)")
    if carried:
        print(f"  ℹ️  {carried} file(s) were left from an earlier archive run and are still "
              f"listed. That is correct for inputs this invocation did not regenerate "
              f"(e.g. per-run geometry when --geometry_root holds only the summaries), and "
              f"WRONG if their source runs changed — re-archive with the full inputs if so.")
    if total / 1e6 > SIZE_WARN_MB:
        print(f"\n⚠️  {total / 1e6:.0f} MB is far above the expected ~4 MB — check the inputs "
              f"before committing this.")


if __name__ == "__main__":
    main()
