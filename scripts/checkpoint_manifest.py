"""Inventory the checkpoints worth preserving, so an off-cluster copy can be verified.

    srun ... python scripts/checkpoint_manifest.py --runs_root $RUNS_ROOT \\
        --out results_archive/CHECKPOINTS.md

`results_archive/` deliberately excludes `*.pt` — gigabytes, reproducible from `config.json`.
That is the right call for the *evidence*, but it means no post-hoc analysis that needs weights
(geometry, Fisher, re-merge, BECAME at a larger sample) is possible once `$WORK` is purged.
The checkpoints therefore have to leave the cluster, and this writes the record that makes the
copy checkable afterwards: what to copy, how large it is, and a SHA-256 per file.

**Scope is derived, not listed.** Every experiment named by *any* file in `analysis_specs/`,
plus every experiment `regenerate_analysis.sh` globs by name, plus the `EXTRA_PREFIXES` families.
Anything else under `$RUNS_ROOT` is superseded, a sweep trial, or a diagnostics run whose CSVs are
already archived and which owns no checkpoints of its own.

⚠️ **The earlier version scoped on `experiment_of_record.csv` plus prefixes, and missed ten
groups holding 0.45 GB.** Three separate causes, none visible from the prefix list:
`etth2_window_W3` does not start with `window_` (the ETT groups put the dataset first);
`etth2_gate_base` is named only in `floor_spec.csv`; and `exch_incremental` / `noisefloor_etth`
*were* experiments of record until §1.29 replaced them with the `selalpha_*` re-runs — so editing
that one spec silently narrowed the backup. Deriving the scope from every spec removes all three
failure modes at once, and means adding a spec entry cannot leave its checkpoints unbacked.

The hashing dominates the runtime (it reads every byte), which is why this is a separate,
sbatch-ed step rather than part of `archive_results.py`.
"""

import argparse
import csv
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("checkpoint_manifest")

EXTRA_PREFIXES = ("opcm2_", "window_", "origin_", "basefrac_", "selalpha_", "n1_",
                  "aeft_", "adfc2_")


def referenced_experiments(spec_dir: Path, regenerate_script: Path) -> set[str]:
    """Every experiment name any spec or the regeneration script refers to.

    Reading the specs rather than one hand-kept list is what makes the backup follow the
    documents automatically: a spec entry added tomorrow brings its checkpoints into scope
    without anyone remembering to widen a prefix.
    """
    names: set[str] = set()
    for path in sorted(spec_dir.glob("*.csv")):
        with path.open() as fh:
            for row in csv.DictReader(fh):
                for key, value in row.items():
                    if not value or "experiment" not in key:
                        continue
                    if not value.replace(".", "").isdigit():
                        names.add(value)
    if regenerate_script.is_file():
        # Brace expansions like `{etth2,ettm2}_merge_n{2,3,5}_diagnostics` are expanded here so
        # the script's own globs count as references.
        text = regenerate_script.read_text()
        for match in re.findall(r'\$RUNS"?/([A-Za-z0-9_{},]+)', text):
            stack = [match]
            while stack:
                item = stack.pop()
                if "{" not in item:
                    names.add(item)
                    continue
                head, rest = item.split("{", 1)
                options, tail = rest.split("}", 1)
                stack.extend(f"{head}{o}{tail}" for o in options.split(","))
    return names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec_dir", type=Path, default=Path("analysis_specs"),
                        help="every CSV here is scanned for experiment names, so the backup "
                             "follows the documents rather than a hand-kept list")
    parser.add_argument("--regenerate_script", type=Path,
                        default=Path("scripts/regenerate_analysis.sh"))
    parser.add_argument("--out", type=Path, default=Path("results_archive/CHECKPOINTS.md"))
    parser.add_argument("--no_hash", action="store_true",
                        help="list sizes only. Faster, but the result cannot verify a copy — "
                             "which is the whole purpose, so this is for a dry run only.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    referenced = referenced_experiments(args.spec_dir, args.regenerate_script)
    groups = sorted(
        d.name for d in args.runs_root.iterdir()
        if d.is_dir() and (d.name in referenced or d.name.startswith(EXTRA_PREFIXES))
        # Diagnostics runs hold no checkpoints of their own — they read the source run's — and
        # their CSVs are already in results_archive/run_diagnostics.
        and any(d.rglob("*.pt"))
    )

    rows: list[tuple[str, str, int, str]] = []
    per_group: dict[str, list[int]] = defaultdict(list)
    for group in groups:
        for checkpoint in sorted((args.runs_root / group).rglob("*.pt")):
            size = checkpoint.stat().st_size
            relative = checkpoint.relative_to(args.runs_root).as_posix()
            rows.append((group, relative, size,
                         "" if args.no_hash else sha256(checkpoint)))
            per_group[group].append(size)
        log.info("  %-40s %3d checkpoints  %6.2f GB", group, len(per_group[group]),
                 sum(per_group[group]) / 1e9)

    total = sum(size for _, _, size, _ in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("# Checkpoints — what to preserve, and how to verify a copy\n\n")
        fh.write(
            "`results_archive/` holds the *evidence* (CSVs, `config.json`, every "
            "`result.json`) and deliberately excludes `*.pt`: they are gigabytes and "
            "reproducible from the configs. But re-merging, geometry, Fisher and any BECAME "
            "resampling need the weights, and `$WORK` is scratch. This is the record that "
            "makes an off-cluster copy verifiable.\n\n"
        )
        fh.write(f"**Scope.** {len(groups)} experiment groups that own checkpoints — every "
                 f"experiment named by any file in `analysis_specs/` or globbed by "
                 f"`regenerate_analysis.sh`, plus the `opcm2_`/`window_`/`origin_`/`basefrac_`/"
                 f"`selalpha_`/`n1_`/`aeft_`/`adfc2_` families. Derived, not listed, so a new "
                 f"spec entry cannot leave its checkpoints unbacked. "
                 f"**{len(rows)} files, {total / 1e9:.1f} GB.**\n\n")
        fh.write("## Copying\n\n```bash\n")
        fh.write(f"# from the cluster, one group per line so a partial copy is resumable\n")
        fh.write(f"rsync -av --info=progress2 \\\n")
        for group in groups[:3]:
            fh.write(f"  {args.runs_root}/{group} \\\n")
        fh.write(f"  ...  # all {len(groups)} groups, listed below\n")
        fh.write(f"  <destination>/incremental-ad-checkpoints/\n```\n\n")
        fh.write("## Verifying afterwards\n\n```bash\n")
        fh.write("python - <<'EOF'\n"
                 "import csv, hashlib\n"
                 "from pathlib import Path\n"
                 "root = Path('<destination>/incremental-ad-checkpoints')\n"
                 "bad = 0\n"
                 "for row in csv.DictReader(open('results_archive/checkpoints.csv')):\n"
                 "    p = root / row['path']\n"
                 "    if not p.exists():\n"
                 "        print('MISSING', row['path']); bad += 1; continue\n"
                 "    h = hashlib.sha256(p.read_bytes()).hexdigest()\n"
                 "    if h != row['sha256']:\n"
                 "        print('CORRUPT', row['path']); bad += 1\n"
                 "print(f'{bad} problem(s)')\n"
                 "EOF\n```\n\n")
        fh.write("## Groups\n\n| experiment | checkpoints | size |\n|---|---|---|\n")
        for group in groups:
            fh.write(f"| `{group}` | {len(per_group[group])} | "
                     f"{sum(per_group[group]) / 1e9:.2f} GB |\n")
        fh.write(f"\nPer-file sizes and SHA-256 are in `results_archive/checkpoints.csv` "
                 f"({len(rows)} rows) — kept as CSV rather than inlined here because a "
                 f"{len(rows)}-row table in Markdown is not readable and not greppable.\n")

    csv_path = args.out.parent / "checkpoints.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "path", "bytes", "sha256"])
        writer.writerows(rows)

    log.info("\n%d checkpoints, %.1f GB across %d groups", len(rows), total / 1e9, len(groups))
    log.info("wrote %s and %s", args.out, csv_path)


if __name__ == "__main__":
    main()
