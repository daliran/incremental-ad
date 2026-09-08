"""Inventory the checkpoints worth preserving, so an off-cluster copy can be verified.

    srun ... python scripts/checkpoint_manifest.py --runs_root $RUNS_ROOT \\
        --out results_archive/CHECKPOINTS.md

`results_archive/` deliberately excludes `*.pt` — gigabytes, reproducible from `config.json`.
That is the right call for the *evidence*, but it means no post-hoc analysis that needs weights
(geometry, Fisher, re-merge, BECAME at a larger sample) is possible once `$WORK` is purged.
The checkpoints therefore have to leave the cluster, and this writes the record that makes the
copy checkable afterwards: what to copy, how large it is, and a SHA-256 per file.

Scope is `analysis_specs/experiment_of_record.csv` — the experiments the documents actually read
— plus the groups added since it was written (`opcm2_`, `window_`, `origin_`, `basefrac_`,
`selalpha_`, `n1_`). Everything else under `$RUNS_ROOT` is either superseded, a diagnostics run
whose CSVs are already archived, or a sweep trial.

The hashing dominates the runtime (it reads every byte), which is why this is a separate,
sbatch-ed step rather than part of `archive_results.py`.
"""

import argparse
import csv
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("checkpoint_manifest")

EXTRA_PREFIXES = ("opcm2_", "window_", "origin_", "basefrac_", "selalpha_", "n1_",
                  "aeft_", "adfc2_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/experiment_of_record.csv"))
    parser.add_argument("--out", type=Path, default=Path("results_archive/CHECKPOINTS.md"))
    parser.add_argument("--no_hash", action="store_true",
                        help="list sizes only. Faster, but the result cannot verify a copy — "
                             "which is the whole purpose, so this is for a dry run only.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with args.spec.open() as fh:
        of_record = {r["experiment"] for r in csv.DictReader(fh)}
    groups = sorted(
        d.name for d in args.runs_root.iterdir()
        if d.is_dir() and (d.name in of_record or d.name.startswith(EXTRA_PREFIXES))
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
        fh.write(f"**Scope.** {len(groups)} experiment groups — the "
                 f"{len(of_record)} in `analysis_specs/experiment_of_record.csv` plus the "
                 f"`opcm2_`/`window_`/`origin_`/`basefrac_`/`selalpha_`/`n1_` groups added "
                 f"since. **{len(rows)} files, {total / 1e9:.1f} GB.**\n\n")
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
