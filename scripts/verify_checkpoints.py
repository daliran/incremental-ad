"""Verify every checkpoint against the hashes recorded in `results_archive/checkpoints.csv`.

    srun ... python scripts/verify_checkpoints.py --runs_root $RUNS_ROOT

`checkpoint_manifest.py` writes a SHA-256 per checkpoint so an off-cluster copy can be checked.
That verification had **never been run against the source**, only against a destination — so the
manifest was trusted to describe the files it was generated from, which is an assumption, not a
check. If a checkpoint changed or was rewritten after the manifest was built, every re-merge
reading it would be working from something the manifest does not describe.

Reads every byte, so it is slow and belongs in a job rather than on a login node. Exits non-zero
on any mismatch or missing file, so it can gate work that depends on the checkpoints.
"""

import argparse
import csv
import hashlib
import logging
from pathlib import Path

log = logging.getLogger("verify_checkpoints")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("results_archive/checkpoints.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results_archive/audit/checkpoint_verification.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with args.manifest.open() as fh:
        entries = list(csv.DictReader(fh))
    log.info("verifying %d checkpoints against %s", len(entries), args.manifest)

    rows, missing, corrupt = [], 0, 0
    for index, entry in enumerate(entries, 1):
        path = args.runs_root / entry["path"]
        if not path.is_file():
            rows.append({**{k: entry[k] for k in ("experiment", "path", "bytes")},
                         "status": "MISSING", "actual_sha256": ""})
            missing += 1
            continue
        actual = sha256(path)
        ok = actual == entry["sha256"]
        if not ok:
            corrupt += 1
        rows.append({"experiment": entry["experiment"], "path": entry["path"],
                     "bytes": entry["bytes"],
                     "status": "ok" if ok else "MISMATCH",
                     "actual_sha256": "" if ok else actual})
        if index % 500 == 0:
            log.info("  %d/%d checked", index, len(entries))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment", "path", "bytes", "status",
                                                "actual_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    verified = len(rows) - missing - corrupt
    log.info("\n%d/%d verified  ·  %d missing  ·  %d mismatched", verified, len(rows),
             missing, corrupt)
    for row in rows:
        if row["status"] != "ok":
            log.info("  %-10s %s", row["status"], row["path"])
    log.info("wrote %s", args.out)
    raise SystemExit(1 if (missing or corrupt) else 0)


if __name__ == "__main__":
    main()
