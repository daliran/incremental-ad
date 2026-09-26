"""Verify an off-cluster copy of the checkpoints against `results_archive/checkpoints.csv`.

    python scripts/verify_checkpoint_copy.py --copy_root /media/disk/incremental-ad-runs
    python scripts/verify_checkpoint_copy.py --copy_root ./restored --inventory checkpoints.csv

Standard library only, so it runs on any machine with Python 3.9+, without this repository's
environment. `--copy_root` is the directory that plays the role of `$RUNS_ROOT`: the inventory's
`path` column is relative to it (`<experiment>/<run_id>/<step>/checkpoints/best.pt`).

Every inventoried file is checked for presence, size and SHA-256. Files under the copy that the
inventory does not list are reported as extra (informational: a copy may legitimately hold more).
Exit status is non-zero if any inventoried file is missing or differs, so it can gate a script.

Copy the checkpoints with their directory structure, e.g.
    rsync -a --prune-empty-dirs --include='*/' --include='*.pt' --exclude='*' \\
        $RUNS_ROOT/ /media/disk/incremental-ad-runs/
(that copies every checkpoint, a superset of the inventory, which this script then verifies).
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--copy_root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "results_archive" / "checkpoints.csv")
    parser.add_argument("--quiet", action="store_true", help="print only problems and totals")
    args = parser.parse_args()
    if not args.copy_root.is_dir():
        sys.exit(f"--copy_root {args.copy_root} is not a directory")
    with args.inventory.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{args.inventory} lists no files")

    missing, wrong_size, wrong_hash, ok = [], [], [], 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        path = args.copy_root / row["path"]
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != int(row["bytes"]):
            wrong_size.append(row["path"])
        elif sha256(path) != row["sha256"]:
            wrong_hash.append(row["path"])
        else:
            ok += 1
        if not args.quiet and i % 250 == 0:
            print(f"  {i}/{total} checked")
    listed = {row["path"] for row in rows}
    extra = [p.relative_to(args.copy_root).as_posix() for p in args.copy_root.rglob("*.pt")
             if p.relative_to(args.copy_root).as_posix() not in listed]

    for label, items in (("MISSING", missing), ("SIZE DIFFERS", wrong_size),
                         ("SHA-256 DIFFERS", wrong_hash)):
        for item in items:
            print(f"  {label:16s} {item}")
    print(f"\n{ok}/{total} inventoried checkpoints verified "
          f"({len(missing)} missing, {len(wrong_size)} wrong size, {len(wrong_hash)} wrong hash); "
          f"{len(extra)} extra .pt file(s) not in the inventory")
    if missing or wrong_size or wrong_hash:
        sys.exit(1)
    print("copy verified: every inventoried checkpoint is present and bit-identical")


if __name__ == "__main__":
    main()
