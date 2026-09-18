"""Make a re-merge's output self-identifying, so a collector cannot read a bad one as good.

Every collector in this project reads results by globbing a directory. That works only while the
filesystem is a truthful record of what ran, and it is not: a job that dies mid-way, a job that
died so early it wrote nothing, and a job that completed against *older code* are all
indistinguishable from a successful current run once you are looking at `result.json` alone. The
82 jobs that died on one ImportError wrote nothing at all, which is the benign case — the
dangerous case is the one that writes something.

Three properties close that gap, and all three have to hold or the file is not read:

1. **`result.json` is the commit marker.** It is written last (after `became_lambdas.csv`) and
   written atomically, via a temporary file and `os.replace`. So its presence means every earlier
   step finished, and a job killed at any point leaves either no file or the previous complete
   one — never a half-written one.
2. **`schema` must match.** Bumped whenever the payload's meaning changes, so a result produced
   before a field existed is rejected rather than silently read as a missing value. `alpha_n_ok`
   would have read 0 for such a file and voided a row that was actually fine.
3. **`code_fingerprint` records the merging code that produced it** — a hash over the merging
   modules and the re-merge entry point. A mismatch does not invalidate a result by itself (a
   comment change moves the hash), so it is reported as a *warning with counts* rather than a
   rejection; what it rules out is a silent mix of two code versions in one table.

**Why `require_schema` exists, and where it is off.** The §1.35 sweep predates this module, so
every one of its 144 archived results is unversioned. Re-running them is not the answer: they are
*archived evidence*, their integrity is already established by `results_archive/MANIFEST.csv`
(SHA-256 per file) and their numbers are verified cell-by-cell by
`scripts/check_tables_against_csv.py`. That is a different provenance mechanism, not an absent
one. So the legacy source is loaded with `require_schema=False` and its unversioned count is
*printed* rather than silently tolerated, while everything written from now on — which has no
such archival backing yet — must carry the stamp. A schema that is present but WRONG is rejected
in both cases: that means the code moved, which the archive cannot vouch for.

`load_result` returns ``(payload, reason)`` with exactly one of them set, and the reason strings
are meant to be tallied and printed by the caller. A collector that drops a file without saying
so is the failure mode this module exists to prevent, so nothing here raises or skips quietly.
"""

import hashlib
import json
import os
import time
from pathlib import Path

# Bump when the payload's MEANING changes, not when a field is merely added downstream of it.
#   1: first versioned payload (adds schema/code_fingerprint/completed_at; the fields
#      implied_alpha_times_n, reverse_order and opcm_* are all present and meaningful)
SCHEMA = 1

_FINGERPRINT_SOURCES = (
    "framework/merging/task_vectors.py",
    "framework/merging/opcm.py",
    "framework/merging/became.py",
    "analysis/remerge.py",
)


def code_fingerprint() -> str:
    """Short hash over the merging code, so two tables cannot silently mix code versions."""
    package = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative in _FINGERPRINT_SOURCES:
        path = package / relative
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    return digest.hexdigest()[:16]


def write_result(out_dir: Path, payload: dict) -> Path:
    """Stamp and atomically commit ``result.json``. Call this LAST, after every sidecar file."""
    path = out_dir / "result.json"
    stamped = {**payload, "schema": SCHEMA, "code_fingerprint": code_fingerprint(),
               "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    temporary = out_dir / "result.json.tmp"
    temporary.write_text(json.dumps(stamped, indent=2))
    os.replace(temporary, path)          # atomic within a filesystem: no torn reads
    return path


def load_result(path: Path, require_metric: str | None = None,
                require_schema: bool = True) -> tuple[dict | None, str]:
    """``(payload, reason)``. Exactly one is set; ``reason`` is "" on success.

    ``require_metric`` is the fully-qualified metric the caller needs (e.g. ``test/forecast/mse``).
    A result whose evaluation failed still writes a payload — `remerge.py` logs the failure and
    carries on so the val numbers are not lost — so "the file exists" does not mean "the number
    exists", and the caller must be told which it got.
    """
    if not path.is_file():
        return None, "absent"
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    schema = payload.get("schema")
    if schema is None and require_schema:
        return None, "unversioned (written before provenance stamping; re-run it)"
    if schema is not None and schema != SCHEMA:
        return None, f"schema {schema} != {SCHEMA} (stale code; re-run it)"
    if require_metric is not None and (payload.get("metrics") or {}).get(require_metric) is None:
        return None, f"no {require_metric} (evaluation failed inside an otherwise finished run)"
    return payload, ""


def report_exclusions(log, reasons: dict[str, int], scanned: int) -> int:
    """Print why results were not used, and return how many were excluded. Never silent."""
    excluded = sum(reasons.values())
    if not excluded:
        log.info("provenance: %d/%d result(s) complete, current-schema and carrying the metric",
                 scanned, scanned)
        return 0
    log.warning("⚠️  provenance: %d of %d result(s) EXCLUDED — these are not in any table below:",
                excluded, scanned)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log.warning("      %4d  %s", count, reason)
    return excluded


def _self_test() -> int:
    """Prove the gate rejects each way a result can be bad. Returns failures.

    A provenance check that has never been seen to fail is indistinguishable from no check, and
    this one guards the thing that is hardest to notice — a table that is quietly built from the
    wrong files. Every rejection path gets a constructed example.
    """
    import tempfile

    good = {"metrics": {"test/forecast/mse": 0.42}, "schema": SCHEMA,
            "code_fingerprint": "deadbeef", "completed_at": "2026-09-18T00:00:00"}
    cases = [
        ("a complete current-schema result", good, True, True),
        ("a result from a crashed job (file never written)", None, False, True),
        ("a result whose evaluation failed (no test metric)",
         {**good, "metrics": {"val/forecast/mse": 0.3}}, False, True),
        ("a result written by older code (no schema)",
         {k: v for k, v in good.items() if k != "schema"}, False, True),
        ("that same legacy result, under require_schema=False",
         {k: v for k, v in good.items() if k != "schema"}, True, False),
        ("a result from a FUTURE/other schema", {**good, "schema": SCHEMA + 1}, False, False),
        ("a torn write (truncated JSON)", "{not json", False, True),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as directory:
        for label, content, should_load, require in cases:
            path = Path(directory) / "result.json"
            path.unlink(missing_ok=True)
            if content is not None:
                path.write_text(content if isinstance(content, str) else json.dumps(content))
            payload, reason = load_result(path, require_metric="test/forecast/mse",
                                          require_schema=require)
            loaded = payload is not None
            ok = loaded == should_load
            failures += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {label}: "
                  f"{'accepted' if loaded else f'rejected ({reason})'}")

    # The atomic write must leave no temporary behind, and must be readable immediately after.
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        write_result(out, {"metrics": {"test/forecast/mse": 0.1}})
        payload, reason = load_result(out / "result.json", require_metric="test/forecast/mse")
        leftovers = sorted(p.name for p in out.iterdir())
        ok = payload is not None and leftovers == ["result.json"]
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} atomic write leaves only result.json: {leftovers}")
    return failures


if __name__ == "__main__":
    import sys
    print("REMERGE PROVENANCE SELF-TEST")
    sys.exit(1 if _self_test() else 0)
