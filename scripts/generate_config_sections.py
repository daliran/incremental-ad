"""Emit EXPERIMENTS.md §2 "Exact configurations" subsections from archived `config.json`.

    python scripts/generate_config_sections.py --archive results_archive/runs \\
        --spec analysis_specs/config_sections.csv > /tmp/sections.md

**Why generated rather than written.** §2 is the record of *how* each result was produced, and
it is the one section whose loss cannot be repaired: the sweep generators build their commands
by reading `config.json` off `$WORK`, so when `$WORK` goes, an unwritten configuration is gone.
Hand-transcribing 60+ argument values per dataset is also precisely the mechanism behind every
documentation error this project has found — §2 was simply large enough that nobody had done it
for the newer groups.

Reads the archived configs, so it runs from the repo with no `$WORK`. Arguments are grouped by
their `ARG_PREFIX` and sorted, matching the existing hand-written subsections; where a companion
run (a joint/standard reference) differs on a value, the difference is annotated inline rather
than given its own table, which is the convention §2.1–§2.6 already use.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Arguments that say nothing about the experiment's identity.
SKIP = {"experiment_name", "seed", "runs_root", "wandb_project", "wandb_entity", "wandb_mode",
        "device", "runner_device", "loader_num_workers", "configurator_debug"}
# Argument prefixes, in the order §2.1-§2.6 present them.
GROUP_ORDER = ["mae_tx", "dataset", "trainer", "finetune_trainer", "loader", "pipeline"]
# Display label per prefix, so the generated tables read like the hand-written §2.1-§2.6.
GROUP_LABEL = {"mae_tx": "model"}


def load_one(archive: Path, experiment: str) -> tuple[str, dict] | None:
    """The first run of `experiment`, as (run_id, args). Configs within a group differ only in
    seed, which is skipped, so any one of them describes the group."""
    group = archive / experiment
    if not group.is_dir():
        return None
    for run in sorted(group.iterdir()):
        config = run / "config.json"
        if config.is_file():
            try:
                return run.name, json.loads(config.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return None


def group_of(key: str) -> tuple[str, str]:
    for prefix in sorted(GROUP_ORDER, key=len, reverse=True):
        if key.startswith(prefix + "_"):
            return prefix, key[len(prefix) + 1:]
    return "", key


def render(title: str, primary: str, companion: str | None, archive: Path) -> str | None:
    loaded = load_one(archive, primary)
    if loaded is None:
        return None
    run_id, config = loaded
    args = config.get("args") or {}
    other_args: dict = {}
    other_id = None
    if companion:
        other = load_one(archive, companion)
        if other is not None:
            other_id, other_config = other
            other_args = other_config.get("args") or {}

    header = f"`{primary}/{run_id}`"
    if other_id:
        header += f" · `{companion}/{other_id}` (reference)"
    header += (f" · model `{config.get('model')}` · dataset `{config.get('dataset')}`"
               f" · task `{config.get('task')}` · pipeline `{config.get('pipeline')}`")

    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(args):
        if key in SKIP or key in ("model", "dataset", "task", "pipeline"):
            continue
        value = args[key]
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue
        prefix, name = group_of(key)
        rendered = f"`{value}`"
        if other_args and key in other_args and other_args[key] != value:
            rendered += f" — reference run uses `{other_args[key]}`"
        buckets[prefix].append((name, rendered))

    lines = [f"### {title}", "", header, "", "| group | argument | value |", "|---|---|---|"]
    for prefix in GROUP_ORDER + sorted(set(buckets) - set(GROUP_ORDER)):
        rows = buckets.get(prefix)
        if not rows:
            continue
        for index, (name, value) in enumerate(rows):
            label = GROUP_LABEL.get(prefix, prefix) if index == 0 else ""
            lines.append(f"| {label} | `{name}` | {value} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, default=Path("results_archive/runs"))
    parser.add_argument("--spec", type=Path, default=Path("analysis_specs/config_sections.csv"),
                        help="CSV: title,experiment,companion — which groups to emit, in order")
    args = parser.parse_args()

    with args.spec.open() as fh:
        entries = list(csv.DictReader(fh))
    missing = []
    for entry in entries:
        section = render(entry["title"], entry["experiment"],
                         entry.get("companion") or None, args.archive)
        if section is None:
            missing.append(entry["experiment"])
            continue
        print(section)
    if missing:
        raise SystemExit(f"absent from the archive: {', '.join(missing)}")


if __name__ == "__main__":
    main()
