"""Emit the adaptive-λ sweep, reproducing each chain run of record's own configuration.

    python scripts/generate_adaptive_lambda_sweep.py --runs_root $RUNS_ROOT \\
        --tier 1 --out $WORK/sweeps

**Not BECAME** — the published method also runs a gradient-projection stage. This is
"adaptive-λ sequential fine-tuning", strategy 6 (CLAUDE.md scope note, EXPERIMENTS.md §1.39).

**Every argument comes from the source run's `config.json`, never from a hand-written list.**
Hand-picking cost three failed submissions: a `--dataset` value invented instead of read (the
registry keys on class names), the `--pipeline_*` prefix guessed as `--continual_*`, and seven
required `--trainer_*` flags dropped on the assumption that skipping base training made them
optional. This is the same discipline `analysis/diagnose` already applies — read the args back
out of the run rather than trusting a retyped copy.

Which flags are store_true is read off the parser's own `--help` rather than inferred from the
value's type: `--pipeline_anchor_to_baseline` is a bool that *requires* `true`/`false`, and
emitting it bare is rejected.

Each run is paired to the source run's **own baseline checkpoint**, so the only difference from
the chain of record is the pullback. That matters because GPU placement alone moves results by
up to 18.8% at a fixed seed (EXPERIMENTS.md §3.2).
"""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

FORECAST = {("ETTh1", "3"), ("ETTh2", "2"), ("ETTh2", "3"), ("ETTh2", "5"),
            ("ETTm2", "2"), ("ETTm2", "3"), ("ETTm2", "5"), ("exchange", "3")}
ALL_CONFIGS = FORECAST | {("SWaT", "3"), ("PSM", "3")}

# tier -> (lambda_source, extra flags, configs, experiment-name prefix)
TIERS = {
    1: ("became", [], ALL_CONFIGS, "adaptive_became"),
    2: ("one_over_t", [], ALL_CONFIGS, "adaptive_one_over_t"),
    # Tier 1 re-run with the corrected Fisher estimator. `diagonal_fisher` squares the gradient
    # of a batch-MEAN loss, so `E[F_hat] = g^2 + sigma^2/B`: at B = 1 the batch mean is the
    # per-sample gradient and the estimator is correct. The B-sweep
    # (`analysis/fisher_scaling_report`) fits the numerator at B^-0.9 and Lambda at B^-0.29,
    # and the published B = 128 suppressed lambda 6-24x at every step past the first.
    #
    # **Forecasting only.** On AD the bias largely self-cancels: random masking makes sigma^2
    # large enough that sigma^2/B plausibly dominates g^2 at the merged points too, so numerator
    # and denominator scale together -- which is why AD's Lambda/F sat at 0.1-11x while
    # forecasting's ran to thousands. That is an inference, not a measurement; AD stays published
    # at B = 128 with a note, and a B-sweep on PSM n=3 would settle it if it ever matters.
    #
    # N = 512, matching the acceptance runs. At B = 1 that agrees with N = 8192 to 1.4% on
    # ETTm2 n=3 seed 7 at t=1, where the two chains are identical by construction -- so the
    # sample count is not carrying anything here, exactly as `E[F_hat]`'s independence of K says.
    3: ("became", ["--pipeline_fisher_batch_size", "1", "--pipeline_fisher_batches", "512"],
        FORECAST, "fisherfix"),
}
SKIP = {"experiment", "run_id", "runs_root"}


def store_true_flags(args: dict) -> set[str]:
    """Flags the parser declares as taking no value, read from its own help text."""
    probe = ["--model", str(args["model"]), "--task", str(args["task"]),
             "--dataset", str(args["dataset"]), "--pipeline", str(args["pipeline"]), "--help"]
    text = subprocess.run([sys.executable, "-m", "incremental_ad.main", *probe],
                          capture_output=True, text=True, timeout=300).stdout
    takes_value = set(re.findall(r"--([a-z0-9_]+) [A-Z_]", text))
    return {f for f in re.findall(r"\[--([a-z0-9_]+)\]", text)} - takes_value


def render(args: dict, store_true: set[str]) -> str:
    parts = []
    for key in sorted(args):
        value = args[key]
        if key in SKIP or value is None:
            continue
        if key in store_true:
            if value:
                parts.append(f"--{key}")
        elif isinstance(value, list):
            parts.append(f"--{key} " + " ".join(shlex.quote(str(v)) for v in value))
        elif isinstance(value, bool):
            parts.append(f"--{key} {'true' if value else 'false'}")
        else:
            parts.append(f"--{key} {shlex.quote(str(value))}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--spec", type=Path,
                        default=Path("analysis_specs/method_comparison_spec.csv"))
    parser.add_argument("--tier", type=int, choices=sorted(TIERS), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source, extra, configs, prefix = TIERS[args.tier]
    with args.spec.open() as fh:
        spec = [r for r in csv.DictReader(fh) if (r["dataset"], r["n"]) in configs]

    commands, missing, cached = [], [], None
    for row in sorted(spec, key=lambda r: (r["dataset"], int(r["n"]))):
        group = args.runs_root / row["seq_experiment"]
        if not group.is_dir():
            missing.append(f"{row['dataset']} n={row['n']}: {row['seq_experiment']}")
            continue
        for run in sorted(p for p in group.iterdir() if p.is_dir()):
            checkpoint = run / "baseline" / "checkpoints" / "best.pt"
            if not checkpoint.is_file():
                continue
            config = json.loads((run / "config.json").read_text())
            config = config.get("args", config)
            if cached is None:
                cached = store_true_flags(config)
            name = f"{prefix}_{row['dataset'].lower()}_n{row['n']}_s{config['seed']}"
            commands.append(
                f"python -m incremental_ad.main {render(config, cached)} "
                f"--pipeline_baseline_checkpoint {checkpoint} "
                f"--pipeline_lambda_source {source} "
                + " ".join(extra) + f" --experiment {name}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"adaptive_tier{args.tier}.sh"
    path.write_text("\n".join(commands) + "\n" if commands else "")
    print(f"  {path.name:26s} {len(commands):>3} runs  (lambda_source={source})")
    for item in missing:
        print(f"  ⚠️  no chain run of record: {item}")


if __name__ == "__main__":
    main()
