"""Emit §1.44's AD headroom screen: SMD and ETTh2-injected, base and joint, seed 42.

    python scripts/generate_ad_screen.py --runs_root $RUNS_ROOT --out $WORK/sweeps

Four runs, written as commands and NOT submitted (§1.44: the screen is launched by hand). Every
argument is copied from PSM's own recorded configuration, so the only differences from PSM are
the dataset, the seed and the experiment name:

- **base** — PSM's incremental run config (`segsweep_psm_merge_n2`) with 0 fine-tune segments,
  i.e. `baseline/test` is a model trained on the first half: the `*_gate_base` pattern §2.18 used.
- **joint** — PSM's joint run config (`noisefloor_std_psm`), baseline_fraction 1.0.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_adaptive_lambda_sweep import render, store_true_flags  # noqa: E402

CANDIDATES = {"smd": "Smd", "etth2inj": "Etth2Injected"}
SOURCES = {"base": "segsweep_psm_merge_n2", "joint": "noisefloor_std_psm"}
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    commands = []
    for role, experiment in SOURCES.items():
        source = sorted((args.runs_root / experiment).glob("*/config.json"))[0]
        config = json.loads(source.read_text())["args"]
        flags = store_true_flags(config)
        for tag, dataset in CANDIDATES.items():
            run = {**config, "dataset": dataset, "seed": SEED, "eval_seed": None}
            if role == "base":
                run.update({"dataset_baseline_fraction": 0.5,
                            "dataset_n_finetune_segments": 0})
            else:
                assert run["dataset_baseline_fraction"] == 1.0, "joint must train on all data"
            commands.append(f"python -m incremental_ad.main {render(run, flags)} "
                            f"--experiment adscreen_{tag}_{role}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ad_screen.sh"
    path.write_text("\n".join(commands) + "\n")
    print(f"  {path}  {len(commands)} runs (NOT submitted)")


if __name__ == "__main__":
    main()
