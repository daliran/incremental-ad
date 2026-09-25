"""§1.44's AD headroom screen: is there a candidate AD dataset where merging has room to matter?

    python -m incremental_ad.analysis.ad_screen_report --runs_root $RUNS_ROOT --out $OUT/ad_screen

Reads the four screen runs (`adscreen_{smd,etth2inj}_{base,joint}`, seed 42) and, for reference,
PSM's and SWaT's own base and joint at seed 42, and applies the decision rule registered in §1.44
before any run:

    headroom = (AUROC_joint − AUROC_base) / (1 − AUROC_base)      — window AUROC, on test
    full three-seed sweep only if headroom >= 10 %

PSM and SWaT are recomputed here under THIS definition (§0.1b's "headroom" column uses a
different one), so the threshold is compared like with like. Drift is `drift_screen`'s own 3-way
and 5-way statistic on the training portion. ETTh2-injected's drift is ETTh2's: injection touches
the test portion only.

⚠️ SMD as distributed is 28 machines concatenated; ETTh2-injected is a controlled test, not a
benchmark. Both caveats are carried into the CSV's `note` column.
"""

import argparse
import csv
import json
from pathlib import Path

RULE_THRESHOLD = 0.10
SEED = 42
FIELDS = ["dataset", "base_run", "joint_run", "auroc_base", "auroc_joint", "headroom",
          "passes_rule", "drift_3way", "drift_5way", "note"]
SCREEN = {
    "SMD": ("adscreen_smd_base", "adscreen_smd_joint",
            "28 machines concatenated; periods can span machine boundaries"),
    "ETTh2-injected": ("adscreen_etth2inj_base", "adscreen_etth2inj_joint",
                       "controlled test, not a benchmark: synthetic anomalies (§1.44 protocol)"),
    # References, recomputed under the same definition at the same seed.
    "PSM": ("segsweep_psm_merge_n2", "noisefloor_std_psm", "reference"),
    "SWaT": ("segsweep_swat_merge_n2", "noisefloor_std_swat", "reference"),
}


def auroc(runs_root: Path, experiment: str, block: str) -> tuple[float | None, str]:
    for run in sorted((runs_root / experiment).glob("*/config.json")):
        if json.loads(run.read_text())["args"].get("seed") != SEED:
            continue
        result = run.parent / block / "result.json"
        if result.is_file():
            payload = json.loads(result.read_text())
            metrics = payload.get("metrics", payload)
            return float(metrics["window_auroc"]), f"{experiment}/{run.parent.name}"
    return None, ""


def drift(name: str) -> tuple[float | str, float | str]:
    import numpy as np

    from incremental_ad.analysis.drift_screen import load_series, segment_statistics
    if name == "SMD":
        from huggingface_hub import hf_hub_download
        values = np.load(hf_hub_download("thuml/Time-Series-Library", "SMD/SMD_train.npy",
                                         repo_type="dataset"))
    elif name == "ETTh2-injected":
        values = load_series("ETTh2", 0.2)
    else:
        return "", ""                    # published in §0.1b; not recomputed here
    return tuple(round(segment_statistics(values, 0.5, k)[0], 3) for k in (3, 5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for name, (base_exp, joint_exp, note) in SCREEN.items():
        base, base_run = auroc(args.runs_root, base_exp, "baseline/test")
        joint, joint_run = auroc(args.runs_root, joint_exp, "train/test")
        headroom = ((joint - base) / (1 - base)) if base is not None and joint is not None else None
        d3, d5 = drift(name)
        rows.append({"dataset": name, "base_run": base_run, "joint_run": joint_run,
                     "auroc_base": base, "auroc_joint": joint,
                     "headroom": round(headroom, 4) if headroom is not None else "",
                     "passes_rule": (headroom >= RULE_THRESHOLD) if headroom is not None else "",
                     "drift_3way": d3, "drift_5way": d5, "note": note})
        print(f"{name:15s} base {base}  joint {joint}  headroom "
              f"{'' if headroom is None else f'{100 * headroom:.2f}%'}  drift3 {d3}  drift5 {d5}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "ad_screen.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
