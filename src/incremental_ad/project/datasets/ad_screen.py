"""Two AD candidates for the headroom screen (EXPERIMENTS.md §1.44): SMD, and ETTh2 with
anomalies injected into its test portion.

Both subclass `Psm` and change ONLY where the data comes from, so windowing, partitioning,
validation slices and the test dataset are PSM's code path, unchanged — the screen tests the
data, not a different loader.

- **`Smd`** — the Server Machine Dataset as `thuml/Time-Series-Library` distributes it
  (`SMD/SMD_{train,test,test_label}.npy`). ⚠️ That file is the **28 machines concatenated**, so
  a "period" of it can span a machine boundary: its drift is partly machine identity, not drift
  of one system over time. Stated wherever SMD is reported.
- **`Etth2Injected`** — ETTh2's first 80% as training data (the forecasting runs' test fraction),
  its last 20% as test, with anomalies injected into the **test portion only** by the protocol
  fixed in `INJECTION` and registered in §1.44 before any run. ⚠️ **A controlled test, not a
  benchmark**: the anomalies are synthetic, and a detector's score on them says how it responds
  to those three perturbations, nothing about real faults.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from incremental_ad.project.datasets.psm import Normalization, Psm

log = logging.getLogger(__name__)

_HF_REPO = "thuml/Time-Series-Library"

# The injection protocol of §1.44 — changing any value here is a different experiment.
INJECTION = {
    "seed": 20260925,
    "anomalous_point_fraction": 0.10,       # stop adding events once >= 10% of test points
    "types": ("spike", "level_shift", "pattern_change"),
    "length": {"spike": (1, 3), "level_shift": (12, 48), "pattern_change": (24, 72)},
    "channels_per_event": (1, 3),           # of ETTh2's 7, uniform
    "spike_sigma": (4.0, 6.0),              # |added value| in training-sd units, random sign
    "level_shift_sigma": (1.5, 3.0),
    "min_gap": 100,                         # points between events, and from either end
}


def _scale(train: np.ndarray, test: np.ndarray, normalization: Normalization):
    if normalization == "standard":
        scaler = StandardScaler().fit(train)
        return (scaler.transform(train).astype(np.float32),
                scaler.transform(test).astype(np.float32))
    return train.astype(np.float32), test.astype(np.float32)


def inject(test: np.ndarray, protocol: dict = INJECTION) -> tuple[np.ndarray, np.ndarray, list]:
    """(perturbed test, point labels, event log). Operates on SCALED data (sd units)."""
    rng = np.random.default_rng(protocol["seed"])
    data, n, channels = test.copy(), len(test), test.shape[1]
    labels = np.zeros(n, dtype=np.int64)
    gap = protocol["min_gap"]
    events, attempts = [], 0
    while labels.mean() < protocol["anomalous_point_fraction"]:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not place enough non-overlapping events")
        kind = protocol["types"][rng.integers(len(protocol["types"]))]
        low, high = protocol["length"][kind]
        length = int(rng.integers(low, high + 1))
        start = int(rng.integers(gap, n - gap - length))
        if labels[max(0, start - gap): start + length + gap].any():
            continue
        picked = rng.choice(channels, size=int(rng.integers(protocol["channels_per_event"][0],
                                                            protocol["channels_per_event"][1] + 1)),
                            replace=False)
        segment = slice(start, start + length)
        for c in picked:
            if kind == "spike":
                data[segment, c] += rng.choice([-1, 1]) * rng.uniform(*protocol["spike_sigma"])
            elif kind == "level_shift":
                data[segment, c] += (rng.choice([-1, 1])
                                     * rng.uniform(*protocol["level_shift_sigma"]))
            else:                                   # same values, reversed in time
                data[segment, c] = data[segment, c][::-1].copy()
        labels[segment] = 1
        events.append({"type": kind, "start": start, "length": length,
                       "channels": sorted(int(c) for c in picked)})
    return data, labels, events


class Smd(Psm):
    """SMD through PSM's code path (see module docstring for the concatenation caveat)."""

    def __init__(self, window_len: int, stride: int, normalization: Normalization,
                 split_config) -> None:
        self._window_len = window_len
        self.stride = stride
        self._eval_stride = 1
        self.split_config = split_config
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return np.load(hf_hub_download(_HF_REPO, f"SMD/{name}", repo_type="dataset"))

        train, test, labels = fetch("SMD_train.npy"), fetch("SMD_test.npy"), \
            fetch("SMD_test_label.npy")
        train_s, test_s = _scale(np.nan_to_num(train), np.nan_to_num(test), normalization)
        log.info("SMD — train %s, test %s, anomalous test points %.2f%%", train.shape,
                 test.shape, 100 * float(np.mean(labels)))
        self._train_data = torch.tensor(train_s)
        self._test_data = torch.tensor(test_s)
        self._test_labels = torch.tensor(labels.astype(np.int64).ravel(), dtype=torch.long)
        self.split_config.validate(len(self._train_data))

    def analyze(self, analysis_dir: Path) -> None:
        log.info("SMD: dataset analysis not implemented for the screen — skipped")


class Etth2Injected(Psm):
    """ETTh2 with registered synthetic anomalies in its test portion. A controlled test."""

    TEST_FRACTION = 0.2

    def __init__(self, window_len: int, stride: int, normalization: Normalization,
                 split_config) -> None:
        self._window_len = window_len
        self.stride = stride
        self._eval_stride = 1
        self.split_config = split_config
        import pandas as pd
        from datasets import load_dataset

        frame = load_dataset(_HF_REPO, "ETTh2")["train"].to_pandas()
        frame = frame.drop(columns=[c for c in frame.columns if c.lower() in ("date",)])
        values = frame.apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy()
        cut = len(values) - int(len(values) * self.TEST_FRACTION)
        train_s, test_s = _scale(values[:cut], values[cut:], normalization)
        test_s, labels, events = inject(test_s)
        log.info("ETTh2-injected (controlled test) — train %d, test %d rows, %d events, "
                 "%.2f%% anomalous test points", len(train_s), len(test_s), len(events),
                 100 * labels.mean())
        self.injected_events = events
        self._train_data = torch.tensor(train_s)
        self._test_data = torch.tensor(test_s)
        self._test_labels = torch.tensor(labels, dtype=torch.long)
        self.split_config.validate(len(self._train_data))

    def analyze(self, analysis_dir: Path) -> None:
        log.info("ETTh2-injected: dataset analysis not implemented for the screen — skipped")
