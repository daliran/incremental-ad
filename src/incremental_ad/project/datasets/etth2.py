"""ETTh2 — hourly electricity transformer temperature, second station.

Source: thuml/Time-Series-Library on HuggingFace (config name: ETTh2).
Same shape as ETTh1 — 17,420 rows, 7 features, hourly — from a different transformer.

**Why this dataset is here.** It is the closest thing to a controlled experiment for drift
that the available data allows: identical row count, feature count, sampling frequency and
sensor type to ETTh1, from a different station, but roughly **double the drift** (segment-shift
0.753 against 0.412 on the 5-way screen in EXECUTION_PLAN.md, "Dataset screen"). Every claim in this
project that is drift-dependent — merging winning under strong drift, routing having headroom
under strong drift, materialising beating accumulation under strong drift — becomes testable
with size, dimensionality and domain held fixed.

It also reuses ETTh1's window sizing (120/24) unchanged, which exchange_rate could not, so the
comparison is not confounded by a different window arithmetic either.

No anomaly labels — forecasting only. See hf_series_forecast.py for the shared implementation;
this file supplies only the HF config name and a display name.
"""

from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset


class Etth2ForecastDataset(HfSeriesForecastDataset):
    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = "ETTh2"
    _DISPLAY_NAME = "ETTh2"
