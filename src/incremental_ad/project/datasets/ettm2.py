"""ETTm2 — 15-minute electricity transformer temperature, second station.

Source: thuml/Time-Series-Library on HuggingFace (config name: ETTm2).
69,680 rows, 7 features — the 15-minute-resolution counterpart of ETTh2.

**Why this dataset is here: it separates drift from data scarcity.**

exchange_rate carries the most distinctive results in this project — merging beating joint
training, +102% routing headroom, materialising beating accumulation, and old data actively
hurting — *and* it is by far the smallest dataset (6,071 training rows, so 607-row shards at
n = 5). Those two facts are confounded: the configuration that gives it the strongest drift also
gives it the thinnest shards, and nothing measured so far separates them.

ETTh2 supplied the first evidence that drift is not the driver: nearly exchange_rate's drift
(0.753 vs 0.833) with twice the data, and it behaved like ETTh1 instead. **ETTm2 completes the
test — the same drift again (0.752) with 55,744 training rows, nine times exchange_rate's.**
If exchange_rate's behaviour reproduces here it is drift; if it disappears, it was scarcity.

**Sizing note.** This reuses ETTh1's window/horizon (120/24) unchanged, which keeps the model
and the split arithmetic identical across the comparison. On 15-minute data that spans 30 hours
rather than the 5 days it spans on hourly ETTh1, so ETTm2 is a *different forecasting problem*
in wall-clock terms — deliberately, because the point of this run is to hold the model fixed and
vary only how much data each shard gets. Absolute errors are therefore not comparable to
ETTh1/ETTh2; the *patterns* (retention crossover, merge versus continual, whether old data
hurts) are what this dataset is for.

No anomaly labels — forecasting only. See hf_series_forecast.py for the shared implementation.
"""

from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset


class Ettm2ForecastDataset(HfSeriesForecastDataset):
    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = "ETTm2"
    _DISPLAY_NAME = "ETTm2"
