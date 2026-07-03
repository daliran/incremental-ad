"""Exchange rate — daily FX rates for 8 countries, ~1990-2010 (~7588 rows).

Source: thuml/Time-Series-Library on HuggingFace (config name: exchange_rate).
Only 8 features and daily (not hourly/10-min) frequency — much smaller than ETTh1,
Weather, or Traffic. Long enough history to plausibly span real financial regime
shifts, but small enough that the standard 96-24-style window_len/forecast_len
(120/24) don't fit safely through the incremental split (see launch.json / sbatch
comments for the exact val-sizing arithmetic) — use a smaller window_len/forecast_len
for this dataset specifically, unlike Weather/Traffic which reuse ETTh1's sizing
as-is. No anomaly labels — forecasting only.

See hf_series_forecast.py for the shared implementation; this file only supplies the
HF config name and a display name for analysis plots.
"""

from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset


class ExchangeRateForecastDataset(HfSeriesForecastDataset):
    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = "exchange_rate"
    _DISPLAY_NAME = "ExchangeRate"
