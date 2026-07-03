"""Weather — Jena (Max Planck Institute) climate station data, 2020, 10-min intervals.

Source: thuml/Time-Series-Library on HuggingFace (config name: weather).
21 meteorological features (temperature, humidity, pressure, ...), one full calendar
year (2020-01-01 to 2021-01-01) — spans all four seasons, so unlike ETTh1 (found to
be fairly homogeneous across its incremental segments) this has a well-documented,
genuine reason to expect real distribution shift between segments if split
chronologically. No anomaly labels — forecasting only.

See hf_series_forecast.py for the shared implementation; this file only supplies the
HF config name and a display name for analysis plots.
"""

from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset


class WeatherForecastDataset(HfSeriesForecastDataset):
    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = "weather"
    _DISPLAY_NAME = "Weather"
