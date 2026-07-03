"""Traffic — San Francisco Bay Area road occupancy rates, ~2 years, hourly.

Source: thuml/Time-Series-Library on HuggingFace (config name: traffic).
862 anonymized sensor features — much wider than ETTh1 (7) or Weather (21), so
training is noticeably heavier per epoch (patch_embedding's input dim scales with
n_features). No anomaly labels — forecasting only.

See hf_series_forecast.py for the shared implementation; this file only supplies the
HF config name and a display name for analysis plots.
"""

from incremental_ad.project.datasets.hf_series_forecast import HfSeriesForecastDataset


class TrafficForecastDataset(HfSeriesForecastDataset):
    ARG_PREFIX = "dataset"
    _HF_DATASET_NAME = "traffic"
    _DISPLAY_NAME = "Traffic"
