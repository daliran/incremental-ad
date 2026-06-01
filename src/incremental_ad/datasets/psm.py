import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal, cast

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from incremental_ad.core.cli import pluck
from incremental_ad.datasets import splitting
from incremental_ad.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATA_NAME = "PSM-data"
HF_LABEL_NAME = "PSM-label"
N_FEATURES = 25

Normalization = Literal["standard", "none"]


@dataclass
class PsmConfig:
    window_len: int
    stride: int
    eval_stride: int
    normalization: Normalization
    val_ratio: float


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--psm-window-len", type=int, required=True)
    parser.add_argument("--psm-stride", type=int, required=True)
    parser.add_argument("--psm-eval-stride", type=int, required=True)
    parser.add_argument(
        "--psm-normalization", choices=["standard", "none"], required=True
    )
    parser.add_argument("--psm-val-ratio", type=float, required=True)


def make_config(args: Namespace) -> PsmConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "psm")
    return PsmConfig(**fields)


class PsmDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        stride: int,
        labels: torch.Tensor | None = None,
    ):
        super().__init__(window_size=window_size, stride=stride, labels=labels)
        self.data = data

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.window_size) // self.stride) + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        return self.data[start : start + self.window_size]


def load_for_analysis() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return raw (unscaled) train/test arrays and feature names for offline analysis."""
    train_df, test_df, label_df = _load_raw()
    train_samples = _extract_features(train_df).ffill().bfill()
    test_samples = _extract_features(test_df)
    test_labels = label_df["label"].astype(int)

    return (
        train_samples.values.astype(np.float32),
        test_samples.values.astype(np.float32),
        np.asarray(test_labels.values, dtype=np.int64),
        list(train_samples.columns),
    )


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    log.info("Loading PSM dataset from HuggingFace...")

    raw = load_dataset(HF_DATASET_PATH, HF_DATA_NAME)
    raw_label = load_dataset(HF_DATASET_PATH, HF_LABEL_NAME)

    train_df = cast(pd.DataFrame, raw["train"].to_pandas())
    test_df = cast(pd.DataFrame, raw["test"].to_pandas())
    label_df = cast(pd.DataFrame, raw_label["test_label"].to_pandas())

    log.info(f"Loaded — train: {len(train_df)} rows, test: {len(test_df)} rows")

    return train_df, test_df, label_df


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the timestamp column; return only the 25 sensor features."""
    return df.drop(columns=["timestamp_(min)"])


def _build_scaler(normalization: Normalization) -> StandardScaler | None:
    if normalization == "standard":
        return StandardScaler()
    if normalization == "none":
        return None
    raise ValueError(f"Unknown normalization '{normalization}'")


def _prepare_dataset(
    config: PsmConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the full scaled train series, the test series and its labels."""
    train_df, test_df, label_df = _load_raw()

    train_samples = _extract_features(train_df)
    test_samples = _extract_features(test_df)

    # Since the dataset contains NaN, ffill these values by repeating the previous non NaN value 
    # up to a non NaN value, then use bfill to cover also the NaN at the beginning of the dataset.
    train_samples = train_samples.ffill().bfill()

    test_labels = label_df["label"].astype(int)

    scaler = _build_scaler(config.normalization)
    log.info(f"Normalization: {config.normalization}")

    if scaler is not None:
        train_scaled = scaler.fit_transform(train_samples.values)
        test_scaled = scaler.transform(test_samples.values)
    else:
        train_scaled = train_samples.values.astype(np.float32)
        test_scaled = test_samples.values.astype(np.float32)

    train_data = torch.tensor(train_scaled, dtype=torch.float32)
    test_data = torch.tensor(test_scaled, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    return train_data, test_data, test_labels_tensor


def get_loaders(
    config: PsmConfig,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    base_data_ratio: float,
) -> tuple[PsmDataset, PsmDataset, PsmDataset]:
    """Build (train, val, test) datasets for the requested slice of the train series."""

    train_data, test_data, test_labels = _prepare_dataset(config)
    n = len(train_data)

    if train_slice == "full":
        start, end = 0, n
    else:
        chunks = splitting.equal_chunks(
            n, partial_ratio, n_finetune, base_ratio=base_data_ratio
        )

        if train_slice not in chunks:
            raise ValueError(
                f"Unknown train_slice '{train_slice}' "
                f"(available: full, {', '.join(chunks)})"
            )

        start, end = chunks[train_slice]

    (t_start, t_end), (v_start, v_end) = splitting.val_tail_split(
        start, end, config.val_ratio
    )

    train_dataset = PsmDataset(
        train_data[t_start:t_end], config.window_len, config.stride
    )

    val_dataset = PsmDataset(
        train_data[v_start:v_end], config.window_len, config.stride
    )

    test_dataset = PsmDataset(test_data, config.window_len, config.stride, test_labels)

    log.info(
        f"Slice '{train_slice}' of {n} train timesteps — "
        f"train: {t_end - t_start}, val: {v_end - v_start}, test: {len(test_data)}"
    )
    log.info(
        f"Windows — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
