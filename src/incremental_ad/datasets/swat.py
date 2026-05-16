import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal, cast

from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from incremental_ad.core.cli import pluck
from incremental_ad.datasets.base_dataset import BaseDataset, Split

log = logging.getLogger(__name__)

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATASET_NAME = "SWaT"
N_FEATURES = 51

Normalization = Literal["standard", "none"]


@dataclass
class SWaTConfig:
    window_len: int
    stride: int
    normalization: Normalization
    val_ratio: float


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--swat-window-len", type=int, required=True)
    parser.add_argument("--swat-stride", type=int, required=True)
    parser.add_argument(
        "--swat-normalization", choices=["standard", "none"], required=True
    )
    parser.add_argument("--swat-val-ratio", type=float, required=True)


def make_config(args: Namespace) -> SWaTConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "swat")
    return SWaTConfig(**fields)


class SwatDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        window_size: int,
        stride: int,
        labels: torch.Tensor | None = None,
    ):
        super().__init__(labels=labels)
        self.data = data
        self.window_size = window_size
        self.stride = stride

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.window_size) // self.stride) + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        return self.data[start : start + self.window_size]


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:

    log.info("Loading SWaT dataset from HuggingFace...")

    # SWAT = Secure Water Treatment
    # This dataset is partially preprocessed compare to the original raw iTrust dataset.
    raw = load_dataset("thuml/Time-Series-Library", "SWaT")

    train_df = cast(pd.DataFrame, raw["train"].to_pandas())
    test_df = cast(pd.DataFrame, raw["test"].to_pandas())

    log.info(f"Loaded — train: {len(train_df)} rows, test: {len(test_df)} rows")

    return train_df, test_df


def _extract_samples_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # 0 = Normal, 1 = Attack (integer encoded).
    # Use to_numeric so that unexpected string labels raise errors.
    labels = pd.to_numeric(df["Normal/Attack"], errors="raise").astype(int)
    samples = df.drop(columns=["Normal/Attack"])
    return samples, labels


def _build_scaler(normalization: Normalization) -> StandardScaler | None:
    if normalization == "standard":
        return StandardScaler()
    if normalization == "none":
        return None
    raise ValueError(f"Unknown normalization '{normalization}'")


def _prepare_dataset(
    config: SWaTConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_df, test_df = _load_raw()

    # getting samples and labels.
    train_samples, _ = _extract_samples_and_labels(train_df)
    test_samples, test_labels = _extract_samples_and_labels(test_df)

    # scaling the data.
    scaler = _build_scaler(config.normalization)
    log.info(f"Normalization: {config.normalization}")

    if scaler is not None:
        train_scaled = scaler.fit_transform(train_samples.values)
        test_scaled = scaler.transform(test_samples.values)
    else:
        train_scaled = train_samples
        test_scaled = test_samples

    # converting dataframe to tensors.
    train_data = torch.tensor(train_scaled, dtype=torch.float32)
    test_data = torch.tensor(test_scaled, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    # creating dataset split.
    total = len(train_data)
    val_size = int(total * config.val_ratio)
    train_size = total - val_size

    # we need to preserve the temporal order, so no shuffling.
    train_data_split = train_data[:train_size]
    val_data_split = train_data[train_size:]

    log.info(f"Split — train: {train_size} timesteps, val: {val_size} timesteps, test: {len(test_data)} timesteps")

    return train_data_split, val_data_split, test_data, test_labels_tensor


def load_train(config: SWaTConfig) -> tuple[SwatDataset, SwatDataset]:
    """Returns (train_dataset, val_dataset)."""

    train_data, val_data, _, _ = _prepare_dataset(config)

    train_dataset = SwatDataset(train_data, config.window_len, config.stride)
    val_dataset = SwatDataset(val_data, config.window_len, config.stride)

    log.info(f"Train windows: {len(train_dataset)}, val windows: {len(val_dataset)}")

    return train_dataset, val_dataset


def load_eval(config: SWaTConfig, split: Split) -> SwatDataset:
    """Returns the dataset for the requested evaluation split."""

    train_data, val_data, test_data, test_labels = _prepare_dataset(config)

    if split == "val":
        dataset = SwatDataset(val_data, config.window_len, config.stride)
    else:
        dataset = SwatDataset(test_data, config.window_len, config.stride, test_labels)

    log.info(f"Eval split '{split}': {len(dataset)} windows")

    return dataset

