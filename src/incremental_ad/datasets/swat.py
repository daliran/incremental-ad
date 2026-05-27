import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Literal, cast

from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from incremental_ad.core.cli import pluck
from incremental_ad.datasets import splitting
from incremental_ad.datasets.base_dataset import BaseDataset

log = logging.getLogger(__name__)

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATASET_NAME = "SWaT"
N_FEATURES = 51

Normalization = Literal["standard", "none"]


@dataclass
class SWaTConfig:
    window_len: int
    stride: int  # windowing stride used for training
    eval_stride: int  # windowing stride used at evaluation (build_for_eval)
    normalization: Normalization
    val_ratio: float 


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--swat-window-len", type=int, required=True)
    parser.add_argument("--swat-stride", type=int, required=True)
    parser.add_argument("--swat-eval-stride", type=int, required=True)
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
        super().__init__(window_size=window_size, stride=stride, labels=labels)
        self.data = data

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the full scaled train series, the test series and its labels.

    The train/val carving and slice selection happen in get_loaders; the scaler
    is fit on the full train series so every slice shares the same normalization.
    """
    train_df, test_df = _load_raw()

    # getting samples and labels.
    train_samples, _ = _extract_samples_and_labels(train_df)
    test_samples, test_labels = _extract_samples_and_labels(test_df)

    # scaling the data.
    scaler = _build_scaler(config.normalization)
    log.info(f"Normalization: {config.normalization}")

    if scaler is not None:
        # standard scaler computes normalization per feature by default, this is not global.
        train_scaled = scaler.fit_transform(train_samples.values)
        test_scaled = scaler.transform(test_samples.values)
    else:
        train_scaled = train_samples
        test_scaled = test_samples

    # converting dataframe to tensors.
    train_data = torch.tensor(train_scaled, dtype=torch.float32)
    test_data = torch.tensor(test_scaled, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    return train_data, test_data, test_labels_tensor


def get_loaders(
    config: SWaTConfig,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    base_data_ratio: float,
) -> tuple[SwatDataset, SwatDataset, SwatDataset]:
    """Build (train, val, test) datasets for the requested slice of the train series.

    train_slice is one of "full" (the whole train series), "partial" (the
    partial-pretrain prefix) or "ft_<i>" (a fine-tuning chunk); partial_ratio and
    n_finetune define the partial/ft chunking (passed in by the phase, not stored
    in the dataset config). The selected slice is split into train + a validation
    tail; the test set is independent of it.
    """

    train_data, test_data, test_labels = _prepare_dataset(config)
    n = len(train_data)

    # Resolve the [start, end) range of the train series for this slice.
    if train_slice == "full":
        start, end = 0, n
    else:
        chunks = splitting.equal_chunks(n, partial_ratio, n_finetune, base_ratio=base_data_ratio)

        if train_slice not in chunks:
            raise ValueError(
                f"Unknown train_slice '{train_slice}' "
                f"(available: full, {', '.join(chunks)})"
            )

        start, end = chunks[train_slice]

    # Carve train + validation tail within the slice (temporal, no shuffling).
    (t_start, t_end), (v_start, v_end) = splitting.val_tail_split(
        start, end, config.val_ratio
    )

    train_dataset = SwatDataset(
        train_data[t_start:t_end], config.window_len, config.stride
    )

    val_dataset = SwatDataset(
        train_data[v_start:v_end], config.window_len, config.stride
    )

    test_dataset = SwatDataset(test_data, config.window_len, config.stride, test_labels)

    log.info(
        f"Slice '{train_slice}' of {n} train timesteps — "
        f"train: {t_end - t_start}, val: {v_end - v_start}, test: {len(test_data)}"
    )
    log.info(
        f"Windows — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
