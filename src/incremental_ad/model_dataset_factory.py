from dataclasses import replace

from incremental_ad.datasets import psm, swat
from incremental_ad.datasets.base_dataset import BaseDataset, Split
from incremental_ad.models.mae_tx import MAETransformer
from incremental_ad.models.base_model import BaseModel


def build_model(
    *,
    model_name: str,
    dataset_name: str,
    model_cfg,
    dataset_cfg,
) -> BaseModel:
    if dataset_name == "swat" and model_name == "mae_tx":
        n_patches = dataset_cfg.window_len // model_cfg.patch_len
        return MAETransformer(
            n_patches=n_patches, n_features=swat.N_FEATURES, config=model_cfg
        )

    if dataset_name == "psm" and model_name == "mae_tx":
        n_patches = dataset_cfg.window_len // model_cfg.patch_len
        return MAETransformer(
            n_patches=n_patches, n_features=psm.N_FEATURES, config=model_cfg
        )

    raise ValueError(
        f"Unknown combination: model='{model_name}', dataset='{dataset_name}'"
    )


def build_datasets(
    *,
    dataset_name: str,
    dataset_cfg,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    base_data_ratio: float,
) -> tuple[BaseDataset, BaseDataset]:
    """Return (train_dataset, val_dataset) for the requested slice."""
    if dataset_name == "swat":
        train_ds, val_ds, _ = swat.get_loaders(
            dataset_cfg, train_slice, partial_ratio, n_finetune, base_data_ratio
        )
        return train_ds, val_ds

    if dataset_name == "psm":
        train_ds, val_ds, _ = psm.get_loaders(
            dataset_cfg, train_slice, partial_ratio, n_finetune, base_data_ratio
        )
        return train_ds, val_ds

    raise ValueError(f"Unknown dataset '{dataset_name}'")


def build_eval_datasets(
    *,
    dataset_name: str,
    dataset_cfg,
    split: Split,
) -> tuple[BaseDataset, BaseDataset]:
    """Return (train_dataset, eval_dataset) for eval at eval_stride on the full series.

    train_dataset covers the full train series (used e.g. for threshold fitting).
    eval_dataset is the val tail or test set depending on split.
    Both use eval_stride rather than the training stride.
    """
    if dataset_name == "swat":
        eval_cfg = replace(dataset_cfg, stride=dataset_cfg.eval_stride)
        train_ds, val_ds, test_ds = swat.get_loaders(eval_cfg, "full", 1.0, 0, 1.0)
        return train_ds, (val_ds if split == "val" else test_ds)

    if dataset_name == "psm":
        eval_cfg = replace(dataset_cfg, stride=dataset_cfg.eval_stride)
        train_ds, val_ds, test_ds = psm.get_loaders(eval_cfg, "full", 1.0, 0, 1.0)
        return train_ds, (val_ds if split == "val" else test_ds)

    raise ValueError(f"Unknown dataset '{dataset_name}'")
