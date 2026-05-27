from dataclasses import dataclass, replace

from incremental_ad.datasets import swat
from incremental_ad.datasets.base_dataset import BaseDataset, Split
from incremental_ad.datasets.swat import SWaTConfig
from incremental_ad.models import mae_tx
from incremental_ad.models.mae_tx import MaeTxConfig, MAETransformer
from incremental_ad.models.base_model import BaseModel


@dataclass
class BuildResult:
    model: BaseModel
    train_dataset: BaseDataset
    val_dataset: BaseDataset


def build_for_training(
    *,
    model_name: str,
    dataset_name: str,
    model_cfg,
    dataset_cfg,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    base_data_ratio: float,
) -> BuildResult:
    if dataset_name == "swat" and model_name == "mae_tx":
        return _build_swat_mae_tx(
            model_cfg, dataset_cfg, train_slice, partial_ratio, n_finetune, base_data_ratio
        )

    raise ValueError(
        f"Unknown combination: model='{model_name}', dataset='{dataset_name}'"
    )


def _build_swat_mae_tx(
    model_cfg: MaeTxConfig,
    dataset_cfg: SWaTConfig,
    train_slice: str,
    partial_ratio: float,
    n_finetune: int,
    base_data_ratio: float,
) -> BuildResult:

    train_dataset, val_dataset, _ = swat.get_loaders(
        dataset_cfg, train_slice, partial_ratio, n_finetune, base_data_ratio
    )

    n_patches = dataset_cfg.window_len // model_cfg.patch_len

    model = MAETransformer(
        n_patches=n_patches,
        n_features=swat.N_FEATURES,
        config=model_cfg,
    )

    return BuildResult(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset
    )


@dataclass
class EvalBuildResult:
    model: BaseModel
    train_dataset: BaseDataset
    eval_dataset: BaseDataset


def build_for_eval(
    *,
    model_name: str,
    dataset_name: str,
    model_cfg,
    dataset_cfg,
    split: Split,
) -> EvalBuildResult:
    if dataset_name == "swat" and model_name == "mae_tx":
        return _build_eval_swat_mae_tx(model_cfg, dataset_cfg, split)

    raise ValueError(
        f"Unknown combination: model='{model_name}', dataset='{dataset_name}'"
    )


def _build_eval_swat_mae_tx(
    model_cfg: MaeTxConfig, dataset_cfg: SWaTConfig, split: Split
) -> EvalBuildResult:

    # Evaluation windows at eval_stride (a fixed dataset-config property), and uses
    # the full train series as the reference (e.g. for the threshold);
    # partial_ratio/n_finetune are unused for the full slice.
    eval_cfg = replace(dataset_cfg, stride=dataset_cfg.eval_stride)

    train_dataset, val_dataset, test_dataset = swat.get_loaders(eval_cfg, "full", 1.0, 0, 1.0)
    eval_dataset = val_dataset if split == "val" else test_dataset

    n_patches = dataset_cfg.window_len // model_cfg.patch_len

    model = MAETransformer(
        n_patches=n_patches,
        n_features=swat.N_FEATURES,
        config=model_cfg,
    )

    return EvalBuildResult(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
