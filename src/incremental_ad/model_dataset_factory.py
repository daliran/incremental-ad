from dataclasses import dataclass

from torch.utils.data import Dataset

from incremental_ad.datasets import swat
from incremental_ad.datasets.swat import SWaTConfig
from incremental_ad.models import mae_tx
from incremental_ad.models.mae_tx import MaeTxConfig, MAETransformer
from incremental_ad.models.base_model import BaseModel


@dataclass
class BuildResult:
    model: BaseModel
    train_dataset: Dataset
    val_dataset: Dataset


def build(
    *,
    model_name: str,
    dataset_name: str,
    model_cfg,
    dataset_cfg,
) -> BuildResult:
    if dataset_name == "swat" and model_name == "mae_tx":
        return _build_swat_mae_tx(model_cfg, dataset_cfg)

    raise ValueError(
        f"Unknown combination: model='{model_name}', dataset='{dataset_name}'"
    )


def _build_swat_mae_tx(model_cfg: MaeTxConfig, dataset_cfg: SWaTConfig) -> BuildResult:

    train_dataset, val_dataset = swat.load(dataset_cfg)

    n_patches = dataset_cfg.window_len // model_cfg.patch_len

    model = MAETransformer(
        n_patches=n_patches,
        n_features=swat.N_FEATURES,
        config=model_cfg,
    )

    return BuildResult(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset
    )
