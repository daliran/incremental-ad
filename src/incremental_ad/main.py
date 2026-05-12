import argparse
import platform

from incremental_ad.datasets import swat
from incremental_ad.models import mae_tx
from incremental_ad.training import trainer

NUM_WORKERS = 0 if platform.system() == "Windows" else 4

DATASETS = {
    "swat": (swat.SWaTConfig, swat.add_args),
}
MODELS = {
    "mae_tx": (mae_tx.MaeTxConfig, mae_tx.add_args),
}


def pluck(args: argparse.Namespace, prefix: str) -> dict:
    """Pull args starting with `<prefix>_`, return dict with the prefix stripped."""
    p = f"{prefix}_"
    return {k.removeprefix(p): v for k, v in vars(args).items() if k.startswith(p)}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", choices=DATASETS, required=True)
    pre.add_argument("--model", choices=MODELS, required=True)
    known, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    dataset_cfg_cls, dataset_add_args = DATASETS[known.dataset]
    model_cfg_cls, model_add_args = MODELS[known.model]
    dataset_add_args(parser)
    model_add_args(parser)
    trainer.add_args(parser)
    args = parser.parse_args()

    dataset_cfg = dataset_cfg_cls(**pluck(args, known.dataset))
    model_cfg = model_cfg_cls(**pluck(args, known.model))
    train_cfg = trainer.TrainingConfig(**pluck(args, "train"))

    print(dataset_cfg)
    print(model_cfg)
    print(train_cfg)


if __name__ == "__main__":
    main()
