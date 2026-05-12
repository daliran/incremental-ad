import argparse
import platform

from incremental_ad.datasets import swat
from incremental_ad.models import mae_tx
from incremental_ad.training import trainer

NUM_WORKERS = 0 if platform.system() == "Windows" else 4

DATASETS = {
    "swat": (swat.add_args, swat.make_config),
}

MODELS = {
    "mae_tx": (mae_tx.add_args, mae_tx.make_config),
}


def load_config():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", choices=DATASETS, required=True)
    pre.add_argument("--model", choices=MODELS, required=True)
    known, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[pre])
    dataset_add_args, dataset_make_config = DATASETS[known.dataset]
    model_add_args, model_make_config = MODELS[known.model]

    dataset_add_args(parser)
    model_add_args(parser)
    trainer.add_args(parser)

    args = parser.parse_args()

    dataset_cfg = dataset_make_config(args)
    model_cfg = model_make_config(args)
    train_cfg = trainer.make_config(args)

    return dataset_cfg, model_cfg, train_cfg


def main():
    dataset_cfg, model_cfg, train_cfg = load_config()
    print(dataset_cfg)
    print(model_cfg)
    print(train_cfg)


if __name__ == "__main__":
    main()
