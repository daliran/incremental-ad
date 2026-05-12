from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from incremental_ad.core.cli import pluck


@dataclass
class MaeTxConfig:
    d_model: int
    patch_len: int
    encoder_layers: int
    encoder_heads: int
    decoder_layers: int
    decoder_heads: int
    mask_ratio: float

    # max_seq_len and n_features should be injected into init when the model is created


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument("--mae-tx-d-model", type=int, required=True)
    parser.add_argument("--mae-tx-patch-len", type=int, required=True)
    parser.add_argument("--mae-tx-encoder-layers", type=int, required=True)
    parser.add_argument("--mae-tx-encoder-heads", type=int, required=True)
    parser.add_argument("--mae-tx-decoder-layers", type=int, required=True)
    parser.add_argument("--mae-tx-decoder-heads", type=int, required=True)
    parser.add_argument("--mae-tx-mask-ratio", type=float, required=True)


def make_config(args: Namespace) -> MaeTxConfig:
    fields = pluck(args, "mae_tx")
    return MaeTxConfig(**fields)
