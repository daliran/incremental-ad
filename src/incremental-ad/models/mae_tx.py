from dataclasses import dataclass

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