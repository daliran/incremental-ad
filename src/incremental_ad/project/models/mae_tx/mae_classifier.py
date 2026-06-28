from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from incremental_ad.project.models.mae_tx.mae import MaeTxEncoderConfig, _MaeTxBase


class MaeTxClassifier(_MaeTxBase):
    """MaeTx encoder backbone extended with a supervised classification head.

    Only the encoder is built — no decoder, no masking, no reconstruction loss.
    Training is supervised: the DataLoader yields (window [W, F], label) pairs
    and compute_loss() uses cross-entropy. At inference score() returns raw logits
    [B, n_classes] — the ClassificationEvaluator handles argmax and metrics.

    CLI args are the encoder-only subset of --mae_tx_* (patch_len, encoder_*,
    patch_norm). Decoder and masking args are not required.
    """

    # ARG_PREFIX inherited from _MaeTxBase ("mae_tx")
    # add_args / from_config inherited from _MaeTxBase — encoder params only

    def __init__(self, config: MaeTxEncoderConfig) -> None:
        super().__init__(config)
        # Injected by MaeTxClassificationConfigurator before _build()
        self.n_classes: int | None = None

    def _build(self) -> None:
        if self.n_classes is None:
            raise RuntimeError(
                "MaeTxClassifier._build() called before n_classes was injected. "
                "Ensure MaeTxClassificationConfigurator.configure() is called first."
            )
        
        self._build_encoder()  # decoder intentionally omitted
        self.classifier = nn.Linear(self.config.encoder_embed_dim, self.n_classes)
        self._log_architecture()

    def compute_loss(self, batch: Any) -> Tensor:
        inputs, labels = batch
        return F.cross_entropy(self.score(inputs), labels.long())

    def score(self, inputs: Tensor) -> Tensor:
        """Encode the full window (no masking) and classify via mean-pooled head.

        Returns class logits [B, n_classes].
        """
        tokens = self._tokenize(inputs)
        embeddings = self.patch_embedding(tokens) + self.encoder_pos_enc
        enc_out = self.encoder(embeddings)   # [B, n_patches, encoder_embed_dim]
        pooled = enc_out.mean(dim=1)         # [B, encoder_embed_dim]
        return self.classifier(pooled)       # [B, n_classes]
