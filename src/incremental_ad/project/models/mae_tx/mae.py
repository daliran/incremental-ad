import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from incremental_ad.framework.contracts.model import Model

log = logging.getLogger(__name__)


class TrainingMode(str, Enum):
    RANDOM_MASK = "random_mask"  # reconstruct randomly masked patches (standard MAE)
    CAUSAL_MASK = "causal_mask"  # reconstruct the second half given the first half
    NEXT_STEP = "next_step"  # reconstruct the last patch given all preceding patches


class InferenceMode(str, Enum):
    AD = "ad"              # anomaly score [B] via reconstruction error
    FORECAST = "forecast"  # future values [B, H, F] via causal masking
    IMPUTATION = "imputation"  # reconstructed values [B, n_masked*patch_len, F] at masked positions only


@dataclass
class MaeTxEncoderConfig:
    """Encoder-only config shared by MaeTx and MaeTxClassifier."""
    patch_len: int
    encoder_embed_dim: int
    encoder_layers: int
    encoder_heads: int
    patch_norm: bool


@dataclass
class MaeTxConfig(MaeTxEncoderConfig):
    """Full MAE config — extends encoder config with decoder and masking fields."""
    decoder_embed_dim: int
    decoder_layers: int
    decoder_heads: int
    mask_ratio: float
    n_eval_passes: int  # forward passes at eval time (random mask only)
    training_mode: TrainingMode


def _str_to_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


# ── Masking utilities ──────────────────────────────────────────────────────────


def _create_random_mask(
    batch_size: int, n_patches: int, mask_ratio: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    n_masked = int(n_patches * mask_ratio)
    if n_masked <= 0:
        raise ValueError(f"mask_ratio {mask_ratio} produces no masked tokens")
    if n_patches - n_masked <= 0:
        raise ValueError(f"mask_ratio {mask_ratio} leaves no unmasked tokens")
    indices = torch.rand(batch_size, n_patches, device=device).argsort(dim=-1)
    return indices[:, :n_masked], indices[:, n_masked:]


def _create_causal_mask(
    batch_size: int, n_patches: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    n_visible = n_patches - n_patches // 2
    visible = torch.arange(n_visible, device=device).unsqueeze(0).expand(batch_size, -1)
    masked = (
        torch.arange(n_visible, n_patches, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    return masked.contiguous(), visible.contiguous()


def _create_next_step_mask(
    batch_size: int, n_patches: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    visible = (
        torch.arange(n_patches - 1, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    masked = torch.full((batch_size, 1), n_patches - 1, dtype=torch.long, device=device)
    return masked.contiguous(), visible.contiguous()


def _get_by_mask(tensor: Tensor, mask_indices: Tensor) -> Tensor:
    batch_idx = torch.arange(tensor.size(0), device=tensor.device).unsqueeze(1)
    return tensor[batch_idx, mask_indices]


def _set_by_mask(tensor: Tensor, mask_indices: Tensor, values: Tensor) -> None:
    batch_idx = torch.arange(tensor.size(0), device=tensor.device).unsqueeze(1)
    tensor[batch_idx, mask_indices] = values


def _complement_indices(visible_idx: Tensor, n: int) -> Tensor:
    """Return the indices in [0, n) not in visible_idx — the complement mask.

    visible_idx: [B, n_visible] — must have the same n_visible in every row.
    Returns:     [B, n - n_visible]
    """
    B, n_visible = visible_idx.shape
    n_masked = n - n_visible
    is_visible = torch.zeros(B, n, dtype=torch.bool, device=visible_idx.device)
    is_visible.scatter_(1, visible_idx, True)
    all_idx = torch.arange(n, device=visible_idx.device).unsqueeze(0).expand(B, -1)
    return all_idx[~is_visible].reshape(B, n_masked)


# ── Sub-modules ────────────────────────────────────────────────────────────────


class _Encoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: norm_first disables the fast path anyway; suppress warning
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.encoder(x))


class _Decoder(nn.Module):
    def __init__(
        self, patch_len: int, n_features: int, d_model: int, n_head: int, n_layer: int
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, patch_len * n_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_projection(self.norm(self.encoder(x)))


# ── Encoder backbone ───────────────────────────────────────────────────────────


class _MaeTxBase(Model):
    """Shared encoder backbone for MaeTx and MaeTxClassifier.

    Holds patch tokenisation, positional encoding, and the transformer encoder.
    Subclasses extend this with task-specific heads and loss functions.
    """

    ARG_PREFIX = "mae_tx"

    def __init__(self, config: MaeTxEncoderConfig) -> None:
        super().__init__()
        self.config = config
        # Injected by the task configurator before _build() is called
        self.n_features: int | None = None
        self.seq_len: int | None = None

    def _build_encoder(self) -> None:
        """Build patch embedding, positional encoding, and transformer encoder.

        Also sets self.n_patches. Requires n_features and seq_len to be injected first.
        """
        if self.n_features is None or self.seq_len is None:
            raise RuntimeError(
                f"{type(self).__name__}._build_encoder() called before n_features/seq_len "
                "were injected. Ensure the task configurator's configure() is called first."
            )

        n_features: int = self.n_features
        seq_len: int = self.seq_len
        config = self.config
        self.n_patches = seq_len // config.patch_len

        if self.n_patches == 0:
            raise ValueError("seq_len // patch_len must be > 0")
        if seq_len % config.patch_len != 0:
            log.warning(
                f"seq_len={seq_len} is not divisible by patch_len={config.patch_len} — "
                f"the last {seq_len % config.patch_len} timestep(s) will be silently dropped during tokenization."
            )

        raw_token_dim = config.patch_len * n_features
        self.patch_embedding = nn.Linear(raw_token_dim, config.encoder_embed_dim)
        self.encoder_pos_enc = nn.Parameter(
            torch.randn(1, self.n_patches, config.encoder_embed_dim)
        )
        self.encoder = _Encoder(
            config.encoder_embed_dim, config.encoder_heads, config.encoder_layers
        )

    def _tokenize(self, x: Tensor) -> Tensor:
        assert self.n_features is not None, "model not built"
        B = x.size(0)
        x_div = x[:, : self.n_patches * self.config.patch_len, :]
        return x_div.reshape(B, self.n_patches, self.config.patch_len * self.n_features)

    def _normalize_patches(self, patches: Tensor) -> Tensor:
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        return (patches - mean) / (var + 1e-6).sqrt()

    def _log_architecture(self) -> None:
        assert self.n_features is not None, "model not built"
        n_params = sum(p.numel() for p in self.parameters())
        c = self.config
        dec = ""
        if hasattr(self, "decoder") and hasattr(c, "decoder_embed_dim"):
            dec = f" | decoder {c.decoder_embed_dim}d x {c.decoder_layers}L x {c.decoder_heads}h"  # type: ignore[union-attr]
        log.info(
            f"{type(self).__name__} | {self.n_patches} patches x {c.patch_len} steps x {self.n_features} features"
            f" | encoder {c.encoder_embed_dim}d x {c.encoder_layers}L x {c.encoder_heads}h"
            f"{dec} | {n_params:,} params"
        )

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_patch_len", type=int, required=True)
        parser.add_argument(f"--{p}_encoder_embed_dim", type=int, required=True)
        parser.add_argument(f"--{p}_encoder_layers", type=int, required=True)
        parser.add_argument(f"--{p}_encoder_heads", type=int, required=True)
        parser.add_argument(f"--{p}_patch_norm", type=_str_to_bool, required=True)

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(MaeTxEncoderConfig(
            patch_len=getattr(cfg, f"{p}_patch_len"),
            encoder_embed_dim=getattr(cfg, f"{p}_encoder_embed_dim"),
            encoder_layers=getattr(cfg, f"{p}_encoder_layers"),
            encoder_heads=getattr(cfg, f"{p}_encoder_heads"),
            patch_norm=getattr(cfg, f"{p}_patch_norm"),
        ))


# ── Full MAE ───────────────────────────────────────────────────────────────────


class MaeTx(_MaeTxBase):
    """Masked Autoencoder for time-series AD, forecasting, and imputation.

    Extends the encoder backbone with a decoder and self-supervised reconstruction
    loss. Task-specific behaviour (scoring, loss, batch format) is selected via
    inference_mode, which is set by the task configurator at configure() time.
    """

    def __init__(self, config: MaeTxConfig) -> None:
        super().__init__(config)
        # Set by the task configurator; not serialised — configurator always re-sets before inference
        self.inference_mode: InferenceMode = InferenceMode.AD
        # Only used when inference_mode == FORECAST
        self.forecast_patches: int | None = None

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build_decoder(self) -> None:
        """Build the encoder-to-decoder bridge, mask token, positional encoding, and transformer decoder."""
        assert self.n_features is not None  # set by _build_encoder
        config = self.config
        assert isinstance(config, MaeTxConfig)
        self.decoder_pos_enc = nn.Parameter(
            torch.randn(1, self.n_patches, config.decoder_embed_dim)
        )
        self.decoder_mask_token = nn.Parameter(
            torch.zeros(1, 1, config.decoder_embed_dim)
        )
        self.encoder_to_decoder = nn.Linear(
            config.encoder_embed_dim, config.decoder_embed_dim
        )
        self.decoder = _Decoder(
            config.patch_len,
            self.n_features,
            config.decoder_embed_dim,
            config.decoder_heads,
            config.decoder_layers,
        )

    def _build(self) -> None:
        """Construct all layers. Called by the task configurator after n_features/seq_len are injected."""
        self._build_encoder()
        self._build_decoder()
        self._log_architecture()

    # ── Configurable interface ─────────────────────────────────────────────────

    @classmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        super().add_args(parser, prefix)  # encoder params
        p = prefix or cls.ARG_PREFIX
        parser.add_argument(f"--{p}_decoder_embed_dim", type=int, required=True)
        parser.add_argument(f"--{p}_decoder_layers", type=int, required=True)
        parser.add_argument(f"--{p}_decoder_heads", type=int, required=True)
        parser.add_argument(f"--{p}_mask_ratio", type=float, required=True)
        parser.add_argument(f"--{p}_n_eval_passes", type=int, required=True)
        parser.add_argument(
            f"--{p}_training_mode",
            type=TrainingMode,
            choices=list(TrainingMode),
            required=True,
        )

    @classmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        p = prefix or cls.ARG_PREFIX
        return cls(MaeTxConfig(
            patch_len=getattr(cfg, f"{p}_patch_len"),
            encoder_embed_dim=getattr(cfg, f"{p}_encoder_embed_dim"),
            encoder_layers=getattr(cfg, f"{p}_encoder_layers"),
            encoder_heads=getattr(cfg, f"{p}_encoder_heads"),
            patch_norm=getattr(cfg, f"{p}_patch_norm"),
            decoder_embed_dim=getattr(cfg, f"{p}_decoder_embed_dim"),
            decoder_layers=getattr(cfg, f"{p}_decoder_layers"),
            decoder_heads=getattr(cfg, f"{p}_decoder_heads"),
            mask_ratio=getattr(cfg, f"{p}_mask_ratio"),
            n_eval_passes=getattr(cfg, f"{p}_n_eval_passes"),
            training_mode=getattr(cfg, f"{p}_training_mode"),
        ))

    # ── Model interface ────────────────────────────────────────────────────────

    def compute_loss(self, batch: Any) -> Tensor:
        # Self-supervised reconstruction (all MAE tasks): batch is a single Tensor or (inputs, _)
        inputs: Tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
        mask_indices, unmask_indices = self._create_mask(inputs.size(0), inputs.device)
        decoder_output, tokens = self._forward(inputs, unmask_indices)
        return self._reconstruction_loss(decoder_output, tokens, mask_indices)

    def predict_step(self, batch: tuple) -> tuple[Tensor, Tensor | None]:
        if self.inference_mode == InferenceMode.IMPUTATION:
            masked_window, (original, visible_idx) = batch
            return self._predict_imputation(masked_window, visible_idx, original)
        return super().predict_step(batch)

    def score(self, inputs: Tensor) -> Tensor:
        """Inference dispatch — routes to the scorer for the configured InferenceMode."""
        match self.inference_mode:
            case InferenceMode.FORECAST:
                return self._score_forecast(inputs)
            case InferenceMode.AD:
                return self._score_ad(inputs)
            case InferenceMode.IMPUTATION:
                raise RuntimeError(
                    "score() is not supported in IMPUTATION mode. "
                    "The dataset provides (masked_window, visible_idx) pairs; "
                    "use predict_step() which handles this format internally."
                )
            case _:
                raise AssertionError(f"unhandled InferenceMode: {self.inference_mode}")

    # ── Core forward ──────────────────────────────────────────────────────────

    def _forward(self, x: Tensor, unmask_indices: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self._tokenize(x)
        embeddings = self.patch_embedding(tokens) + self.encoder_pos_enc
        visible = _get_by_mask(embeddings, unmask_indices)
        enc_out = self.encoder(visible)

        B = enc_out.size(0)
        dec_in = self.decoder_mask_token.repeat(B, self.n_patches, 1)
        _set_by_mask(dec_in, unmask_indices, self.encoder_to_decoder(enc_out))
        dec_in = dec_in + self.decoder_pos_enc

        return self.decoder(dec_in), tokens

    # ── Loss and error helpers ─────────────────────────────────────────────────

    def _reconstruction_loss(
        self, prediction: Tensor, ground_truth: Tensor, mask_indices: Tensor
    ) -> Tensor:
        gt = _get_by_mask(ground_truth, mask_indices)
        pred = _get_by_mask(prediction, mask_indices)
        if self.config.patch_norm:
            gt = self._normalize_patches(gt)
        return F.mse_loss(pred, gt, reduction="mean")

    def _accumulate_errors(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
        mask_indices: Tensor,
        error_sum: Tensor,
        error_counts: Tensor,
    ) -> None:
        gt = _get_by_mask(ground_truth, mask_indices)
        pred = _get_by_mask(prediction, mask_indices)
        if self.config.patch_norm:
            gt = self._normalize_patches(gt)
        errors = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)  # [B, n_masked]
        error_sum.scatter_add_(1, mask_indices, errors)
        error_counts.scatter_add_(1, mask_indices, torch.ones_like(errors))

    # ── Masking ───────────────────────────────────────────────────────────────

    def _create_mask(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        assert isinstance(config, MaeTxConfig)
        mode = config.training_mode
        if mode == TrainingMode.RANDOM_MASK:
            return _create_random_mask(
                batch_size, self.n_patches, config.mask_ratio, device
            )
        if mode == TrainingMode.CAUSAL_MASK:
            return _create_causal_mask(batch_size, self.n_patches, device)
        return _create_next_step_mask(batch_size, self.n_patches, device)

    # ── Predict ───────────────────────────────────────────────────────────────

    def _score_forecast(self, inputs: Tensor) -> Tensor:
        """Causal inference: context patches visible, forecast patches predicted.

        Returns [B, forecast_patches * patch_len, F].
        Requires training_mode=CAUSAL_MASK and forecast_patches to be set.
        """
        assert self.forecast_patches is not None
        B = inputs.size(0)
        n_context = self.n_patches - self.forecast_patches

        visible = (
            torch.arange(n_context, device=inputs.device)
            .unsqueeze(0).expand(B, -1).contiguous()
        )

        decoder_output, _ = self._forward(inputs, visible)

        forecast_idx = (
            torch.arange(n_context, self.n_patches, device=inputs.device)
            .unsqueeze(0).expand(B, -1).contiguous()
        )

        pred_patches = _get_by_mask(decoder_output, forecast_idx)   # [B, fp, patch_len*F]
        forecast_len = self.forecast_patches * self.config.patch_len
        
        assert self.n_features is not None
        return pred_patches.reshape(B, forecast_len, self.n_features)  # [B, H, F]

    def _predict_imputation(
        self,
        masked_window: Tensor,
        visible_idx: Tensor,
        original: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fixed-mask inference: encode only visible patches, return predictions vs ground truth
        at the masked positions.

        Returns (pred [B, n_masked*patch_len, F], orig [B, n_masked*patch_len, F]).
        The evaluator computes MSE/MAE only at the positions the model could NOT see.
        """
        assert self.n_features is not None
        B = masked_window.size(0)
        n_masked = self.n_patches - visible_idx.size(1)

        decoder_output, _ = self._forward(masked_window, visible_idx)         # [B, n_patches, patch_dim]
        mask_idx = _complement_indices(visible_idx, self.n_patches)            # [B, n_masked]

        pred_at_mask = _get_by_mask(decoder_output, mask_idx)                 # [B, n_masked, patch_dim]
        orig_at_mask = _get_by_mask(self._tokenize(original), mask_idx)       # [B, n_masked, patch_dim]

        H = n_masked * self.config.patch_len
        return (
            pred_at_mask.reshape(B, H, self.n_features),
            orig_at_mask.reshape(B, H, self.n_features),
        )

    def _score_ad(self, inputs: Tensor) -> Tensor:
        """AD scorer: delegates to the appropriate sub-method for the training mode."""
        config = self.config
        assert isinstance(config, MaeTxConfig)
        if config.training_mode == TrainingMode.RANDOM_MASK:
            return self._predict_random_mask(inputs)
        return self._predict_deterministic(inputs)

    def _predict_random_mask(self, batch: Tensor) -> Tensor:
        config = self.config
        assert isinstance(config, MaeTxConfig)
        B = batch.size(0)
        error_sum = torch.zeros(B, self.n_patches, device=batch.device)
        error_counts = torch.zeros(B, self.n_patches, device=batch.device)
        for _ in range(config.n_eval_passes):
            mask_indices, unmask_indices = _create_random_mask(
                B, self.n_patches, config.mask_ratio, batch.device
            )
            decoder_output, tokens = self._forward(batch, unmask_indices)
            self._accumulate_errors(
                decoder_output, tokens, mask_indices, error_sum, error_counts
            )
        mean_token_errors = error_sum / error_counts.clamp(min=1)  # [B, n_patches]
        return mean_token_errors.mean(dim=-1)  # [B]

    def _predict_deterministic(self, batch: Tensor) -> Tensor:
        B = batch.size(0)
        mask_indices, unmask_indices = self._create_mask(B, batch.device)
        decoder_output, tokens = self._forward(batch, unmask_indices)
        gt = _get_by_mask(tokens, mask_indices)
        pred = _get_by_mask(decoder_output, mask_indices)
        if self.config.patch_norm:
            gt = self._normalize_patches(gt)
        return F.mse_loss(pred, gt, reduction="none").mean(dim=-1).mean(dim=-1)  # [B]
