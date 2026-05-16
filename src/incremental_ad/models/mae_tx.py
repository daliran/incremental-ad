import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from incremental_ad.core.cli import pluck
from incremental_ad.models.base_model import BaseModel

log = logging.getLogger(__name__)


@dataclass
class MaeTxConfig:
    patch_len: int
    encoder_embed_dim: int
    encoder_layers: int
    encoder_heads: int
    decoder_embed_dim: int
    decoder_layers: int
    decoder_heads: int
    mask_ratio: float
    patch_norm: bool
    n_eval_passes: int


def add_args(parser: ArgumentParser) -> None:
    """Adds the specific argparser arguments."""
    parser.add_argument("--mae-tx-patch-len", type=int, required=True)
    parser.add_argument("--mae-tx-encoder-embed-dim", type=int, required=True)
    parser.add_argument("--mae-tx-encoder-layers", type=int, required=True)
    parser.add_argument("--mae-tx-encoder-heads", type=int, required=True)
    parser.add_argument("--mae-tx-decoder-embed-dim", type=int, required=True)
    parser.add_argument("--mae-tx-decoder-layers", type=int, required=True)
    parser.add_argument("--mae-tx-decoder-heads", type=int, required=True)
    parser.add_argument("--mae-tx-mask-ratio", type=float, required=True)
    parser.add_argument("--mae-tx-patch-norm", action="store_true")
    parser.add_argument("--mae-tx-n-eval-passes", type=int, required=True)


def make_config(args: Namespace) -> MaeTxConfig:
    """Extracts the arguments from the argparse namespace and creates the dataclass."""
    fields = pluck(args, "mae_tx")
    return MaeTxConfig(**fields)


def create_mask(
    batch_size: int, number_of_patches: int, mask_ratio: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:

    # check if due to masking, at least a masked token is returned.
    number_of_masked_tokens = int(number_of_patches * mask_ratio)

    if number_of_masked_tokens <= 0:
        raise ValueError(f"mask_ratio {mask_ratio} provides no masked tokens")

    # check if due to masking, at least an unmasked token is returned.
    number_of_unmasked_tokens = number_of_patches - number_of_masked_tokens

    if number_of_unmasked_tokens <= 0:
        raise ValueError(f"mask_ratio {mask_ratio} leaves no unmasked tokens")

    # create a tensor of shape (batch size, number of patches) fllled with random values between [0, 1].
    # sort the tensor by number of patches (last dimension) in ascending order and return their indices (not the values).
    # this returns for each batch, the sorted patches.
    random_indices_sequence = torch.rand(
        batch_size, number_of_patches, device=device
    ).argsort(dim=-1)

    # mask_indices.shape = (batch size, number of masked patches).
    mask_indices = random_indices_sequence[:, :number_of_masked_tokens]

    # unmask_indices.shape = (batch size, number of unmasked patches).
    unmask_indices = random_indices_sequence[:, number_of_masked_tokens:]

    return mask_indices, unmask_indices


def get_by_mask(tensor: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
    batch_size = tensor.size(0)

    # batch_idx.shape = (batch size, 1).
    batch_idx = torch.arange(batch_size, device=tensor.device).unsqueeze(1)

    # masked.shape = (batch size, number of masked patches, transformer depth).
    # Advanced pytorch indexing to select for each batch the correct masked indices.
    # When two index tensor of the same shape are passed in [], then we get tokens[b, mask_indices[b]], selecting the correct subset.
    masked = tensor[batch_idx, mask_indices]

    return masked


def set_by_mask(
    tensor: torch.Tensor, mask_indices: torch.Tensor, values: torch.Tensor
) -> None:
    batch_size = tensor.size(0)

    # batch_idx.shape = (batch size, 1)
    batch_idx = torch.arange(batch_size, device=tensor.device).unsqueeze(1)

    # Advanced pytorch indexing to select for each batch the correct masked indices.
    # When two index tensor of the same shape are passed in [], then we get tokens[b, mask_indices[b]], selecting the correct subset.
    tensor[batch_idx, mask_indices] = values


class AEEncoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layer: int) -> None:
        super().__init__()

        # apply norm_first to follow the modern pre-norm convention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layer
        )

        # final LayerNorm required in the pre-norm convention.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch size, visible tokens, encoder depth).
        return self.norm(self.transformer_encoder(x))


class AEDecoder(nn.Module):
    def __init__(
        self, patch_len: int, n_features: int, d_model: int, n_head: int, n_layer: int
    ) -> None:
        super().__init__()

        # apply norm_first to follow the modern pre norm convention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layer
        )

        # final LayerNorm required in the pre-norm convention.
        self.norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, patch_len * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch size, n_patches, decoder depth).
        # projection_out.shape = (batch size, n_patches, patch length * features).
        norm = self.norm(self.transformer_encoder(x))
        return self.output_projection(norm)


# number_of_patches = time_series_length // self.config.patch_len.
class MAETransformer(BaseModel):
    def __init__(self, n_patches: int, n_features: int, config: MaeTxConfig) -> None:
        super().__init__()

        self.config = config
        self.n_patches = n_patches
        self.n_features = n_features

        if self.n_patches == 0:
            raise ValueError("The number of patches needs to be > 0")

        self.patch_embedding_layer = nn.Linear(
            self.config.patch_len * self.n_features, self.config.encoder_embed_dim
        )

        # (batch size, number of patches, encoder depth).
        # Each token in the sequence gets a different positional encoding but this positional encoding is the same across all batches.
        # This starts random instead of zeros because having different values per position helps the model to learn that the positions are different.
        self.encoder_positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, self.config.encoder_embed_dim)
        )

        # (batch size, number of patches, decoder depth).
        self.decoder_positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, self.config.decoder_embed_dim)
        )

        # ("batch size", "number of patches", transformer depth).
        # Learned token used as a place older for the masked tokens in the decoder input.
        self.decoder_mask_token = nn.Parameter(
            torch.zeros(1, 1, self.config.decoder_embed_dim)
        )

        self.encoder = AEEncoder(
            d_model=self.config.encoder_embed_dim,
            n_head=self.config.encoder_heads,
            n_layer=self.config.encoder_layers,
        )

        # Used to project the encoder output from encoder_embed_dim to decoder_embed_dim before passing to the decoder.
        self.encoder_to_decoder = nn.Linear(
            self.config.encoder_embed_dim, self.config.decoder_embed_dim
        )

        self.decoder = AEDecoder(
            patch_len=self.config.patch_len,
            n_features=n_features,
            d_model=self.config.decoder_embed_dim,
            n_head=self.config.decoder_heads,
            n_layer=self.config.decoder_layers,
        )

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch size, time series length, time series features).
        batch_size = x.size(0)

        # makes x divisible by number of patches, by selecting only the part which makes x divisible (discarding the rest at the bottom).
        x_div = x[:, : self.n_patches * self.config.patch_len, :]

        # each token is a concatenation of the features of multiple timesteps.
        tokens = x_div.reshape(
            batch_size, self.n_patches, self.config.patch_len * self.n_features
        )

        return tokens

    def _build_encoder_input(
        self, tokens: torch.Tensor, unmask_indices: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = (
            self.patch_embedding_layer(tokens) + self.encoder_positional_encoding
        )
        visible_tokens = get_by_mask(token_embeddings, unmask_indices)
        return visible_tokens

    def _build_decoder_input(
        self, encoder_output: torch.Tensor, unmask_indices: torch.Tensor
    ) -> torch.Tensor:

        batch_size = encoder_output.size(0)

        # decoder_mask_token.shape = (1, 1, decoder depth).
        # all_tokens.shape = (batch size, number of patches, decoder depth).
        # create an input tensor where all tokens are mask tokens.
        all_tokens = self.decoder_mask_token.repeat(batch_size, self.n_patches, 1)

        # set the encoder output tokens into the correct sequence position by using the unmask_indices.
        set_by_mask(all_tokens, unmask_indices, encoder_output)

        return all_tokens + self.decoder_positional_encoding

    def _forward(
        self, x: torch.Tensor, unmask_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(x)

        encoder_input = self._build_encoder_input(tokens, unmask_indices)
        encoder_output = self.encoder(encoder_input)

        # projects the encoder input to the decoder output embed dim
        encoder_output_d = self.encoder_to_decoder(encoder_output)

        decoder_input = self._build_decoder_input(encoder_output_d, unmask_indices)
        decoder_output = self.decoder(decoder_input)

        return decoder_output, tokens

    def _normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        # Normalize each patch independently on the last dimension.
        # (batch_size, n_patches, patch_len * n_features).
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        return (patches - mean) / (var + 1e-6).sqrt()

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:

        batch_size = batch.size(0)

        mask_indices, unmask_indices = create_mask(
            batch_size, self.n_patches, self.config.mask_ratio, batch.device
        )

        decoder_output, tokens = self._forward(batch, unmask_indices)

        loss = self._compute_training_loss(decoder_output, tokens, mask_indices)

        return loss

    def _compute_training_loss(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:

        # the training loss is calculated between masked patches/tokens.
        ground_truth_patches = get_by_mask(ground_truth, mask_indices)
        predicted_patches = get_by_mask(prediction, mask_indices)

        # enables patch level normalization for the ground truth patches.
        # this allows the model to focus on the shape and not on the magnitude of the values.
        # this is important even after global normalization.
        if self.config.patch_norm:
            ground_truth_patches = self._normalize_patches(ground_truth_patches)

        # (1) single tensor value.
        return F.mse_loss(predicted_patches, ground_truth_patches, reduction="mean")

    def eval_step(self, batch: torch.Tensor) -> torch.Tensor:
        #Note: at eval, the stride used in the dataset is 1.

        batch_size = batch.size(0)

        # (batch_size, masked patches).
        # token level reconstruction error.
        # since multiple forward passes are run, the masks used can be different.
        # this tensors keep track of the visited tokens the relative accumulated error. 
        token_error_sum = torch.zeros(batch_size, self.n_patches, device=batch.device)

        # (batch_size, masked patches).
        # tensors which keeps track how many time a token has been masked.
        # this is used to calculate the average token level reconstruction using token level information.
        token_error_counts = torch.zeros(
            batch_size, self.n_patches, device=batch.device
        )

        for _ in range(self.config.n_eval_passes):

            mask_indices, unmask_indices = create_mask(
                batch_size, self.n_patches, self.config.mask_ratio, batch.device
            )

            decoder_output, tokens = self._forward(batch, unmask_indices)

            # fills the error sum and error counts tensors.
            self._accumulate_errors(
                decoder_output,
                tokens,
                mask_indices,
                token_error_sum,
                token_error_counts,
            )
    
        # (batch_size, masked patches).
        # average token error considering each token accumulated error and the number of times the token appeared.
        # the clamp is used to avoid division by 0 in case a token is never visited.
        mean_token_errors = token_error_sum / token_error_counts.clamp(min=1)

        # (batch_size).
        # average reconstruction error of all tokens in a batch/sequence.
        # this is basically a single anomaly score per window.

        # with a single forward pass this would have been:
        # errors = F.mse_loss(predicted_patches, ground_truth_patches, reduction="none").mean(dim=-1)
        # score = errors.mean(dim=-1)

        return mean_token_errors.mean(dim=-1)

    def _accumulate_errors(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        mask_indices: torch.Tensor,
        token_error_sum: torch.Tensor,
        token_error_counts: torch.Tensor,
    ) -> None:
        
        ground_truth_patches = get_by_mask(ground_truth, mask_indices)
        predicted_patches = get_by_mask(prediction, mask_indices)

        # is this needed?
        if self.config.patch_norm:
            ground_truth_patches = self._normalize_patches(ground_truth_patches)

        # (batch_size, masked patches, patch_len * n_features) -> (batch_size, masked patches).
        # reconstruction error per patch/token.
        errors = F.mse_loss(
            predicted_patches, ground_truth_patches, reduction="none"
        ).mean(dim=-1)

        # Adds the error value in the tensor at the location of the mask indices.
        token_error_sum.scatter_add_(1, mask_indices, errors)

        # sets 1 if there is error, 0 otherwise.
        visited_tokens = torch.ones_like(errors)

        # Adds the increment value in the tensor at the location of the mask indices.
        token_error_counts.scatter_add_(1, mask_indices, visited_tokens)


# Moved at the bottom because it needs the model to be definend first
def build_model(seq_len: int, n_features: int, config: MaeTxConfig) -> MAETransformer:
    n_patches = seq_len // config.patch_len
    return MAETransformer(n_patches=n_patches, n_features=n_features, config=config)
