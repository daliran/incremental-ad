# Incremental anomaly detection

## Transformer MAE

* Architecture
    * Fixed sequence length.
    * Learned encoder positional encoding.
    * Learned decoder positional encoding (different than encoder).
    * Learned decoder mask token.
    * Learned projection between encoder d_model to decoder d_model.
* AE encoder/decoder
    * Use pre-norm instead of post-norm.
    * FFN dim = 4*d_model.
    * Encoder d_model > decoder d_model.
* Training loss
    * Calculated between masked patches/tokens.
    * Patch level normalization on the ground truth patches.
    * MSE loss between predicted patches and ground truth patches.