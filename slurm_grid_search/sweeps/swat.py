"""SWaT sweeps: model architecture, then training parameters for both pipelines.

Base config anchors on the current proven recipe in
scripts/sbatch_mae_tx_swat_ad_standard.sh / _incremental.sh (patch_len=10, embed_dim=256,
encoder_layers=2, encoder_heads=2, decoder_embed_dim=128, decoder_layers=1,
decoder_heads=2, mask_ratio=0.80) -- not the older patch_len=5 config from the (now
removed) local grid_search_ad.py, which predates this anchor being updated.

Architecture args are kept separate from everything else: MODEL_SWEEP anchors them at
the proven values above and sweeps around them; TRAIN_STANDARD_SWEEP/
TRAIN_INCREMENTAL_SWEEP omit them entirely -- submit.py appends the MODEL_SWEEP winner's
architecture args at submission time (--arch-args), since the winner isn't known until
that sweep's results are collected.
"""

from __future__ import annotations

from harness import Sweep, SlurmConfig, expand_trials

# The 5 axes actually being searched (patch_norm excluded per instruction).
ARCH_ARGS = [
    "--mae_tx_decoder_embed_dim", "128",
    "--mae_tx_decoder_layers", "1",
    "--mae_tx_decoder_heads", "2",
    "--mae_tx_encoder_embed_dim", "256",
    "--mae_tx_encoder_layers", "2",
    "--mae_tx_encoder_heads", "2",
    "--mae_tx_patch_len", "10",
    "--mae_tx_mask_ratio", "0.80",
]

# mae_tx_* flags that are required but not part of the architecture search -- fixed in
# every sweep below, including MODEL_SWEEP.
_NON_ARCH_MAE_ARGS = [
    "--mae_tx_patch_norm", "false",
    "--mae_tx_n_eval_passes", "30",
    "--mae_tx_training_mode", "random_mask",
]

_COMMON_ARGS = [
    "--model", "MaeTx",
    "--dataset", "Swat",
    "--task", "ad",

    "--dataset_window_len", "100",
    "--dataset_stride", "50",
    "--dataset_normalization", "standard",
    "--dataset_val_fraction", "0.15",

    "--loader_batch_size", "64",
    "--loader_num_workers", "4",

    "--trainer_n_epochs", "300",
    "--trainer_patience", "30",
    "--trainer_optimizer", "adamw",
    "--trainer_weight_decay", "1e-2",
    "--trainer_learning_rate", "1e-4",
    "--trainer_grad_clip", "0.5",
    "--trainer_scheduler", "cosine",
    "--trainer_warmup_ratio", "0.1",
    "--trainer_checkpoint_interval", "0",

    "--configurator_threshold_strategy", "oracle",
    "--configurator_threshold_percentile", "99",
]

_INCREMENTAL_EXTRA_ARGS = [
    "--dataset_baseline_fraction", "0.5",
    "--dataset_baseline_use_fraction", "1.0",
    "--dataset_n_finetune_segments", "3",

    "--finetune_trainer_n_epochs", "50",
    "--finetune_trainer_patience", "10",
    "--finetune_trainer_weight_decay", "1e-2",
    "--finetune_trainer_learning_rate", "1e-5",
    "--finetune_trainer_grad_clip", "0.5",
    "--finetune_trainer_scheduler", "cosine",
]

# patch_len must divide dataset_window_len=100 -- valid divisors used below: 5, 10, 20.
MODEL_SWEEP = Sweep(
    name="swat_model",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_swat_model",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *ARCH_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="swat_model", time="01:30:00"),
    trials=expand_trials(
        cross={
            "mae_tx_patch_len": ["5", "10", "20"],
            "mae_tx_mask_ratio": ["0.5", "0.65", "0.80"],
        },
        one_at_a_time={
            "mae_tx_encoder_layers": ["3", "4"],
            "mae_tx_encoder_heads": ["4"],
            "mae_tx_encoder_embed_dim": ["128", "512"],
            "mae_tx_decoder_layers": ["2"],
            "mae_tx_decoder_heads": ["4"],
            "mae_tx_decoder_embed_dim": ["64", "256"],
        },
    ),
)

# base_args intentionally omits ARCH_ARGS -- submit.py appends the MODEL_SWEEP winner's
# architecture flags at submission time (--arch-args).
TRAIN_STANDARD_SWEEP = Sweep(
    name="swat_train_standard",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_swat_train_standard",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="swat_train_std", time="01:30:00"),
    trials=expand_trials(
        cross={
            "dataset_val_fraction": ["0.10", "0.15", "0.20"],
            "trainer_learning_rate": ["1e-4", "3e-4"],
        },
        one_at_a_time={
            "trainer_weight_decay": ["1e-3"],
        },
    ),
)

TRAIN_INCREMENTAL_SWEEP = Sweep(
    name="swat_train_incremental",
    pipeline="IncrementalTaskArithmeticPipeline",
    experiment_name="slurm_grid_swat_train_incremental",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *_INCREMENTAL_EXTRA_ARGS,
               "--pipeline", "IncrementalTaskArithmeticPipeline"],
    slurm=SlurmConfig(job_name="swat_train_inc", time="03:00:00"),
    trials=expand_trials(
        cross={
            "pipeline_merge_scale": ["0.3", "0.5", "1.0"],
            "finetune_trainer_reg_lambda": ["0.0", "1e-3", "1e-2"],
        },
        one_at_a_time={
            "dataset_baseline_fraction": ["0.3", "0.7"],
            "dataset_val_fraction": ["0.10", "0.20"],
            "dataset_n_finetune_segments": ["2", "5"],
        },
    ),
)
