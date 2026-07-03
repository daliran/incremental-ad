"""PSM sweeps: model architecture, then training parameters for both pipelines.

Same shape and base config as swat.py (identical proven architecture anchor) -- only
--dataset and --configurator_threshold_percentile differ (PSM's anomaly rate is ~28%,
hence 72 instead of SWaT's 99, matching scripts/sbatch_mae_tx_psm_ad_*.sh).
"""

from __future__ import annotations

from harness import Sweep, SlurmConfig, expand_trials

ARCH_ARGS = [
    "--mae_tx_decoder_embed_dim", "128",
    "--mae_tx_decoder_layers", "1",
    "--mae_tx_decoder_heads", "2",
    "--mae_tx_encoder_embed_dim", "256",
    "--mae_tx_encoder_layers", "2",
    "--mae_tx_encoder_heads", "2",
    "--mae_tx_patch_len", "5",
    "--mae_tx_mask_ratio", "0.8",
]

_NON_ARCH_MAE_ARGS = [
    "--mae_tx_patch_norm", "false",
    "--mae_tx_n_eval_passes", "30",
    "--mae_tx_training_mode", "random_mask",
]

_COMMON_ARGS = [
    "--model", "MaeTx",
    "--dataset", "Psm",
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
    "--configurator_threshold_percentile", "72",
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

MODEL_SWEEP = Sweep(
    name="psm_model",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_psm_model",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *ARCH_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="psm_model", time="01:30:00"),
    trials=expand_trials(
        cross={
            "mae_tx_patch_len": ["5", "10", "20"],
            "mae_tx_mask_ratio": ["0.5", "0.65", "0.8"],
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

TRAIN_STANDARD_SWEEP = Sweep(
    name="psm_train_standard",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_psm_train_standard",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="psm_train_std", time="01:30:00"),
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
    name="psm_train_incremental",
    pipeline="IncrementalTaskArithmeticPipeline",
    experiment_name="slurm_grid_psm_train_incremental",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *_INCREMENTAL_EXTRA_ARGS,
               "--pipeline", "IncrementalTaskArithmeticPipeline"],
    slurm=SlurmConfig(job_name="psm_train_inc", time="03:00:00"),
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
