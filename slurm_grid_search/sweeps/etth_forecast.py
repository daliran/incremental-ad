"""ETTh1 forecasting sweeps: model architecture, then training parameters for both
pipelines.

Base config anchors on the recipe found by the (now-superseded) local grid search,
already reflected in scripts/sbatch_mae_tx_etth_forecast_*.sh and EXPERIMENTS.md:
patch_len=4, embed_dim=128, encoder_layers=3, encoder_heads=4, decoder_embed_dim=64,
decoder_layers=2, decoder_heads=4, instance_norm=false. window_len=120/forecast_len=24
is the "96-24" context/horizon convention.

mask_ratio is NOT part of MODEL_SWEEP's headline cross here, unlike swat.py/psm.py: this
dataset trains with --mae_tx_training_mode causal_mask, and CLAUDE.md's own gotcha notes
that causal_mask forecasting masks exactly forecast_patches (derived from
forecast_len // patch_len), not mask_ratio -- mask_ratio is a required CLI arg but has
*zero effect* on this task/mode. Sweeping it here would be a no-op. patch_len x
encoder_embed_dim are the headline cross instead.
"""

from __future__ import annotations

from harness import Sweep, SlurmConfig, expand_trials

ARCH_ARGS = [
    "--mae_tx_decoder_embed_dim", "64",
    "--mae_tx_decoder_layers", "2",
    "--mae_tx_decoder_heads", "4",
    "--mae_tx_encoder_embed_dim", "128",
    "--mae_tx_encoder_layers", "3",
    "--mae_tx_encoder_heads", "4",
    "--mae_tx_patch_len", "4",
    # required but functionally a no-op for causal_mask forecasting -- see module docstring.
    "--mae_tx_mask_ratio", "0.5",
]

_NON_ARCH_MAE_ARGS = [
    "--mae_tx_patch_norm", "false",
    "--mae_tx_n_eval_passes", "1",
    "--mae_tx_training_mode", "causal_mask",
    "--mae_tx_instance_norm", "false",  # found to win over true at this horizon -- EXPERIMENTS.md
]

_COMMON_ARGS = [
    "--model", "MaeTx",
    "--dataset", "EtthForecastDataset",
    "--task", "forecast",

    "--dataset_window_len", "120",
    "--dataset_forecast_len", "24",
    "--dataset_stride", "1",
    "--dataset_normalization", "standard",
    "--dataset_val_fraction", "0.1",
    "--dataset_test_fraction", "0.2",

    "--loader_batch_size", "128",
    "--loader_num_workers", "4",

    "--trainer_n_epochs", "100",
    "--trainer_patience", "15",
    "--trainer_optimizer", "adamw",
    "--trainer_weight_decay", "1e-4",
    "--trainer_learning_rate", "3e-4",
    "--trainer_grad_clip", "1.0",
    "--trainer_scheduler", "cosine",
    "--trainer_warmup_ratio", "0.1",
    "--trainer_checkpoint_interval", "0",
]

_INCREMENTAL_EXTRA_ARGS = [
    "--dataset_baseline_fraction", "0.5",
    "--dataset_baseline_use_fraction", "1.0",
    "--dataset_n_finetune_segments", "3",

    "--finetune_trainer_n_epochs", "30",
    "--finetune_trainer_patience", "10",
    "--finetune_trainer_weight_decay", "1e-4",
    "--finetune_trainer_learning_rate", "1e-4",
    "--finetune_trainer_grad_clip", "1.0",
    "--finetune_trainer_scheduler", "cosine",
]

# patch_len must divide both dataset_window_len=120 and dataset_forecast_len=24 --
# divisors of gcd(120,24)=24 used below: 4, 6, 8.
#
# Full grid actually run this session (24 trials, see etth_forecast_model_results.csv):
# patch_len=4 and encoder_embed_dim in {64,128} came out best (256 and patch_len 6/8 all
# consistently worse); decoder_embed_dim=128 looked like a further win at seed=42 alone,
# but a 3-seed check (SEED_REPRO_CHECK below) showed that ranking flips at seed 123 and
# doesn't hold on average -- decoder_embed_dim stays at 64. See EXPERIMENTS.md §1.8.
MODEL_SWEEP = Sweep(
    name="etth_forecast_model",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_etth_forecast_model",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *ARCH_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="etth_fc_model", time="02:00:00"),
    trials=expand_trials(
        cross={
            "mae_tx_patch_len": ["4", "6", "8"],
            "mae_tx_encoder_embed_dim": ["64", "128", "256"],
        },
        one_at_a_time={
            "mae_tx_encoder_layers": ["2", "4"],
            "mae_tx_encoder_heads": ["2"],
            "mae_tx_decoder_layers": ["1", "3"],
            "mae_tx_decoder_heads": ["2"],
            "mae_tx_decoder_embed_dim": ["32", "128"],
        },
    ),
)

# One-off diagnostic (not wired into submit.py's --stage dispatch -- run directly via
# harness.submit_sweep, see EXPERIMENTS.md §1.8/§4): repro-checks the apparent
# decoder_embed_dim=128 winner against the base recipe's 64 across 3 seeds. Kept here,
# not folded into MODEL_SWEEP, since it varies `seeds` per-trial rather than sweeping an
# architecture axis -- mixing the two would silently apply 3 seeds to every MODEL_SWEEP
# trial. Result: 128 wins at seeds 42/7, loses badly at 123: not a real effect.
SEED_REPRO_CHECK = Sweep(
    name="etth_forecast_model",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_etth_forecast_model",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *ARCH_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="etth_fc_model", time="02:00:00"),
    trials=[
        {"mae_tx_decoder_embed_dim": "64"},
        {"mae_tx_decoder_embed_dim": "128"},
    ],
    seeds=["42", "123", "7"],
)

TRAIN_STANDARD_SWEEP = Sweep(
    name="etth_forecast_train_standard",
    pipeline="StandardPipeline",
    experiment_name="slurm_grid_etth_forecast_train_standard",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, "--pipeline", "StandardPipeline"],
    slurm=SlurmConfig(job_name="etth_fc_train_std", time="02:00:00"),
    trials=expand_trials(
        cross={
            "dataset_val_fraction": ["0.05", "0.1", "0.15"],
            "trainer_learning_rate": ["1e-3", "3e-4"],
        },
        one_at_a_time={
            "trainer_weight_decay": ["1e-3"],
        },
    ),
)

TRAIN_INCREMENTAL_SWEEP = Sweep(
    name="etth_forecast_train_incremental",
    pipeline="IncrementalTaskArithmeticPipeline",
    experiment_name="slurm_grid_etth_forecast_train_incremental",
    base_args=[*_COMMON_ARGS, *_NON_ARCH_MAE_ARGS, *_INCREMENTAL_EXTRA_ARGS,
               "--pipeline", "IncrementalTaskArithmeticPipeline"],
    slurm=SlurmConfig(job_name="etth_fc_train_inc", time="01:30:00"),
    # dataset_val_fraction=0.05 alone (default n_finetune_segments=3) fails a split-size
    # guard -- each finetune segment's val slice ends up shorter than window_len=120. Paired
    # here with n_finetune_segments=2 instead, which clears the guard (see EXPERIMENTS.md
    # §1.9); the plain val_fraction=0.05/segments=3 combo is intentionally not listed since
    # it deterministically fails, not a trial worth resubmitting.
    trials=expand_trials(
        cross={
            "pipeline_merge_scale": ["0.3", "0.5", "1.0"],
            "finetune_trainer_reg_lambda": ["0.0", "1e-3", "1e-2"],
        },
        one_at_a_time={
            "dataset_baseline_fraction": ["0.3", "0.7"],
            "dataset_val_fraction": ["0.15"],
            "dataset_n_finetune_segments": ["2", "5"],
        },
    ) + [{"dataset_val_fraction": "0.05", "dataset_n_finetune_segments": "2"}],
)
