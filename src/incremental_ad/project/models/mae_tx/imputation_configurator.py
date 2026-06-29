from incremental_ad.framework.contracts.dataset import Dataset, ImputationDataset, TimeSeriesDataset
from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.project.task import Task
from incremental_ad.framework.evaluators.imputation_evaluator import ImputationEvaluator
from incremental_ad.project.models.mae_tx.mae import InferenceMode, MaeTx


@TaskModelConfigurator.register(Task.IMPUTATION, MaeTx)
class MaeTxImputationConfigurator(TaskModelConfigurator):
    """Switches MaeTx into imputation mode.

    Training is self-supervised (random-mask MAE, same as AD). At inference the
    dataset supplies a fixed per-window mask and visible_idx; the model encodes
    only visible patches and predicts at masked positions only.

    The dataset must expose mask_patch_len (via ImputationDataset protocol) and it
    must equal model.config.patch_len — otherwise masked token positions would not
    align with the model's tokenisation.
    """

    def _configure(self, model: Model, dataset: Dataset) -> None:
        assert isinstance(model, MaeTx), f"expected MaeTx, got {type(model).__name__}"
        
        assert isinstance(dataset, TimeSeriesDataset), (
            f"expected TimeSeriesDataset, got {type(dataset).__name__}"
        )

        assert isinstance(dataset, ImputationDataset), (
            f"{type(dataset).__name__} must expose a 'mask_patch_len' property "
            "(implement the ImputationDataset protocol) for use with MaeTxImputationConfigurator."
        )

        if dataset.mask_patch_len != model.config.patch_len:
            raise ValueError(
                f"patch_len mismatch: dataset.mask_patch_len={dataset.mask_patch_len} "
                f"but model.config.patch_len={model.config.patch_len}. "
                "Set --dataset_patch_len equal to --mae_tx_patch_len."
            )
        
        if model.config.patch_norm:
            raise ValueError(
                "patch_norm=True is not supported for imputation: the decoder outputs patches "
                "in per-patch-normalized space, but the evaluator compares against raw values. "
                "Set --mae_tx_patch_norm False."
            )
        
        model.n_features = dataset.n_features
        model.seq_len = dataset.window_len
        model.inference_mode = InferenceMode.IMPUTATION

    def create_val_evaluator(self) -> Evaluator:
        return ImputationEvaluator()

    def create_test_evaluator(self) -> Evaluator:
        return ImputationEvaluator()
