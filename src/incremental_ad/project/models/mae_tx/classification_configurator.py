from incremental_ad.framework.contracts.dataset import ClassificationDataset, Dataset, TimeSeriesDataset
from incremental_ad.framework.contracts.evaluator import Evaluator
from incremental_ad.framework.contracts.model import Model, TaskModelConfigurator
from incremental_ad.framework.contracts.task import Task
from incremental_ad.framework.evaluators.classification_evaluator import ClassificationEvaluator
from incremental_ad.project.models.mae_tx.mae_classifier import MaeTxClassifier


@TaskModelConfigurator.register(Task.CLASSIFICATION, MaeTxClassifier)
class MaeTxClassificationConfigurator(TaskModelConfigurator):
    """Configures MaeTxClassifier for supervised time-series classification.

    Injects n_features, seq_len, and n_classes so that _build() can construct
    both the MAE backbone and the classification head.
    """

    def _configure(self, model: Model, dataset: Dataset) -> None:
        assert isinstance(model, MaeTxClassifier), (
            f"expected MaeTxClassifier, got {type(model).__name__}"
        )

        assert isinstance(dataset, TimeSeriesDataset), (
            f"expected TimeSeriesDataset, got {type(dataset).__name__}"
        )

        assert isinstance(dataset, ClassificationDataset), (
            f"{type(dataset).__name__} must expose a 'n_classes' property "
            "(implement the ClassificationDataset protocol) for use with MaeTxClassificationConfigurator."
        )
        
        model.n_features = dataset.n_features
        model.seq_len = dataset.window_len
        model.n_classes = dataset.n_classes

    def create_val_evaluator(self) -> Evaluator:
        return ClassificationEvaluator()

    def create_test_evaluator(self) -> Evaluator:
        return ClassificationEvaluator()
