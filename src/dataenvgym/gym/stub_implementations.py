from pathlib import Path
from typing import Collection, Sequence
from ulid import ULID
from .domain_models import (
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    VqaTrainingDatum,
    TrainableVqaPredictorInterface,
    implements,
)
from typing import Type, TypeVar, Collection, Callable
from functools import wraps


# @implements2(TrainableVqaPredictorInterface)
class StubTrainablePredictor:
    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        # Placeholder implementation for prediction
        return ["dummy_response"] * len(task_instances)

    def train(self, training_data: Sequence[VqaTrainingDatum]) -> None:
        # Placeholder implementation for training
        print(f"Training on {len(training_data)} instances.")

    def save(self, path: Path) -> None:
        # Placeholder implementation for saving the model
        print(f"Saving model to {path}")


implements(TrainableVqaPredictorInterface)(StubTrainablePredictor)
