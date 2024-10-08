from src.dataenvgym.gym.domain_models import (
    MathPredictorInterface,
    CompletedMathTaskInstance,
    MathTaskInstance,
    implements,
)
from src.dataenvgym.utils import PydanticJSONLinesReader
from typing import Sequence
from pathlib import Path


class RewindPredictor:
    def __init__(self, path_to_replayable_predictions: str | Path):
        self.file_path = path_to_replayable_predictions
        self.previous_predictions = self._load_previous_predictions()

    def _load_previous_predictions(self) -> dict[str, str]:
        reader = PydanticJSONLinesReader(self.file_path, CompletedMathTaskInstance)
        predictions = {}
        for completed_instance in reader():
            predictions[completed_instance.task_instance.instance_id] = (
                completed_instance.predictor_response
            )
        return predictions

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return [
            self.previous_predictions[task_instance.instance_id]
            for task_instance in task_instances
        ]


implements(MathPredictorInterface)(RewindPredictor)
