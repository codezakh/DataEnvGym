from dataenvgym.gym.domain_models import (
    VqaTrainingDatum,
    implements,
    VqaDataGenerationAgent,
    VqaPredictorInterface,
    CompletedVqaTaskInstance,
    SerializableImage,
    PredictorInterface,
    OpenEndedVqaTaskInstance,
)
from dataenvgym.gym.tasks.vqa.gqa import GqaTask
from typing import Collection, Sequence
import random
from ulid import ULID


class RandomSelectionDataGenerationAgent:
    def __init__(self, datum_to_generate_per_error: int = 3):
        self.task_instances = GqaTask(split="train_100k").task_instances
        self.datum_to_generate_per_error = datum_to_generate_per_error

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: PredictorInterface[OpenEndedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDatum]:
        errors = [_ for _ in completed_task_instances if not _.was_correct]

        num_training_data_to_generate = len(errors) * self.datum_to_generate_per_error

        training_data: list[VqaTrainingDatum] = []
        for _ in range(num_training_data_to_generate):
            task_instance = random.choice(self.task_instances)

            if isinstance(task_instance.ground_truth_label, str):
                response = task_instance.ground_truth_label
            else:
                response = random.choice(task_instance.ground_truth_label)

            training_datum = VqaTrainingDatum(
                ulid=ULID(),
                instruction=task_instance.instruction,
                image=SerializableImage.from_pil_image(task_instance.image),
                response=response,
            )
            training_data.append(training_datum)

        return training_data

    def step(self) -> None:
        pass


implements(VqaDataGenerationAgent)(RandomSelectionDataGenerationAgent)
