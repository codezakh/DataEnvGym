import random
from typing import Collection, Sequence
from ulid import ULID

from dataenvgym.gym.domain_models import (
    CompletedMathTaskInstance,
    MathDataGenerationAgent,
    MathTrainingDatum,
    implements,
    MathPredictorInterface,
)
from dataenvgym.gym.tasks.math.MATH.task import MATHTask


class RandomSelectionDataGenerationAgent:
    def __init__(self, num_training_data_per_invocation: int):
        self.num_training_data_per_invocation = num_training_data_per_invocation
        self.task_instances = MATHTask("train_all").task_instances

    def generate_training_data(
        self,
        completed_task_instances: Collection[CompletedMathTaskInstance],
    ) -> Sequence[MathTrainingDatum]:
        # Randomly select a subset of task instances
        selected_instances = random.sample(
            self.task_instances, self.num_training_data_per_invocation
        )

        # Convert selected task instances to training data
        training_data = [
            MathTrainingDatum(
                ulid=ULID(),
                instruction=instance.instruction,
                response=instance.ground_truth_label,
            )
            for instance in selected_instances
        ]

        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]:
        return self.generate_training_data(completed_task_instances)

    def step(self) -> None:
        # This method can be used to update any internal state if needed
        pass


implements(MathDataGenerationAgent)(RandomSelectionDataGenerationAgent)
