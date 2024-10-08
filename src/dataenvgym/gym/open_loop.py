from ulid import ULID
from typing import Collection, Sequence, Callable
import random
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .domain_models import (
    OpenEndedVqaTaskInstance,
    SerializableImage,
    CompletedVqaTaskInstance,
    VqaPredictorInterface,
    TaskPerformanceReport,
    VqaTrainingDatum,
    VqaTaskInterface,
    VqaDataGenerationAgent,
    TrainableVqaPredictorInterface,
    MultipleChoiceVqaTaskInstance,
    VqaPreferenceDataGenerationAgent,
    VqaPreferenceTrainingDatum,
    PreferenceTrainableVqaPredictorInterface,
    PreferenceVqaTrainerInterface,
    implements,
)


class PredictColorVqaTask:
    def __init__(
        self,
        length: int = 8,
        image_size: tuple[int, int] = (256, 256),
        color: str = "turquoise",
    ):
        self.length = length
        self.image_size = image_size
        self.task_instances = [
            OpenEndedVqaTaskInstance(
                task_name="random_vqa",
                instance_id=str(_),
                instruction="What is the color of the static?",
                image=SerializableImage.from_random(image_size).pil_image,
                ground_truth_label=color,
            )
            for _ in range(length)
        ]

    def evaluate(
        self, predictor: VqaPredictorInterface
    ) -> Collection[CompletedVqaTaskInstance]:
        predictor_responses = predictor.predict(self.task_instances)
        return [
            CompletedVqaTaskInstance(
                ulid=ULID(),
                task_instance=task_instance,
                predictor_response=predictor_response,
                was_correct=predictor_response == task_instance.ground_truth_label,
            )
            for task_instance, predictor_response in zip(
                self.task_instances, predictor_responses
            )
        ]

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> TaskPerformanceReport:
        return TaskPerformanceReport(
            task_name="PredictColorVqaTask",
            overall_performance=sum(
                completed_task_instance.was_correct
                for completed_task_instance in completed_task_instances
            )
            / len(completed_task_instances),
            slices=[],
        )


class RandomTrainingDataProductionStrategy:
    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDatum]:
        return [
            VqaTrainingDatum(
                ulid=ULID(),
                instruction=completed_task_instance.task_instance.instruction,
                image=SerializableImage.from_random(),
                response="".join(
                    random.choices(string.ascii_letters + string.digits, k=10)
                ),
            )
            for completed_task_instance in completed_task_instances
        ]

    def step(self):
        pass


class TrainingDataOfSpecificColorProductionStrategy:
    def __init__(self, color: str):
        self.color = color

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: VqaPredictorInterface,
    ) -> Sequence[VqaTrainingDatum]:
        return [
            VqaTrainingDatum(
                ulid=ULID(),
                instruction=completed_task_instance.task_instance.instruction,
                image=SerializableImage.from_pil_image(
                    completed_task_instance.task_instance.image
                ),
                response=self.color,
            )
            for completed_task_instance in completed_task_instances
        ]

    def step(self):
        pass


implements(VqaDataGenerationAgent)(TrainingDataOfSpecificColorProductionStrategy)


class PreferenceTrainingDataOfSpecificColorProductionStrategy:
    def __init__(self, color: str):
        self.color = color

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: VqaPredictorInterface,
    ) -> Sequence[VqaPreferenceTrainingDatum]:
        return [
            VqaPreferenceTrainingDatum(
                ulid=ULID(),
                instruction=completed_task_instance.task_instance.instruction,
                image=SerializableImage.from_pil_image(
                    completed_task_instance.task_instance.image
                ),
                chosen_response=self.color,
                rejected_response="".join(
                    random.choices(string.ascii_letters + string.digits, k=10)
                ),
            )
            for completed_task_instance in completed_task_instances
        ]

    def step(self):
        pass


implements(VqaPreferenceDataGenerationAgent)(
    PreferenceTrainingDataOfSpecificColorProductionStrategy
)


class PredictMostCommonResponseTrainablePredictor(TrainableVqaPredictorInterface):
    def __init__(self):
        self.counter = Counter()

    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        if len(self.counter) == 0:
            return [""] * len(task_instances)
        return [self.counter.most_common(1)[0][0] for _ in task_instances]

    def train(self, training_data: Collection[VqaTrainingDatum]) -> None:
        for data in training_data:
            self.counter[data.response] += 1

    def save(self, path: Path):
        pass


class PreferencePredictMostCommonResponseTrainablePredictor:
    def __init__(self):
        self.counter = Counter()

    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        if len(self.counter) == 0:
            return [""] * len(task_instances)
        return [self.counter.most_common(1)[0][0] for _ in task_instances]

    def train_preference(
        self,
        training_data: Sequence[VqaPreferenceTrainingDatum],
    ) -> None:
        for data in training_data:
            self.counter[data.chosen_response] += 1

    def save(self, path: Path) -> None:
        pass


implements(PreferenceTrainableVqaPredictorInterface)(
    PreferencePredictMostCommonResponseTrainablePredictor
)


class PredictConstantAnswerTrainablePredictor:
    def __init__(self, constant_answer: str = "foo"):
        self.constant_answer = constant_answer

    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        for task_instance in task_instances:
            task_instance.image.verify()
        return [self.constant_answer] * len(task_instances)

    def train(self, training_data: Collection[VqaTrainingDatum]) -> None:
        pass


@dataclass
class OutputSink:
    completed_val_task_instances: Callable[[Collection[CompletedVqaTaskInstance]], None]
    completed_test_task_instances: Callable[
        [Collection[CompletedVqaTaskInstance]], None
    ]
    training_data: Callable[[Collection[VqaTrainingDatum]], None]
    performance_reports: Callable[[Collection[TaskPerformanceReport]], None]
    pre_performance_reports: Callable[[Collection[TaskPerformanceReport]], None]


NULL_OUTPUT_SINK = OutputSink(
    completed_val_task_instances=lambda _: None,
    completed_test_task_instances=lambda _: None,
    training_data=lambda _: None,
    performance_reports=lambda _: None,
    pre_performance_reports=lambda _: None,
)


def run_open_loop(
    validation_vqa_tasks: Collection[VqaTaskInterface],
    test_vqa_tasks: Collection[VqaTaskInterface],
    trainable_vqa_predictor: TrainableVqaPredictorInterface,
    training_data_production_strategy: VqaDataGenerationAgent,
    write_outputs: OutputSink = NULL_OUTPUT_SINK,
) -> Collection[TaskPerformanceReport]:

    completed_val_task_instances: list[CompletedVqaTaskInstance] = []
    pre_performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in validation_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        pre_performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )
        write_outputs.completed_val_task_instances(completed_instances_for_task)
        completed_val_task_instances.extend(completed_instances_for_task)
    write_outputs.pre_performance_reports(pre_performance_reports)

    training_data = training_data_production_strategy(
        completed_val_task_instances, trainable_vqa_predictor
    )
    write_outputs.training_data(training_data)
    trainable_vqa_predictor.train(training_data)

    completed_test_task_instances: list[CompletedVqaTaskInstance] = []
    performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in test_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        write_outputs.completed_test_task_instances(completed_instances_for_task)
        completed_test_task_instances.extend(completed_instances_for_task)
        performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )

    write_outputs.performance_reports(performance_reports)
    return performance_reports
