from dataenvgym.gym.tasks.vqa.naturalbench import NaturalBenchTask
from dataenvgym.gym.domain_models import (
    OpenEndedVqaTaskInstance,
    implements,
    VqaPredictorInterface,
    MultipleChoiceVqaTaskInstance,
)
from typing import Sequence


class AlwaysCorrectPredictor:
    def predict(
        self,
        task_instances: Sequence[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        responses: list[str] = []
        for task_instance in task_instances:
            if isinstance(task_instance.ground_truth_label, str):
                responses.append(task_instance.ground_truth_label)
            else:
                responses.append(task_instance.ground_truth_label[0])
        return responses


implements(VqaPredictorInterface)(AlwaysCorrectPredictor)


class AlwaysIncorrectPredictor:
    def predict(
        self,
        task_instances: Sequence[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        return ["spongebob"] * len(task_instances)


implements(VqaPredictorInterface)(AlwaysIncorrectPredictor)


def test_naturalbench_task_always_correct():
    task = NaturalBenchTask()
    predictor = AlwaysCorrectPredictor()
    completed_task_instances = task.evaluate(predictor)
    for completed_task_instance in completed_task_instances:
        assert completed_task_instance.was_correct


def test_naturalbench_task_always_incorrect():
    task = NaturalBenchTask()
    predictor = AlwaysIncorrectPredictor()
    completed_task_instances = task.evaluate(predictor)
    for completed_task_instance in completed_task_instances:
        assert not completed_task_instance.was_correct


def test_generating_performance_report():
    task = NaturalBenchTask()
    predictor = AlwaysCorrectPredictor()
    completed_task_instances = task.evaluate(predictor)
    report = task.generate_performance_report(completed_task_instances)
    assert report.overall_performance == 1.0
