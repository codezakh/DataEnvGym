from unittest.mock import MagicMock
from pathlib import Path
from typing import Sequence, Collection
from ulid import ULID
from dataenvgym.gym.closed_loop import (
    single_math_iteration,
    run_math_closed_loop,
    IterationMetadata,
    IoProvider,
)
from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathTrainingDatum,
    CompletedMathTaskInstance,
    MathPredictorInterface,
    MathTrainerInterface,
    TrainableMathPredictorInterface,
    MathTaskInterface,
    MathDataGenerationAgent,
    MathTaskPerformanceReport,
    implements,
)


class MockMathTask:
    def __init__(self, num_instances: int, correct_label: str = "42"):
        self.num_instances = num_instances
        self.correct_label = correct_label
        self.task_instances = [
            MathTaskInstance(
                task_name="mock_math_task",
                instance_id=str(i),
                instruction=f"Math task {i}",
                ground_truth_label=self.correct_label,
            )
            for i in range(num_instances)
        ]

    def evaluate(
        self, predictor: MathPredictorInterface
    ) -> Collection[CompletedMathTaskInstance]:
        predictor_responses = predictor.predict(self.task_instances)
        return [
            CompletedMathTaskInstance(
                ulid=ULID(),
                task_instance=task_instance,
                predictor_response=response,
                was_correct=response == task_instance.ground_truth_label,
            )
            for task_instance, response in zip(self.task_instances, predictor_responses)
        ]

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
    ) -> MathTaskPerformanceReport:
        correct_count = sum(
            instance.was_correct for instance in completed_task_instances
        )
        overall_performance = correct_count / len(completed_task_instances)
        return MathTaskPerformanceReport(
            task_name="MockMathTask",
            overall_performance=overall_performance,
            slices=[],
        )


implements(MathTaskInterface)(MockMathTask)


class MockMathPredictor(MathPredictorInterface, MathTrainerInterface):
    def __init__(self, correct_label="42"):
        self.correct_label = correct_label

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return [self.correct_label] * len(task_instances)

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        pass

    def save(self, path: Path) -> None:
        pass


implements(TrainableMathPredictorInterface)(MockMathPredictor)


class MockMathTrainingDataProductionStrategy:
    def __call__(
        self,
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]:
        return [
            MathTrainingDatum(
                ulid=ULID(),
                instruction=instance.task_instance.instruction,
                response=instance.predictor_response,
            )
            for instance in completed_task_instances
        ]

    def step(self):
        pass


implements(MathDataGenerationAgent)(MockMathTrainingDataProductionStrategy)


def test_single_math_iteration(tmp_path):
    mock_task = MockMathTask(num_instances=5)
    mock_predictor = MockMathPredictor()
    data_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    performance_reports = single_math_iteration(
        iteration_metadata=IterationMetadata(iteration=1),
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=data_strategy,
        io_provider=io_provider,
    )

    assert len(performance_reports) == 1
    assert performance_reports[0].overall_performance == 1.0


def test_entire_math_closed_loop(tmp_path):
    mock_task = MockMathTask(num_instances=5)
    mock_predictor = MockMathPredictor()
    data_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    performance_reports = run_math_closed_loop(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=data_strategy,
        num_iterations=10,
        io_provider=io_provider,
    )

    assert len(performance_reports) == 10
    for iteration, performance_reports in performance_reports.items():
        for performance_report in performance_reports:
            print(iteration)
            assert performance_report.overall_performance == 1.0


def test_math_closed_loop_steps_data_strategy(tmp_path, monkeypatch):
    mock_task = MockMathTask(num_instances=5)
    mock_predictor = MockMathPredictor()
    data_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    step_mock = MagicMock()
    monkeypatch.setattr(data_strategy, "step", step_mock)

    run_math_closed_loop(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=data_strategy,
        num_iterations=10,
        io_provider=io_provider,
    )

    assert step_mock.call_count == 10
