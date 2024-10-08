from dataenvgym.gym.accumulate_train_loop import (
    run_math_loop,
    run_math_loop_with_multiple_cycles,
    IoProvider,
    IterationMetadata,
    evaluate_predictor_on_tasks,
)
from dataenvgym.gym.domain_models import (
    MathTaskInterface,
    MathPredictorInterface,
    MathDataGenerationAgent,
    CompletedMathTaskInstance,
    MathTrainingDatum,
    TaskPerformanceReport,
    MathTaskInstance,
)
from typing import List, Collection, Sequence
from ulid import ULID
from pathlib import Path


class MockMathTask(MathTaskInterface):
    def __init__(self):
        self.task_instances = [
            MathTaskInstance(
                task_name="MATH",
                instance_id="test_id",
                instruction="What is 2+2?",
                ground_truth_label="The answer is \\boxed{4}.",
            )
        ]

    def evaluate(
        self, predictor: MathPredictorInterface
    ) -> List[CompletedMathTaskInstance]:
        responses = predictor.predict(self.task_instances)
        return [
            CompletedMathTaskInstance(
                ulid=ULID(),
                task_instance=self.task_instances[0],
                predictor_response=responses[0],
                was_correct=responses[0] == "The answer is \\boxed{4}.",
            )
        ]

    def generate_performance_report(
        self, completed_instances: Collection[CompletedMathTaskInstance]
    ) -> TaskPerformanceReport:
        overall_performance = sum(
            1 for instance in completed_instances if instance.was_correct
        ) / len(completed_instances)
        return TaskPerformanceReport(
            task_name="MATH", overall_performance=overall_performance, slices=[]
        )


class AlwaysRightMathPredictor:
    def __init__(self):
        self.train_count = 0

    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return ["The answer is \\boxed{4}."] * len(task_instances)

    def save(self, path: Path) -> None:
        pass

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        self.train_count += 1


class AlwaysWrongMathPredictor(MathPredictorInterface):
    def __init__(self):
        self.train_count = 0

    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return ["wrong"] * len(task_instances)

    def save(self, path: Path) -> None:
        pass

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        self.train_count += 1


class MockMathTrainingDataProductionStrategy(MathDataGenerationAgent):
    def __call__(
        self,
        completed_task_instances: List[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> List[MathTrainingDatum]:
        return [
            MathTrainingDatum(
                ulid=ULID(),
                instruction=instance.task_instance.instruction,
                response=instance.predictor_response,
            )
            for instance in completed_task_instances
        ]

    def step(self) -> None:
        return


def test_run_math_loop(tmp_path):
    mock_task = MockMathTask()
    mock_predictor = AlwaysRightMathPredictor()
    mock_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    final_val_reports, final_test_reports = run_math_loop(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=mock_strategy,
        io_provider=io_provider,
        accumulation_iterations=3,
    )

    assert len(final_val_reports) == 1
    assert len(final_test_reports) == 1
    assert final_val_reports[0].overall_performance == 1.0
    assert final_test_reports[0].overall_performance == 1.0

    # Check that the io_provider methods were called
    assert any(file.name == "training_data.jsonl" for file in tmp_path.rglob("*"))
    assert any(
        file.name == "val_performance_reports.jsonl" for file in tmp_path.rglob("*")
    )
    assert any(
        file.name == "test_performance_reports.jsonl" for file in tmp_path.rglob("*")
    )


def test_run_math_loop_with_always_wrong_predictor(tmp_path):
    mock_task = MockMathTask()
    mock_predictor = AlwaysWrongMathPredictor()
    mock_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    final_val_reports, final_test_reports = run_math_loop(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=mock_strategy,
        io_provider=io_provider,
        accumulation_iterations=3,
    )

    assert len(final_val_reports) == 1
    assert len(final_test_reports) == 1
    assert final_val_reports[0].overall_performance == 0.0
    assert final_test_reports[0].overall_performance == 0.0


def test_evaluate_predictor_on_tasks(tmp_path):
    mock_task = MockMathTask()
    mock_predictor = AlwaysRightMathPredictor()
    io_provider = IoProvider(tmp_path)
    iteration_metadata = IterationMetadata(iteration=0)

    completed_instances, performance_reports = evaluate_predictor_on_tasks(
        tasks=[mock_task],
        predictor=mock_predictor,
        io_provider=io_provider,
        iteration_metadata=iteration_metadata,
        val_or_test="val",
    )

    assert len(completed_instances) == 1
    assert len(performance_reports) == 1
    assert performance_reports[0].overall_performance == 1.0


def test_run_math_loop_with_multiple_cycles(tmp_path):
    mock_task = MockMathTask()
    mock_predictor = AlwaysRightMathPredictor()
    mock_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    final_val_reports, final_test_reports = run_math_loop_with_multiple_cycles(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=mock_strategy,
        io_provider=io_provider,
        accumulation_iterations_per_cycle=2,
        num_cycles=2,
    )

    assert len(final_val_reports) == 1
    assert len(final_test_reports) == 1
    assert final_val_reports[0].overall_performance == 1.0
    assert final_test_reports[0].overall_performance == 1.0

    # Check that the io_provider methods were called
    assert any(file.name == "training_data.jsonl" for file in tmp_path.rglob("*"))
    assert any(
        file.name == "val_performance_reports.jsonl" for file in tmp_path.rglob("*")
    )
    assert any(
        file.name == "test_performance_reports.jsonl" for file in tmp_path.rglob("*")
    )

    # Check that the predictor was trained multiple times
    assert mock_predictor.train_count == 2


def test_run_math_loop_with_multiple_cycles_always_wrong_predictor(tmp_path):
    mock_task = MockMathTask()
    mock_predictor = AlwaysWrongMathPredictor()
    mock_strategy = MockMathTrainingDataProductionStrategy()
    io_provider = IoProvider(tmp_path)

    final_val_reports, final_test_reports = run_math_loop_with_multiple_cycles(
        validation_math_tasks=[mock_task],
        test_math_tasks=[mock_task],
        trainable_math_predictor=mock_predictor,
        training_data_production_strategy=mock_strategy,
        io_provider=io_provider,
        accumulation_iterations_per_cycle=2,
        num_cycles=2,
    )

    assert len(final_val_reports) == 1
    assert len(final_test_reports) == 1
    assert final_val_reports[0].overall_performance == 0.0
    assert final_test_reports[0].overall_performance == 0.0

    # Check that the predictor was trained multiple times
    assert mock_predictor.train_count == 2
