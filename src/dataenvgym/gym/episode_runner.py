from typing import Collection, TypeVar, Union, Sequence, Optional, Literal, Mapping
from dataenvgym.gym.domain_models import (
    TaskPerformanceReport,
    CompletedVqaTaskInstance,
    VqaTrainingDatum,
    VqaPreferenceTrainingDatum,
    MathTrainingDatum,
    CompletedMathTaskInstance,
    MathTaskInterface,
    TrainableMathPredictorInterface,
    MathDataGenerationAgent,
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTaskInterface,
    TrainableCodeGenerationPredictorInterface,
    CodeGenerationDataGenerationAgent,
    CodeGenerationTrainingDatum,
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    MathTaskInstance,
    CodeGenerationTaskInstance,
    TaskInstanceCovariant,
    TaskInterface,
    CompletedTaskInstance,
    PredictorInterface,
    TrainablePredictorInterface,
    VqaDataGenerationAgent,
    TrainingDatum,
    DataGenerationAgent,
    TaskInstance,
)
from pydantic import BaseModel
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesWriter
from enum import Enum
from dataenvgym.gym.environments.base_environment import EnvironmentInterface


# TODO: This currently has no notion of order.
# It should be changed to clearly indicate that these are
# stages in the pipeline that happen one afte the other.
class OutputPath(Enum):
    completed_test_task_instances = "completed_test_task_instances.jsonl"
    completed_val_task_instances = "completed_val_task_instances.jsonl"
    training_data = "training_data.jsonl"
    val_performance_reports = "val_performance_reports.jsonl"
    test_performance_reports = "test_performance_reports.jsonl"
    model_save_path = "trainable_predictor"
    test_performance_history = "test_performance_history.jsonl"
    val_performance_history = "val_performance_history.jsonl"


# Currently this has no other fields, but we might add some in the future.
class IterationMetadata(BaseModel):
    iteration: int


class IoProvider:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir

    def get_output_path_for_category(
        self, iteration_metadata: IterationMetadata, output_category: OutputPath
    ) -> Path:
        output_path = (
            self.experiment_dir
            / Path(f"iteration_{iteration_metadata.iteration}")
            / output_category.value
        )
        return output_path

    def make_and_return_output_path(
        self, iteration_metadata: IterationMetadata, output_category: OutputPath
    ) -> Path:
        output_path = self.get_output_path_for_category(
            iteration_metadata, output_category
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def write_completed_val_task_instances(
        self,
        completed_val_task_instances: Collection[CompletedTaskInstance],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.completed_val_task_instances
        )

        writer: PydanticJSONLinesWriter[CompletedTaskInstance] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(completed_val_task_instances)

    def write_completed_test_task_instances(
        self,
        completed_test_task_instances: Collection[CompletedTaskInstance],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.completed_test_task_instances
        )
        writer: PydanticJSONLinesWriter[CompletedTaskInstance] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(completed_test_task_instances)

    def write_training_data(
        self,
        training_data: Collection[TrainingDatum],
        iteration_metadata: IterationMetadata,
    ):

        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.training_data
        )

        writer: PydanticJSONLinesWriter[TrainingDatum] = PydanticJSONLinesWriter(
            output_path
        )

        writer.write_batch(training_data)

    def write_test_performance_reports(
        self,
        pre_performance_reports: Collection[TaskPerformanceReport],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.test_performance_reports
        )
        writer: PydanticJSONLinesWriter[TaskPerformanceReport] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(pre_performance_reports)

    def write_val_performance_reports(
        self,
        performance_reports: Collection[TaskPerformanceReport],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.val_performance_reports
        )

        writer: PydanticJSONLinesWriter[TaskPerformanceReport] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(performance_reports)

    def get_trainable_predictor_save_path(
        self, iteration_metadata: IterationMetadata
    ) -> Path:
        return self.make_and_return_output_path(
            iteration_metadata, OutputPath.model_save_path
        )

    def get_performance_report_path(
        self, iteration_metadata: IterationMetadata, val_or_test: Literal["val", "test"]
    ) -> Path:
        return self.get_output_path_for_category(
            iteration_metadata,
            (
                OutputPath.val_performance_reports
                if val_or_test == "val"
                else OutputPath.test_performance_reports
            ),
        )

    def save_trainable_predictor(
        self,
        trainable_predictor: TrainablePredictorInterface[TaskInstance, TrainingDatum],
        iteration_metadata: IterationMetadata,
    ):
        save_path = self.get_trainable_predictor_save_path(iteration_metadata)
        trainable_predictor.save(save_path)


def run_episode(
    environment: EnvironmentInterface[
        CompletedTaskInstance, TrainingDatum, TaskInstanceCovariant
    ],
    data_generation_agent: DataGenerationAgent[
        CompletedTaskInstance, TrainingDatum, TaskInstanceCovariant
    ],
    io_provider: IoProvider,
    stop_at_num_training_data: Optional[int] = None,
    num_iterations: int = 10,
) -> Mapping[int, Sequence[TaskPerformanceReport]]:

    val_performance_history: dict[int, Sequence[TaskPerformanceReport]] = {}

    iteration_metadata = IterationMetadata(iteration=0)
    completed_val_task_instances, val_performance_reports = environment.reset()
    val_performance_history[iteration_metadata.iteration] = val_performance_reports
    io_provider.write_completed_val_task_instances(
        completed_val_task_instances, iteration_metadata
    )
    io_provider.write_val_performance_reports(
        val_performance_reports, iteration_metadata
    )

    for iteration in range(1, num_iterations + 1):
        iteration_metadata = IterationMetadata(iteration=iteration)
        training_data = data_generation_agent(
            completed_val_task_instances, environment.trainable_predictor
        )
        io_provider.write_training_data(training_data, iteration_metadata)

        completed_val_task_instances, val_performance_reports = environment.step(
            training_data
        )
        val_performance_history[iteration_metadata.iteration] = val_performance_reports
        io_provider.write_completed_val_task_instances(
            completed_val_task_instances, iteration_metadata
        )
        io_provider.write_val_performance_reports(
            val_performance_reports, iteration_metadata
        )

        io_provider.save_trainable_predictor(
            environment.trainable_predictor, iteration_metadata
        )

        stop_condition_met = (
            stop_at_num_training_data is not None
            and len(training_data) >= stop_at_num_training_data
        )

        if stop_condition_met:
            break

    return val_performance_history
