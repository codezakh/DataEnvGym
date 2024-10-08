from typing import Collection, TypeVar, Union, Sequence, Optional, Literal
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
)
from pydantic import BaseModel
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesWriter
from enum import Enum
from loguru import logger


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


DatumType = TypeVar(
    "DatumType",
    bound=Union[
        VqaTrainingDatum,
        VqaPreferenceTrainingDatum,
        MathTrainingDatum,
        CodeGenerationTrainingDatum,
    ],
)

CompletedTaskInstanceType = TypeVar(
    "CompletedTaskInstanceType",
    bound=Union[
        CompletedVqaTaskInstance,
        CompletedMathTaskInstance,
        CodeGenerationCompletedTaskInstance,
    ],
)


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
        completed_val_task_instances: Collection[CompletedTaskInstanceType],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.completed_val_task_instances
        )

        writer: PydanticJSONLinesWriter[CompletedTaskInstanceType] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(completed_val_task_instances)

    def write_completed_test_task_instances(
        self,
        completed_test_task_instances: Collection[CompletedTaskInstanceType],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.completed_test_task_instances
        )
        writer: PydanticJSONLinesWriter[CompletedTaskInstanceType] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(completed_test_task_instances)

    def write_training_data(
        self,
        training_data: Collection[DatumType],
        iteration_metadata: IterationMetadata,
    ):

        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.training_data
        )

        writer: PydanticJSONLinesWriter[DatumType] = PydanticJSONLinesWriter(
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


def evaluate_predictor_on_tasks(
    tasks: Collection[TaskInterface[CompletedTaskInstance, TaskInstanceCovariant]],
    predictor: PredictorInterface[TaskInstanceCovariant],
    io_provider: IoProvider,
    iteration_metadata: IterationMetadata,
    val_or_test: Literal["val", "test"],
) -> tuple[list[CompletedTaskInstance], list[TaskPerformanceReport]]:
    completed_task_instances: list[CompletedTaskInstance] = []
    performance_reports: list[TaskPerformanceReport] = []
    for task in tasks:
        completed_instances_for_task = task.evaluate(predictor)
        performance_reports.append(
            task.generate_performance_report(completed_instances_for_task)
        )
        completed_task_instances.extend(completed_instances_for_task)

    if val_or_test == "val":
        io_provider.write_val_performance_reports(
            performance_reports, iteration_metadata
        )
        io_provider.write_completed_val_task_instances(
            completed_task_instances, iteration_metadata
        )
    elif val_or_test == "test":
        io_provider.write_test_performance_reports(
            performance_reports, iteration_metadata
        )
        io_provider.write_completed_test_task_instances(
            completed_task_instances, iteration_metadata
        )
    else:
        raise ValueError(f"Invalid value for val_or_test: {val_or_test}")

    return completed_task_instances, performance_reports


def run_generic_accumulation_train_loop(
    validation_tasks: Collection[
        TaskInterface[CompletedTaskInstance, TaskInstanceCovariant]
    ],
    test_tasks: Collection[TaskInterface[CompletedTaskInstance, TaskInstanceCovariant]],
    trainable_predictor: TrainablePredictorInterface[
        TaskInstanceCovariant, TrainingDatum
    ],
    training_data_production_strategy: DataGenerationAgent[
        CompletedTaskInstance, TrainingDatum, TaskInstanceCovariant
    ],
    io_provider: IoProvider,
    accumulation_iterations_per_cycle: int = 10,
    num_cycles: int = 3,
    stop_at_num_training_data: Optional[int] = None,
) -> tuple[list[TaskPerformanceReport], list[TaskPerformanceReport]]:

    completed_val_task_instances_start_of_cycle: Optional[
        list[CompletedTaskInstance]
    ] = None
    accumulated_training_data: list[TrainingDatum] = []

    for cycle in range(num_cycles):
        logger.info(f"Starting cycle {cycle + 1} of {num_cycles}")
        # These should be reset at the start of each cycle.
        completed_val_task_instances_start_of_cycle = None

        for within_cycle_iteration in range(accumulation_iterations_per_cycle):
            if (
                stop_at_num_training_data
                and len(accumulated_training_data) >= stop_at_num_training_data
            ):
                logger.info(
                    f"Stopping accumulation at {stop_at_num_training_data} training data."
                )
                break

            # Determine iteration metadata
            # The iteration metadata is calculated as the current cycle number multiplied by the number of iterations per cycle,
            # plus the current iteration within the cycle. This ensures that each iteration across all cycles has a unique identifier.
            iteration_metadata = IterationMetadata(
                iteration=cycle * accumulation_iterations_per_cycle
                + within_cycle_iteration
            )

            # If we are in the first iteration of a cycle, evaluate the predictor.
            # This will also refresh the completed validation task instances.
            if within_cycle_iteration == 0:
                completed_val_task_instances_start_of_cycle, _ = (
                    evaluate_predictor_on_tasks(
                        validation_tasks,
                        trainable_predictor,
                        io_provider,
                        iteration_metadata,
                        val_or_test="val",
                    )
                )

                evaluate_predictor_on_tasks(
                    test_tasks,
                    trainable_predictor,
                    io_provider,
                    iteration_metadata,
                    val_or_test="test",
                )
            # Continue accumulating training data.
            assert completed_val_task_instances_start_of_cycle is not None
            training_data = training_data_production_strategy(
                completed_val_task_instances_start_of_cycle,
                trainable_predictor,
            )

            io_provider.write_training_data(training_data, iteration_metadata)
            accumulated_training_data.extend(training_data)

        # Train the predictor after each cycle of accumulation.
        trainable_predictor.train(accumulated_training_data)
        trainable_predictor.save(
            io_provider.get_trainable_predictor_save_path(iteration_metadata)
        )

    # Evaluate the predictor on the validation set after all cycles.
    _, final_val_performance_reports = evaluate_predictor_on_tasks(
        validation_tasks,
        trainable_predictor,
        io_provider,
        iteration_metadata,
        val_or_test="val",
    )

    # Evaluate the predictor on the test set after all cycles.
    _, final_test_performance_reports = evaluate_predictor_on_tasks(
        test_tasks,
        trainable_predictor,
        io_provider,
        iteration_metadata,
        val_or_test="test",
    )

    return final_val_performance_reports, final_test_performance_reports
