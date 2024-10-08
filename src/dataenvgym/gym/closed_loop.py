from typing import Collection, TypeVar, Union, Sequence, Generic, TypeGuard
from dataenvgym.gym.domain_models import (
    VqaTaskInterface,
    TrainableVqaPredictorInterface,
    VqaDataGenerationAgent,
    TaskPerformanceReport,
    CompletedVqaTaskInstance,
    VqaTrainingDatum,
    VqaPreferenceDataGenerationAgent,
    PreferenceTrainableVqaPredictorInterface,
    VqaPreferenceTrainingDatum,
    MathTrainingDatum,
    CompletedMathTaskInstance,
    MathTaskInterface,
    TrainableMathPredictorInterface,
    MathDataGenerationAgent,
    CodeGenerationTaskInterface,
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTrainingDatum,
    TrainableCodeGenerationPredictorInterface,
    CodeGenerationDataGenerationAgent,
)
from pydantic import BaseModel
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesWriter
from enum import Enum


# TODO: This currently has no notion of order.
# It should be changed to clearly indicate that these are
# stages in the pipeline that happen one afte the other.
class OutputPath(Enum):
    completed_test_task_instances = "completed_test_task_instances.jsonl"
    completed_val_task_instances = "completed_val_task_instances.jsonl"
    training_data = "training_data.jsonl"
    performance_reports = "performance_reports.jsonl"
    pre_performance_reports = "pre_performance_reports.jsonl"
    data_for_generation = "data_for_generation.jsonl"
    model_save_path = "trainable_predictor"


# Currently this has no other fields, but we might add some in the future.
class IterationMetadata(BaseModel):
    iteration: int


TrainingDatumType = TypeVar(
    "TrainingDatumType",
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

TrainablePredictorType = TypeVar(
    "TrainablePredictorType",
    bound=Union[
        TrainableVqaPredictorInterface,
        TrainableMathPredictorInterface,
        TrainableCodeGenerationPredictorInterface,
    ],
)

DataProductionStrategyType = TypeVar(
    "DataProductionStrategyType",
    bound=Union[
        VqaDataGenerationAgent,
        VqaPreferenceDataGenerationAgent,
        MathDataGenerationAgent,
        CodeGenerationDataGenerationAgent,
    ],
)

TaskInterfaceType = TypeVar(
    "TaskInterfaceType",
    bound=Union[VqaTaskInterface, MathTaskInterface, CodeGenerationTaskInterface],
)


class IoProvider(Generic[CompletedTaskInstanceType, TrainingDatumType]):
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
        training_data: Collection[TrainingDatumType],
        iteration_metadata: IterationMetadata,
    ):

        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.training_data
        )

        writer: PydanticJSONLinesWriter[TrainingDatumType] = PydanticJSONLinesWriter(
            output_path
        )

        writer.write_batch(training_data)

    def write_pre_performance_reports(
        self,
        pre_performance_reports: Collection[TaskPerformanceReport],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.pre_performance_reports
        )
        writer: PydanticJSONLinesWriter[TaskPerformanceReport] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(pre_performance_reports)

    def write_performance_reports(
        self,
        performance_reports: Collection[TaskPerformanceReport],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.performance_reports
        )

        writer: PydanticJSONLinesWriter[TaskPerformanceReport] = (
            PydanticJSONLinesWriter(output_path)
        )

        writer.write_batch(performance_reports)

    def write_data_for_generation(
        self,
        data_for_generation: Collection[TrainingDatumType],
        iteration_metadata: IterationMetadata,
    ):
        output_path = self.make_and_return_output_path(
            iteration_metadata, OutputPath.data_for_generation
        )

        writer: PydanticJSONLinesWriter[TrainingDatumType] = PydanticJSONLinesWriter(
            output_path
        )

        writer.write_batch(data_for_generation)

    def get_trainable_predictor_save_path(
        self, iteration_metadata: IterationMetadata
    ) -> Path:
        return self.make_and_return_output_path(
            iteration_metadata, OutputPath.model_save_path
        )


def single_iteration(
    iteration_metadata: IterationMetadata,
    validation_vqa_tasks: Collection[VqaTaskInterface],
    test_vqa_tasks: Collection[VqaTaskInterface],
    trainable_vqa_predictor: TrainableVqaPredictorInterface,
    training_data_production_strategy: VqaDataGenerationAgent,
    io_provider: IoProvider,
) -> Collection[TaskPerformanceReport]:

    completed_val_task_instances: list[CompletedVqaTaskInstance] = []
    pre_performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in validation_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        pre_performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )
        io_provider.write_completed_val_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_val_task_instances.extend(completed_instances_for_task)
    io_provider.write_pre_performance_reports(
        pre_performance_reports, iteration_metadata
    )

    training_data = training_data_production_strategy(
        completed_val_task_instances, trainable_vqa_predictor
    )
    io_provider.write_training_data(training_data, iteration_metadata)
    trainable_vqa_predictor.train(training_data)
    trainable_vqa_predictor.save(
        io_provider.get_trainable_predictor_save_path(iteration_metadata)
    )

    completed_test_task_instances: list[CompletedVqaTaskInstance] = []
    post_performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in test_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        io_provider.write_completed_test_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_test_task_instances.extend(completed_instances_for_task)
        post_performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )

    io_provider.write_performance_reports(post_performance_reports, iteration_metadata)
    training_data_production_strategy.step()
    return post_performance_reports


def single_preference_iteration(
    iteration_metadata: IterationMetadata,
    validation_vqa_tasks: Collection[VqaTaskInterface],
    test_vqa_tasks: Collection[VqaTaskInterface],
    trainable_vqa_predictor: PreferenceTrainableVqaPredictorInterface,
    training_data_production_strategy: VqaPreferenceDataGenerationAgent,
    io_provider: IoProvider,
) -> Collection[TaskPerformanceReport]:

    completed_val_task_instances: list[CompletedVqaTaskInstance] = []
    pre_performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in validation_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        pre_performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )
        io_provider.write_completed_val_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_val_task_instances.extend(completed_instances_for_task)
    io_provider.write_pre_performance_reports(
        pre_performance_reports, iteration_metadata
    )

    training_data = training_data_production_strategy(
        completed_val_task_instances, trainable_vqa_predictor
    )
    io_provider.write_training_data(training_data, iteration_metadata)
    trainable_vqa_predictor.train_preference(training_data)
    trainable_vqa_predictor.save(
        io_provider.get_trainable_predictor_save_path(iteration_metadata)
    )

    completed_test_task_instances: list[CompletedVqaTaskInstance] = []
    post_performance_reports: list[TaskPerformanceReport] = []
    for vqa_task in test_vqa_tasks:
        completed_instances_for_task = vqa_task.evaluate(trainable_vqa_predictor)
        io_provider.write_completed_test_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_test_task_instances.extend(completed_instances_for_task)
        post_performance_reports.append(
            vqa_task.generate_performance_report(completed_instances_for_task)
        )

    io_provider.write_performance_reports(post_performance_reports, iteration_metadata)
    training_data_production_strategy.step()
    return post_performance_reports


def run_closed_loop(
    validation_vqa_tasks: Collection[VqaTaskInterface],
    test_vqa_tasks: Collection[VqaTaskInterface],
    trainable_vqa_predictor: TrainableVqaPredictorInterface,
    training_data_production_strategy: VqaDataGenerationAgent,
    io_provider: IoProvider,
    num_iterations: int = 2,
) -> dict[int, Collection[TaskPerformanceReport]]:
    performance_reports = {}
    for i in range(num_iterations):
        iteration_metadata = IterationMetadata(iteration=i)
        performance_reports[i] = single_iteration(
            iteration_metadata=iteration_metadata,
            validation_vqa_tasks=validation_vqa_tasks,
            test_vqa_tasks=test_vqa_tasks,
            trainable_vqa_predictor=trainable_vqa_predictor,
            training_data_production_strategy=training_data_production_strategy,
            io_provider=io_provider,
        )
    return performance_reports


def run_closed_preference_loop(
    validation_vqa_tasks: Collection[VqaTaskInterface],
    test_vqa_tasks: Collection[VqaTaskInterface],
    trainable_vqa_predictor: PreferenceTrainableVqaPredictorInterface,
    training_data_production_strategy: VqaPreferenceDataGenerationAgent,
    io_provider: IoProvider,
    num_iterations: int = 2,
) -> dict[int, Sequence[TaskPerformanceReport]]:
    performance_reports = {}
    for i in range(num_iterations):
        iteration_metadata = IterationMetadata(iteration=i)
        performance_reports[i] = single_preference_iteration(
            iteration_metadata=iteration_metadata,
            validation_vqa_tasks=validation_vqa_tasks,
            test_vqa_tasks=test_vqa_tasks,
            trainable_vqa_predictor=trainable_vqa_predictor,
            training_data_production_strategy=training_data_production_strategy,
            io_provider=io_provider,
        )
    return performance_reports


def single_math_iteration(
    iteration_metadata: IterationMetadata,
    validation_math_tasks: Collection[MathTaskInterface],
    test_math_tasks: Collection[MathTaskInterface],
    trainable_math_predictor: TrainableMathPredictorInterface,
    training_data_production_strategy: MathDataGenerationAgent,
    io_provider: IoProvider,
) -> Sequence[TaskPerformanceReport]:

    completed_val_task_instances: list[CompletedMathTaskInstance] = []
    pre_performance_reports: list[TaskPerformanceReport] = []
    for math_task in validation_math_tasks:
        completed_instances_for_task = math_task.evaluate(trainable_math_predictor)
        pre_performance_reports.append(
            math_task.generate_performance_report(completed_instances_for_task)
        )
        io_provider.write_completed_val_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_val_task_instances.extend(completed_instances_for_task)
    io_provider.write_pre_performance_reports(
        pre_performance_reports, iteration_metadata
    )

    training_data = training_data_production_strategy(
        completed_val_task_instances, trainable_math_predictor
    )
    io_provider.write_training_data(training_data, iteration_metadata)
    trainable_math_predictor.train(training_data)
    trainable_math_predictor.save(
        io_provider.get_trainable_predictor_save_path(iteration_metadata)
    )

    completed_test_task_instances: list[CompletedMathTaskInstance] = []
    post_performance_reports: list[TaskPerformanceReport] = []
    for math_task in test_math_tasks:
        completed_instances_for_task = math_task.evaluate(trainable_math_predictor)
        io_provider.write_completed_test_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_test_task_instances.extend(completed_instances_for_task)
        post_performance_reports.append(
            math_task.generate_performance_report(completed_instances_for_task)
        )

    io_provider.write_performance_reports(post_performance_reports, iteration_metadata)
    training_data_production_strategy.step()
    return post_performance_reports


def run_math_closed_loop(
    validation_math_tasks: Collection[MathTaskInterface],
    test_math_tasks: Collection[MathTaskInterface],
    trainable_math_predictor: TrainableMathPredictorInterface,
    training_data_production_strategy: MathDataGenerationAgent,
    io_provider: IoProvider,
    num_iterations: int = 2,
) -> dict[int, Sequence[TaskPerformanceReport]]:
    performance_reports = {}
    for i in range(num_iterations):
        iteration_metadata = IterationMetadata(iteration=i)
        performance_reports[i] = single_math_iteration(
            iteration_metadata=iteration_metadata,
            validation_math_tasks=validation_math_tasks,
            test_math_tasks=test_math_tasks,
            trainable_math_predictor=trainable_math_predictor,
            training_data_production_strategy=training_data_production_strategy,
            io_provider=io_provider,
        )
    return performance_reports


def single_code_generation_iteration(
    iteration_metadata: IterationMetadata,
    validation_code_tasks: Collection[CodeGenerationTaskInterface],
    test_code_tasks: Collection[CodeGenerationTaskInterface],
    trainable_code_predictor: TrainableCodeGenerationPredictorInterface,
    training_data_production_strategy: CodeGenerationDataGenerationAgent,
    io_provider: IoProvider[
        CodeGenerationCompletedTaskInstance, CodeGenerationTrainingDatum
    ],
) -> Sequence[TaskPerformanceReport]:

    completed_val_task_instances: list[CodeGenerationCompletedTaskInstance] = []
    pre_performance_reports: list[TaskPerformanceReport] = []
    for code_task in validation_code_tasks:
        completed_instances_for_task = code_task.evaluate(trainable_code_predictor)
        pre_performance_reports.append(
            code_task.generate_performance_report(completed_instances_for_task)
        )
        io_provider.write_completed_val_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_val_task_instances.extend(completed_instances_for_task)
    io_provider.write_pre_performance_reports(
        pre_performance_reports, iteration_metadata
    )

    training_data = training_data_production_strategy(
        completed_val_task_instances, trainable_code_predictor
    )
    io_provider.write_training_data(training_data, iteration_metadata)
    trainable_code_predictor.train(training_data)
    trainable_code_predictor.save(
        io_provider.get_trainable_predictor_save_path(iteration_metadata)
    )

    completed_test_task_instances: list[CodeGenerationCompletedTaskInstance] = []
    post_performance_reports: list[TaskPerformanceReport] = []
    for code_task in test_code_tasks:
        completed_instances_for_task = code_task.evaluate(trainable_code_predictor)
        io_provider.write_completed_test_task_instances(
            completed_instances_for_task, iteration_metadata
        )
        completed_test_task_instances.extend(completed_instances_for_task)
        post_performance_reports.append(
            code_task.generate_performance_report(completed_instances_for_task)
        )

    io_provider.write_performance_reports(post_performance_reports, iteration_metadata)
    training_data_production_strategy.step()
    return post_performance_reports


def run_code_generation_closed_loop(
    validation_code_tasks: Collection[CodeGenerationTaskInterface],
    test_code_tasks: Collection[CodeGenerationTaskInterface],
    trainable_code_predictor: TrainableCodeGenerationPredictorInterface,
    training_data_production_strategy: CodeGenerationDataGenerationAgent,
    io_provider: IoProvider[
        CodeGenerationCompletedTaskInstance, CodeGenerationTrainingDatum
    ],
    num_iterations: int = 2,
) -> dict[int, Sequence[TaskPerformanceReport]]:
    performance_reports = {}
    for i in range(num_iterations):
        iteration_metadata = IterationMetadata(iteration=i)
        performance_reports[i] = single_code_generation_iteration(
            iteration_metadata=iteration_metadata,
            validation_code_tasks=validation_code_tasks,
            test_code_tasks=test_code_tasks,
            trainable_code_predictor=trainable_code_predictor,
            training_data_production_strategy=training_data_production_strategy,
            io_provider=io_provider,
        )
    return performance_reports
