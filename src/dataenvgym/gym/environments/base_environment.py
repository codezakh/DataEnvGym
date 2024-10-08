from typing import Collection, Generic, Sequence

from dataenvgym.gym.domain_models import (
    CompletedTaskInstance,
    PredictorInterface,
    TaskInstanceCovariant,
    TaskInterface,
    TrainablePredictorInterface,
    TrainingDatum,
    TaskPerformanceReport,
    EnvironmentInterface,
    implements,
)

from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathTrainingDatum,
    CompletedMathTaskInstance,
)


from dataenvgym.gym.domain_models import (
    CodeGenerationTaskInstance,
    CodeGenerationTrainingDatum,
    CodeGenerationCompletedTaskInstance,
)

from dataenvgym.gym.domain_models import (
    OpenEndedVqaTaskInstance,
    VqaTrainingDatum,
    CompletedVqaTaskInstance,
)


class BaseEnvironment(
    Generic[
        CompletedTaskInstance,
        TrainingDatum,
        TaskInstanceCovariant,
    ]
):
    def __init__(
        self,
        validation_tasks: Collection[
            TaskInterface[CompletedTaskInstance, TaskInstanceCovariant]
        ],
        trainable_predictor: TrainablePredictorInterface[
            TaskInstanceCovariant, TrainingDatum
        ],
    ):
        self.validation_tasks = validation_tasks
        self.trainable_predictor = trainable_predictor

        self.accumulated_training_data: list[TrainingDatum] = []

    @staticmethod
    def _evaluate_predictor_on_tasks(
        tasks: Collection[TaskInterface[CompletedTaskInstance, TaskInstanceCovariant]],
        predictor: PredictorInterface[TaskInstanceCovariant],
    ) -> tuple[list[CompletedTaskInstance], list[TaskPerformanceReport]]:
        completed_task_instances: list[CompletedTaskInstance] = []
        performance_reports: list[TaskPerformanceReport] = []
        for task in tasks:
            completed_instances_for_task = task.evaluate(predictor)
            performance_reports.append(
                task.generate_performance_report(completed_instances_for_task)
            )
            completed_task_instances.extend(completed_instances_for_task)

        return completed_task_instances, performance_reports

    def reset(
        self,
    ) -> tuple[Sequence[CompletedTaskInstance], Sequence[TaskPerformanceReport]]:
        self.accumulated_training_data = []

        completed_val_task_instances, val_performance_reports = (
            self._evaluate_predictor_on_tasks(
                self.validation_tasks,
                self.trainable_predictor,
            )
        )

        return completed_val_task_instances, val_performance_reports

    def step(
        self, training_data: Sequence[TrainingDatum]
    ) -> tuple[Sequence[CompletedTaskInstance], Sequence[TaskPerformanceReport]]:

        # Add the training data to the accumulated training data.
        self.accumulated_training_data.extend(training_data)

        # Train the predictor.
        self.trainable_predictor.train(self.accumulated_training_data)

        # Evaluate the predictor on the validation set.
        completed_val_task_instances, val_performance_reports = (
            self._evaluate_predictor_on_tasks(
                self.validation_tasks,
                self.trainable_predictor,
            )
        )
        return completed_val_task_instances, val_performance_reports


implements(EnvironmentInterface)(BaseEnvironment)


MathEnvironment = BaseEnvironment[
    CompletedMathTaskInstance, MathTrainingDatum, MathTaskInstance
]


CodeGenerationEnvironment = BaseEnvironment[
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTrainingDatum,
    CodeGenerationTaskInstance,
]


VqaEnvironment = BaseEnvironment[
    CompletedVqaTaskInstance, VqaTrainingDatum, OpenEndedVqaTaskInstance
]
