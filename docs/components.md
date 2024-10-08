# Repository Structure & Abstractions

`dataenvgym` relies on a few core abstractions:

- `TaskInterface`: A class representing a collection of task instances, such as VQA questions or MATH problems that we will try to improve a student model on.
    - This can be found in `dataenvgym/gym/tasks/`.
- `TrainablePredictorInterface`: A class representing a model that can be trained on a sequence of `TrainingDatum`s, which are used to update the model's parameters.
    - This can be found in `dataenvgym/gym/trainable_predictors/`.
- `DataGenerationAgentInterface`: A class representing an agent that can generate new training data for a task.
    - This can be found in `dataenvgym/gym/data_generation_agents/`.
- `EnvironmentInterface`: A class representing an environment that can be used to evaluate a data generation agent.
    - This can be found in `dataenvgym/gym/environments/`.
    - Note that there is currently a `base_environment.py` file. Each of the environments (skill-list, skill-tree, open-ended) can be run within this file by providing the correct arguments. See TODO for more details.

Each of these abstractions have only 1-2 methods that need to be implemented to create a new task, trainable predictor, or data generation agent, and no other methods or attributes of these classes are relevant to the core logic of a data environment.
## Trainable Predictor
```python
class TrainablePredictorInterface(Protocol):
    def train(self, training_data: Sequence[TrainingDatum]) -> None:
        """Train the predictor on the provided training data."""
    def save(self, path: Path) -> None:
        """Save the predictor to the provided path."""
    def predict(self, task_instances: Sequence[TaskInstance]) -> list[str]:
        """Use the predictor to generate predictions for the provided task instances."""
```

## Tasks
```python
class TaskInterface(Protocol):
    def evaluate(
        self, predictor: PredictorInterface[TaskInstance]
    ) -> Collection[CompletedTaskInstance]:
        """Evaluate the predictor on the task and return evaluated predictions."""
    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedTaskInstance]
    ) -> TaskPerformanceReport:
        """Consume the evaluated predictions and return a report on the performance of the predictor on the task."""
```


## Data Generation Agent
```python
class DataGenerationAgent(Protocol):
    def __call__(
        self,
        completed_task_instances: Collection[CompletedTaskInstance],
        predictor: PredictorInterface[TaskInstance],
    ) -> Sequence[TrainingDatum]:
        """Generate new training data for a task given a predictor and the responses to some task instances."""

    def step(self) -> None:
        """Perform any necessary updates to the data generation agent after a step of the environment."""
```

## Environment
```python
class EnvironmentInterface(Protocol):
    def reset(
        self,
    ) -> tuple[
        Sequence[CompletedTaskInstance], Sequence[TaskPerformanceReport]
    ]:
        """Reset the environment to the initial state and return the initial observations and performance reports."""

    def step(
        self, training_data: Sequence[TrainingDatum]
    ) -> tuple[
        Sequence[CompletedTaskInstance], Sequence[TaskPerformanceReport]
    ]:
        """Train the predictor on the provided training data and return new observations and performance reports."""

    @property
    def trainable_predictor(
        self,
    ) -> TrainablePredictorInterface[TaskInstance, TrainingDatum]: 
        """Return the trainable predictor so management code can do things like save/load it."""
```