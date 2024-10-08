import base64
import functools
from io import BytesIO
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Literal,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


class SerializableImage(BaseModel):
    base64_payload: str = Field(repr=False)
    model_config = ConfigDict(frozen=True)

    @functools.cached_property
    def pil_image(self) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(self.base64_payload)))

    @classmethod
    def from_pil_image(cls, pil_image: Image.Image) -> "SerializableImage":
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return cls(base64_payload=base64.b64encode(buffered.getvalue()).decode())

    @classmethod
    def from_file(cls, file_path: str) -> "SerializableImage":
        image = Image.open(file_path).convert("RGB")
        return cls.from_pil_image(image)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "SerializableImage":
        image = Image.fromarray(array)
        return cls.from_pil_image(image)

    @classmethod
    def from_random(cls, size: tuple[int, int] = (256, 256)) -> "SerializableImage":
        return cls.from_array(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))

    def __repr__(self):
        return f"SerializableImage({self.pil_image.size})"


class OpenEndedVqaTaskInstance(BaseModel):
    task_name: str
    instance_id: str
    instruction: str
    ground_truth_label: str | list[str]
    image: Image.Image = Field(exclude=True)
    vqa_task_type: Literal["open_ended"] = "open_ended"

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class MathTaskInstance(BaseModel):
    task_name: str
    instance_id: str
    instruction: str
    ground_truth_label: str

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)


class MultipleChoiceVqaTaskInstance(BaseModel):
    task_name: str
    instance_id: str
    instruction: str
    answer_choices: list[str]
    ground_truth_label: str
    image: Image.Image = Field(exclude=True)
    vqa_task_type: Literal["multiple_choice"] = "multiple_choice"

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VqaPredictorInterface(Protocol):
    def predict(
        self,
        task_instances: Sequence[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]: ...


class MathPredictorInterface(Protocol):
    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]: ...


class CompletedVqaTaskInstance(BaseModel):
    ulid: ULID  # To uniquely identify the task instance.
    task_instance: OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance = Field(
        ..., discriminator="vqa_task_type"
    )
    predictor_response: str
    was_correct: bool

    model_config = ConfigDict(frozen=True)


class CompletedMathTaskInstance(BaseModel):
    ulid: ULID
    task_instance: MathTaskInstance
    predictor_response: str
    was_correct: bool

    model_config = ConfigDict(frozen=True)


class TaskSlicePerformance(BaseModel):
    # Ex: attribute / color
    slice_name: str
    # Ex: color for attribute / color
    # TODO: Should be a computed property
    slice_relname: str
    # Ex: accuracy
    metric_name: str
    # Ex: 0.8
    metric_value: float
    # Ex: If slice_name = "/" then the children might be
    # ["attribute", "relation"] and the children of "attribute"
    # might be ["color", "shape"].
    children: list["TaskSlicePerformance"] = []
    # Number of instances inside this slice, including all the children.
    # TODO: Should probably be a computed property.
    count: int


class TaskPerformanceReport(BaseModel):
    task_name: str
    overall_performance: float
    slices: list[TaskSlicePerformance]


MathTaskPerformanceReport = TaskPerformanceReport


class VqaTaskInterface(Protocol):
    def evaluate(
        self, predictor: VqaPredictorInterface
    ) -> Collection[CompletedVqaTaskInstance]: ...

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> TaskPerformanceReport: ...


class MathTaskInterface(Protocol):
    def evaluate(
        self, predictor: MathPredictorInterface
    ) -> Collection[CompletedMathTaskInstance]: ...

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
    ) -> TaskPerformanceReport: ...


class VqaDataSpec(BaseModel):
    instruction: str
    image_description: str
    response: str

    model_config = ConfigDict(frozen=True)


class PreferenceVqaDataSpec(BaseModel):
    instruction: str
    image_description: str
    chosen_response: str
    rejected_response: str

    model_config = ConfigDict(frozen=True)


# Note: I'm not sure if it is beter to have separate classes for preference
# and non-preference data. I'm going to keep them separate for now rather than creating
# a union because it simplifies the type-checking of data strategies, since in a
# preference data strategy, we can use only the preference types without needing
# to bother with any conditionals.
class VqaDataHypothesis(BaseModel):
    inferred_weak_skill: str
    data_specs: list[VqaDataSpec]


class VqaPreferenceDataHypothesis(BaseModel):
    inferred_weak_skill: str
    data_specs: list[PreferenceVqaDataSpec]


class VqaTrainingDatum(BaseModel):
    ulid: ULID
    instruction: str
    image: SerializableImage
    response: str

    model_config = ConfigDict(frozen=True)


class MathDataSpec(BaseModel):
    problem: str
    chain_of_thought: str
    final_answer: str

    model_config = ConfigDict(frozen=True)


class MathDataHypothesis(BaseModel):
    inferred_weak_skill: str
    data_specs: list[MathDataSpec]


class MathTrainingDatum(BaseModel):
    ulid: ULID
    instruction: str
    response: str

    model_config = ConfigDict(frozen=True)


class VqaTrainerInterface(Protocol):
    def train(self, training_data: Sequence[VqaTrainingDatum]) -> None: ...


class MathTrainerInterface(Protocol):
    def train(self, training_data: Sequence[MathTrainingDatum]) -> None: ...


class TrainableVqaPredictorInterface(
    VqaPredictorInterface, VqaTrainerInterface, Protocol
):
    pass

    def save(self, path: Path) -> None: ...


class TrainableMathPredictorInterface(
    MathPredictorInterface, MathTrainerInterface, Protocol
):
    pass

    def save(self, path: Path) -> None: ...


class VqaPreferenceTrainingDatum(BaseModel):
    ulid: ULID
    instruction: str
    image: SerializableImage
    chosen_response: str
    rejected_response: str

    model_config = ConfigDict(frozen=True)


class PreferenceVqaTrainerInterface(Protocol):
    def train_preference(
        self,
        training_data: Sequence[VqaPreferenceTrainingDatum],
    ) -> None: ...


class PreferenceTrainableVqaPredictorInterface(
    VqaPredictorInterface, PreferenceVqaTrainerInterface, Protocol
):
    def save(self, path: Path) -> None: ...


class VqaDataGenerationAgent(Protocol):
    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: VqaPredictorInterface,
    ) -> Sequence[VqaTrainingDatum]: ...

    def step(self) -> None: ...


class MathDataGenerationAgent(Protocol):
    def __call__(
        self,
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]: ...

    def step(self) -> None: ...


class VqaPreferenceDataGenerationAgent(Protocol):
    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: VqaPredictorInterface,
    ) -> Sequence[VqaPreferenceTrainingDatum]: ...

    def step(self) -> None: ...


class VqaSkillDiscoveryInterface(Protocol):
    def discover_skills(
        self, task_instances: Collection[OpenEndedVqaTaskInstance]
    ) -> None: ...
    def get_skill_category_name_for_task_instance(
        self, task_instance: OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
    ) -> str: ...


class MathSkillDiscoveryInterface(Protocol):
    def discover_skills(self, task_instances: Collection[MathTaskInstance]) -> None: ...
    def get_skill_category_name_for_task_instance(
        self, task_instance: MathTaskInstance
    ) -> str: ...


class MathQualityCheckAnswerKnownByPredictor(BaseModel):
    predictor_response: str
    ground_truth_answer: str
    answered_correctly: bool


class MathTrainingDatumQualityCheck(BaseModel):
    ulid: ULID
    training_datum_ulid: ULID
    annotated_answer_passes_scoring_code: bool
    student_already_knows_answer: (
        MathQualityCheckAnswerKnownByPredictor
        | Literal["not applicable: training data failed scoring code"]
        | Literal["not applicable: was not checked"]
    )
    qa_passed: bool

    @property
    def student_accuracy(self) -> float | None:
        # This method exists to decouple the accuracy check logic from the data model.
        # The data model is bit confusing and requires isinstance checks. This is
        # simpler, since calling code can just check this property without knowing
        # the details of the data model.

        # If we checked whether the student knew the answer, we return the student's
        # score converted to a float.
        if isinstance(
            self.student_already_knows_answer, MathQualityCheckAnswerKnownByPredictor
        ):
            return 1.0 if self.student_already_knows_answer.answered_correctly else 0.0
        else:
            # We did not check whether the student knew the answer, so we return None.
            return None


class CodeGenerationTrainingDataQualityCheck(BaseModel):
    ulid: ULID
    training_datum_ulid: ULID
    student_accuracy: float | None


class VqaTrainingDataQualityCheck(BaseModel):
    ulid: ULID
    training_datum_ulid: ULID
    qa_passed: bool
    student_accuracy: float | None


class MathTrainingDataQualityCheckerInterface(Protocol):
    def __call__(
        self,
        training_data: Sequence[MathTrainingDatum],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatumQualityCheck]: ...


class CodeGenerationTaskInstance(BaseModel):
    task_name: str
    instance_id: str
    instruction: str
    solution: str | None = None
    starter_code: str | None = None


class CodeGenerationDataSpec(BaseModel):
    instruction: str
    solution: str
    starter_code: str | None = None
    ulid: ULID = Field(default_factory=ULID)


class CodeGenerationTrainingDatum(BaseModel):
    ulid: ULID
    instruction: str
    response: str


class CodeGenerationCompletedTaskInstance(BaseModel):
    ulid: ULID
    task_instance: CodeGenerationTaskInstance
    predictor_response: str
    was_correct: bool


class CodeGenerationPredictorInterface(Protocol):
    def predict(
        self, task_instances: Sequence[CodeGenerationTaskInstance]
    ) -> list[str]: ...


class CodeGenerationTrainerInterface(Protocol):
    def train(self, training_data: Sequence[CodeGenerationTrainingDatum]) -> None: ...


class TrainableCodeGenerationPredictorInterface(
    CodeGenerationPredictorInterface, CodeGenerationTrainerInterface, Protocol
):
    def save(self, path: Path) -> None: ...


class CodeGenerationDataGenerationAgent(Protocol):
    def __call__(
        self,
        completed_task_instances: Collection[CodeGenerationCompletedTaskInstance],
        predictor: CodeGenerationPredictorInterface,
    ) -> Sequence[CodeGenerationTrainingDatum]: ...

    def step(self) -> None: ...


class CodeGenerationTaskInterface(Protocol):
    def evaluate(
        self, predictor: CodeGenerationPredictorInterface
    ) -> Collection[CodeGenerationCompletedTaskInstance]: ...

    def generate_performance_report(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> TaskPerformanceReport: ...


class CodeGenerationSkillDiscoveryInterface(Protocol):
    def discover_skills(
        self, task_instances: Collection[CodeGenerationTaskInstance]
    ) -> None: ...
    def get_skill_category_name_for_task_instance(
        self, task_instance: CodeGenerationTaskInstance
    ) -> str: ...
    def get_skill_categories(self) -> list[str]: ...


P = TypeVar("P")
Q = TypeVar("Q")


def implements(protocol: Type[P]) -> Callable[[Type[P]], Type[P]]:
    def decorator(cls: Type[P]) -> Type[P]:
        # The type checker will enforce that `cls` matches the `protocol` without casting.
        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        # Returning the original class, which must be type-compatible with the protocol
        return cls

    return decorator


# These are generic type variables for use in environments and trajectory runners.

TrainingDatum = TypeVar(
    "TrainingDatum",
    bound=Union[VqaTrainingDatum, MathTrainingDatum, CodeGenerationTrainingDatum],
    contravariant=True,
)
TaskInstance = TypeVar(
    "TaskInstance",
    contravariant=True,
    bound=Union[
        OpenEndedVqaTaskInstance,
        MultipleChoiceVqaTaskInstance,
        MathTaskInstance,
        CodeGenerationTaskInstance,
    ],
)

TaskInstanceCovariant = TypeVar(
    "TaskInstanceCovariant",
    bound=Union[
        OpenEndedVqaTaskInstance,
        MultipleChoiceVqaTaskInstance,
        MathTaskInstance,
        CodeGenerationTaskInstance,
    ],
    covariant=True,
)
TaskInstanceInvariant = TypeVar(
    "TaskInstanceInvariant",
    bound=Union[
        OpenEndedVqaTaskInstance,
        MultipleChoiceVqaTaskInstance,
        MathTaskInstance,
        CodeGenerationTaskInstance,
    ],
)
CompletedTaskInstance = TypeVar(
    "CompletedTaskInstance",
    bound=Union[
        CompletedVqaTaskInstance,
        CompletedMathTaskInstance,
        CodeGenerationCompletedTaskInstance,
    ],
)

CompletedTaskInstanceContravariant = TypeVar(
    "CompletedTaskInstanceContravariant",
    bound=Union[
        CompletedVqaTaskInstance,
        CompletedMathTaskInstance,
        CodeGenerationCompletedTaskInstance,
    ],
    contravariant=True,
)

CompletedTaskInstanceCovariant = TypeVar(
    "CompletedTaskInstanceCovariant",
    bound=Union[
        CompletedVqaTaskInstance,
        CompletedMathTaskInstance,
        CodeGenerationCompletedTaskInstance,
    ],
    covariant=True,
)


TrainingDatumCovariant = TypeVar(
    "TrainingDatumCovariant",
    bound=Union[
        VqaTrainingDatum,
        MathTrainingDatum,
        CodeGenerationTrainingDatum,
    ],
    covariant=True,
)

TrainingDataQualityCheck = TypeVar(
    "TrainingDataQualityCheck",
    bound=Union[
        VqaTrainingDataQualityCheck,
        MathTrainingDatumQualityCheck,
        CodeGenerationTrainingDataQualityCheck,
    ],
)

TrainingDataQualityCheckCovariant = TypeVar(
    "TrainingDataQualityCheckCovariant",
    bound=Union[
        VqaTrainingDataQualityCheck,
        MathTrainingDatumQualityCheck,
        CodeGenerationTrainingDataQualityCheck,
    ],
    covariant=True,
)


class PredictorInterface(Protocol[TaskInstance]):
    def predict(self, task_instances: Sequence[TaskInstance]) -> list[str]: ...


class TaskInterface(Protocol[CompletedTaskInstance, TaskInstanceCovariant]):
    def evaluate(
        self, predictor: PredictorInterface[TaskInstanceCovariant]
    ) -> Collection[CompletedTaskInstance]: ...

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedTaskInstance]
    ) -> TaskPerformanceReport: ...


class TrainerInterface(Protocol[TrainingDatum]):
    def train(self, training_data: Sequence[TrainingDatum]) -> None: ...


class TrainablePredictorInterface(Protocol[TaskInstance, TrainingDatum]):
    def train(self, training_data: Sequence[TrainingDatum]) -> None: ...
    def save(self, path: Path) -> None: ...
    def predict(self, task_instances: Sequence[TaskInstance]) -> list[str]: ...


class SkillDiscoveryInterface(Protocol[TaskInstance]):
    def discover_skills(self, task_instances: Collection[TaskInstance]) -> None: ...
    def get_skill_category_name_for_task_instance(
        self, task_instance: TaskInstance
    ) -> str: ...


class QualityCheckerInterface(
    Protocol[TrainingDatum, TrainingDataQualityCheckCovariant, TaskInstanceCovariant]
):
    def __call__(
        self,
        training_data: Sequence[TrainingDatum],
        predictor: PredictorInterface[TaskInstanceCovariant],
    ) -> Sequence[TrainingDataQualityCheckCovariant]: ...


class DataGenerationAgent(
    Protocol[
        CompletedTaskInstanceContravariant,
        TrainingDatumCovariant,
        TaskInstanceCovariant,
    ]
):
    def __call__(
        self,
        completed_task_instances: Collection[CompletedTaskInstanceContravariant],
        predictor: PredictorInterface[TaskInstanceCovariant],
    ) -> Sequence[TrainingDatumCovariant]: ...

    def step(self) -> None: ...


class StubVqaPredictor:
    def predict(
        self,
        task_instances: Sequence[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]: ...


implements(VqaPredictorInterface)(StubVqaPredictor)


implements(PredictorInterface[OpenEndedVqaTaskInstance])(StubVqaPredictor)


class EnvironmentInterface(
    Protocol[CompletedTaskInstanceCovariant, TrainingDatum, TaskInstance]
):
    def reset(
        self,
    ) -> tuple[
        Sequence[CompletedTaskInstanceCovariant], Sequence[TaskPerformanceReport]
    ]: ...

    def step(
        self, training_data: Sequence[TrainingDatum]
    ) -> tuple[
        Sequence[CompletedTaskInstanceCovariant], Sequence[TaskPerformanceReport]
    ]: ...

    @property
    def trainable_predictor(
        self,
    ) -> TrainablePredictorInterface[TaskInstance, TrainingDatum]: ...
