from datasets import load_dataset
from .evaluator import MnmsRecordRaw, MnmScorable, MNMEvaluator
from typing import Sequence, cast, Collection
from enum import Enum
from typing_extensions import assert_never, Self
import ast
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, Any, Iterable
from dataenvgym.gym.domain_models import (
    CodeGenerationTaskInstance,
    PredictorInterface,
    CodeGenerationCompletedTaskInstance,
    TaskPerformanceReport,
    TaskSlicePerformance,
    implements,
    CodeGenerationTaskInterface,
)
from ulid import ULID


class MnmsSplit(Enum):
    VAL = "validation"
    TEST = "test"


class MnmsTool(Enum):
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"
    IMAGE_GENERATION = "image_generation"
    IMAGE_CAPTIONING = "image_captioning"
    IMAGE_EDITING = "image_editing"
    IMAGE_CLASSIFICATION = "image_classification"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    OPTICAL_CHARACTER_RECOGNITION = "optical_character_recognition"
    IMAGE_CROP = "image_crop"
    IMAGE_CROP_LEFT = "image_crop_left"
    IMAGE_CROP_RIGHT = "image_crop_right"
    IMAGE_CROP_TOP = "image_crop_top"
    IMAGE_CROP_BOTTOM = "image_crop_bottom"
    BACKGROUND_BLUR = "background_blur"
    COLOR_POP = "color_pop"
    COUNT = "count"
    TAG = "tag"
    EMOJI = "emoji"
    SELECT_OBJECT = "select_object"
    GET_DATE_FACT = "get_date_fact"
    GET_YEAR_FACT = "get_year_fact"
    GET_MATH_FACT = "get_math_fact"
    GET_TRIVIA_FACT = "get_trivia_fact"
    LOVE_CALCULATOR = "love_calculator"
    GET_LOCATION = "get_location"
    SEARCH_MOVIE = "search_movie"
    GET_WEATHER = "get_weather"
    WIKIPEDIA_SIMPLE_SEARCH = "wikipedia_simple_search"


class MnmsPlanStep(BaseModel):
    id: int
    name: str
    args: Optional[dict[str, Any]]


class MnmsRecord(BaseModel):
    id: int
    user_request: str
    plan: list[MnmsPlanStep]
    code: str
    alt_plans: list[list[MnmsPlanStep]]

    @classmethod
    def from_raw(cls, raw: MnmsRecordRaw) -> Self:
        user_request = raw["user_request"]
        plan = ast.literal_eval(raw["plan_str"])
        for step in plan:
            step["name"] = step["name"].replace(" ", "_")
        code = raw["code_str"]
        if raw["alt_plans_str"] is None:
            alt_plans = None
        else:
            alt_plans = ast.literal_eval(raw["alt_plans_str"])
            for alt_plan in alt_plans:
                for step in alt_plan:
                    step["name"] = step["name"].replace(" ", "_")

        payload = {
            "user_request": user_request,
            "plan": plan,
            "code": code,
            "alt_plans": alt_plans,
            "id": raw["id"],
        }
        return cls.model_validate(payload)

    @classmethod
    def sequence_from_ds(cls, ds: Iterable[MnmsRecordRaw]) -> Sequence[Self]:
        return [cls.from_raw(record) for record in ds]

    @property
    def required_tools(self) -> set[MnmsTool]:
        return {MnmsTool(step.name) for step in self.plan}


def record_to_task_instance(
    record: MnmsRecord, split: MnmsSplit
) -> CodeGenerationTaskInstance:
    return CodeGenerationTaskInstance(
        task_name="mnms",
        instance_id=f"{split.value}_{record.id}",
        instruction=record.user_request,
        solution=record.code,
    )


def get_subset_requiring_all_tools(records: Sequence[MnmsRecord]) -> list[MnmsRecord]:
    """
    Get a subset of records that covers all required tools.
    """
    uncovered = {_ for _ in MnmsTool}
    chosen: list[MnmsRecord] = []
    candidates = list(records)
    while uncovered and candidates:
        best = max(candidates, key=lambda x: len(x.required_tools & uncovered))
        chosen.append(best)
        uncovered -= best.required_tools
        candidates.remove(best)
    return chosen


def check_which_tools_missing(records: Sequence[MnmsRecord]) -> set[MnmsTool]:
    all_tools = {tool for tool in MnmsTool}
    uncovered = all_tools.copy()
    for record in records:
        uncovered -= record.required_tools
    return uncovered


def load_mnms_human_verified_filtered() -> Sequence[MnmsRecordRaw]:
    ds = load_dataset(
        "zixianma/mnms",
        split="test_human_verified_filtered",
        revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
    )
    return cast(Sequence[MnmsRecordRaw], [_ for _ in ds])


def load_mnms_human_verified() -> Sequence[MnmsRecordRaw]:
    ds = load_dataset(
        "zixianma/mnms",
        split="test_human_verified",
        revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
    )
    return cast(Sequence[MnmsRecordRaw], [_ for _ in ds])


def get_split(split: MnmsSplit) -> Sequence[MnmsRecordRaw]:
    test_set = load_mnms_human_verified_filtered()
    test_set_ids = {record["id"] for record in test_set}
    # The test set is a filtered subset of `human_verified`.
    # We use whatever is not in the test set as the validation set.
    superset = load_mnms_human_verified()
    # Note: The validation set does not cover all tools. There
    # are some tools in the test set that are not covered by the validation set.
    validation_records = [
        record for record in superset if record["id"] not in test_set_ids
    ]
    match split:
        case MnmsSplit.VAL:
            return validation_records
        case MnmsSplit.TEST:
            return test_set
        case _:
            assert_never(split)


class MnmsTask:
    def __init__(self, split: MnmsSplit):
        self.split = split
        self.raw_records = get_split(split)
        self.task_instances = [
            record_to_task_instance(MnmsRecord.from_raw(raw_record), split)
            for raw_record in self.raw_records
        ]
        self.task_instance_id_to_raw_record = {
            task_instance.instance_id: raw_record
            for task_instance, raw_record in zip(self.task_instances, self.raw_records)
        }

    def evaluate(
        self, predictor: PredictorInterface[CodeGenerationTaskInstance]
    ) -> Collection[CodeGenerationCompletedTaskInstance]:
        predictions = predictor.predict(self.task_instances)
        completed_task_instances: list[CodeGenerationCompletedTaskInstance] = []
        for task_instance, prediction in zip(self.task_instances, predictions):
            raw_record_for_task_instance = self.task_instance_id_to_raw_record[
                task_instance.instance_id
            ]
            scorable = MnmScorable(
                id=raw_record_for_task_instance["id"],
                prediction=prediction,
            )

            evaluator = MNMEvaluator(
                gt_data=[raw_record_for_task_instance],
                pred_data=[scorable],
                plan_format="code",
            )
            score, _ = evaluator.evaluate()

            was_correct = score["plan_tool"]["accuracy"] > 0

            completed_task_instances.append(
                CodeGenerationCompletedTaskInstance(
                    ulid=ULID(),
                    task_instance=task_instance,
                    predictor_response=prediction,
                    was_correct=was_correct,
                )
            )

        return completed_task_instances

    def generate_performance_report(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> TaskPerformanceReport:
        raw_records = [
            self.task_instance_id_to_raw_record[
                completed_task_instance.task_instance.instance_id
            ]
            for completed_task_instance in completed_task_instances
        ]

        scorables = [
            MnmScorable(
                id=raw_record["id"],
                prediction=completed_task_instance.predictor_response,
            )
            for raw_record, completed_task_instance in zip(
                raw_records, completed_task_instances
            )
        ]

        evaluator = MNMEvaluator(
            gt_data=raw_records,
            pred_data=scorables,
            plan_format="code",
        )
        score, _ = evaluator.evaluate()

        return TaskPerformanceReport(
            task_name="mnms",
            overall_performance=score["plan_tool"]["accuracy"],
            slices=[
                TaskSlicePerformance(
                    slice_name="all",
                    slice_relname="all",
                    metric_name="tool_macro_precision",
                    metric_value=score["tool_macro"]["precision"],
                    count=len(raw_records),
                ),
                TaskSlicePerformance(
                    slice_name="all",
                    slice_relname="all",
                    metric_name="tool_macro_recall",
                    metric_value=score["tool_macro"]["recall"],
                    count=len(raw_records),
                ),
                TaskSlicePerformance(
                    slice_name="all",
                    slice_relname="all",
                    metric_name="tool_macro_f1",
                    metric_value=score["tool_macro"]["f1"],
                    count=len(raw_records),
                ),
            ],
        )


implements(CodeGenerationTaskInterface)(MnmsTask)
