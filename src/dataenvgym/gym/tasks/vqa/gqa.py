from dataenvgym.gym.domain_models import (
    VqaTaskInterface,
    VqaPredictorInterface,
    CompletedVqaTaskInstance,
    TaskPerformanceReport,
    OpenEndedVqaTaskInstance,
    TaskSlicePerformance,
    implements,
    TaskInterface,
    PredictorInterface,
)
from typing import Collection
from dataenvgym.gym.preprocessing.gqa import GqaRecord, MachineSpecificGqaLayout
from typing import Literal
import json
import os
from ulid import ULID
from tqdm.auto import tqdm
from loguru import logger
from PIL import Image
from datasets import load_dataset
from typing import cast, Sequence

UNC_NLP_LAYOUT = MachineSpecificGqaLayout(
    gqa_root_abspath="/nas-ssd/zaidkhan/datasets/gqa",
    gqa_images_abspath="/nas-ssd/zaidkhan/datasets/gqa/images",
)

GqaSplitChoices = Literal["val", "testdev", "debug"]


def load_gqa_split(split: GqaSplitChoices) -> Collection[GqaRecord]:
    try:
        if split in ("val", "testdev"):
            if split == "val":
                split_path = "/nas-ssd/zaidkhan/envgenpp_workspace/task_data/vqa/gqa/val_balanced_subset_5_per_qtype.jsonl"
            elif split == "testdev":
                split_path = "/nas-ssd/zaidkhan/envgenpp_workspace/task_data/vqa/gqa/testdev_balanced_subset_5_per_qtype.jsonl"
            with open(split_path, "r") as f:
                records = [GqaRecord(**json.loads(line)) for line in f]  # type: ignore
        elif split == "debug":
            split_path = "/nas-ssd/zaidkhan/envgenpp_workspace/task_data/vqa/gqa/val_balanced_subset_1_per_qtype.jsonl"
            with open(split_path, "r") as f:
                records = [GqaRecord(**json.loads(line)) for line in f][:1]  # type: ignore
        elif split == "val_50_per_qtype":
            split_path = "/nas-ssd/zaidkhan/envgenpp_workspace/task_data/vqa/gqa/val_balanced_subset_50_per_qtype.jsonl"
            with open(split_path, "r") as f:
                records = [GqaRecord(**json.loads(line)) for line in f]
    except FileNotFoundError:
        records = load_dataset("codezakh/dataenvgym-gqa-raw", split=split)
        records = cast(Collection[GqaRecord], records)

    return records


def gqa_record_to_open_ended_vqa_task_instance(
    record: GqaRecord, gqa_image_root: str
) -> OpenEndedVqaTaskInstance:
    return OpenEndedVqaTaskInstance(
        task_name="gqa",
        instance_id=record["question_id"],
        instruction=record["question"],
        image=Image.open(
            os.path.join(gqa_image_root, f'{record["image_name"]}.jpg')
        ).convert("RGB"),
        ground_truth_label=record["answer"],
    )


def load_gqa_from_huggingface(
    split: GqaSplitChoices,
) -> tuple[dict[str, GqaRecord], Sequence[OpenEndedVqaTaskInstance]]:
    dataset = load_dataset("codezakh/dataenvgym-gqa-raw", split=split)
    # cast to GqaRecord
    records = cast(Sequence[GqaRecord], dataset)

    gqa_id_to_gqa_record: dict[str, GqaRecord] = {
        record["question_id"]: record for record in records
    }

    task_instances = load_dataset("codezakh/dataenvgym-gqa", split=split)
    task_instances = [
        OpenEndedVqaTaskInstance.model_validate(instance) for instance in task_instances
    ]

    return gqa_id_to_gqa_record, task_instances


def load_gqa_from_filesystem(
    split: GqaSplitChoices,
    gqa_layout: MachineSpecificGqaLayout = UNC_NLP_LAYOUT,
) -> tuple[dict[str, GqaRecord], Sequence[OpenEndedVqaTaskInstance]]:
    gqa_id_to_gqa_record: dict[str, GqaRecord] = {
        record["question_id"]: record
        for record in tqdm(load_gqa_split(split), desc="Loading GQA records")
    }
    task_instances = [
        gqa_record_to_open_ended_vqa_task_instance(
            record, gqa_layout["gqa_images_abspath"]
        )
        for record in tqdm(
            gqa_id_to_gqa_record.values(), desc="Converting to task instances"
        )
    ]

    return gqa_id_to_gqa_record, task_instances


class GqaTask:
    def __init__(
        self,
        split: GqaSplitChoices,
        gqa_layout: MachineSpecificGqaLayout = UNC_NLP_LAYOUT,
    ):
        self.split = split
        self.gqa_layout = gqa_layout

        try:
            self.gqa_id_to_gqa_record, self.task_instances = load_gqa_from_filesystem(
                split, gqa_layout
            )
        except FileNotFoundError:
            self.gqa_id_to_gqa_record, self.task_instances = load_gqa_from_huggingface(
                split
            )

        logger.info(
            f"Loaded {len(self.gqa_id_to_gqa_record)} GQA records for split {split}"
        )

        self.gqa_id_to_task_instance: dict[str, OpenEndedVqaTaskInstance] = {
            instance.instance_id: instance for instance in self.task_instances
        }

    def get_task_instance_by_id(self, instance_id: str) -> OpenEndedVqaTaskInstance:
        return self.gqa_id_to_task_instance[instance_id]

    def evaluate(
        self, predictor: PredictorInterface[OpenEndedVqaTaskInstance]
    ) -> Collection[CompletedVqaTaskInstance]:

        responses = predictor.predict(self.task_instances)

        return [
            CompletedVqaTaskInstance(
                ulid=ULID(),
                task_instance=task_instance,
                predictor_response=response,
                was_correct=response == task_instance.ground_truth_label,
            )
            for task_instance, response in zip(self.task_instances, responses)
        ]

    def calculate_slice_performance(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> list[TaskSlicePerformance]:
        # Group completed instances by question type
        slice_performance: dict[str, dict[str, int]] = {}

        for completed_instance in completed_task_instances:
            question_type = self.gqa_id_to_gqa_record[
                completed_instance.task_instance.instance_id
            ]["question_type"]

            if question_type not in slice_performance:
                slice_performance[question_type] = {"correct": 0, "total": 0}

            slice_performance[question_type]["total"] += 1
            if completed_instance.was_correct:
                slice_performance[question_type]["correct"] += 1

        # Convert slice performance to a list of VqaTaskSlicePerformance
        slice_performance_list = []
        for question_type, performance in slice_performance.items():
            accuracy = performance["correct"] / performance["total"]
            slice_performance_list.append(
                TaskSlicePerformance(
                    slice_name="question_type",
                    slice_relname=question_type,
                    metric_name="accuracy",
                    metric_value=accuracy,
                    count=performance["total"],
                )
            )

        return slice_performance_list

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> TaskPerformanceReport:
        overall_performance = sum(
            completed_task_instance.was_correct
            for completed_task_instance in completed_task_instances
        ) / len(completed_task_instances)

        logger.info(f"GQA overall performance: {overall_performance}")

        return TaskPerformanceReport(
            task_name="gqa",
            overall_performance=overall_performance,
            slices=[],
        )


implements(VqaTaskInterface)(GqaTask)
implements(TaskInterface[CompletedVqaTaskInstance, OpenEndedVqaTaskInstance])(GqaTask)
