from datasets import load_dataset, disable_caching
from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    implements,
    CompletedMathTaskInstance,
    MathTaskInterface,
    MathPredictorInterface,
    TaskPerformanceReport,
    TaskSlicePerformance,
)
from ulid import ULID
from typing import Literal, cast, Collection
from datasets import DatasetDict
from typing import TypedDict, Sequence, cast, Literal
import random
from collections import defaultdict
from loguru import logger
from datasets import load_dataset, disable_caching, DatasetDict
import pandas as pd
from .scoring import (
    process_docs,
    score_candidate_answer,
    list_fewshot_samples,
    EXPECTED_ANSWER_PREFIX,
    render_solution_for_scoring,
)
from tqdm.auto import tqdm
import jinja2
from dataenvgym.gym.domain_models import PredictorInterface

MATHSplitChoices = Literal[
    "train_all",
    "test_all",
    "val_balanced_subset_1",
    "val_balanced_subset_10",
    "val_balanced_subset_50",
    "train_level_1_balanced_1",
    "test_balanced_subset_10",
    "debug",
    "algebra_train_500",
    "algebra_test_500",
    "algebra_test_all",
    "number_theory_train_500",
    "number_theory_test_500",
    "number_theory_test_all",
]


MATHLevels = Literal["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
MATHTypes = Literal[
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]

allowed_math_levels: set[MATHLevels] = {
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
    "Level 5",
}

allowed_math_types: set[MATHTypes] = {
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
}


class MATHRecord(TypedDict):
    problem: str
    level: MATHLevels
    type: MATHTypes
    solution: str


class MATHRecordWithIDAndAnswer(MATHRecord):
    """
    A MATH record with an ID and answer.
    The answer is a processed version of the solution that contains
    only the final answer extracted from the solution.
    """

    record_id: str
    answer: str


def balanced_sampler(
    records: Sequence[MATHRecordWithIDAndAnswer],
    samples_per_level_per_type: int,
) -> Sequence[MATHRecordWithIDAndAnswer]:
    """
    Sample a balanced set of problems from the MATH dataset.

    This function does stratified sampling so that each cell in the
    level x type matrix has the same number of samples.

    Parameters
    ----------
    records : Sequence[MATHRecord]
        The records to sample from.
    samples_per_level_per_type : int
        The number of samples to take from each cell in the level x type matrix.

    Returns
    -------
    Sequence[MATHRecord]
        The sampled records.
    """
    # Create a dictionary to hold records categorized by (level, type)
    level_type_dict: dict[tuple[str, str], list[MATHRecordWithIDAndAnswer]] = (
        defaultdict(list)
    )

    # Populate the dictionary with records
    for record in records:
        if record["level"] not in allowed_math_levels:
            logger.warning(
                "Record with invalid level {} found. Skipping record.",
                record["level"],
            )
            continue
        if record["type"] not in allowed_math_types:
            logger.warning(
                "Record with invalid type {} found. Skipping record.",
                record["type"],
            )
            continue
        level_type_dict[(record["level"], record["type"])].append(record)

    sampled_records: list[MATHRecordWithIDAndAnswer] = []

    # Perform stratified sampling
    for group, records_for_group in level_type_dict.items():
        if len(records_for_group) < samples_per_level_per_type:
            logger.warning(
                "Group {} has fewer records {} than the requested number of samples {}",
                group,
                len(records_for_group),
                samples_per_level_per_type,
            )
            # Add them all
            sampled_records.extend(records_for_group)
        else:
            # Sample the records. Instead of using random.sample, we will sample
            # the first samples_per_level_per_type records from the group.
            # This is deterministic and will allow us to reproduce the same
            # sample each time.
            sampled_records.extend(records_for_group[:samples_per_level_per_type])
    return sampled_records


def load_math_dataset() -> DatasetDict:
    dataset = load_dataset("lighteval/MATH")
    dataset = cast(DatasetDict, dataset)

    dataset = process_docs(dataset)  # type: ignore
    dataset = cast(DatasetDict, dataset)

    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            lambda example, idx: {"record_id": f"MATH_{split}_{idx}"}, with_indices=True
        )

    return dataset


def load_split(
    split: MATHSplitChoices, math_dataset: DatasetDict
) -> Sequence[MATHRecordWithIDAndAnswer]:
    if split == "debug":
        sample = math_dataset["train"][0]
        sample = cast(MATHRecordWithIDAndAnswer, sample)
        return [sample]
    elif split == "val_balanced_subset_10":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        sample = balanced_sampler(train_split, 10)
        return sample
    elif split == "test_balanced_subset_10":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        sample = balanced_sampler(test_split, 10)
        return sample
    elif split == "train_all":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        return train_split
    elif split == "test_all":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        return test_split
    elif split == "train_level_1_balanced_1":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        sample = balanced_sampler(train_split, 1)
        # Filter out records that are not level 1
        sample = [record for record in sample if record["level"] == "Level 1"]
        return sample
    elif split == "val_balanced_subset_50":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        sample = balanced_sampler(train_split, 50)
        return sample
    elif split == "val_balanced_subset_1":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        sample = balanced_sampler(train_split, 1)
        return sample
    elif split == "algebra_train_500":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        # Sample 100 records from each level; this will result in 500 records
        # for algebra.
        sample = balanced_sampler(train_split, 100)
        # Filter out records that are not algebra
        sample = [record for record in sample if record["type"] == "Algebra"]
        return sample
    elif split == "algebra_test_500":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        # Sample 100 records from each level; this will result in 500 records
        # for algebra.
        sample = balanced_sampler(test_split, 100)
        # Filter out records that are not algebra
        sample = [record for record in sample if record["type"] == "Algebra"]
        return sample
    elif split == "algebra_test_all":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        # Filter out records that are not algebra
        sample = [record for record in test_split if record["type"] == "Algebra"]
        return sample
    elif split == "number_theory_train_500":
        train_split = math_dataset["train"]
        train_split = cast(Sequence[MATHRecordWithIDAndAnswer], train_split)
        # Sample 100 records from each level; this will result in 500 records
        # for number theory.
        sample = balanced_sampler(train_split, 100)
        # Filter out records that are not number theory
        sample = [record for record in sample if record["type"] == "Number Theory"]
        return sample
    elif split == "number_theory_test_500":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        # Sample 100 records from each level; this will result in 500 records
        # for number theory.
        sample = balanced_sampler(test_split, 100)
        # Filter out records that are not number theory
        sample = [record for record in sample if record["type"] == "Number Theory"]
        return sample
    elif split == "number_theory_test_all":
        test_split = math_dataset["test"]
        test_split = cast(Sequence[MATHRecordWithIDAndAnswer], test_split)
        # Filter out records that are not number theory
        sample = [record for record in test_split if record["type"] == "Number Theory"]
        return sample
    else:
        raise ValueError(f"Invalid split {split}")


def math_record_to_math_task_instance(
    record: MATHRecordWithIDAndAnswer,
) -> MathTaskInstance:
    return MathTaskInstance(
        task_name="MATH",
        instance_id=record["record_id"],
        instruction=record["problem"],
        # The MATH scoring code will only give correct scores if the final answer
        # is prefixed with the expected answer prefix and occurs in a \boxed{} block.
        # We prepare the ground truth label carefully to ensure that this field can
        # be used as a target for training a model whose responses will be scored
        # correctly by the MATH scoring code.
        ground_truth_label=render_solution_for_scoring(
            chain_of_thought=record["solution"],
            final_answer=record["answer"],
        ),
    )


MATH_PROMPT = jinja2.Template(
    """Answer the math problem. Begin your final answer with "{{ expected_answer_prefix }}" and put the final answer in a \\boxed{} block.
    {% if few_shot_examples %}
    {% for few_shot_example in few_shot_examples %}
    Problem: {{ few_shot_example.problem }}
    Solution: {{ few_shot_example.solution }}
    {% endfor %}
    {% endif %}

    Problem: {{ problem }}
    Solution:
    """
)

FEW_SHOT_PROMPT = jinja2.Template(
    """{% for few_shot_example in few_shot_examples %}
    Problem: {{ few_shot_example.problem }}
    Solution: {{ few_shot_example.solution }}
    {% endfor %}
    Problem: {{ problem }}
    Solution:"""
)


def prepare_math_prompt(task_instance: MathTaskInstance) -> str:
    few_shot_examples = list_fewshot_samples()
    return MATH_PROMPT.render(
        expected_answer_prefix=EXPECTED_ANSWER_PREFIX,
        problem=task_instance.instruction,
        few_shot_examples=few_shot_examples,
    )


def prepare_few_shot_prompt(task_instance: MathTaskInstance) -> str:
    few_shot_examples = list_fewshot_samples()
    return FEW_SHOT_PROMPT.render(
        problem=task_instance.instruction,
        few_shot_examples=few_shot_examples,
    )


class MATHTask:
    def __init__(self, split: MATHSplitChoices):
        self.split = split
        self.math_id_to_math_record: dict[str, MATHRecordWithIDAndAnswer] = {
            record["record_id"]: record
            for record in tqdm(
                load_split(split, load_math_dataset()), desc="Loading MATH records"
            )
        }
        self.task_instances = [
            math_record_to_math_task_instance(math_record)
            for math_record in tqdm(
                self.math_id_to_math_record.values(),
                desc="Converting to task instances",
            )
        ]

    def check_response_correct_for_instance_id(
        self, instance_id: str, response: str
    ) -> bool:
        # Look up the original record for the task instance.
        math_record = self.math_id_to_math_record[instance_id]
        # Get the processed answer _only_ rather than the full solution.
        answer = math_record["answer"]
        # Check if the answer was correct.
        score = score_candidate_answer(ground_truth_answer=answer, candidate=response)

        if score in (0, 1):
            return bool(score)

        raise ValueError(f"Invalid score {score}, expected a 1 or 0")

    def evaluate(
        self,
        predictor: PredictorInterface[MathTaskInstance],
    ) -> Collection[CompletedMathTaskInstance]:
        responses = predictor.predict(self.task_instances)

        return [
            CompletedMathTaskInstance(
                ulid=ULID(),
                task_instance=task_instance,
                predictor_response=response,
                was_correct=self.check_response_correct_for_instance_id(
                    task_instance.instance_id, response
                ),
            )
            for task_instance, response in tqdm(
                zip(self.task_instances, responses), desc="Evaluating"
            )
        ]

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
    ) -> TaskPerformanceReport:
        overall_performance = sum(
            completed_task_instance.was_correct
            for completed_task_instance in completed_task_instances
        ) / len(completed_task_instances)

        # Stratify performance by the `type` field in `MATHRecordWithIDAndAnswer`
        type_to_performance = {}
        type_to_count = {}

        for completed_instance in completed_task_instances:
            task_instance = completed_instance.task_instance
            math_record = self.math_id_to_math_record[task_instance.instance_id]
            question_type = math_record["type"]

            if question_type not in type_to_performance:
                type_to_performance[question_type] = 0
                type_to_count[question_type] = 0

            type_to_performance[question_type] += completed_instance.was_correct
            type_to_count[question_type] += 1

        slices = [
            TaskSlicePerformance(
                slice_name=question_type,
                slice_relname=question_type,
                metric_name="accuracy",
                metric_value=type_to_performance[question_type]
                / type_to_count[question_type],
                children=[],
                count=type_to_count[question_type],
            )
            for question_type in type_to_performance
        ]

        return TaskPerformanceReport(
            task_name=f"math/{self.split}",
            overall_performance=overall_performance,
            slices=slices,
        )


implements(MathTaskInterface)(MATHTask)
