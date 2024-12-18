from datasets import load_dataset, Dataset
import re
from typing import Literal, Sequence, TypedDict, cast, Collection
from typing_extensions import assert_never
from pydantic import BaseModel
from PIL import Image
import rich
from dataclasses import dataclass
from tqdm.auto import tqdm
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
from enum import Enum
from ulid import ULID
import itertools
from collections import defaultdict


class NaturalBenchSplitChoice(Enum):
    VAL_200 = "val_200"
    TEST_200 = "test_200"
    ALL = "all"


SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option.",
}


class NaturalBenchMetrics(TypedDict):
    question_score: float
    image_score: float
    binary_score: float
    group_score: float


class NaturalBenchScorable(TypedDict):
    q0_i0: int
    q0_i1: int
    q1_i0: int
    q1_i1: int


class NaturalBenchSlug(Enum):
    Q0_I0 = "q0_i0"
    Q0_I1 = "q0_i1"
    Q1_I0 = "q1_i0"
    Q1_I1 = "q1_i1"


@dataclass
class NaturalBenchRecord:
    instance_id: str
    question: str
    image: Image.Image
    answer: str
    question_type: Literal["yes_no", "multiple_choice"]
    slug_for_scoring: NaturalBenchSlug
    naturalbench_index: int

    def to_task_instance(self) -> OpenEndedVqaTaskInstance:
        return OpenEndedVqaTaskInstance(
            instance_id=self.instance_id,
            task_name="naturalbench",
            instruction=self.question,
            ground_truth_label=self.answer,
            image=self.image,
        )


class RawNaturalBenchRecord(TypedDict):
    Index: int
    Image_0: Image.Image
    Image_1: Image.Image
    Question_0: str
    Image_0_Question_0: str
    Image_1_Question_0: str
    Question_1: str
    Image_0_Question_1: str
    Image_1_Question_1: str
    Question_Type: Literal["yes_no", "multiple_choice"]
    Source: str


# binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
# binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
# binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
# binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0


def load_naturalbench_split(
    split: NaturalBenchSplitChoice,
) -> Collection[NaturalBenchRecord]:
    dataset = load_dataset(
        "BaiqiL/NaturalBench",
        split="train",
        revision="21c415e3d60cdbabad4a02feaa2e4643789612f4",
    )
    dataset = cast(Sequence[RawNaturalBenchRecord], dataset)

    if split == NaturalBenchSplitChoice.VAL_200:
        records: list[NaturalBenchRecord] = []
        for idx, item in enumerate(dataset):
            records.extend(explode_raw_natural_bench_record_into_records(item))
            if idx >= 50:
                break
        return records
    elif split == NaturalBenchSplitChoice.TEST_200:
        records: list[NaturalBenchRecord] = []
        for idx, item in enumerate(dataset):
            # Only take the last 50 out of 1900 items.
            if idx >= 1850:
                records.extend(explode_raw_natural_bench_record_into_records(item))
        return records
    elif split == NaturalBenchSplitChoice.ALL:
        records: list[NaturalBenchRecord] = []
        for item in dataset:
            records.extend(explode_raw_natural_bench_record_into_records(item))
        return records
    else:
        raise ValueError(f"Invalid split choice: {split}")


def get_binary_score_for_response(response: str, record: NaturalBenchRecord) -> bool:
    answer_code = extract_answer(response, record.question_type)
    match record.slug_for_scoring:
        case NaturalBenchSlug.Q0_I0:
            return answer_code == 1
        case NaturalBenchSlug.Q0_I1:
            return answer_code == 0
        case NaturalBenchSlug.Q1_I0:
            return answer_code == 0
        case NaturalBenchSlug.Q1_I1:
            return answer_code == 1
        case _:
            assert_never(record.slug_for_scoring)


class NaturalBenchTask:
    def __init__(
        self, split: NaturalBenchSplitChoice = NaturalBenchSplitChoice.VAL_200
    ):
        self.split = split
        self.records = load_naturalbench_split(split)
        self.map_instance_id_to_record = {
            record.instance_id: record for record in self.records
        }

    def get_task_instances(self) -> list[OpenEndedVqaTaskInstance]:
        return [record.to_task_instance() for record in self.records]

    def evaluate(
        self, predictor: PredictorInterface[OpenEndedVqaTaskInstance]
    ) -> Collection[CompletedVqaTaskInstance]:

        task_instances = self.get_task_instances()
        responses = predictor.predict(task_instances)
        completed_task_instances: list[CompletedVqaTaskInstance] = []
        for response, task_instance in zip(responses, task_instances):
            record = self.map_instance_id_to_record[task_instance.instance_id]
            was_correct = get_binary_score_for_response(response, record)
            completed_task_instances.append(
                CompletedVqaTaskInstance(
                    ulid=ULID(),
                    task_instance=task_instance,
                    predictor_response=response,
                    was_correct=was_correct,
                )
            )
        return completed_task_instances

    def generate_performance_report(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> TaskPerformanceReport:

        scorables_partial: dict[int, dict[NaturalBenchSlug, int]] = defaultdict(dict)

        for completed_task_instance in completed_task_instances:
            # Get the record the instance corresponds to
            record = self.map_instance_id_to_record[
                completed_task_instance.task_instance.instance_id
            ]
            # Extract the answer code from the response
            answer_code = extract_answer(
                completed_task_instance.predictor_response, record.question_type
            )

            # Each completed task instance contributes to one of the four scorables
            # based on which question and image it corresponds to.
            scorables_partial[record.naturalbench_index][
                record.slug_for_scoring
            ] = answer_code

        scorables_final: list[NaturalBenchScorable] = []
        for naturalbench_index, partial_scorable in scorables_partial.items():
            # Make sure all four scorables are present
            assert all(
                [slug in partial_scorable for slug in NaturalBenchSlug]
            ), f"Missing scorables for naturalbench index {naturalbench_index}"

            scorable: NaturalBenchScorable = {
                "q0_i0": partial_scorable[NaturalBenchSlug.Q0_I0],
                "q0_i1": partial_scorable[NaturalBenchSlug.Q0_I1],
                "q1_i0": partial_scorable[NaturalBenchSlug.Q1_I0],
                "q1_i1": partial_scorable[NaturalBenchSlug.Q1_I1],
            }
            scorables_final.append(scorable)

        metrics = get_scores(scorables_final)

        slices: list[TaskSlicePerformance] = []
        for metric_name, metric_value in metrics.items():
            slices.append(
                TaskSlicePerformance(
                    metric_name=metric_name,
                    metric_value=cast(float, metric_value),
                    count=len(scorables_final),
                    slice_name="/",
                    slice_relname="",
                )
            )

        return TaskPerformanceReport(
            task_name="naturalbench",
            slices=slices,
            overall_performance=cast(float, metrics["group_score"]),
        )


implements(VqaTaskInterface)(NaturalBenchTask)


def explode_raw_natural_bench_record_into_records(
    raw_record: RawNaturalBenchRecord,
) -> list[NaturalBenchRecord]:
    records: list[NaturalBenchRecord] = [
        NaturalBenchRecord(
            instance_id=f"naturalbench_{raw_record['Index']}_image_0_question_0",
            question=raw_record["Question_0"]
            + SUFFIX_FOR_VQA[raw_record["Question_Type"]],
            image=raw_record["Image_0"],
            answer=raw_record["Image_0_Question_0"],
            question_type=raw_record["Question_Type"],
            slug_for_scoring=NaturalBenchSlug.Q0_I0,
            naturalbench_index=raw_record["Index"],
        ),
        NaturalBenchRecord(
            instance_id=f"naturalbench_{raw_record['Index']}_image_1_question_0",
            question=raw_record["Question_0"]
            + SUFFIX_FOR_VQA[raw_record["Question_Type"]],
            image=raw_record["Image_1"],
            answer=raw_record["Image_1_Question_0"],
            question_type=raw_record["Question_Type"],
            slug_for_scoring=NaturalBenchSlug.Q0_I1,
            naturalbench_index=raw_record["Index"],
        ),
        NaturalBenchRecord(
            instance_id=f"naturalbench_{raw_record['Index']}_image_0_question_1",
            question=raw_record["Question_1"]
            + SUFFIX_FOR_VQA[raw_record["Question_Type"]],
            image=raw_record["Image_0"],
            answer=raw_record["Image_0_Question_1"],
            question_type=raw_record["Question_Type"],
            slug_for_scoring=NaturalBenchSlug.Q1_I0,
            naturalbench_index=raw_record["Index"],
        ),
        NaturalBenchRecord(
            instance_id=f"naturalbench_{raw_record['Index']}_image_1_question_1",
            question=raw_record["Question_1"]
            + SUFFIX_FOR_VQA[raw_record["Question_Type"]],
            image=raw_record["Image_1"],
            answer=raw_record["Image_1_Question_1"],
            question_type=raw_record["Question_Type"],
            slug_for_scoring=NaturalBenchSlug.Q1_I1,
            naturalbench_index=raw_record["Index"],
        ),
    ]

    return records


def extract_answer(
    output_string: str, task_type: Literal["yes_no", "multiple_choice"]
) -> int:
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int:
        1 if "yes" or "A"
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string: str, word: str) -> int:
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1

    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError(
            "Task type not supported. Must be 'yes_no' or 'multiple_choice'."
        )

    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1


def get_scores(
    scores: list[NaturalBenchScorable] | dict[int, NaturalBenchScorable]
) -> NaturalBenchMetrics:
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0
        - "q0_i1" means question_0 on image_1
        - "q1_i0" means question_1 on image_0
        - "q1_i1" means question_1 on image_1

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'question_score': Average question score
            - 'image_score': Average image score
            - 'binary_score': Average binary VQA score
            - 'group_score': Average group score
    """
    question_score = 0.0
    image_score = 0.0
    binary_score = 0.0
    group = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct

    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1

        return group_correct

    if isinstance(scores, dict):
        for _, result in tqdm(scores.items()):
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)
    else:
        for result in tqdm(scores):
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)

    results: NaturalBenchMetrics = {
        "question_score": question_score / float(num_samples * 2),
        "image_score": image_score / float(num_samples * 2),
        "binary_score": binary_score / float(num_samples * 4),
        "group_score": group / num_samples,
    }

    return results


if __name__ == "__main__":
    # 1.Load dataset from HuggingFace
    dataset = load_dataset("BaiqiL/NaturalBench")

    # 2.Use NaturalBench: construct 1900*4 [question, image, correct_answer] samples from the dataset with 1900 samples
    raw_records = cast(list[RawNaturalBenchRecord], [_ for _ in dataset["train"]])  # type: ignore
    naturalbench: list[NaturalBenchRecord] = []
    for item in raw_records:
        naturalbench.extend(explode_raw_natural_bench_record_into_records(item))

    # 3. Test Models: use the naturalbench dataset to test your own models and get the "output_file" of your model
    outputs = [_.answer for _ in naturalbench]

    # 4. Extract the answer: extract the answer from the outputs (you could also use LLMs such as ChatGPT to extract the answer)
    assert len(outputs) == 1900 * 4
    answers: dict[int, NaturalBenchScorable] = {}
    number_answered_samples = len(outputs) // 4
    for i in range(number_answered_samples):
        scorable: NaturalBenchScorable = {
            "q0_i0": extract_answer(outputs[i * 4], naturalbench[i * 4].question_type),
            "q0_i1": extract_answer(
                outputs[i * 4 + 1], naturalbench[i * 4 + 1].question_type
            ),
            "q1_i0": extract_answer(
                outputs[i * 4 + 2], naturalbench[i * 4 + 2].question_type
            ),
            "q1_i1": extract_answer(
                outputs[i * 4 + 3], naturalbench[i * 4 + 3].question_type
            ),
        }
        answers[i] = scorable
    # 5. Calculate the scores
    scores = get_scores(answers)
    rich.print(scores)
