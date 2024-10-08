from pathlib import Path
import click
import json
from collections import defaultdict
import random
from typing import TypedDict
from enum import Enum
from loguru import logger


class GqaSplitToAnnotationFileRelPath(Enum):
    val_balanced = "val_balanced_questions.json"
    test_balanced = "test_balanced_questions.json"
    testdev_balanced = "testdev_balanced_questions.json"
    train_balanced = "train_balanced_questions.json"


class GqaRecord(TypedDict):
    question_id: str
    image_name: str
    question: str
    question_type: str
    answer: str


class MachineSpecificGqaLayout(TypedDict):
    gqa_root_abspath: str
    gqa_images_abspath: str


@click.command()
@click.option(
    "--gqa-dir",
    help="The directory containing the GQA dataset.",
    required=True,
    type=str,
)
@click.option(
    "--split",
    help="The split of the GQA dataset to use.",
    required=True,
    type=click.Choice([split for split in GqaSplitToAnnotationFileRelPath.__members__]),
)
@click.option(
    "--output-dir",
    help="The directory to write the output to.",
    required=True,
    type=str,
)
@click.option(
    "--samples-per-question-type",
    help="The number of questions to sample for each question type.",
    required=True,
    type=int,
)
@click.option(
    "--dont-overwrite",
    help="If set, will not overwrite existing files.",
    is_flag=True,
)
def main(
    gqa_dir: str,
    split: str,
    output_dir: str,
    samples_per_question_type: int,
    dont_overwrite: bool = False,
):
    output_filename = f"{split}_subset_{samples_per_question_type}_per_qtype.jsonl"
    output_path = Path(output_dir) / output_filename

    if output_path.exists() and dont_overwrite:
        logger.info(
            f"Output file {output_path} already exists and --dont-overwrite was set. Exiting."
        )
        return

    gqa_directory = Path(gqa_dir)
    gqa_validation_json_path = (
        gqa_directory / GqaSplitToAnnotationFileRelPath[split].value
    )
    with open(gqa_validation_json_path, "r") as f:
        gqa_validation_json = json.load(f)
    has_equivalent: set[str] = set()
    unique_questions: dict[str, dict] = dict()
    for gqa_id, gqa_annotation in gqa_validation_json.items():
        # Skip this, we already have an equivalent question.
        if gqa_id in has_equivalent:
            continue
        # No questions we've seen so far are equivalent to this one.
        # We could keep it as a list, but we'll keep it as a dicionary
        # in case we need to retrieve specific questions by id later.
        unique_questions[gqa_id] = gqa_annotation
        # Add any equivalents of this question to the set so we can ignore them.
        for equivalent in gqa_annotation["equivalent"]:
            has_equivalent.add(equivalent)

    gqa_questions_by_detailed_type = defaultdict(list)

    for gqa_id, gqa_annotation in unique_questions.items():
        gqa_questions_by_detailed_type[gqa_annotation["types"]["detailed"]].append(
            gqa_id
        )

    # Now sample 10 questions of each type. If there are less than 10 questions of a type, sample all of them.
    sampled_questions = dict()
    for detailed_type, questions in gqa_questions_by_detailed_type.items():
        if len(questions) <= samples_per_question_type:
            sampled_questions[detailed_type] = questions
        else:
            sampled_questions[detailed_type] = random.sample(
                questions, samples_per_question_type
            )

    # Now we have a dictionary of 10 questions for each detailed type.
    # We'll grab the records corresponding to each of these questions and put them in a list.
    sampled_question_records = dict()
    for detailed_type, questions in sampled_questions.items():
        for gqa_id in questions:
            sampled_question_records[gqa_id] = unique_questions[gqa_id]

    gqa_records = []
    for gqa_id, gqa_annotation in sampled_question_records.items():
        gqa_records.append(
            GqaRecord(
                question_id=gqa_id,
                question=gqa_annotation["question"],
                image_name=gqa_annotation["imageId"],
                answer=gqa_annotation["answer"],
                question_type=gqa_annotation["types"]["detailed"],
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in gqa_records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Wrote {len(gqa_records)} records to {output_path}")


if __name__ == "__main__":
    main()
