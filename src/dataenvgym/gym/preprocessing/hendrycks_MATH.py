from typing import TypedDict, Sequence, cast, Literal
import random
from collections import defaultdict
from loguru import logger
from datasets import load_dataset, disable_caching, DatasetDict
import pandas as pd

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


def balanced_sampler(
    records: Sequence[MATHRecord],
    samples_per_level_per_type: int,
) -> Sequence[MATHRecord]:
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
    level_type_dict: dict[tuple[str, str], list[MATHRecord]] = defaultdict(list)

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

    sampled_records: list[MATHRecord] = []

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
            # Sample the records
            sampled_records.extend(
                random.sample(records_for_group, samples_per_level_per_type)
            )

    return sampled_records


def main():
    disable_caching()
    # The dataset is only split into train and test by default.
    dataset = load_dataset("lighteval/MATH")
    dataset = cast(DatasetDict, dataset)
    train = dataset["train"]
    test = dataset["test"]

    train = cast(Sequence[MATHRecord], train)
    test = cast(Sequence[MATHRecord], test)

    sampled_train = balanced_sampler(train, 10)
    sampled_test = balanced_sampler(test, 10)

    sampled_train_df = pd.DataFrame(sampled_train)
    sampled_test_df = pd.DataFrame(sampled_test)

    # Print out the level x type matrix for the sampled train set
    # We want the counts for each level x type so we can inspect them.
    level_type_counts_train = sampled_train_df.groupby(["level", "type"]).size()
    print(level_type_counts_train)
    print(f"Total samples in train set: {len(sampled_train)}")

    # Print out the level x type matrix for the sampled test set
    # We want the counts for each level x type so we can inspect them.
    level_type_counts_test = sampled_test_df.groupby(["level", "type"]).size()
    print(level_type_counts_test)
    print(f"Total samples in test set: {len(sampled_test)}")


if __name__ == "__main__":
    main()
