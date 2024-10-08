import pytest
from dataenvgym.gym.tasks.math.MATH.task import MATHTask, MATHRecordWithIDAndAnswer
from dataenvgym.gym.tasks.math.MATH.scoring import score_candidate_answer
from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathTrainingDatum,
    MathTrainingDatumQualityCheck,
    MathQualityCheckAnswerKnownByPredictor,
    MathPredictorInterface,
    implements,
)
from dataenvgym.gym.quality_checking.math.minimal import MathTrainingDataQualityChecker
from dataenvgym.gym.tasks.math.MATH.scoring import render_solution_for_scoring
from typing import Collection, Sequence
from ulid import ULID


class FixedAnswerListMathPredictor:
    def __init__(self, answers: list[str]):
        self.answers = answers

    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return self.answers


class AnswerByLookupPredictor:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def predict(self, task_instances: Collection[MathTaskInstance]) -> list[str]:
        return [
            self.answers[task_instance.instruction] for task_instance in task_instances
        ]


implements(MathPredictorInterface)(FixedAnswerListMathPredictor)
implements(MathPredictorInterface)(AnswerByLookupPredictor)


@pytest.fixture(scope="module")
def math_task() -> MATHTask:
    return MATHTask("val_balanced_subset_10")


def task_instance_to_training_datum(
    task_instance: MathTaskInstance,
    math_task: MATHTask,
) -> MathTrainingDatum:
    chain_of_thought = math_task.math_id_to_math_record[task_instance.instance_id][
        "solution"
    ]
    answer_for_instance = math_task.math_id_to_math_record[task_instance.instance_id][
        "answer"
    ]
    return MathTrainingDatum(
        ulid=ULID(),
        instruction=task_instance.instruction,
        response=render_solution_for_scoring(
            chain_of_thought=chain_of_thought, final_answer=answer_for_instance
        ),
    )


@pytest.fixture
def correct_training_datum(math_task: MATHTask) -> MathTrainingDatum:
    task_instance = math_task.task_instances[0]
    return task_instance_to_training_datum(task_instance, math_task)


@pytest.fixture
def correct_training_data(math_task: MATHTask) -> list[MathTrainingDatum]:
    return [
        task_instance_to_training_datum(task_instance, math_task)
        for task_instance in math_task.task_instances
    ]


@pytest.fixture
def corrupted_training_datum(
    correct_training_datum: MathTrainingDatum,
) -> MathTrainingDatum:
    return MathTrainingDatum(
        ulid=ULID(),
        instruction=correct_training_datum.instruction,
        response="some wrong answer",
    )


@pytest.fixture
def corrupted_training_data(
    correct_training_data: list[MathTrainingDatum],
) -> list[MathTrainingDatum]:
    return [
        MathTrainingDatum(
            ulid=ULID(),
            instruction=datum.instruction,
            response="some wrong answer",
        )
        for datum in correct_training_data
    ]


def test_scoring_consistency_check_passes_for_correct_answer(
    correct_training_datum: MathTrainingDatum,
):
    assert MathTrainingDataQualityChecker.check_annotated_answer_passes_scoring_code(
        correct_training_datum
    )


def test_scoring_consistency_check_fails_for_corrupted_answer(
    corrupted_training_datum: MathTrainingDatum,
):
    assert (
        not MathTrainingDataQualityChecker.check_annotated_answer_passes_scoring_code(
            corrupted_training_datum
        )
    )


def test_predictor_knows_answer(correct_training_datum: MathTrainingDatum):
    assert MathTrainingDataQualityChecker.check_predictor_knows_answer(
        training_datum=correct_training_datum,
        predictor_response=correct_training_datum.response,
    ).answered_correctly


def test_predictor_does_not_know_answer(correct_training_datum: MathTrainingDatum):
    assert not MathTrainingDataQualityChecker.check_predictor_knows_answer(
        training_datum=correct_training_datum,
        predictor_response="wrong_answer",
    ).answered_correctly


def test_training_data_passes_scoring_code(
    correct_training_data: list[MathTrainingDatum],
):
    quality_checker = MathTrainingDataQualityChecker()
    failed_quality_checks, _ = quality_checker.check_training_data_passes_scoring_code(
        training_data=correct_training_data
    )
    assert len(failed_quality_checks) == 0


def test_training_data_fails_scoring_code(
    corrupted_training_data: list[MathTrainingDatum],
):
    quality_checker = MathTrainingDataQualityChecker()
    failed_quality_checks, _ = quality_checker.check_training_data_passes_scoring_code(
        training_data=corrupted_training_data
    )
    assert len(failed_quality_checks) == len(corrupted_training_data)


def test_training_data_known_by_predictor(
    correct_training_data: list[MathTrainingDatum],
):
    quality_checker = MathTrainingDataQualityChecker()
    failed_quality_checks, _ = quality_checker.check_training_data_known_by_predictor(
        training_data=correct_training_data,
        predictor=FixedAnswerListMathPredictor(
            answers=[datum.response for datum in correct_training_data]
        ),
    )
    assert len(failed_quality_checks) == len(correct_training_data)
    # Assert all quality checks failed due to student already knowing the answer.
    for quality_check in failed_quality_checks:
        if not isinstance(
            quality_check.student_already_knows_answer,
            MathQualityCheckAnswerKnownByPredictor,
        ):
            pytest.fail(
                f"Expected a {MathQualityCheckAnswerKnownByPredictor.__class__.__name__}, got {quality_check}"
            )
        else:
            assert quality_check.student_already_knows_answer.answered_correctly


def test_training_data_not_known_by_predictor(
    correct_training_data: list[MathTrainingDatum],
):
    quality_checker = MathTrainingDataQualityChecker()
    failed_quality_checks, _ = quality_checker.check_training_data_known_by_predictor(
        training_data=correct_training_data,
        predictor=FixedAnswerListMathPredictor(
            answers=["wrong answer"] * len(correct_training_data)
        ),
    )
    assert len(failed_quality_checks) == 0


def test_quality_checker(
    correct_training_data: list[MathTrainingDatum],
    corrupted_training_data: list[MathTrainingDatum],
):
    training_data = []
    # Switch between the correct and corrupted training data on odd / even index.
    for i in range(len(correct_training_data)):
        if i % 2 == 0:
            training_data.append(correct_training_data[i])
        else:
            training_data.append(corrupted_training_data[i])

    # A predictor that answers correctly for all training data that is not a multiple of 4.
    predictor = AnswerByLookupPredictor(
        answers={
            datum.instruction: (
                "I do not know this answer" if i % 4 == 0 else datum.response
            )
            for i, datum in enumerate(training_data)
        }
    )

    quality_checker = MathTrainingDataQualityChecker()
    quality_checks = quality_checker(training_data=training_data, predictor=predictor)

    # Assert that the quality checks are in the same order as the training data.
    assert len(quality_checks) == len(training_data)
    for i in range(len(training_data)):
        assert quality_checks[i].training_datum_ulid == training_data[i].ulid

    # All odd quality checks should have failed with the data not passing the scoring code.
    odd_quality_checks = [
        quality_checks[i] for i in range(len(training_data)) if i % 2 == 1
    ]
    for quality_check in odd_quality_checks:
        assert not quality_check.annotated_answer_passes_scoring_code

    # All even quality checks should have passed the scoring code.
    even_quality_checks = [
        quality_checks[i] for i in range(len(training_data)) if i % 2 == 0
    ]
    for quality_check in even_quality_checks:
        assert quality_check.annotated_answer_passes_scoring_code

    # All even quality checks that are not a multiple of 4 should fail because the predictor knows the answer.
    not_multiple_of_4_even_quality_checks = [
        quality_checks[i]
        for i in range(len(training_data))
        if i % 4 != 0 and i % 2 == 0
    ]
    for quality_check in not_multiple_of_4_even_quality_checks:
        assert isinstance(
            quality_check.student_already_knows_answer,
            MathQualityCheckAnswerKnownByPredictor,
        ), f"Didn't get a {MathQualityCheckAnswerKnownByPredictor.__class__.__name__}; this means the predictor was not checked."
        assert quality_check.student_already_knows_answer.answered_correctly

    # All even quality checks that _are_ a multiple of 4 should pass because the predictor does not know the answer.
    multiple_of_4_even_quality_checks = [
        quality_checks[i]
        for i in range(len(training_data))
        if i % 4 == 0 and i % 2 == 0
    ]
    for quality_check in multiple_of_4_even_quality_checks:
        assert isinstance(
            quality_check.student_already_knows_answer,
            MathQualityCheckAnswerKnownByPredictor,
        ), f"Didn't get a {MathQualityCheckAnswerKnownByPredictor.__class__.__name__}; this means the predictor was not checked."
        assert not quality_check.student_already_knows_answer.answered_correctly

    # Check that the only training data that should have passed the quality checks
    # is even data that is a multiple of 4.
    for i in range(len(training_data)):
        if i % 4 == 0 and i % 2 == 0:
            assert quality_checks[i].qa_passed
        else:
            assert not quality_checks[i].qa_passed


def test_math_quality_checker_with_empty_list():
    quality_checker = MathTrainingDataQualityChecker()
    mock_predictor = FixedAnswerListMathPredictor(answers=[])

    empty_training_data: list[MathTrainingDatum] = []

    quality_checks = quality_checker(empty_training_data, mock_predictor)

    assert (
        len(quality_checks) == 0
    ), "Quality checks should be an empty list for empty input"
