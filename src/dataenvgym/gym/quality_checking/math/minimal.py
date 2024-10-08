from dataenvgym.gym.domain_models import (
    MathTrainingDataQualityCheckerInterface,
    MathTrainingDatumQualityCheck,
    MathTrainingDatum,
    MathPredictorInterface,
    implements,
    MathTaskInstance,
    CompletedMathTaskInstance,
    MathQualityCheckAnswerKnownByPredictor,
)
from dataenvgym.gym.tasks.math.MATH.scoring import (
    score_candidate_answer,
    get_unnormalized_answer,
    normalize_final_answer,
)
from typing import Sequence
from ulid import ULID
from loguru import logger
from tqdm.auto import tqdm


# TODO: This class has a lot of code for what it does. We should simplify it
# and also consider simplifying the output type.
class MathTrainingDataQualityChecker:
    @staticmethod
    def convert_training_datum_to_task_instance(
        training_datum: MathTrainingDatum,
    ) -> MathTaskInstance:
        return MathTaskInstance(
            task_name="MATH",
            instance_id=str(training_datum.ulid),
            instruction=training_datum.instruction,
            ground_truth_label=normalize_final_answer(
                get_unnormalized_answer(training_datum.response)
            ),
        )

    @staticmethod
    def check_annotated_answer_passes_scoring_code(
        training_datum: MathTrainingDatum,
    ) -> bool:

        # We have to put the "response" through the same preprocessing as
        # the answers in the MATH dataset.
        unnormalized_answer = get_unnormalized_answer(training_datum.response)
        ground_truth_answer = normalize_final_answer(unnormalized_answer)

        # If we see "[invalid_answer]", we do not pass it through to the
        # scoring code, because two [invalid_answer]s are equivalent and
        # will be falsely marked as correct by the scoring code.
        if ground_truth_answer == "[invalidanswer]":
            return False

        score = score_candidate_answer(
            ground_truth_answer=ground_truth_answer,
            candidate=training_datum.response,
        )

        if score == 0:
            return False
        else:
            return True

    @staticmethod
    def check_predictor_knows_answer(
        training_datum: MathTrainingDatum, predictor_response: str
    ) -> MathQualityCheckAnswerKnownByPredictor:
        ground_truth_answer = normalize_final_answer(
            get_unnormalized_answer(training_datum.response)
        )
        score = score_candidate_answer(
            ground_truth_answer=ground_truth_answer,
            candidate=predictor_response,
        )
        if score == 1:
            return MathQualityCheckAnswerKnownByPredictor(
                predictor_response=predictor_response,
                ground_truth_answer=ground_truth_answer,
                answered_correctly=True,
            )
        else:
            return MathQualityCheckAnswerKnownByPredictor(
                predictor_response=predictor_response,
                ground_truth_answer=ground_truth_answer,
                answered_correctly=False,
            )

    def check_training_data_known_by_predictor(
        self,
        training_data: Sequence[MathTrainingDatum],
        predictor: MathPredictorInterface,
    ) -> tuple[
        Sequence[MathTrainingDatumQualityCheck], Sequence[MathTrainingDatumQualityCheck]
    ]:
        failed_quality_checks: list[MathTrainingDatumQualityCheck] = []
        passed_quality_checks: list[MathTrainingDatumQualityCheck] = []
        task_instances = [
            self.convert_training_datum_to_task_instance(datum)
            for datum in training_data
        ]
        predictor_responses = predictor.predict(task_instances)
        for training_datum, task_instance, predictor_response in tqdm(
            zip(training_data, task_instances, predictor_responses)
        ):
            predictor_knows_answer = self.check_predictor_knows_answer(
                training_datum, predictor_response
            )
            if predictor_knows_answer.answered_correctly:
                failed_quality_checks.append(
                    MathTrainingDatumQualityCheck(
                        ulid=ULID(),
                        training_datum_ulid=training_datum.ulid,
                        annotated_answer_passes_scoring_code=True,
                        student_already_knows_answer=predictor_knows_answer,
                        qa_passed=False,
                    )
                )
            else:
                passed_quality_checks.append(
                    MathTrainingDatumQualityCheck(
                        ulid=ULID(),
                        training_datum_ulid=training_datum.ulid,
                        annotated_answer_passes_scoring_code=True,
                        student_already_knows_answer=predictor_knows_answer,
                        qa_passed=True,
                    )
                )
        logger.info(
            "Checked {num_training_data} data, {failed}/{total}={percent_failed} had"
            " ground truth answers that were known by the predictor.",
            num_training_data=len(training_data),
            failed=len(failed_quality_checks),
            total=len(training_data),
            percent_failed=len(failed_quality_checks) / len(training_data),
        )
        return failed_quality_checks, passed_quality_checks

    def check_training_data_passes_scoring_code(
        self,
        training_data: Sequence[MathTrainingDatum],
    ) -> tuple[
        Sequence[MathTrainingDatumQualityCheck], Sequence[MathTrainingDatumQualityCheck]
    ]:
        failed_quality_checks: list[MathTrainingDatumQualityCheck] = []
        passed_quality_checks: list[MathTrainingDatumQualityCheck] = []
        for datum in tqdm(
            training_data, desc="Checking training data passes scoring code"
        ):
            if not self.check_annotated_answer_passes_scoring_code(datum):
                failed_quality_checks.append(
                    MathTrainingDatumQualityCheck(
                        ulid=ULID(),
                        training_datum_ulid=datum.ulid,
                        annotated_answer_passes_scoring_code=False,
                        student_already_knows_answer="not applicable: training data failed scoring code",
                        qa_passed=False,
                    )
                )
            else:
                passed_quality_checks.append(
                    MathTrainingDatumQualityCheck(
                        ulid=ULID(),
                        training_datum_ulid=datum.ulid,
                        annotated_answer_passes_scoring_code=True,
                        student_already_knows_answer="not applicable: was not checked",
                        qa_passed=True,
                    )
                )
        logger.info(
            "Checked {num_training_data} data, {failed}/{total}={percent_failed} did not have"
            " ground truth answers that passed scoring code.",
            num_training_data=len(training_data),
            failed=len(failed_quality_checks),
            total=len(training_data),
            percent_failed=len(failed_quality_checks) / len(training_data),
        )
        return failed_quality_checks, passed_quality_checks

    def __call__(
        self,
        training_data: Sequence[MathTrainingDatum],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatumQualityCheck]:

        # Otherwise we are going to get zero division errors in places.
        if len(training_data) == 0:
            logger.warning("No training data to quality check.")
            return []

        quality_checks_map: dict[ULID, MathTrainingDatumQualityCheck] = {}

        # First, we check if the annotated answers pass the scoring code.
        # If they do not, we mark them as as not passing the scoring code and fail the quality check.
        failed_quality_checks, _ = self.check_training_data_passes_scoring_code(
            training_data
        )
        for quality_check in failed_quality_checks:
            quality_checks_map[quality_check.training_datum_ulid] = quality_check

        # If they do pass the scoring code, we check if the student already knows the answer.
        # We will collect all the training data that passes the scoring code and pass it to the predictor.
        # Then we check if the predictor says the student already knows the answer.
        not_yet_quality_checked = [
            datum for datum in training_data if datum.ulid not in quality_checks_map
        ]
        logger.info(
            "{num_not_yet_checked} training data will be checked if the student already knows the answer.",
            num_not_yet_checked=len(not_yet_quality_checked),
        )
        failed_quality_checks, passed_quality_checks = (
            self.check_training_data_known_by_predictor(
                not_yet_quality_checked, predictor
            )
        )
        for quality_check in failed_quality_checks:
            quality_checks_map[quality_check.training_datum_ulid] = quality_check
        for quality_check in passed_quality_checks:
            quality_checks_map[quality_check.training_datum_ulid] = quality_check

        # Check that every ULID that is in the training data list has been quality checked.
        assert len(quality_checks_map) == len(training_data)

        # Return the quality checks in the order of the input training data.
        return [
            quality_checks_map[training_datum.ulid] for training_datum in training_data
        ]


implements(MathTrainingDataQualityCheckerInterface)(MathTrainingDataQualityChecker)
