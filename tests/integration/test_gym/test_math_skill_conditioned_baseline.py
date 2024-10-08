from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    DataGenerationAgent,
)
from dataenvgym.gym.skill_discovery.llm_metacognitive import MathSkillDiscoverer
from dataenvgym.gym.tasks.math.MATH.task import MATHTask
from dataenvgym.gym.domain_models import (
    CompletedMathTaskInstance,
    MathTaskInstance,
    MathSkillDiscoveryInterface,
    MathDataHypothesis,
    MathDataSpec,
)
from ulid import ULID
import pytest
from typing import Sequence
from dataenvgym.gym.tasks.math.MATH.scoring import score_candidate_answer
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesReader
import itertools


@pytest.fixture(scope="module")
def math_task_instances() -> Sequence[MathTaskInstance]:
    return MATHTask("train_level_1_balanced_1").task_instances


@pytest.fixture(scope="module")
def skill_discoverer(
    math_task_instances: Sequence[MathTaskInstance],
) -> MathSkillDiscoveryInterface:
    skill_discoverer = MathSkillDiscoverer(num_skill_categories=5)
    skill_discoverer.discover_skills(math_task_instances)
    return skill_discoverer


def test_math_data_strategy(
    skill_discoverer: MathSkillDiscoveryInterface,
    math_task_instances: Sequence[MathTaskInstance],
):
    # skill_discoverer = MathSkillDiscoverer()
    # skill_discoverer.set_precomputed_skills("path/to/precomputed/math_skills.json")
    data_strategy = DataGenerationAgent(
        skill_discoverer, data_specs_per_hypothesis=2, hypotheses_per_skill_category=2
    )

    math_task_instances = math_task_instances[:1]

    errors = [
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            was_correct=False,
            predictor_response="not sure",
        )
        for task_instance in math_task_instances
    ]

    training_data = data_strategy.generate_training_data(errors)

    assert len(training_data) >= 4


def test_math_data_strategy_with_no_errors(
    skill_discoverer: MathSkillDiscoveryInterface,
    math_task_instances: Sequence[MathTaskInstance],
):
    data_strategy = DataGenerationAgent(
        skill_discoverer,
        data_specs_per_hypothesis=2,
        hypotheses_per_skill_category=2,
        generate_data_only_for_errors=True,
    )

    math_task_instances = math_task_instances[:1]

    completed_task_instances = [
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            was_correct=True,
            predictor_response="correct",
        )
        for task_instance in math_task_instances
    ]

    training_data = data_strategy.generate_training_data(completed_task_instances)

    assert len(training_data) == 0

    data_strategy = DataGenerationAgent(
        skill_discoverer,
        data_specs_per_hypothesis=2,
        hypotheses_per_skill_category=2,
        generate_data_only_for_errors=False,
    )

    training_data = data_strategy.generate_training_data(completed_task_instances)

    assert len(training_data) >= 4


def test_generated_training_data_is_correct(
    skill_discoverer: MathSkillDiscoveryInterface,
    math_task_instances: Sequence[MathTaskInstance],
    tmp_path: Path,
):
    math_task_instances = math_task_instances[:1]
    data_strategy = DataGenerationAgent(
        skill_discoverer,
        data_specs_per_hypothesis=2,
        hypotheses_per_skill_category=2,
        generate_data_only_for_errors=False,
        logging_folder=tmp_path,
    )
    completed_task_instances = [
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            was_correct=True,
            predictor_response="correct",
        )
        for task_instance in math_task_instances
    ]

    training_data = data_strategy.generate_training_data(completed_task_instances)
    data_specs_path = data_strategy.get_path_to_data_hypotheses(
        tmp_path, data_strategy.generation_index
    )
    reader = PydanticJSONLinesReader(data_specs_path, MathDataHypothesis)
    data_hypotheses: list[MathDataHypothesis] = list(reader())
    data_specs: list[MathDataSpec] = list(
        itertools.chain(*[h.data_specs for h in data_hypotheses])
    )

    for data_spec, training_data in zip(data_specs, training_data):
        assert (
            score_candidate_answer(
                ground_truth_answer=data_spec.final_answer,
                candidate=training_data.response,
            )
            == 1.0
        )
