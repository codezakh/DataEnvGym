import pytest
from typing import Callable, Any
from ulid import ULID
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    Skill,
    Subskill,
    SkillExperience,
    SkillState,
    SkillTree,
    MathTrainingDatumWithSkillCategory,
    InMemoryKeyValueCache,
    Exploit,
    SkillHistory,
)
from dataenvgym.gym.data_generation_agents.math.bandit_utilities import (
    get_data_generated_for_skill_over_history,
    get_ulids_for_data_generated_for_skill_over_history,
    sum_data_generated_over_forest_history,
    sum_data_generated_over_tree_history,
)

SkillHistoryFactory = Callable[[int, int, int, int], SkillHistory]


@pytest.fixture
def skill_history_factory() -> SkillHistoryFactory:
    def _factory(
        num_skills: int,
        num_subskills_per_skill: int,
        num_training_data_per_subskill: int,
        num_experiences: int,
    ) -> SkillHistory:
        skills = [Skill(f"skill{i}") for i in range(num_skills)]
        subskills = [
            [Subskill(f"subskill{i}_{j}") for j in range(num_subskills_per_skill)]
            for i in range(num_skills)
        ]
        training_data_cache = InMemoryKeyValueCache()
        skill_histories = []

        for _ in range(num_experiences):
            experience_entry = {}
            for skill, subskill_list in zip(skills, subskills):
                training_data = {}
                data_allocation = {}
                quality_checks = {}
                perf_on_training_data = {}
                for subskill in subskill_list:
                    ulids = [ULID() for _ in range(num_training_data_per_subskill)]
                    training_data[subskill] = ulids
                    data_allocation[subskill] = num_training_data_per_subskill
                    quality_checks[subskill] = ulids
                    perf_on_training_data[subskill] = 0.9  # Example performance value
                    for ulid in ulids:
                        training_data_cache[ulid] = MathTrainingDatumWithSkillCategory(
                            ulid=ulid,
                            instruction=f"Instruction {ulid}",
                            response=f"Response {ulid}",
                            skill_category=subskill,
                        )

                skill_tree = SkillTree(
                    subskills=subskill_list,
                    data_allocation=data_allocation,
                    training_data=training_data,
                    skill=skill,
                    quality_checks=quality_checks,
                    perf_on_training_data=perf_on_training_data,
                )
                skill_state = SkillState(skill_tree=skill_tree, past_performance=0.85)
                skill_experience = SkillExperience(
                    state=skill_state,
                    reward=0.1,
                    action=Exploit(data_allocation_delta={}),
                    skill=skill,
                )
                experience_entry[skill] = skill_experience
            skill_histories.append(experience_entry)

        return skill_histories

    return _factory


def test_get_ulids_for_data_generated_for_skill_over_history(
    skill_history_factory: SkillHistoryFactory,
) -> None:
    num_skills = 1
    num_subskills_per_skill = 2
    num_training_data_per_subskill = 2
    num_experiences = 1
    skill_history = skill_history_factory(
        num_skills,
        num_subskills_per_skill,
        num_training_data_per_subskill,
        num_experiences,
    )

    expected_data = sum_data_generated_over_tree_history(
        [_[Skill("skill0")] for _ in skill_history]
    )
    total_expected_data = sum(expected_data.values())

    ulids = get_ulids_for_data_generated_for_skill_over_history(
        skill_history=skill_history,
        skill=Skill("skill0"),
    )

    assert len(ulids) == total_expected_data


def test_get_ulids_for_data_generated_multiple_skills(
    skill_history_factory: SkillHistoryFactory,
) -> None:
    num_skills = 2
    num_subskills_per_skill = 2
    num_training_data_per_subskill = 2
    num_experiences = 1
    skill_history = skill_history_factory(
        num_skills,
        num_subskills_per_skill,
        num_training_data_per_subskill,
        num_experiences,
    )

    target_skill = Skill("skill0")

    ulids = get_ulids_for_data_generated_for_skill_over_history(
        skill_history=skill_history,
        skill=target_skill,
    )

    expected_data = sum_data_generated_over_forest_history(
        skill_history,
    )

    expected_data_for_target_skill = sum(
        [v for k, v in expected_data[target_skill].items()]
    )

    assert len(ulids) == expected_data_for_target_skill


def test_get_ulids_for_data_generated_single_skill_multiple_experiences(
    skill_history_factory: SkillHistoryFactory,
) -> None:
    num_skills = 1
    num_subskills_per_skill = 2
    num_training_data_per_subskill = 2
    num_experiences = 2
    skill_history = skill_history_factory(
        num_skills,
        num_subskills_per_skill,
        num_training_data_per_subskill,
        num_experiences,
    )

    target_skill = Skill("skill0")

    ulids = get_ulids_for_data_generated_for_skill_over_history(
        skill_history=skill_history,
        skill=target_skill,
    )

    expected_data = sum_data_generated_over_forest_history(
        skill_history,
    )

    expected_data_for_target_skill = sum(
        [v for k, v in expected_data[target_skill].items()]
    )

    assert len(ulids) == expected_data_for_target_skill * num_experiences
