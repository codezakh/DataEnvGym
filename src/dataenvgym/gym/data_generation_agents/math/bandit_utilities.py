from collections import defaultdict
from typing import Sequence
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from ulid import ULID
import itertools

from dataenvgym.gym.data_generation_agents.skill_tree import (
    SkillExperience,
    Subskill,
    SkillHistory,
    TrainingDataCache,
    Skill,
)


def get_ulids_for_data_generated_for_skill_over_history(
    skill_history: SkillHistory,
    skill: Skill,
) -> list[ULID]:
    ulids: list[ULID] = []
    for experience_bundle in skill_history:
        experience_for_skill = experience_bundle[skill]
        ulids.extend(
            itertools.chain.from_iterable(
                experience_for_skill.state.skill_tree.training_data.values()
            )
        )
    return ulids


def get_data_generated_for_skill_over_history(
    skill_history: SkillHistory,
    skill: Skill,
    training_data_cache: TrainingDataCache,
) -> list[MathTrainingDatumWithSkillCategory]:
    data_for_skill: list[MathTrainingDatumWithSkillCategory] = []
    ulids = get_ulids_for_data_generated_for_skill_over_history(
        skill_history,
        skill,
    )
    for ulid in ulids:
        data_for_skill.append(training_data_cache[ulid])
    return data_for_skill


def sum_data_generated_over_forest_history(
    skill_history: SkillHistory,
) -> dict[Skill, dict[Subskill, int]]:

    # The number of skills never change over time so it is safe
    # to prefill the dictionary with the skills in the first experience.
    data_generated: dict[Skill, dict[Subskill, int]] = {
        _: defaultdict(int) for _ in skill_history[0]
    }

    # Sum the data generated for each skill and subskill.
    for experience in skill_history:
        for skill, experience in experience.items():
            for (
                subskill,
                training_data_ulids,
            ) in experience.state.skill_tree.training_data.items():
                data_generated[skill][subskill] += len(training_data_ulids)

    return data_generated


def sum_data_generated_over_tree_history(
    skill_history: Sequence[SkillExperience],
) -> dict[Subskill, int]:
    data_generated: dict[Subskill, int] = defaultdict(int)
    for experience in skill_history:
        for (
            subskill,
            training_data_ulids,
        ) in experience.state.skill_tree.training_data.items():
            data_generated[subskill] += len(training_data_ulids)
    return data_generated
