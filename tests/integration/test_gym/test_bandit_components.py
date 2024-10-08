from dataenvgym.gym.data_generation_agents.math.bandit_components import (
    ProposeSubskillsWithAzureOpenAI,
    GenerateSubskillDataWithAzureOpenAI,
    AzureOpenAIActionPolicy,
    SkillExperience,
    accumulate_train_prompt_formatter,
    ProposeSubskillsFromFixedList,
)
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    PartialSkillTree,
    Subskill,
    Skill,
    SkillTree,
    Action,
    Explore,
    SkillState,
)
import pytest
from dataenvgym.utils import JSONLinesKeyValueCache
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from pathlib import Path
from ulid import ULID
from typing import Sequence
import rich


@pytest.fixture
def empty_skill_tree() -> PartialSkillTree:
    return PartialSkillTree(subskills=[], data_allocation={}, skill=Skill("Algebra"))


def test_propose_subskills_with_azure_openai(empty_skill_tree: PartialSkillTree):
    proposer = ProposeSubskillsWithAzureOpenAI()
    new_subskills = proposer(empty_skill_tree, 2)
    assert len(new_subskills) == 2


def test_propose_subskills_with_azure_openai_non_empty_skill_tree():
    skill_tree = PartialSkillTree(
        subskills=[
            Subskill("Solving Linear Equations"),
            Subskill("Graphing Quadratic Functions"),
            Subskill("Factoring Polynomials"),
            Subskill("Simplifying Rational Expressions"),
            Subskill("Solving Systems of Equations"),
        ],
        data_allocation={},
        skill=Skill("Algebra"),
    )
    proposer = ProposeSubskillsWithAzureOpenAI()
    new_subskills = proposer(skill_tree, 3)
    assert len(new_subskills) == 3
    for subskill in new_subskills:
        assert subskill not in skill_tree.subskills


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini"])
def test_generate_subskill_data_with_azure_openai(model):
    generator = GenerateSubskillDataWithAzureOpenAI(model=model)
    subskill = Subskill("Solving Differential Equations By Separation of Variables")
    data_budget = 5
    training_data = generator(subskill, data_budget)
    assert len(training_data) == data_budget


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini"])
def test_generate_subskill_data_with_azure_openai_zero_budget(model):
    generator = GenerateSubskillDataWithAzureOpenAI(model=model)
    subskill = Subskill("Solving Differential Equations By Separation of Variables")
    data_budget = 0
    training_data = generator(subskill, data_budget)
    assert len(training_data) == data_budget


def test_file_based_training_data_cache(tmp_path: Path):
    cache = JSONLinesKeyValueCache(
        tmp_path / "cache.jsonl", MathTrainingDatumWithSkillCategory
    )
    to_save = MathTrainingDatumWithSkillCategory(
        ulid=ULID(),
        instruction="What is the derivative of sin(x)?",
        response="We can use the chain rule to find the derivative of sin(x).",
        skill_category="Calculus",
    )
    cache[to_save.ulid] = to_save
    assert len(cache) == 1
    assert cache[to_save.ulid] == to_save

    new_cache = JSONLinesKeyValueCache(
        tmp_path / "cache.jsonl", MathTrainingDatumWithSkillCategory
    )
    assert len(new_cache) == 1
    assert new_cache[to_save.ulid] == to_save

    del cache[to_save.ulid]
    assert len(cache) == 0
    with pytest.raises(KeyError):
        cache[to_save.ulid]


def test_azure_openai_action_policy():
    policy = AzureOpenAIActionPolicy()
    history = [
        SkillExperience(
            skill=Skill("Algebra"),
            state=SkillState(
                skill_tree=SkillTree(
                    skill=Skill("Algebra"),
                    subskills=[
                        Subskill("Solving Linear Equations"),
                        Subskill("Graphing Quadratic Functions"),
                    ],
                    data_allocation={
                        Subskill("Solving Linear Equations"): 10,
                        Subskill("Graphing Quadratic Functions"): 10,
                    },
                    training_data=dict(),
                    quality_checks={},
                    perf_on_training_data={
                        Subskill("Solving Linear Equations"): 0.5,
                        Subskill("Graphing Quadratic Functions"): 0.6,
                    },
                ),
                past_performance=0.40,
            ),
            reward=0.05,
            action=Explore(num_new_skills=2, data_allocation_for_new_skills=10),
        ),
    ]
    prompt = policy.make_prompt(history)
    rich.print("Prompt:")
    rich.print(prompt)
    action = policy(history)
    rich.print("Action:")
    rich.print(action)


def test_azure_openai_action_policy_with_accumulate_train_prompt_formatter():
    policy = AzureOpenAIActionPolicy(accumulate_train_prompt_formatter)
    history = [
        SkillExperience(
            skill=Skill("Algebra"),
            state=SkillState(
                skill_tree=SkillTree(
                    skill=Skill("Algebra"),
                    subskills=[
                        Subskill("Solving Linear Equations"),
                        Subskill("Graphing Quadratic Functions"),
                    ],
                    data_allocation={
                        Subskill("Solving Linear Equations"): 10,
                        Subskill("Graphing Quadratic Functions"): 10,
                    },
                    training_data=dict(),
                    quality_checks={},
                    perf_on_training_data={
                        Subskill("Solving Linear Equations"): 0.5,
                        Subskill("Graphing Quadratic Functions"): 0.6,
                    },
                ),
                past_performance=0.40,
            ),
            reward=0.05,
            action=Explore(num_new_skills=2, data_allocation_for_new_skills=10),
        ),
    ]
    prompt = policy.make_prompt(history)
    rich.print("Prompt:")
    rich.print(prompt)
    action = policy(history)
    rich.print("Action:")
    rich.print(action)


def test_propose_subskills_from_fixed_list(empty_skill_tree: PartialSkillTree):
    subskills_dict = {
        Skill("Algebra"): [
            Subskill("Solving Linear Equations"),
            Subskill("Graphing Quadratic Functions"),
            Subskill("Factoring Polynomials"),
        ]
    }
    proposer = ProposeSubskillsFromFixedList(subskills_dict)

    # First call, should return the first 2 subskills
    new_subskills = proposer(empty_skill_tree, 2)
    assert len(new_subskills) == 2
    assert new_subskills == [
        Subskill("Solving Linear Equations"),
        Subskill("Graphing Quadratic Functions"),
    ]

    # Second call, should return the remaining subskill
    new_subskills = proposer(empty_skill_tree, 2)
    assert len(new_subskills) == 1
    assert new_subskills == [Subskill("Factoring Polynomials")]

    # Third call, should return an empty list as all subskills have been returned
    new_subskills = proposer(empty_skill_tree, 2)
    assert len(new_subskills) == 0
