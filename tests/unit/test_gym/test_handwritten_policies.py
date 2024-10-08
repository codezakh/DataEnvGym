import pytest
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    MathBanditDataStrategy,
    Explore,
    Exploit,
    SkillTree,
    PartialSkillTree,
    SkillExperience,
    SkillState,
    Subskill,
    Skill,
)
from dataenvgym.gym.data_generation_agents.math.bandit_utilities import (
    sum_data_generated_over_forest_history,
)
from dataenvgym.gym.data_generation_agents.math.handwritten_policies import (
    AlternatingExploreExploitPolicy,
    FillBalancedSkillTreePolicy,
)
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from typing import Sequence
from ulid import ULID
from tests.unit.test_gym.test_math_bandit_data_strategy import (
    StubMathPredictor,
    StubMathSkillDiscoverer,
)
from dataenvgym.gym.domain_models import CompletedMathTaskInstance, MathTaskInstance
from typing import Callable


@pytest.fixture
def completed_task_instances_factory() -> (
    Callable[[int], list[CompletedMathTaskInstance]]
):
    def _factory(num_instances: int) -> list[CompletedMathTaskInstance]:
        return [
            CompletedMathTaskInstance(
                ulid=ULID(),
                task_instance=MathTaskInstance(
                    task_name="MATH",
                    instance_id=f"MATH_{i:03d}",
                    instruction="Solve for x: 2x + 5 = 13",
                    ground_truth_label="The answer is \\boxed{4}",
                ),
                predictor_response="Let's solve this step by step:\n1) First, subtract 5 from both sides:\n   2x + 5 - 5 = 13 - 5\n   2x = 8\n\n2) Now, divide both sides by 2:\n   2x/2 = 8/2\n   x = 4\n\nTherefore, the answer is \\boxed{4}",
                was_correct=True,
            )
            for i in range(num_instances)
        ]

    return _factory


@pytest.mark.parametrize(
    "num_skills, num_task_instances, max_data_per_skill",
    [
        (1, 1, 50),
        (1, 2, 50),
        (2, 4, 50),
        (2, 5, 50),
        (3, 6, 50),
        (3, 7, 50),
    ],
)
def test_alternating_explore_exploit_policy_action_alternation(
    completed_task_instances_factory: Callable[[int], list[CompletedMathTaskInstance]],
    num_skills: int,
    num_task_instances: int,
    max_data_per_skill: int,
) -> None:
    """
    Test that the AlternatingExploreExploitPolicy alternates between explore and exploit actions.
    """
    explore_action = Explore(num_new_skills=1, data_allocation_for_new_skills=10)
    policy = AlternatingExploreExploitPolicy(explore_action, max_data_per_skill)
    strategy = MathBanditDataStrategy(
        action_policy=policy,
        skill_discovery_module=StubMathSkillDiscoverer(num_skills=num_skills),
        initial_explore_action=explore_action,
    )
    predictor = StubMathPredictor()
    completed_task_instances = completed_task_instances_factory(num_task_instances)

    # Run the data strategy for a fixed number of iterations
    for _ in range(10):
        strategy(completed_task_instances=completed_task_instances, predictor=predictor)

    # Assert that Explore and Exploit were alternated
    history: list[dict[Skill, SkillExperience]] = strategy.past_experiences
    for skill in history[0].keys():
        expected_action_type = (
            "explore"  # The first action for each skill should be explore.
        )
        for experience in history:
            assert experience[skill].action.action_type == expected_action_type
            # The next action should be exploit if the previous action was explore, and vice versa.
            expected_action_type = (
                "exploit" if expected_action_type == "explore" else "explore"
            )


@pytest.mark.parametrize(
    "num_skills, num_task_instances, max_data_per_skill",
    [
        (1, 1, 50),
        (1, 2, 50),
        (2, 4, 50),
        (2, 5, 50),
        (3, 6, 50),
        (3, 7, 50),
    ],
)
def test_alternating_explore_exploit_policy_data_allocation(
    completed_task_instances_factory: Callable[[int], list[CompletedMathTaskInstance]],
    num_skills: int,
    num_task_instances: int,
    max_data_per_skill: int,
) -> None:
    """
    Test that at each iteration, either all skills had a data allocation of zero or only n skills had a nonzero data allocation.
    """
    explore_action = Explore(num_new_skills=1, data_allocation_for_new_skills=10)
    policy = AlternatingExploreExploitPolicy(explore_action, max_data_per_skill)
    strategy = MathBanditDataStrategy(
        action_policy=policy,
        skill_discovery_module=StubMathSkillDiscoverer(num_skills=num_skills),
        initial_explore_action=explore_action,
    )
    predictor = StubMathPredictor()
    completed_task_instances = completed_task_instances_factory(num_task_instances)

    # Run the data strategy for a fixed number of iterations
    for _ in range(10):
        strategy(completed_task_instances=completed_task_instances, predictor=predictor)

    # Assert that at each iteration, either all skills had a data allocation of zero or only n skills had a nonzero data allocation
    history: list[dict[Skill, SkillExperience]] = strategy.past_experiences
    for experience in history:
        for skill, experience in experience.items():
            data_allocations = experience.state.skill_tree.data_allocation.values()
            non_zero_allocations = [alloc for alloc in data_allocations if alloc != 0]
            assert (
                len(non_zero_allocations) == 0
                or len(non_zero_allocations) == explore_action.num_new_skills
            )


@pytest.mark.parametrize(
    "num_skills, num_task_instances, max_data_per_skill",
    [
        (3, 9, 3),
        (3, 8, 3),
    ],
)
def test_alternating_explore_exploit_policy_max_data_per_skill(
    completed_task_instances_factory: Callable[[int], list[CompletedMathTaskInstance]],
    num_skills: int,
    num_task_instances: int,
    max_data_per_skill: int,
) -> None:
    """
    Test that no skill's data allocation exceeds max_data_per_skill.
    """
    explore_action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    policy = AlternatingExploreExploitPolicy(
        explore_action, max_data_for_skill=max_data_per_skill
    )
    strategy = MathBanditDataStrategy(
        action_policy=policy,
        skill_discovery_module=StubMathSkillDiscoverer(num_skills=num_skills),
        initial_explore_action=explore_action,
    )
    predictor = StubMathPredictor()
    completed_task_instances = completed_task_instances_factory(num_task_instances)

    # Run the data strategy for a fixed number of iterations
    for _ in range(10):
        strategy(completed_task_instances=completed_task_instances, predictor=predictor)

    # Assert that no skill's data allocation exceeds max_data_per_skill
    history: list[dict[Skill, SkillExperience]] = strategy.past_experiences
    data_generated = sum_data_generated_over_forest_history(history)
    for skill, data_per_subskill in data_generated.items():
        data_for_skill = sum(data_per_subskill.values())
        assert data_for_skill <= max_data_per_skill


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "Single Skill, Single Subskill",
            "num_skills": 1,
            "num_task_instances": 10,
            "max_subskills": 1,
            "max_data_per_subskill": 15,
            "max_allocation_per_subskill": 5,
        },
        {
            "name": "Single Skill, Multiple Subskills",
            "num_skills": 1,
            "num_task_instances": 10,
            "max_subskills": 3,
            "max_data_per_subskill": 15,
            "max_allocation_per_subskill": 5,
        },
        {
            "name": "Multiple Skills, Multiple Subskills",
            "num_skills": 3,
            "num_task_instances": 10,
            "max_subskills": 3,
            "max_data_per_subskill": 15,
            "max_allocation_per_subskill": 5,
        },
        {
            "name": "Multiple Skills, Multiple Subskills, Relatively Prime Data Allocation",
            "num_skills": 3,
            "num_task_instances": 10,
            "max_subskills": 3,
            "max_data_per_subskill": 20,
            "max_allocation_per_subskill": 6,
        },
    ],
    ids=lambda x: x["name"],
)
def test_fill_balanced_skill_tree_policy(
    completed_task_instances_factory,
    test_case,
):
    explore_action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    policy = FillBalancedSkillTreePolicy(
        explore_action=explore_action,
        max_subskills=test_case["max_subskills"],
        max_data_per_subskill=test_case["max_data_per_subskill"],
        max_allocation_per_subskill=test_case["max_allocation_per_subskill"],
    )
    strategy = MathBanditDataStrategy(
        action_policy=policy,
        skill_discovery_module=StubMathSkillDiscoverer(
            num_skills=test_case["num_skills"]
        ),
        initial_explore_action=explore_action,
    )
    predictor = StubMathPredictor()
    completed_task_instances = completed_task_instances_factory(
        test_case["num_task_instances"]
    )

    # Run the data strategy for a fixed number of iterations
    for _ in range(50):  # Increase iterations to ensure policy completion
        strategy(completed_task_instances=completed_task_instances, predictor=predictor)

    # Get the final skill forest
    skill_forest = strategy.get_current_skill_forest()

    # Check conditions for each skill tree
    for skill, skill_tree in skill_forest.items():
        # 1. Check if the skill tree has max_subskills subskills
        epsilon = 1  # Define an integer epsilon for leniency
        assert (
            abs(len(skill_tree.subskills) - test_case["max_subskills"]) <= epsilon
        ), f"Test case '{test_case['name']}': Skill {skill} does not have {test_case['max_subskills']} ± {epsilon} subskills"

        # 2. Check if we reached max_data_per_subskill for each subskill
        data_generated = sum_data_generated_over_forest_history(
            strategy.past_experiences
        )
        for subskill, data_count in data_generated[skill].items():
            assert (
                abs(data_count - test_case["max_data_per_subskill"]) <= epsilon
            ), f"Test case '{test_case['name']}': Subskill {subskill} did not reach {test_case['max_data_per_subskill']} ± {epsilon}"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "maximum allocation",
            "max_data_per_subskill": 10,
            "max_allocation_delta": 5,
            "data_already_generated": {
                Subskill("subskill_1"): 5,
                Subskill("subskill_2"): 3,
                Subskill("subskill_3"): 2,
            },
            "expected_allocation": {
                Subskill("subskill_1"): 4,
                Subskill("subskill_2"): 4,
                Subskill("subskill_3"): 4,
            },
            "current_data_allocation": {
                Subskill("subskill_1"): 1,
                Subskill("subskill_2"): 1,
                Subskill("subskill_3"): 1,
            },
        },
        {
            "name": "fully allocated skill",
            "max_data_per_subskill": 10,
            "max_allocation_delta": 5,
            "data_already_generated": {
                Subskill("subskill_1"): 10,
                Subskill("subskill_2"): 3,
                Subskill("subskill_3"): 2,
            },
            "expected_allocation": {
                Subskill("subskill_1"): -5,
                Subskill("subskill_2"): 0,
                Subskill("subskill_3"): 0,
            },
            "current_data_allocation": {
                Subskill("subskill_1"): 5,
                Subskill("subskill_2"): 5,
                Subskill("subskill_3"): 5,
            },
        },
        {
            "name": "near max allocation",
            "max_data_per_subskill": 10,
            "max_allocation_delta": 5,
            "data_already_generated": {
                Subskill("subskill_1"): 8,
                Subskill("subskill_2"): 7,
                Subskill("subskill_3"): 9,
            },
            "expected_allocation": {
                Subskill("subskill_1"): 1,
                Subskill("subskill_2"): 2,
                Subskill("subskill_3"): 0,
            },
            "current_data_allocation": {
                Subskill("subskill_1"): 1,
                Subskill("subskill_2"): 1,
                Subskill("subskill_3"): 1,
            },
        },
        {
            "name": "data allocation zero'd out",
            "max_data_per_subskill": 10,
            "max_allocation_delta": 5,
            "data_already_generated": {
                Subskill("subskill_1"): 10,
                Subskill("subskill_2"): 10,
                Subskill("subskill_3"): 10,
            },
            "expected_allocation": {
                Subskill("subskill_1"): -1,
                Subskill("subskill_2"): -1,
                Subskill("subskill_3"): -1,
            },
            "current_data_allocation": {
                Subskill("subskill_1"): 1,
                Subskill("subskill_2"): 1,
                Subskill("subskill_3"): 1,
            },
        },
    ],
)
def test_calculate_next_exploit_action(test_case):
    """
    Test that the calculate_next_exploit_action method works correctly for various scenarios.
    """
    next_exploit_action = FillBalancedSkillTreePolicy._calculate_next_exploit_action(
        test_case["max_data_per_subskill"],
        test_case["max_allocation_delta"],
        test_case["data_already_generated"],
        test_case["current_data_allocation"],
    )

    assert (
        next_exploit_action.data_allocation_delta == test_case["expected_allocation"]
    ), f"Failed for test case: {test_case['name']}"

    # Additional check: ensure that adding allocation to existing data doesn't exceed max_data_per_subskill
    for subskill, allocation in next_exploit_action.data_allocation_delta.items():
        total_data = test_case["data_already_generated"][subskill] + allocation
        assert (
            total_data <= test_case["max_data_per_subskill"]
        ), f"Total data ({total_data}) exceeds max_data_per_subskill ({test_case['max_data_per_subskill']}) for {subskill} in test case: {test_case['name']}"


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "Last action explore",
            "last_action": Explore(num_new_skills=1, data_allocation_for_new_skills=10),
            "skill_tree": SkillTree(
                subskills=[Subskill("subskill1"), Subskill("subskill2")],
                data_allocation={Subskill("subskill1"): 5, Subskill("subskill2"): 5},
                training_data={
                    Subskill("subskill1"): [ULID() for _ in range(5)],
                    Subskill("subskill2"): [ULID() for _ in range(5)],
                },
                skill=Skill("test_skill"),
                quality_checks={Subskill("subskill1"): [], Subskill("subskill2"): []},
                perf_on_training_data={
                    Subskill("subskill1"): 0.8,
                    Subskill("subskill2"): 0.7,
                },
            ),
            "expected_action_type": Exploit,
            "expected_data_allocation_delta": {"subskill1": -5, "subskill2": -5},
        },
        {
            "name": "Last action exploit",
            "last_action": Exploit(
                data_allocation_delta={
                    Subskill("subskill1"): 5,
                    Subskill("subskill2"): 5,
                }
            ),
            "skill_tree": SkillTree(
                subskills=[Subskill("subskill1"), Subskill("subskill2")],
                data_allocation={Subskill("subskill1"): 10, Subskill("subskill2"): 10},
                training_data={
                    Subskill("subskill1"): [ULID() for _ in range(10)],
                    Subskill("subskill2"): [ULID() for _ in range(10)],
                },
                skill=Skill("test_skill"),
                quality_checks={Subskill("subskill1"): [], Subskill("subskill2"): []},
                perf_on_training_data={
                    Subskill("subskill1"): 0.8,
                    Subskill("subskill2"): 0.7,
                },
            ),
            "expected_action_type": Explore,
            "expected_data_allocation_delta": None,
        },
    ],
)
def test_alternating_explore_exploit(test_case):
    explore_action = Explore(num_new_skills=1, data_allocation_for_new_skills=10)

    result = FillBalancedSkillTreePolicy._alternating_explore_exploit(
        test_case["last_action"], test_case["skill_tree"], explore_action
    )

    assert isinstance(
        result, test_case["expected_action_type"]
    ), f"Test case '{test_case['name']}' failed: expected {test_case['expected_action_type']}, got {type(result)}"

    if test_case["expected_data_allocation_delta"] is not None:
        assert isinstance(result, Exploit)
        assert (
            result.data_allocation_delta == test_case["expected_data_allocation_delta"]
        ), f"Test case '{test_case['name']}' failed: expected data allocation delta {test_case['expected_data_allocation_delta']}, got {result.data_allocation_delta}"
    else:
        assert (
            result == explore_action
        ), f"Test case '{test_case['name']}' failed: expected explore action, got {result}"
