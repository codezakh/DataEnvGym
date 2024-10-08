from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    MathBanditDataStrategy,
    StubActionPolicy,
    Explore,
    Subskill,
    JsonExperienceCheckpointer,
)
from dataenvgym.gym.domain_models import (
    CompletedMathTaskInstance,
)
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)

import pytest
from ulid import ULID
from pathlib import Path
from typing import Collection, Sequence
from dataenvgym.gym.domain_models import MathTaskInstance


class StubMathSkillDiscoverer:
    """
    A stub implementation of a math skill discoverer for testing purposes.

    This class simulates the discovery of skills for math task instances by assigning
    skills in a round-robin fashion. It maintains a fixed number of skills and cycles
    through them as it processes task instances.

    Attributes:
        num_skills (int): The number of distinct skills to cycle through.
        skill_mapping (dict): A mapping of task instance IDs to assigned skills.
        current_skill (int): The current skill number in the rotation.

    The round-robin assignment works as follows:
    1. Each task instance is assigned a skill in the format "Skill X", where X is a number.
    2. The skill number increments with each assignment, wrapping back to 1 after reaching num_skills.
    3. This ensures an even distribution of skills across task instances for testing purposes.
    """

    def __init__(self, num_skills: int = 3):
        self.num_skills = num_skills
        self.skill_mapping = {}
        self.current_skill = 1

    def discover_skills(self, task_instances: Collection[MathTaskInstance]) -> None:
        """
        Assigns skills to a collection of task instances using the round-robin method.

        Args:
            task_instances (Collection[MathTaskInstance]): The task instances to process.
        """
        for task in task_instances:
            self.skill_mapping[task.instance_id] = f"Skill {self.current_skill}"
            self.current_skill = (self.current_skill % self.num_skills) + 1

    def get_skill_category_name_for_task_instance(
        self, task_instance: MathTaskInstance
    ) -> str:
        """
        Retrieves the skill category name for a given task instance.

        If the task instance hasn't been assigned a skill yet, it assigns one using
        the round-robin method before returning it.

        Args:
            task_instance (MathTaskInstance): The task instance to get the skill for.

        Returns:
            str: The assigned skill category name.
        """
        if task_instance.instance_id not in self.skill_mapping:
            self.skill_mapping[task_instance.instance_id] = (
                f"Skill {self.current_skill}"
            )
            self.current_skill = (self.current_skill % self.num_skills) + 1
        return self.skill_mapping[task_instance.instance_id]


class StubMathPredictor:
    def __init__(self, constant_answer: str = "The answer is 42"):
        self.constant_answer = constant_answer

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return [self.constant_answer] * len(task_instances)


@pytest.fixture
def completed_task_instances():
    return [
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_001",
                instruction="Solve for x: 2x + 5 = 13",
                ground_truth_label="The answer is \\boxed{4}",
            ),
            predictor_response="Let's solve this step by step:\n1) First, subtract 5 from both sides:\n   2x + 5 - 5 = 13 - 5\n   2x = 8\n\n2) Now, divide both sides by 2:\n   2x/2 = 8/2\n   x = 4\n\nTherefore, the answer is \\boxed{4}",
            was_correct=True,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_002",
                instruction="What is the area of a circle with radius 3 cm?",
                ground_truth_label="The answer is \\boxed{9\\pi} square cm",
            ),
            predictor_response="Let's approach this step-by-step:\n1) The formula for the area of a circle is A = πr^2, where r is the radius.\n2) We're given that the radius is 3 cm.\n3) Let's substitute this into our formula:\n   A = π(3)^2\n   A = π(9)\n   A = 9π\n\nTherefore, the answer is \\boxed{9\\pi} square cm",
            was_correct=True,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_003",
                instruction="If a triangle has angles measuring 30°, 60°, and 90°, what is the ratio of its shortest to its longest side?",
                ground_truth_label="The answer is \\boxed{1:2}",
            ),
            predictor_response="Let's think through this:\n1) This is a 30-60-90 triangle, which has special properties.\n2) If we denote the shortest side as x, then:\n   - The hypotenuse (longest side) will be 2x\n   - The middle side will be x√3\n3) The ratio of shortest to longest side is therefore x : 2x\n4) This simplifies to 1 : 2\n\nTherefore, the answer is \\boxed{1:2}",
            was_correct=True,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_004",
                instruction="What is the sum of the first 10 positive integers?",
                ground_truth_label="The answer is \\boxed{55}",
            ),
            predictor_response="Let's add them up: 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 45\nTherefore, the answer is \\boxed{45}",
            was_correct=False,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_005",
                instruction="Simplify: (x^2 + 2x + 1) - (x^2 - 2x + 1)",
                ground_truth_label="The answer is \\boxed{4x}",
            ),
            predictor_response="Let's subtract term by term:\n(x^2 + 2x + 1) - (x^2 - 2x + 1)\n= x^2 + 2x + 1 - x^2 + 2x - 1\n= 2x + 2x\n= 4x\nTherefore, the answer is \\boxed{2x}",
            was_correct=False,
        ),
        CompletedMathTaskInstance(
            ulid=ULID(),
            task_instance=MathTaskInstance(
                task_name="MATH",
                instance_id="MATH_006",
                instruction="If f(x) = 2x + 3 and g(x) = x^2, what is f(g(2))?",
                ground_truth_label="The answer is \\boxed{11}",
            ),
            predictor_response="Let's solve this step-by-step:\n1) First, we need to find g(2):\n   g(2) = 2^2 = 4\n2) Now we need to find f(4):\n   f(4) = 2(4) + 3 = 8 + 3 = 11\nTherefore, the answer is \\boxed{13}",
            was_correct=False,
        ),
    ]


def test_initializing_math_bandit_data_strategy(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
    )

    required_skills = data_strategy.determine_required_skills(completed_task_instances)
    assert len(required_skills) == num_skills

    performance_per_skill = data_strategy.determine_performance_per_skill(
        list(required_skills.keys()), completed_task_instances
    )
    # Assert each skill has 50% accuracy
    for skill, performance in performance_per_skill.items():
        assert performance == 0.5

    initial_skill_forest = data_strategy.create_initial_skill_forest(
        completed_task_instances
    )
    assert len(initial_skill_forest) == num_skills
    for skill, skill_tree in initial_skill_forest.items():
        assert len(skill_tree.subskills) == 0

    actions = data_strategy.choose_initial_explore_actions(list(required_skills.keys()))
    assert len(actions) == num_skills

    data_strategy.apply_actions_to_skill_forest(actions, initial_skill_forest)

    # Check that the skill forest now has subskills and a data allocation for each subskill.
    for skill, skill_tree in initial_skill_forest.items():
        assert len(skill_tree.subskills) > 0
        assert skill_tree.data_allocation is not None
        assert len(skill_tree.data_allocation) > 0

    # Generate data for the skill forest.
    finalized_skill_forest = data_strategy.generate_data_for_partial_skill_forest(
        initial_skill_forest, predictor=StubMathPredictor()
    )
    assert len(finalized_skill_forest) == num_skills
    # Assert that we have generated data for each skill.
    for skill, skill_tree in finalized_skill_forest.items():
        assert len(skill_tree.subskills) > 0
        assert skill_tree.training_data is not None
        assert len(skill_tree.training_data) > 0

    partial_experiences = data_strategy.construct_partial_experience(
        skill_forest=finalized_skill_forest,
        past_performance_per_skill=performance_per_skill,
        actions=actions,
    )
    assert len(partial_experiences) == num_skills
    for skill, partial_experience in partial_experiences.items():
        assert partial_experience.skill == skill
        assert partial_experience.action == actions[skill]

    expected_training_data_size = data_strategy.calculate_expected_training_data_size(
        finalized_skill_forest
    )
    assert expected_training_data_size > 0

    training_data = data_strategy.pull_training_data_for_skill_forest(
        finalized_skill_forest
    )
    assert len(training_data) == expected_training_data_size

    # Assert that we got all the ULIDs that we expected from the skill forest.
    expected_ulids: set[ULID] = set()
    for skill, skill_tree in finalized_skill_forest.items():
        for subskill in skill_tree.subskills:
            expected_ulids.update(skill_tree.training_data[subskill])
    assert len(expected_ulids) == len(training_data)
    assert all(
        training_datum.ulid in expected_ulids for training_datum in training_data
    )


def test_initializing_math_bandit_data_strategy_method(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
    )
    training_data = data_strategy.init_state(
        completed_task_instances, predictor=StubMathPredictor()
    )
    skill_forest = data_strategy.get_current_skill_forest()
    expected_training_data_size = data_strategy.calculate_expected_training_data_size(
        skill_forest
    )
    assert len(training_data) == expected_training_data_size


def test_updating_state(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
    )
    data_strategy.init_state(completed_task_instances, predictor=StubMathPredictor())
    assert len(data_strategy.past_experiences) == 0

    data_strategy.update_state(completed_task_instances)

    # Check that the state has been updated correctly.
    assert len(data_strategy.past_experiences) == 1
    # Check that the reward is 0 for all skills.
    last_experience = data_strategy.past_experiences[-1]
    for skill, experience in last_experience.items():
        assert experience.reward == 0

    # Check that the performance is 0.5 for all skills.
    for skill, experience in last_experience.items():
        assert experience.state.past_performance == 0.5

    # Now we change all the task instances to be correct.
    # The task instances are immutable, so we need to make a new copy of them.
    # Copy every attribute except for .was_correct.
    new_completed_task_instances = []
    for task_instance in completed_task_instances:
        new_task_instance = task_instance.model_copy(update={"was_correct": True})
        new_completed_task_instances.append(new_task_instance)

    data_strategy.update_state(new_completed_task_instances)
    # Check that the state has been updated correctly.
    assert len(data_strategy.past_experiences) == 2

    # Check that the reward is 0.5 for all skills.
    last_experience = data_strategy.past_experiences[-1]
    for skill, experience in last_experience.items():
        assert experience.reward == 0.5


def test_mechanics_of_transition_to_next_state(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    # Create a fixed action policy that always chooses to explore and adds
    # 5 new subskills and 3 data points for each subskill.
    fixed_action = Explore(num_new_skills=5, data_allocation_for_new_skills=3)
    action_policy = StubActionPolicy(fixed_action)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
    )
    data_strategy.init_state(completed_task_instances, predictor=StubMathPredictor())
    assert len(data_strategy.past_experiences) == 0

    # Now we pretend that we used the training data and got fresh results on the
    # task instances. But we just re-use the same task instances.
    data_strategy.update_state(completed_task_instances)
    assert len(data_strategy.past_experiences) == 1

    last_experience = data_strategy.past_experiences[-1]
    required_skills = list(last_experience.keys())
    actions = data_strategy.choose_next_actions(
        required_skills, data_strategy.past_experiences
    )

    # Assert all the actions are the fixed action we set above.
    for skill, chosen_action in actions.items():
        assert chosen_action == fixed_action

    # Copy the last experience to a new partial skill forest.
    new_partial_skill_forest = data_strategy.copy_partial_skill_forest_from_experience(
        last_experience
    )

    # Apply the actions to the new partial skill forest.
    data_strategy.apply_actions_to_skill_forest(actions, new_partial_skill_forest)

    # Now we check that the new skill forest has an increased number of subskills
    # and a data allocation for each new subskill.
    for skill, skill_tree in new_partial_skill_forest.items():
        last_skill_tree = last_experience[skill].state.skill_tree
        # Check that new skills have been added.
        assert (
            len(skill_tree.subskills)
            == len(last_skill_tree.subskills) + fixed_action.num_new_skills
        )
        # Check that the total data allocated has been increased by the expected amount.
        expected_increase = (
            fixed_action.data_allocation_for_new_skills * fixed_action.num_new_skills
        )
        assert (
            skill_tree.total_data_allocated - last_skill_tree.total_data_allocated
            == expected_increase
        )

    # Now we generate data for the new partial skill forest.
    finalized_skill_forest = data_strategy.generate_data_for_partial_skill_forest(
        new_partial_skill_forest, predictor=StubMathPredictor()
    )
    # Check that the total amount of training data in each finalized skill tree
    # is equal to the number of data points we allocated.
    for skill, skill_tree in finalized_skill_forest.items():
        partial_skill_tree = new_partial_skill_forest[skill]
        assert (
            skill_tree.total_data_allocated == partial_skill_tree.total_data_allocated
        )

    # Construct partial experiences for each skill.
    partial_experiences = (
        data_strategy.construct_partial_experiences_given_past_experiences(
            past_experiences=last_experience,
            current_skill_forest=finalized_skill_forest,
            chosen_actions=actions,
        )
    )
    assert len(partial_experiences) == num_skills
    # Check that the action for each partial experience is the fixed action we set above.
    for skill, partial_experience in partial_experiences.items():
        assert partial_experience.action == fixed_action


def test_transition_to_next_state(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    # Create a fixed action policy that always chooses to explore and adds
    # one new skill and one data point for that skill.
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    action_policy = StubActionPolicy(action)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
    )
    training_data_0 = data_strategy.init_state(
        completed_task_instances, predictor=StubMathPredictor()
    )
    assert len(data_strategy.past_experiences) == 0

    # Now we pretend that we used the training data and got fresh results on the
    # task instances. But we just re-use the same task instances.
    data_strategy.update_state(completed_task_instances)
    assert len(data_strategy.past_experiences) == 1

    # Transition to the next state.
    training_data_1 = data_strategy.transition_to_next_state(
        predictor=StubMathPredictor()
    )

    # Now we check some properties of the data strategy and training data.
    # There should be more training data after the transition, the amount
    # of new data should be equal to the number of skills times the number of
    # new subskills for each skill times the number of data points per subskill.
    num_new_datapoints = (
        num_skills * action.num_new_skills * action.data_allocation_for_new_skills
    )
    assert len(training_data_0) + num_new_datapoints == len(training_data_1)

    # We check that all rewards for the past experiences are 0. It should be 0
    # because we re-used the same task instances, so there were no changes.
    for skill, skill_experience in data_strategy.past_experiences[-1].items():
        assert skill_experience.reward == 0


def test_repeatedly_running_the_data_strategy(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    # Create a fixed action policy that always chooses to explore and adds
    # one new skill and one data point for that skill.
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    action_policy = StubActionPolicy(action)
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
    )

    # Check that the training data is increasing with each call.
    size_of_prev_training_data = None
    for i in range(10):
        training_data = data_strategy(
            completed_task_instances, predictor=StubMathPredictor()
        )
        if size_of_prev_training_data is not None:
            assert len(training_data) > size_of_prev_training_data
        size_of_prev_training_data = len(training_data)

    # Look at the history of past experiences and confirm that the width of the
    # skill tree (the number of subskills) is increasing with each call.
    for i in range(1, len(data_strategy.past_experiences)):
        previous_experience = data_strategy.past_experiences[i - 1]
        current_experience = data_strategy.past_experiences[i]
        for skill in previous_experience:
            previous_skill_tree = previous_experience[skill].state.skill_tree
            current_skill_tree = current_experience[skill].state.skill_tree
            assert len(current_skill_tree.subskills) > len(
                previous_skill_tree.subskills
            )


def test_repeatedly_running_the_data_strategy_with_checkpointing(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    # Create a fixed action policy that always chooses to explore and adds
    # one new skill and one data point for that skill.
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    action_policy = StubActionPolicy(action)
    checkpointer = JsonExperienceCheckpointer(output_path=tmp_path / "checkpoints")
    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
        experience_checkpointer=checkpointer,
    )

    # Check that the training data is increasing with each call.
    size_of_prev_training_data = None
    for i in range(10):
        training_data = data_strategy(
            completed_task_instances, predictor=StubMathPredictor()
        )
        past_experiences, current_experiences = checkpointer.load_checkpoint()
        print(len(past_experiences))
        if current_experiences is not None:
            print(len(current_experiences))
        else:
            print("Current experiences is None")
        if size_of_prev_training_data is not None:
            assert len(training_data) > size_of_prev_training_data
        size_of_prev_training_data = len(training_data)

    # Look at the history of past experiences and confirm that the width of the
    # skill tree (the number of subskills) is increasing with each call.
    for i in range(1, len(data_strategy.past_experiences)):
        previous_experience = data_strategy.past_experiences[i - 1]
        current_experience = data_strategy.past_experiences[i]
        for skill in previous_experience:
            previous_skill_tree = previous_experience[skill].state.skill_tree
            current_skill_tree = current_experience[skill].state.skill_tree
            assert len(current_skill_tree.subskills) > len(
                previous_skill_tree.subskills
            )

    # Load the experiences from the checkpoint and check that we have 10 experiences saved.
    past_experiences, current_experiences = checkpointer.load_checkpoint()

    assert len(past_experiences) == 9
    assert current_experiences is not None


class SubSkillDataGeneratorThatFailsAtIteration:
    def __init__(self, num_iterations_to_fail_at: int):
        self.num_iterations_to_fail_at = num_iterations_to_fail_at
        self.num_iterations = 0

    def __call__(
        self, subskill: Subskill, data_budget: int
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
        self.num_iterations += 1
        if self.num_iterations == self.num_iterations_to_fail_at:
            raise ValueError("Failed to generate data")
        return [
            MathTrainingDatumWithSkillCategory(
                ulid=ULID(),
                instruction=f"This is the {_}th training datum for subskill: {subskill}",
                response=f"This is the {_}th response for the 1st training datum of subskill: {subskill}",
                skill_category=str(subskill),
            )
            for _ in range(data_budget)
        ]


@pytest.mark.xfail(
    reason="Don't have a good idea right now for how to do checkpointing."
)
def test_can_recover_from_failure(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    initial_explore_action = Explore(num_new_skills=2, data_allocation_for_new_skills=5)
    action_policy = StubActionPolicy(action)

    # We start off with 3 skills, and in the first iteration we will add 2 subskills
    # for each skill. In subsequent iterations we will add one subskill for each skill.
    # The count of subskills looks like this:
    # 1. 3*2 = 6
    # 2. 6 + 3 * 1 = 9
    # 3. 9 + 3 * 1 = 12
    # If we want it to die halfway through the 3rd iteration, we will need
    # to set num_iterations_to_fail_at somewhere between 9 and 12.

    num_iterations_to_fail_at = 11
    subskill_data_generator = SubSkillDataGeneratorThatFailsAtIteration(
        num_iterations_to_fail_at
    )

    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
        initial_explore_action=initial_explore_action,
        subskill_data_generator=subskill_data_generator,
    )

    for i in [1, 2, 3]:
        data_strategy(completed_task_instances, predictor=StubMathPredictor())


def test_quality_checks_added_to_skill_tree(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    initial_explore_action = Explore(num_new_skills=2, data_allocation_for_new_skills=5)
    action_policy = StubActionPolicy(action)

    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
        initial_explore_action=initial_explore_action,
    )

    for i in range(10):
        data_strategy(completed_task_instances, predictor=StubMathPredictor())

    # Check that each subskill has as many quality checks as there are training data.
    # This may be a false negative is no training data is ever produced.
    for skill_experience_bundle in data_strategy.past_experiences:
        for skill, skill_experience in skill_experience_bundle.items():
            for subskill in skill_experience.state.skill_tree.subskills:
                num_training_data_for_subskill = len(
                    skill_experience.state.skill_tree.training_data[subskill]
                )
                num_quality_checks_for_subskill = len(
                    skill_experience.state.skill_tree.quality_checks[subskill]
                )
                assert num_training_data_for_subskill == num_quality_checks_for_subskill


def test_quality_checks_retrievable_by_ulid(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    initial_explore_action = Explore(num_new_skills=2, data_allocation_for_new_skills=5)
    action_policy = StubActionPolicy(action)
    predictor = StubMathPredictor()

    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
        initial_explore_action=initial_explore_action,
    )

    for i in range(10):
        data_strategy(completed_task_instances, predictor=predictor)

    # Check that all quality checks are retrievable by ulid.
    for skill_experience_bundle in data_strategy.past_experiences:
        for skill, skill_experience in skill_experience_bundle.items():
            for subskill in skill_experience.state.skill_tree.subskills:
                for ulid in skill_experience.state.skill_tree.quality_checks[subskill]:
                    data_strategy.get_quality_check_by_ulid(ulid)


def test_performance_on_training_data_is_calculated(
    tmp_path: Path, completed_task_instances: Collection[CompletedMathTaskInstance]
):
    num_skills = 3
    skill_discoverer = StubMathSkillDiscoverer(num_skills=num_skills)
    action = Explore(num_new_skills=1, data_allocation_for_new_skills=1)
    initial_explore_action = Explore(num_new_skills=2, data_allocation_for_new_skills=5)
    action_policy = StubActionPolicy(action)

    predictor = StubMathPredictor()

    data_strategy = MathBanditDataStrategy(
        skill_discoverer,
        logging_folder=tmp_path,
        action_policy=action_policy,
        initial_explore_action=initial_explore_action,
    )

    for i in range(10):
        data_strategy(completed_task_instances, predictor=predictor)

    # Check that the performance on training data is calculated for each subskill.
    for skill_experience_bundle in data_strategy.past_experiences:
        for skill, skill_experience in skill_experience_bundle.items():
            for subskill in skill_experience.state.skill_tree.subskills:
                assert (
                    skill_experience.state.skill_tree.perf_on_training_data[subskill]
                    is not None
                )
