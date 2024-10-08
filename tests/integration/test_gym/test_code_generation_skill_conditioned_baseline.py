from dataenvgym.gym.domain_models import (
    CodeGenerationSkillDiscoveryInterface,
    implements,
    CodeGenerationTaskInstance,
    CodeGenerationCompletedTaskInstance,
)
from typing import Collection
import pytest
from ulid import ULID
from dataenvgym.gym.data_generation_agents.code.baselines.skill_list import (
    DataGenerationAgent,
)
from .test_codegen_error_conditioned_baseline import StubCodeGenerationPredictor


class StubSkillDiscoverer:
    def __init__(
        self, instance_id_to_skill_category: dict[str, str], skill_categories: list[str]
    ):
        self.instance_id_to_skill_category = instance_id_to_skill_category
        self.skill_categories = skill_categories
        # Assert that self.instance_id_to_skill_category is a subset of self.skill_categories
        assert set(self.instance_id_to_skill_category.values()).issubset(
            set(self.skill_categories)
        )

    def get_skill_categories(self) -> list[str]:
        return self.skill_categories

    def get_skill_category_name_for_task_instance(
        self, task_instance: CodeGenerationTaskInstance
    ) -> str:
        return self.instance_id_to_skill_category[task_instance.instance_id]

    def discover_skills(
        self, task_instances: Collection[CodeGenerationTaskInstance]
    ) -> None:
        pass


implements(CodeGenerationSkillDiscoveryInterface)(StubSkillDiscoverer)


@pytest.fixture
def completed_task_instances() -> list[CodeGenerationCompletedTaskInstance]:
    # Two LeetCode problems from different categories
    task_instances = [
        CodeGenerationTaskInstance(
            task_name="Two Sum",
            instance_id="two_sum_001",
            instruction="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            solution="def two_sum(nums, target):\n    num_dict = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_dict:\n            return [num_dict[complement], i]\n        num_dict[num] = i\n    return []",
            starter_code="def two_sum(nums, target):\n    # Your code here\n    pass",
        ),
        CodeGenerationTaskInstance(
            task_name="Valid Parentheses",
            instance_id="valid_parentheses_001",
            instruction="Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
            solution="""def is_valid(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return len(stack) == 0""",
            starter_code="""def is_valid(s):
    # Your code here
    pass""",
        ),
    ]

    completed_instances = [
        CodeGenerationCompletedTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            predictor_response="I don't know!",
            was_correct=False,
        )
        for task_instance in task_instances
    ]

    return completed_instances


def test_skill_conditioned_baseline(
    completed_task_instances: list[CodeGenerationCompletedTaskInstance],
) -> None:
    instance_id_to_skill_category = {
        "two_sum_001": "Array Manipulation and Hashing",
        "valid_parentheses_001": "Stack and String Parsing",
    }
    skill_categories = [
        "Array Manipulation and Hashing",
        "Stack and String Parsing",
        "Dynamic Programming",
    ]
    skill_discoverer = StubSkillDiscoverer(
        instance_id_to_skill_category, skill_categories
    )
    data_strategy = DataGenerationAgent(
        skill_discovery_module=skill_discoverer,
        data_specs_per_skill_category=2,
    )

    generated_data = data_strategy(
        completed_task_instances, StubCodeGenerationPredictor()
    )

    assert len(generated_data) == 6
