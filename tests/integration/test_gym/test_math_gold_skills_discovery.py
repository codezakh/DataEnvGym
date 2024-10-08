import pytest
from dataenvgym.gym.skill_discovery.math_gold_skills import AssignAnnotatedTopicsAsSkills
from dataenvgym.gym.tasks.math.MATH.task import MATHTask, MATHSplitChoices


def test_assign_annotated_topics_as_skills():
    # Initialize the MATHTask with the split 'number_theory_train_500'
    math_task = MATHTask(split="number_theory_train_500")

    # Initialize the AssignAnnotatedTopicsAsSkills
    skill_discovery = AssignAnnotatedTopicsAsSkills()

    # Check that each MathTaskInstance is assigned the "Number Theory" skill
    for task_instance in math_task.task_instances:
        skill_category = skill_discovery.get_skill_category_name_for_task_instance(
            task_instance
        )
        assert (
            skill_category == "Number Theory"
        ), f"Expected 'Number Theory', got {skill_category}"
