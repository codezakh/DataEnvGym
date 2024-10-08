from dataenvgym.gym.tasks.vqa.gqa import GqaTask
from dataenvgym.gym.skill_discovery.gqa_gold_skills import (
    UseGoldSkillsFromGqaQuestionTypes,
)


def test_using_gold_skills_from_gqa_question_types():
    gqa_task = GqaTask(split="val")
    skill_discovery = UseGoldSkillsFromGqaQuestionTypes(gqa_task=gqa_task)
    skill_discovery.get_skill_category_name_for_task_instance(
        gqa_task.task_instances[0]
    )
