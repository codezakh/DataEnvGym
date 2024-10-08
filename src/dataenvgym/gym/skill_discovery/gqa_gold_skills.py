from ..domain_models import (
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    VqaSkillDiscoveryInterface,
    implements,
)
from typing import Collection
from pydantic import BaseModel
from dataenvgym.utils import PydanticJSONLinesReader
from dataenvgym.gym.tasks.vqa.gqa import GqaTask

QUESTION_TYPE_TO_SKILL_NAME_RECORDS_PATH = "workspace/notebooks__008_turn_gqa_question_types_into_human_readable_skill_names/question_type_to_skill_name.jsonl"


class QuestionTypeToSkillName(BaseModel):
    skill_name: str
    reason_for_skill_name: str
    original_question_type: str


class UseGoldSkillsFromGqaQuestionTypes:
    """A skill discovery module that uses the pre-labeled gold skills
    for GQA questions."""

    def __init__(
        self,
        gqa_task: GqaTask,
        question_type_to_skill_name_mapping_path: str = QUESTION_TYPE_TO_SKILL_NAME_RECORDS_PATH,
    ) -> None:
        self.gqa_task = gqa_task
        self.question_type_to_skill_name_mapping_path = (
            question_type_to_skill_name_mapping_path
        )

        reader = PydanticJSONLinesReader(
            question_type_to_skill_name_mapping_path, QuestionTypeToSkillName
        )

        self.question_type_to_skill_name_mapping = {
            record.original_question_type: record.skill_name for record in reader()
        }

    def discover_skills(
        self, task_instances: Collection[OpenEndedVqaTaskInstance]
    ) -> None:
        return

    def get_skill_category_name_for_task_instance(
        self, task_instance: OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
    ) -> str:

        gqa_record_for_task_instance = self.gqa_task.gqa_id_to_gqa_record[
            task_instance.instance_id
        ]

        question_type = gqa_record_for_task_instance["question_type"]

        return self.question_type_to_skill_name_mapping[question_type]


implements(VqaSkillDiscoveryInterface)(UseGoldSkillsFromGqaQuestionTypes)
