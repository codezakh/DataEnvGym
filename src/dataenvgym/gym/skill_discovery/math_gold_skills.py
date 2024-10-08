from dataenvgym.gym.domain_models import (
    MathSkillDiscoveryInterface,
    MathTaskInstance,
    implements,
)
from dataenvgym.gym.tasks.math.MATH.task import (
    load_math_dataset,
    load_split,
)
from typing import Collection
import itertools


class AssignAnnotatedTopicsAsSkills:
    """
    Uses the predefined topics in the MATH dataset as skills.
    """

    def __init__(self):
        self.instance_id_to_type = self._build_instance_id_to_type_mapping()

    def _build_instance_id_to_type_mapping(self) -> dict[str, str]:
        math_dataset = load_math_dataset()
        train_records = load_split("train_all", math_dataset)
        test_records = load_split("test_all", math_dataset)

        instance_id_to_type = {
            record["record_id"]: record["type"]
            for record in itertools.chain(train_records, test_records)
        }
        print(len(instance_id_to_type))
        return instance_id_to_type

    def discover_skills(self, task_instances: Collection[MathTaskInstance]) -> None:
        return

    def get_skill_category_name_for_task_instance(
        self, task_instance: MathTaskInstance
    ) -> str:
        return self.instance_id_to_type.get(task_instance.instance_id, "Unknown")


implements(MathSkillDiscoveryInterface)(AssignAnnotatedTopicsAsSkills)
