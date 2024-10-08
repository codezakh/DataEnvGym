from dataenvgym.gym.data_generation_agents.vqa.baselines.skill_list import (
    DataGenerationAgent,
)
from dataenvgym.gym.skill_discovery.llm_metacognitive import VqaSkillDiscoverer
from dataenvgym.gym.tasks.vqa.gqa import GqaTask
from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
)
from ulid import ULID


def test_data_strategy():
    skill_discoverer = VqaSkillDiscoverer()
    skill_discoverer.set_precomputed_skills(
        "workspace/notebooks__006_precompute_skills_for_gqa/skills_num_categories=5.json"
    )
    data_strategy = DataGenerationAgent(
        skill_discoverer, data_specs_per_hypothesis=2, hypotheses_per_skill_category=2
    )

    task_instances = GqaTask("val").task_instances[:5]

    errors = [
        CompletedVqaTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            was_correct=False,
            predictor_response="not sure",
        )
        for task_instance in task_instances
    ]

    training_data = data_strategy.generate_training_data(errors)

    assert len(training_data) >= 4


def test_data_strategy_with_no_errors():
    skill_discoverer = VqaSkillDiscoverer()
    skill_discoverer.set_precomputed_skills(
        "workspace/notebooks__006_precompute_skills_for_gqa/skills_num_categories=5.json"
    )
    data_strategy = DataGenerationAgent(
        skill_discoverer,
        data_specs_per_hypothesis=2,
        hypotheses_per_skill_category=2,
        generate_data_only_for_errors=True,
    )

    task_instances = GqaTask("val").task_instances[:5]

    completed_task_instances = [
        CompletedVqaTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            was_correct=True,
            predictor_response="yes",
        )
        for task_instance in task_instances
    ]

    training_data = data_strategy.generate_training_data(completed_task_instances)

    assert len(training_data) == 0

    data_strategy = DataGenerationAgent(
        skill_discoverer,
        data_specs_per_hypothesis=2,
        hypotheses_per_skill_category=2,
        generate_data_only_for_errors=False,
    )

    training_data = data_strategy.generate_training_data(completed_task_instances)

    assert len(training_data) >= 4
