import pytest
from dataenvgym.gym.skill_discovery.vqa.llm_metacognitive import (
    VqaSkillDiscoverer,
)
from dataenvgym.gym.domain_models import OpenEndedVqaTaskInstance
from PIL import Image
from pathlib import Path


@pytest.fixture(scope="module")
def vqa_discovery_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[
    VqaSkillDiscoverer,
    list[OpenEndedVqaTaskInstance],
    list[OpenEndedVqaTaskInstance],
]:
    example_image = Image.new("RGB", (100, 100))

    requires_color_identification = [
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="123",
            instruction="What is the color of the object?",
            ground_truth_label="blue",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="124",
            instruction="What is the color of the object?",
            ground_truth_label="green",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="125",
            instruction="What is the color of the object?",
            ground_truth_label="red",
            image=example_image,
        ),
    ]
    requires_shape_identification = [
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="126",
            instruction="What shape is the object?",
            ground_truth_label="pentagon",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="127",
            instruction="What shape is the object?",
            ground_truth_label="rhombus",
            image=example_image,
        ),
        OpenEndedVqaTaskInstance(
            task_name="gqa",
            instance_id="128",
            instruction="What shape is the object?",
            ground_truth_label="ellipse",
            image=example_image,
        ),
    ]

    checkpoint_path = tmp_path_factory.mktemp("checkpoint")

    discoverer = VqaSkillDiscoverer(checkpoint_path=checkpoint_path)
    discoverer.discover_skills(
        requires_color_identification + requires_shape_identification
    )

    return discoverer, requires_color_identification, requires_shape_identification


def test_skills_discovered(vqa_discovery_setup):
    discoverer, _, _ = vqa_discovery_setup
    assert discoverer.discovery_result is not None
    assert len(discoverer.discovery_result.skill_categories) >= 2


def test_skill_category_for_color_identification(vqa_discovery_setup):
    discoverer, requires_color_identification, _ = vqa_discovery_setup
    category_for_color_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_color_identification[0]
        )
    )
    assert category_for_color_identification is not None


def test_skill_category_for_shape_identification(vqa_discovery_setup):
    discoverer, _, requires_shape_identification = vqa_discovery_setup
    category_for_shape_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_shape_identification[0]
        )
    )
    assert category_for_shape_identification is not None


def test_new_shape_identification_task(vqa_discovery_setup):
    discoverer, _, requires_shape_identification = vqa_discovery_setup
    example_image = Image.new("RGB", (100, 100))
    new_shape_identification_task = OpenEndedVqaTaskInstance(
        task_name="gqa",
        instance_id="129",
        instruction="What shape is the object?",
        ground_truth_label="circle",
        image=example_image,
    )

    category_for_shape_identification = (
        discoverer.get_skill_category_name_for_task_instance(
            requires_shape_identification[0]
        )
    )

    assert (
        discoverer.get_skill_category_name_for_task_instance(
            new_shape_identification_task
        )
        == category_for_shape_identification
    )


def test_loading_and_saving_results(tmp_path: Path, vqa_discovery_setup) -> None:
    discoverer, _, _ = vqa_discovery_setup
    save_path = tmp_path / "discovery_result.json"
    discoverer.save_precomputed_skills(save_path)

    new_discoverer = VqaSkillDiscoverer()
    assert new_discoverer.discovery_result is None
    new_discoverer.set_precomputed_skills(save_path)

    assert new_discoverer.discovery_result is not None

    example_image = Image.new("RGB", (100, 100))
    new_shape_identification_task = OpenEndedVqaTaskInstance(
        task_name="gqa",
        instance_id="129",
        instruction="What shape is the object?",
        ground_truth_label="circle",
        image=example_image,
    )

    # Check that the new and old discoverer give the same category for the same task.
    assert discoverer.get_skill_category_name_for_task_instance(
        new_shape_identification_task
    ) == new_discoverer.get_skill_category_name_for_task_instance(
        new_shape_identification_task
    )
