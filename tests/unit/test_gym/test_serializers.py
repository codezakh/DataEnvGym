from dataenvgym.gym.serializers import VqaTaskInstanceSerializer
from dataenvgym.gym.domain_models import OpenEndedVqaTaskInstance
from PIL import Image
from pathlib import Path


def test_serialization(cat_with_carrot_image: Image.Image, tmp_path: Path) -> None:
    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="carrot",
    )

    task_instances = [task_instance] * 2

    save_dir = tmp_path / "save_dir"

    VqaTaskInstanceSerializer.serialize(task_instances, save_dir)

    loaded_instances = VqaTaskInstanceSerializer.deserialize(save_dir)

    assert len(loaded_instances) == 2
    assert all(
        isinstance(instance, OpenEndedVqaTaskInstance) for instance in loaded_instances
    )
