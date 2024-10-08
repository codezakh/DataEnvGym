from dataenvgym.gym.domain_models import OpenEndedVqaTaskInstance


def test_open_ended_vqa_task_instance(example_image):
    OpenEndedVqaTaskInstance(
        task_name="",
        instance_id="",
        instruction="",
        image=example_image,
        ground_truth_label="",
    ).model_dump()
