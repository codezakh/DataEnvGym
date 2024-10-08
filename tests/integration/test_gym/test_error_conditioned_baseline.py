from dataenvgym.gym.data_generation_agents.vqa.baselines.open_ended import (
    DataGenerationAgent,
    KandinskyText2ImagePipelineWrapper,
)

from dataenvgym.gym.data_generation_agents.vqa.baselines.error_conditioned_preference_baseline import (
    DataGenerationAgent,
)

from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
    OpenEndedVqaTaskInstance,
    PreferenceVqaDataSpec,
)
from dataenvgym.gym.stub_implementations import StubTrainablePredictor
from PIL import Image
from ulid import ULID


def test_data_strategy(cat_with_carrot_image: Image.Image) -> None:
    strategy = DataGenerationAgent(datum_to_generate_per_error=2)

    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="carrot",
    )
    completed_task_with_error = CompletedVqaTaskInstance(
        ulid=ULID(),
        task_instance=task_instance,
        predictor_response="banana",
        was_correct=False,
    )

    second_task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_002",
        instruction="What natural pesticide is present in this vegetable?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    completed_task_with_error_2 = CompletedVqaTaskInstance(
        ulid=ULID(),
        task_instance=second_task_instance,
        predictor_response="carrot",
        was_correct=False,
    )

    training_data = strategy(
        [completed_task_with_error, completed_task_with_error_2],
        StubTrainablePredictor(),
    )

    assert len(training_data) == 4


def test_preference_data_strategy(cat_with_carrot_image: Image.Image) -> None:
    strategy = DataGenerationAgent(datum_to_generate_per_error=2)

    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="carrot",
    )

    completed_task_with_error = CompletedVqaTaskInstance(
        ulid=ULID(),
        task_instance=task_instance,
        predictor_response="banana",
        was_correct=False,
    )

    second_task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_002",
        instruction="What natural pesticide is present in this vegetable?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    completed_task_with_error_2 = CompletedVqaTaskInstance(
        ulid=ULID(),
        task_instance=second_task_instance,
        predictor_response="carrot",
        was_correct=False,
    )

    training_data = strategy(
        [completed_task_with_error, completed_task_with_error_2],
        StubTrainablePredictor(),
    )

    assert len(training_data) == 4


def test_generating_real_images():
    pipeline = KandinskyText2ImagePipelineWrapper()
    image = pipeline("A cat with a carrot")
