from dataenvgym.gym.trainable_predictors.vqa.paligemma import (
    PaliGemmaTrainablePredictor,
    PaliGemmaPreferenceTrainablePredictor,
)
from dataenvgym.gym.domain_models import (
    OpenEndedVqaTaskInstance,
    SerializableImage,
    VqaTrainingDatum,
    VqaPreferenceTrainingDatum,
)
from PIL import Image
from ulid import ULID
from pathlib import Path
import ray


def test_predicting(cat_with_carrot_image: Image.Image) -> None:
    ray.init(ignore_reinit_error=True)
    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="carrot",
    )

    predictor = PaliGemmaTrainablePredictor()
    prediction = predictor.predict([task_instance])
    assert prediction == ["carrot"]


# CUDA_VISIBLE_DEVICES=1,2 pytest -s tests/integration/test_gym/test_paligemma_trainable_predictor.py::test_training
def test_training(cat_with_carrot_image: Image.Image, tmp_path: Path) -> None:
    ray.init(ignore_reinit_error=True)
    # Create a training example
    training_example = VqaTrainingDatum(
        ulid=ULID(),
        instruction="What compound in this vegetable is a natural pesticide?",
        image=SerializableImage.from_pil_image(cat_with_carrot_image),
        response="falcarinol",
    )

    # Create a test instance using the same image and instruction
    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What compound in this vegetable is a natural pesticide?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    predictor = PaliGemmaTrainablePredictor()

    # Test that the predictor does not know the answer.
    prediction = predictor.predict([task_instance])
    assert prediction != [task_instance.ground_truth_label]

    # Create multiple instances of the same example to ensure overfitting
    training_data: list[VqaTrainingDatum] = [training_example] * 10

    predictor.train(training_data)

    prediction = predictor.predict([task_instance])

    assert prediction == [task_instance.ground_truth_label]


# CUDA_VISIBLE_DEVICES=1,2 pytest -s tests/integration/test_gym/test_paligemma_trainable_predictor.py::test_preference_training
def test_preference_training(
    cat_with_carrot_image: Image.Image, tmp_path: Path
) -> None:
    ray.init(ignore_reinit_error=True)

    # Create a training example
    training_example = VqaPreferenceTrainingDatum(
        ulid=ULID(),
        instruction="What compound in this vegetable is a natural pesticide?",
        image=SerializableImage.from_pil_image(cat_with_carrot_image),
        chosen_response="falcarinol",
        rejected_response="carrot",
    )

    # Create a test instance using the same image and instruction
    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What compound in this vegetable is a natural pesticide?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    predictor = PaliGemmaPreferenceTrainablePredictor()

    # Test that the predictor does not know the answer.
    prediction = predictor.predict([task_instance])
    assert prediction != [task_instance.ground_truth_label]

    # Create multiple instances of the same example to ensure overfitting
    training_data: list[VqaPreferenceTrainingDatum] = [training_example] * 10

    predictor.train_preference(training_data)

    prediction = predictor.predict([task_instance])

    assert prediction == [task_instance.ground_truth_label]
