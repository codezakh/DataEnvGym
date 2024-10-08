from dataenvgym.gym.trainable_predictors.vqa.blip import (
    BlipTrainablePredictor,
    OpenEndedVqaTaskInstance,
)
from dataenvgym.gym.domain_models import SerializableImage, VqaTrainingDatum
from PIL import Image
from ulid import ULID


def test_predicting_with_blip(cat_with_carrot_image: Image.Image) -> None:
    task_instance = OpenEndedVqaTaskInstance(
        task_name="test",
        instance_id="test_001",
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="carrot",
    )

    predictor = BlipTrainablePredictor()
    prediction = predictor.predict([task_instance])

    assert prediction == ["carrot"]


def test_training_with_blip(cat_with_carrot_image: Image.Image) -> None:
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
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    predictor = BlipTrainablePredictor()

    # Test that the predictor does not know the answer.
    prediction = predictor.predict([task_instance])
    assert prediction != [task_instance.ground_truth_label]

    # Create multiple instances of the same example to ensure overfitting
    training_data: list[VqaTrainingDatum] = [training_example] * 10

    predictor.train(training_data)

    prediction = predictor.predict([task_instance])

    assert prediction == [task_instance.ground_truth_label]


def test_training_with_blip_gpu(cat_with_carrot_image: Image.Image) -> None:
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
        instruction="What vegetable is present?",
        image=cat_with_carrot_image,
        ground_truth_label="falcarinol",
    )

    predictor = BlipTrainablePredictor(device="cuda:0")

    # Test that the predictor does not know the answer.
    prediction = predictor.predict([task_instance])
    assert prediction != [task_instance.ground_truth_label]

    # Create multiple instances of the same example to ensure overfitting
    training_data: list[VqaTrainingDatum] = [training_example] * 10

    predictor.train(training_data)

    prediction = predictor.predict([task_instance])

    assert prediction == [task_instance.ground_truth_label]
