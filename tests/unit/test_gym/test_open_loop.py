from pydantic import BaseModel
from dataenvgym.gym.open_loop import (
    PredictColorVqaTask,
    TrainingDataOfSpecificColorProductionStrategy,
    PredictMostCommonResponseTrainablePredictor,
    run_open_loop,
)
import numpy as np
from dataenvgym.gym.domain_models import SerializableImage


def test_pil_image_in_pydantic_model(example_image, tmp_path) -> None:
    class Model(BaseModel):
        image: SerializableImage

    my_model = Model(image=SerializableImage.from_pil_image(example_image))

    with open(tmp_path / "my_model.json", "w") as f:
        f.write(my_model.model_dump_json())

    with open(tmp_path / "my_model.json", "r") as f:
        loaded_model = Model.model_validate_json(f.read())

    original_image_as_array = np.array(example_image)
    loaded_image_as_array = np.array(loaded_model.image.pil_image)

    assert np.array_equal(original_image_as_array, loaded_image_as_array)


def test_entire_open_loop_with_stubs():
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PredictMostCommonResponseTrainablePredictor()
    data_strategy = TrainingDataOfSpecificColorProductionStrategy(color="red")

    performance_reports = run_open_loop(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
    )

    assert len(performance_reports) == 1
    performance_report = performance_reports[0]
    assert performance_report.overall_performance == 1.0
