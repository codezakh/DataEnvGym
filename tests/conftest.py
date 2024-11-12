import os
import pytest
from PIL import Image
from typing import Sequence

from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathPredictorInterface,
    implements,
)


class StubMathPredictor:
    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        # Return a dummy response for each task instance
        return ["42" for _ in task_instances]


implements(MathPredictorInterface)(StubMathPredictor)


@pytest.fixture
def path_to_example_image():
    return os.path.abspath("tests/example_image.png")


@pytest.fixture
def example_image() -> Image.Image:
    return Image.open("tests/example_image.png").convert("RGB")


@pytest.fixture
def cat_with_carrot_image() -> Image.Image:
    return Image.open("tests/data/cat_with_carrot.jpg").convert("RGB")


@pytest.fixture
def stub_math_predictor() -> MathPredictorInterface:
    return StubMathPredictor()
