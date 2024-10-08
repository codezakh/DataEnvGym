import os

import pytest
from PIL import Image


@pytest.fixture
def path_to_example_image():
    return os.path.abspath("tests/example_image.png")


@pytest.fixture
def example_image() -> Image.Image:
    return Image.open("tests/example_image.png").convert("RGB")


@pytest.fixture
def cat_with_carrot_image() -> Image.Image:
    return Image.open("tests/data/cat_with_carrot.jpg").convert("RGB")
