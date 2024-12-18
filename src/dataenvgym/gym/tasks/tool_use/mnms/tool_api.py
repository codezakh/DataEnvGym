# BEGIN PREAMBLE #
# This file contains the tool api for the m&ms task.
# This file is read verbatim by functions which generate prompts.
# Do not put any comments or docstrings in this file other than those
# you _want_ to appear in the generated prompts.
# This premable will be stripped from the source code when the prompts are generated.
# END PREAMBLE #
from typing import TypedDict, List, Union, TypeAlias
from PIL import Image

BBox: TypeAlias = List[float]  # [x1, y1, x2, y2] coordinates


class Detection(TypedDict):
    bbox: BBox
    label: str


class Segmentation(TypedDict):
    mask: List[List[float]]
    inst_id: int
    label: str
    bbox: BBox


class PILImage(TypedDict):
    image: Image.Image


class Text(TypedDict):
    text: str


class Answer(TypedDict):
    answer: str


class Number(TypedDict):
    number: int


class Love(TypedDict):
    number: int
    message: str


class Location(TypedDict):
    lon: float
    lat: float


class SelectedObject(TypedDict):
    object: Detection


class ObjectDetections(TypedDict):
    image: Image.Image
    objects: List[Detection]


class ImageSegmentation(TypedDict):
    image: Image.Image
    objects: List[Segmentation]


class Wind10m(TypedDict):
    direction: str
    speed: int


class WeatherPoint(TypedDict):
    timepoint: int
    cloudcover: int
    lifted_index: int
    prec_type: str
    prec_amount: int
    temp2m: int
    rh2m: str
    wind10m: Wind10m
    weather: str


class Weather(TypedDict):
    objects: List[WeatherPoint]


def text_generation(text: str) -> Text: ...


def text_summarization(text: str) -> Text: ...


def text_classification(text: str) -> Text: ...


def question_answering(question: str, text: str) -> Text: ...


def automatic_speech_recognition(audio: str) -> Text: ...


def image_generation(text: str) -> PILImage: ...


def image_captioning(image: Union[str, Image.Image]) -> Text: ...


def image_editing(image: Union[str, Image.Image], prompt: str) -> PILImage: ...


def image_classification(image: Union[str, Image.Image]) -> Text: ...


def visual_question_answering(
    image: Union[str, Image.Image], question: str
) -> Text: ...


def object_detection(image: Union[str, Image.Image]) -> ObjectDetections: ...


def image_segmentation(image: Union[str, Image.Image]) -> ImageSegmentation: ...


def optical_character_recognition(image: Union[str, Image.Image]) -> Text: ...


def image_crop(image: Union[str, Image.Image], object: Detection) -> PILImage: ...


def image_crop_left(image: Union[str, Image.Image]) -> PILImage: ...


def image_crop_right(image: Union[str, Image.Image]) -> PILImage: ...


def image_crop_top(image: Union[str, Image.Image]) -> PILImage: ...


def image_crop_bottom(image: Union[str, Image.Image]) -> PILImage: ...


def background_blur(image: Union[str, Image.Image], object: Detection) -> PILImage: ...


def color_pop(image: Union[str, Image.Image], object: Detection) -> PILImage: ...


def count(objects: List[Detection]) -> Number: ...


def tag(image: Union[str, Image.Image], objects: List[Detection]) -> PILImage: ...


def emoji(
    image: Union[str, Image.Image], object: Detection, emoji: str
) -> PILImage: ...


def select_object(objects: List[Detection], object_name: str) -> SelectedObject: ...


def get_date_fact(date: str) -> Text: ...


def get_year_fact(year: Union[str, int]) -> Text: ...


def get_math_fact(number: Union[str, int]) -> Text: ...


def get_trivia_fact(number: Union[str, int]) -> Text: ...


def love_calculator(first_name: str, second_name: str) -> Love: ...


def get_location(city: str) -> Location: ...


def search_movie(movie_title: str, movie_year: Union[str, int]) -> Text: ...


def get_weather(lon: Union[str, float], lat: Union[str, float]) -> Weather: ...


def wikipedia_simple_search(text: str) -> Text: ...
