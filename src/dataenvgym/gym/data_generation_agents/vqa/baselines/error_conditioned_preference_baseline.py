from pathlib import Path
from typing import Callable, Collection, Optional, Sequence

import instructor
import jinja2
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm
from ulid import ULID

from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
    VqaPreferenceDataGenerationAgent,
    SerializableImage,
    VqaPreferenceDataHypothesis,
    VqaPreferenceTrainingDatum,
    implements,
    VqaPredictorInterface,
)
from dataenvgym.utils import PydanticJSONLinesWriter

from .open_ended import RandomImageGenerator

DEFAULT_TEMPLATE = jinja2.Template(
    """
You are a experienced engineer and your role is to provide training data to correct a model. 

The model was given the following instruction and responded incorrectly.
Instruction: {{ vqa_task_error.task_instance.instruction }}
Model Response: {{ vqa_task_error.predictor_response }}
Correct Response: {{ vqa_task_error.task_instance.ground_truth_label }}

Craft training data to improve the model. The model will be trained on the data you provide.
The training data you produce must be valid JSON with the following fields:
- "inferred_weak_skill": A concise to-the-point description of why you think the model responded incorrectly and how you'll fix it. Produce this first to give yourself a chance to think.
- "instruction": The instruction you want the model to respond to.
- "image": The description of an image the instruction is about. 
- "rejected_response": A plausible but incorrect response to the instruction.
- "chosen_response": The correct response to the instruction. 

When crafting the training data, consider the following:
- the instructions should be similar in style, length, and complexity to the original instruction
- the images should be relevant to the instruction
- the responses should be similar in style, length, and complexity to the original response
- think about what knowledge the model might be missing that would help it answer the question correctly, and craft your training data to give it that knowledge
- think about what knowledge the model might have that would lead it to give the incorrect response, and craft your training data to correct that knowledge
- each chosen_response should be a logically _correct_ response to the instruction in the context of the image description
- the training data should be diverse enough to help the model generalize to new examples

Produce no more than {{ num_training_data }} training data examples.
"""
)


class PromptFormatter:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self, vqa_task_error: CompletedVqaTaskInstance, num_training_data: int
    ) -> str:
        return self.template.render(
            vqa_task_error=vqa_task_error, num_training_data=num_training_data
        )


DEFAULT_PROMPT_FORMATTER = PromptFormatter()


class DataGenerationAgent:
    def __init__(
        self,
        format_error_for_gpt_fn: Callable[
            [CompletedVqaTaskInstance, int], str
        ] = DEFAULT_PROMPT_FORMATTER,
        datum_to_generate_per_error: int = 3,
        text_to_image_fn: Callable[[str], SerializableImage] = RandomImageGenerator(),
        logging_folder: Optional[Path] = None,
    ):
        self.client = instructor.from_openai(OpenAI())
        self.format_error_for_gpt_fn = format_error_for_gpt_fn
        self.datum_to_generate_per_error = datum_to_generate_per_error
        self.text_to_image_fn = text_to_image_fn
        self.generation_index = 0
        self.logging_folder = logging_folder

        # Make sure the logging folder exists, create it if it doesn't.
        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_path_to_data_for_generation(logging_folder: Path, generation_index: int):
        return (
            logging_folder / f"data_for_generation_generation_{generation_index}.jsonl"
        )

    def _data_for_generation_sink(
        self, data: VqaPreferenceDataHypothesis, generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[VqaPreferenceDataHypothesis] = (
                PydanticJSONLinesWriter(
                    self.get_path_to_data_for_generation(
                        self.logging_folder, generation_index
                    )
                )
            )
            writer(data)

    def generate_image_from_text(self, text: str) -> SerializableImage:
        return self.text_to_image_fn(text)

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_llm_completion(self, prompt: str) -> VqaPreferenceDataHypothesis:
        data_for_generation = self.client.chat.completions.create(
            model="gpt-4o",
            response_model=VqaPreferenceDataHypothesis,
            messages=[{"role": "user", "content": prompt}],
        )
        return data_for_generation

    def generate_data_for_error(
        self, vqa_task_error: CompletedVqaTaskInstance
    ) -> Sequence[VqaPreferenceTrainingDatum]:
        prompt = self.format_error_for_gpt_fn(
            vqa_task_error, self.datum_to_generate_per_error
        )

        data_for_generation = self.get_llm_completion(prompt)

        self._data_for_generation_sink(data_for_generation, self.generation_index)

        training_data: list[VqaPreferenceTrainingDatum] = []
        for datum_to_generate in data_for_generation.data_specs:
            image = self.generate_image_from_text(datum_to_generate.image_description)
            training_datum = VqaPreferenceTrainingDatum(
                ulid=ULID(),
                instruction=datum_to_generate.instruction,
                image=image,
                chosen_response=datum_to_generate.chosen_response,
                rejected_response=datum_to_generate.rejected_response,
            )
            training_data.append(training_datum)

        return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> Sequence[VqaPreferenceTrainingDatum]:
        errors = [_ for _ in completed_task_instances if not _.was_correct]
        training_data: list[VqaPreferenceTrainingDatum] = []
        for error in tqdm(errors, desc="Generating training data", unit="error"):
            training_data.extend(self.generate_data_for_error(error))
        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: VqaPredictorInterface,
    ) -> Sequence[VqaPreferenceTrainingDatum]:
        generated_training_data = self.generate_training_data(completed_task_instances)
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(VqaPreferenceDataGenerationAgent)(DataGenerationAgent)
