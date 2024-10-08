import os
from pathlib import Path
from typing import Callable, Collection, Literal, Optional, Sequence

import instructor
import jinja2
import torch
from diffusers import AutoPipelineForText2Image  # type: ignore
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm
from ulid import ULID

from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
    OpenEndedVqaTaskInstance,
    PredictorInterface,
    SerializableImage,
    VqaDataGenerationAgent,
    VqaDataHypothesis,
    VqaTrainingDatum,
    implements,
)
from dataenvgym.utils import PydanticJSONLinesWriter

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
- "response": The correct response to the instruction. 

When crafting the training data, consider the following:
- the instructions should be similar in style, length, and complexity to the original instruction
- the images should be relevant to the instruction
- the responses should be similar in style, length, and complexity to the original response
- think about what knowledge the model might be missing that would help it answer the question correctly, and craft your training data to give it that knowledge
- each response should be a logically _correct_ response to the instruction in the context of the image description
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


class KandinskyText2ImagePipelineWrapper:
    def __init__(
        self,
        model_name_or_path: str = "kandinsky-community/kandinsky-2-2-decoder",
        device: str = "cuda:1",
    ):
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        ).to(device)
        self.pipeline.enable_model_cpu_offload()  # type: ignore
        self.pipeline.set_progress_bar_config(disable=True)

    def __call__(self, text: str) -> SerializableImage:
        return SerializableImage.from_pil_image(self.pipeline(text).images[0])  # type: ignore


class SdxlTurboText2ImagePipelineWrapper:
    def __init__(
        self,
        model_name_or_path: str = "stabilityai/sdxl-turbo",
        device: str = "cuda:1",
    ):
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipeline.set_progress_bar_config(disable=True)

    def __call__(self, text: str) -> SerializableImage:
        return SerializableImage.from_pil_image(self.pipeline(text, num_inference_steps=4, guidance_scale=0.0).images[0])  # type: ignore


class RandomImageGenerator:
    def __call__(self, text: str) -> SerializableImage:
        return SerializableImage.from_random()


class DataGenerationAgent:
    def __init__(
        self,
        format_error_for_gpt_fn: Callable[
            [CompletedVqaTaskInstance, int], str
        ] = DEFAULT_PROMPT_FORMATTER,
        datum_to_generate_per_error: int = 3,
        text_to_image_fn: Callable[[str], SerializableImage] = RandomImageGenerator(),
        logging_folder: Optional[Path] = None,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
    ):
        self.model = model
        if self.model == "gpt-4o":
            self.client = instructor.patch(
                AzureOpenAI(
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_version="2023-03-15-preview",
                )
            )
        elif self.model == "gpt-4o-mini":
            self.client = instructor.patch(
                AzureOpenAI(
                    api_key=os.environ["AZURE_OPENAI_API_KEY_GPT4O_MINI"],
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT4O_MINI"],
                    api_version="2023-03-15-preview",
                )
            )
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
        self, data: VqaDataHypothesis, generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[VqaDataHypothesis] = (
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
    def get_llm_completion(self, prompt: str) -> VqaDataHypothesis:
        data_for_generation = self.client.chat.completions.create(
            model=self.model,
            response_model=VqaDataHypothesis,  # type: ignore
            messages=[{"role": "user", "content": prompt}],  # type: ignore
        )
        return data_for_generation

    def generate_data_for_error(
        self, vqa_task_error: CompletedVqaTaskInstance
    ) -> Sequence[VqaTrainingDatum]:
        prompt = self.format_error_for_gpt_fn(
            vqa_task_error, self.datum_to_generate_per_error
        )

        data_for_generation = self.get_llm_completion(prompt)

        self._data_for_generation_sink(data_for_generation, self.generation_index)

        training_data: list[VqaTrainingDatum] = []
        for datum_to_generate in data_for_generation.data_specs:
            image = self.generate_image_from_text(datum_to_generate.image_description)
            training_datum = VqaTrainingDatum(
                ulid=ULID(),
                instruction=datum_to_generate.instruction,
                image=image,
                response=datum_to_generate.response,
            )
            training_data.append(training_datum)

        return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> Sequence[VqaTrainingDatum]:
        errors = [_ for _ in completed_task_instances if not _.was_correct]
        training_data: list[VqaTrainingDatum] = []
        for error in tqdm(errors, desc="Generating training data", unit="error"):
            training_data.extend(self.generate_data_for_error(error))
        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: PredictorInterface[OpenEndedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDatum]:
        generated_training_data = self.generate_training_data(completed_task_instances)
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(VqaDataGenerationAgent)(DataGenerationAgent)
