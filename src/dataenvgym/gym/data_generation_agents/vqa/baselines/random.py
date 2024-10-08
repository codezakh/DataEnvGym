import os
from pathlib import Path
from typing import Callable, Collection, Iterable, Sequence, cast

import instructor
import jinja2
from loguru import logger
from openai import AzureOpenAI
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm
from ulid import ULID

from dataenvgym.gym.data_generation_agents.vqa.baselines.open_ended import (
    RandomImageGenerator,
)
from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
    OpenEndedVqaTaskInstance,
    PredictorInterface,
    SerializableImage,
    VqaDataGenerationAgent,
    VqaDataSpec,
    VqaTrainingDatum,
    implements,
)
from dataenvgym.utils import PydanticJSONLinesWriter

DEFAULT_TEMPLATE = jinja2.Template(
    """
You are a experienced engineer and your role is to provide training data to improve a visual question answering model.

Craft training data to improve the model. The model will be trained on the data you provide.
The training data you produce must be valid JSON with the following fields:
- "instruction": The instruction you want the model to respond to.
- "image": The description of an image the instruction is about. 
- "response": The correct response to the instruction. 

When crafting the training data, consider the following:
- the images should be relevant to the instruction
- the responses should be short and concise, no more than 1-2 words 
- each response should be a logically _correct_ response to the instruction in the context of the image description
- the training data should be diverse enough to help the model generalize to new examples

Produce no more than {{ num_training_data }} training data examples.
""",
    undefined=jinja2.StrictUndefined,
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


class VqaDataSpecWithIndex(VqaDataSpec):
    index: int


class DataGenerationAgent:
    def __init__(
        self,
        logging_folder: Path,
        data_specs_per_llm_call: int = 10,
        num_training_data_per_invocation: int = 120,
        text_to_image_fn: Callable[[str], SerializableImage] = RandomImageGenerator(),
    ):
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.generation_index = 0
        self.data_specs_per_llm_call = data_specs_per_llm_call
        self.logging_folder = logging_folder
        self.num_training_data_per_invocation = num_training_data_per_invocation
        self.text_to_image_fn = text_to_image_fn

        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_path_to_log_data_specs(logging_folder: Path, generation_index: int):
        return logging_folder / f"data_specs_{generation_index}.jsonl"

    def log_data_specs(
        self, data: Collection[VqaDataSpec], generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[VqaDataSpec] = PydanticJSONLinesWriter(
                self.get_path_to_log_data_specs(self.logging_folder, generation_index)
            )
            writer.write_batch(data)

    def render_prompt(self) -> str:
        return DEFAULT_TEMPLATE.render(
            num_training_data=self.data_specs_per_llm_call,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_specs_from_llm(self, prompt: str) -> Iterable[VqaDataSpec]:
        # Make llm count using an explicit index.
        data_specs = self.client.chat.completions.create(
            model="gpt-4o",
            response_model=Iterable[VqaDataSpecWithIndex],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_specs = cast(Iterable[VqaDataSpecWithIndex], data_specs)
        return data_specs

    def get_num_llm_calls_needed(self) -> int:
        return self.num_training_data_per_invocation // self.data_specs_per_llm_call

    def convert_data_spec_to_training_datum(
        self, data_spec: VqaDataSpec
    ) -> VqaTrainingDatum:
        image = self.text_to_image_fn(data_spec.image_description)
        return VqaTrainingDatum(
            ulid=ULID(),
            instruction=data_spec.instruction,
            image=image,
            response=data_spec.response,
        )

    def generate_training_data(self) -> Sequence[VqaTrainingDatum]:
        prompt = self.render_prompt()

        data_specs: list[VqaDataSpec] = []

        llm_calls_needed = self.get_num_llm_calls_needed()

        logger.info(
            f"Generating {self.num_training_data_per_invocation} training data specifications, requiring {llm_calls_needed} LLM calls."
        )

        for _ in tqdm(
            range(llm_calls_needed), desc="LLM calls", total=llm_calls_needed
        ):
            try:
                data_specs_batch = list(self.get_data_specs_from_llm(prompt))
                self.log_data_specs(data_specs_batch, self.generation_index)
                data_specs.extend(data_specs_batch)
            except RetryError:
                logger.opt(exception=True).error(
                    "A call to get data specs from the LLM failed."
                )
                continue

        logger.info(f"Generated {len(data_specs)} data specifications.")
        self.log_data_specs(data_specs, self.generation_index)

        training_data = [
            self.convert_data_spec_to_training_datum(_) for _ in data_specs
        ]

        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: PredictorInterface[OpenEndedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDatum]:
        generated_training_data = self.generate_training_data()
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(VqaDataGenerationAgent)(DataGenerationAgent)
