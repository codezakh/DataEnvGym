import os
import random
from pathlib import Path
from typing import Collection, Iterable, Sequence, cast

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

from dataenvgym.gym.domain_models import (
    CompletedMathTaskInstance,
    MathDataSpec,
    MathPredictorInterface,
    MathDataGenerationAgent,
    MathTrainingDatum,
    implements,
)
from dataenvgym.gym.tasks.math.MATH.scoring import render_solution_for_scoring
from dataenvgym.gym.tasks.math.MATH.task import MATHTask
from dataenvgym.utils import PydanticJSONLinesWriter

DEFAULT_TEMPLATE = jinja2.Template(
    """
You are an expert math educator and your task is to create training data for improving a model's skills in solving math problems.

Here are examples of math problems:
{% for task_instance in task_instances %}
{{ task_instance.instruction }}
{% endfor %}

The training data you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "problem": The math problem you want the model to solve. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "chain_of_thought": A step-by-step explanation of how the model should solve the problem. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "final_answer": The final answer to the problem as a LaTeX string. For example '17' or '\\frac{1}{2} or `\\matrix{1 & 2 \\cr 3 & 4}`. Do not write a sentence here, just the answer.

Produce {{ num_data_specs }} problems. 
"""
)


class DataStrategy:
    def __init__(
        self,
        logging_folder: Path,
        data_specs_per_llm_call: int = 10,
        num_training_data_per_invocation: int = 120,
    ):
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.generation_index = 0
        # We will randomly sample from these to generate training data,
        # these will serve as in-context examples for the data generating LLM.
        self.task_instances = MATHTask("train_all").task_instances
        self.data_specs_per_llm_call = data_specs_per_llm_call
        self.logging_folder = logging_folder
        self.num_training_data_per_invocation = num_training_data_per_invocation
        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_path_to_log_data_specs(logging_folder: Path, generation_index: int):
        return logging_folder / f"data_specs_{generation_index}.jsonl"

    def log_data_specs(
        self, data: Collection[MathDataSpec], generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[MathDataSpec] = PydanticJSONLinesWriter(
                self.get_path_to_log_data_specs(self.logging_folder, generation_index)
            )
            writer.write_batch(data)

    def render_prompt(
        self,
    ) -> str:
        in_context_examples = random.sample(self.task_instances, k=1)
        return DEFAULT_TEMPLATE.render(
            num_data_specs=self.data_specs_per_llm_call,
            trim_blocks=True,
            lstrip_blocks=True,
            task_instances=in_context_examples,
            undefined=jinja2.StrictUndefined,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_specs_from_llm(self, prompt: str) -> Iterable[MathDataSpec]:
        data_specs = self.client.chat.completions.create(
            model="gpt-4o",
            response_model=Iterable[MathDataSpec],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_specs = cast(Iterable[MathDataSpec], data_specs)
        return data_specs

    def get_num_llm_calls_needed(self) -> int:
        return self.num_training_data_per_invocation // self.data_specs_per_llm_call

    @staticmethod
    def convert_data_spec_to_training_datum(
        data_spec: MathDataSpec,
    ) -> MathTrainingDatum:
        return MathTrainingDatum(
            ulid=ULID(),
            instruction=data_spec.problem,
            response=render_solution_for_scoring(
                chain_of_thought=data_spec.chain_of_thought,
                final_answer=data_spec.final_answer,
            ),
        )

    def generate_training_data(
        self,
    ) -> Sequence[MathTrainingDatum]:
        prompt = self.render_prompt()

        data_specs: list[MathDataSpec] = []

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
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]:
        generated_training_data = self.generate_training_data()
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(MathDataGenerationAgent)(DataStrategy)
