import os
from pathlib import Path
from typing import Callable, Collection, Iterable, Literal, Optional, Sequence
import random

import instructor
import jinja2
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm

from dataenvgym.gym.domain_models import (
    CodeGenerationCompletedTaskInstance,
    CodeGenerationDataGenerationAgent,
    CodeGenerationDataSpec,
    CodeGenerationPredictorInterface,
    CodeGenerationTrainingDatum,
    implements,
)
from dataenvgym.gym.trainable_predictors.tool_use.vllm_predictor import render_data_spec
from dataenvgym.utils import PydanticJSONLinesWriter, extract_code_in_markdown_backticks
from dataenvgym.gym.tasks.tool_use.mnms.constants import (
    CODE_DEMO_EXAMPLES,
    TOOL_SIGNATURES,
)

DEFAULT_TEMPLATE = jinja2.Template(
    """# Tool Descriptions
The code snippet below describes the available tools and their signatures.
```python
{{ tool_descriptions }}
```

# Examples
The examples below show how to use the tools to solve a user request.
{% for example in examples %}
User Request: {{ example.user_request }}
Response:
```python
{{ example.prediction }}
```
{% endfor %}

# Instruction
Create training data to help an agent learn to solve user requests using the tools.
The agent has made the following error on a user request:

User Request: {{ completed_task_instance.task_instance.instruction }}
Bad Response:
```python
{{ completed_task_instance.predictor_response }}
```
Correct Response:
```python
{{ completed_task_instance.task_instance.solution }}
```

Write {{ num_data_specs }} new user requests that are similar in style to the user request above.
For each new user request, write a correct response using the tools.
Ensure that the new user requests are each different from the original user request and the other new user requests.
Write your solution to each user request in the same style as the correct response above.
Surround your response with ```python and ``` to be a valid Python code block.
""",
    undefined=jinja2.StrictUndefined,
)


class ResponseModel(BaseModel):
    index: int
    user_request: str
    response: str

    def to_code_generation_data_spec(self) -> CodeGenerationDataSpec:
        return CodeGenerationDataSpec(
            instruction=self.user_request,
            solution=extract_code_in_markdown_backticks(self.response),
        )


class PromptFormatter:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        tool_use_error: CodeGenerationCompletedTaskInstance,
        num_data_specs: int,
    ) -> str:
        return self.template.render(
            completed_task_instance=tool_use_error,
            examples=CODE_DEMO_EXAMPLES,
            tool_descriptions=TOOL_SIGNATURES,
            num_data_specs=num_data_specs,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )


DEFAULT_PROMPT_FORMATTER = PromptFormatter()


class DataGenerationAgent:
    def __init__(
        self,
        format_error_for_gpt_fn: Callable[
            [CodeGenerationCompletedTaskInstance, int], str
        ] = DEFAULT_PROMPT_FORMATTER,
        datum_to_generate_per_error: int = 3,
        logging_folder: Optional[Path] = None,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
        data_generation_per_invocation_limit: Optional[int] = None,
    ):
        self.model = model
        self.data_generation_per_invocation_limit = data_generation_per_invocation_limit
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
        self, data: Sequence[CodeGenerationDataSpec], generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[CodeGenerationDataSpec] = (
                PydanticJSONLinesWriter(
                    self.get_path_to_data_for_generation(
                        self.logging_folder, generation_index
                    )
                )
            )
            writer.write_batch(data)

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_training_data_from_llm(self, prompt: str) -> list[ResponseModel]:
        responses = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[ResponseModel],
            messages=[{"role": "user", "content": prompt}],  # type: ignore
        )
        return responses

    def generate_data_for_error(
        self, code_task_error: CodeGenerationCompletedTaskInstance
    ) -> Sequence[CodeGenerationTrainingDatum]:
        prompt = self.format_error_for_gpt_fn(
            code_task_error,
            self.datum_to_generate_per_error,
        )

        llm_responses = self.get_training_data_from_llm(prompt)

        data_specs = [
            response.to_code_generation_data_spec() for response in llm_responses
        ]

        self._data_for_generation_sink(data_specs, self.generation_index)

        training_data: list[CodeGenerationTrainingDatum] = []
        for data_spec in data_specs:
            training_datum = render_data_spec(data_spec)
            training_data.append(training_datum)

        return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> Sequence[CodeGenerationTrainingDatum]:
        errors = [_ for _ in completed_task_instances if not _.was_correct]
        if self.data_generation_per_invocation_limit:
            num_errors_to_sample = (
                self.data_generation_per_invocation_limit
                // self.datum_to_generate_per_error
            )
            errors = random.sample(errors, num_errors_to_sample)
        training_data: list[CodeGenerationTrainingDatum] = []
        for error in tqdm(errors, desc="Generating training data", unit="error"):
            training_data.extend(self.generate_data_for_error(error))
        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CodeGenerationCompletedTaskInstance],
        predictor: CodeGenerationPredictorInterface,
    ) -> Sequence[CodeGenerationTrainingDatum]:
        generated_training_data = self.generate_training_data(completed_task_instances)
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(CodeGenerationDataGenerationAgent)(DataGenerationAgent)
