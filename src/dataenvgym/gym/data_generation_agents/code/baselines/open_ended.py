from dataenvgym.gym.domain_models import (
    CodeGenerationDataGenerationAgent,
    CodeGenerationCompletedTaskInstance,
    CodeGenerationTrainingDatum,
    CodeGenerationDataSpec,
    CodeGenerationPredictorInterface,
    implements,
)
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import render_data_spec
from typing import Collection, Callable, Sequence, Optional, Literal
import instructor
from openai import OpenAI
import jinja2
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesWriter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import Iterable
from tqdm.auto import tqdm
from pydantic import BaseModel
from openai import AzureOpenAI
import os
import random
from dataenvgym.gym.trainable_predictors.code.openai_predictor import (
    answer_code_generation_question,
)

DEFAULT_TEMPLATE = jinja2.Template(
    """You are an expert Python engineer and competitive programming tutor.
You are helping a junior engineer improve their coding skills.

The junior engineer was given the following problem:
{{ code_task_error.task_instance.instruction }}

{%- if code_task_error.task_instance.starter_code -%}
{{ code_task_error.task_instance.starter_code }}
{%- endif -%}

They wrote the following code to solve the problem:
{{ code_task_error.predictor_response }}

However, their code was incorrect.
{%- if code_task_error.task_instance.solution -%}
The correct code is:
{{ code_task_error.task_instance.solution }}
{%- endif %}

First, look at the problem and the junior engineer's code, and think about what knowledge the junior engineer is missing.
Give yourself a chance to think and write down what knowledge the junior engineer is missing.
Next, write out some new problems that will help the junior engineer improve.

Here are some guidelines:
- the problems should be similar in style, length, and complexity to the original problem
- the problems should require the junior engineer to use the knowledge they are missing
- you should know how to solve the problem! This is important.

Here is the output format:
- instruction: A complete problem statement that would be found in a place like LeetCode. This will be shown verbatim to the junior engineer.
    - This should include an example input / output and a concise explanation for why the output is correct.
    - Do not write "### Question", just output the problem statement.
- starter_code: The starter code to the problem. Not all problems need starter code.

If you are including starter code, it should be formatted as follows:
```python
class Solution: 
    def functionWithMeaningfulName(self, parameter_1: list[SomeType], parameter_2: AnotherType):
        # YOUR CODE HERE
```

Keep "# YOUR CODE HERE" in the code block so the junior engineer knows where to fill in the solution.
You can change functionWithMeaningfulName to anything you want.
Don't forget to also change the parameter names to something that makes sense for the problem.

Propose no more than {{ num_data_specs }} new problems.
{{ num_no_starter_code_problems }} should have no starter code. 
The remaining {{ num_data_specs - num_no_starter_code_problems }} should have starter code.
"""
)


class ResponseModel(BaseModel):
    index: int
    instruction: str
    starter_code: str | None = None

    def to_code_generation_data_spec(self, solution: str) -> CodeGenerationDataSpec:
        return CodeGenerationDataSpec(
            instruction=self.instruction,
            solution=solution,
            starter_code=self.starter_code,
        )


class PromptFormatter:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        code_task_error: CodeGenerationCompletedTaskInstance,
        num_data_specs: int,
        num_no_starter_code_problems: int,
    ) -> str:
        return self.template.render(
            code_task_error=code_task_error,
            num_data_specs=num_data_specs,
            num_no_starter_code_problems=num_no_starter_code_problems,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )


DEFAULT_PROMPT_FORMATTER = PromptFormatter()


class DataGenerationAgent:
    def __init__(
        self,
        format_error_for_gpt_fn: Callable[
            [CodeGenerationCompletedTaskInstance, int, int], str
        ] = DEFAULT_PROMPT_FORMATTER,
        datum_to_generate_per_error: int = 3,
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
    def get_problems(self, prompt: str) -> list[ResponseModel]:
        responses = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[ResponseModel],  # type: ignore
            messages=[{"role": "user", "content": prompt}],  # type: ignore
        )
        return responses

    def get_solutions(self, problems: list[ResponseModel]) -> list[str]:
        solutions = []
        for problem in problems:
            solutions.append(
                answer_code_generation_question(
                    client=self.client,
                    question=problem.instruction,
                    starter_code=problem.starter_code,
                )
            )
        return solutions

    def generate_data_for_error(
        self, code_task_error: CodeGenerationCompletedTaskInstance
    ) -> Sequence[CodeGenerationTrainingDatum]:
        num_no_starter_code_problems = random.randint(
            0, self.datum_to_generate_per_error
        )
        prompt = self.format_error_for_gpt_fn(
            code_task_error,
            self.datum_to_generate_per_error,
            num_no_starter_code_problems,
        )

        problems = self.get_problems(prompt)
        solutions = self.get_solutions(problems)

        data_specs = [
            problem.to_code_generation_data_spec(solution)
            for problem, solution in zip(problems, solutions)
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
