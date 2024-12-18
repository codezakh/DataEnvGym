import os
from collections import defaultdict
from pathlib import Path
from typing import Collection, Iterable, Literal, Optional, Sequence, Protocol

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
from dataenvgym.gym.tasks.tool_use.mnms.evaluator import get_code_labels
from loguru import logger
import random
from dataenvgym.gym.tasks.tool_use.mnms.constants import (
    TOOL_SIGNATURES,
    CODE_DEMO_EXAMPLES,
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
Create training data that will improve an agents ability to solve user requests that require the following tools: 
{% for tool in tools_used_in_skill %}
- {{ tool }}
{% endfor %}

Below are examples of user requests requiring the tools above, the correct responses to those user requests, and whether the agent knows the correct response.

{% for skill_example in skill_examples %}
User Request: {{ skill_example.task_instance.instruction }}
Correct Response:
```python
{{ skill_example.task_instance.solution }}
```
Agent Knows Correct Response: {{ skill_example.was_correct }}
{% endfor %}

Write {{ num_data_specs }} new user requests that are similar in style to the user requests above.
For each new user request, write a correct response using the tools.
Ensure that the new user requests are each different from the original user request and the other new user requests.
Write your solution to each user request in the same style as the correct response above.
Ensure that the new user requests require all the tools in {{ tools_used_in_skill }}. 
You may use the tools in other ways than in the examples above and you may write user requests that require more than the tools in {{ tools_used_in_skill }}.
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


class PromptFormatterInterface(Protocol):
    def __call__(
        self,
        skill: tuple[str, ...],
        examples: list[CodeGenerationCompletedTaskInstance],
        num_data_specs: int,
    ) -> str: ...


class PromptFormatter:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        skill: tuple[str, ...],
        examples: list[CodeGenerationCompletedTaskInstance],
        num_data_specs: int,
    ) -> str:
        return self.template.render(
            tools_used_in_skill=skill,
            tool_descriptions=TOOL_SIGNATURES,
            examples=CODE_DEMO_EXAMPLES,  # Fixed ICL examples for MNMs.
            skill_examples=examples,  # Examples specifically for the skill.
            num_data_specs=num_data_specs,
            trim_blocks=True,
            lstrip_blocks=True,
        )


DEFAULT_PROMPT_FORMATTER = PromptFormatter()


class DataGenerationAgent:
    def __init__(
        self,
        format_error_for_gpt_fn: PromptFormatterInterface = DEFAULT_PROMPT_FORMATTER,
        datum_to_generate_per_skill: int = 3,
        logging_folder: Optional[Path] = None,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
        max_skills_to_use_during_generation: Optional[int] = None,
        sampling_method: Literal[
            "top_n_most_common",
            "top_n_with_highest_error_rate",
            "randomly_sample_imperfect_skills",
        ] = "top_n_most_common",
        imperfect_sampling_threshold: float = 0.8,
    ):
        self.sampling_method = sampling_method
        # Only used when sampling_method == "randomly_sample_imperfect_skills"
        self.imperfect_sampling_threshold = imperfect_sampling_threshold
        self.max_skills_to_use_during_generation = max_skills_to_use_during_generation
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
        self.datum_to_generate_per_skill = datum_to_generate_per_skill
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

    def generate_data_for_skill(
        self,
        skill: tuple[str, ...],
        task_instances_for_skill: list[CodeGenerationCompletedTaskInstance],
    ) -> Sequence[CodeGenerationTrainingDatum]:
        prompt = self.format_error_for_gpt_fn(
            skill=skill,
            examples=task_instances_for_skill,
            num_data_specs=self.datum_to_generate_per_skill,
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

    def bucket_task_instances_into_skills_by_tools_required(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> dict[tuple[str, ...], list[CodeGenerationCompletedTaskInstance]]:
        bucket_by_tools_required: dict[
            tuple[str, ...], list[CodeGenerationCompletedTaskInstance]
        ] = defaultdict(list)

        for task_instance in completed_task_instances:
            if task_instance.task_instance.solution is None:
                continue
            tool_names = get_code_labels(task_instance.task_instance.solution)
            bucket_by_tools_required[tuple(tool_names)].append(task_instance)

        return bucket_by_tools_required

    def get_top_n_most_common_skill_buckets(
        self,
        bucket_by_tools_required: dict[
            tuple[str, ...], list[CodeGenerationCompletedTaskInstance]
        ],
        n: int,
    ) -> list[tuple[str, ...]]:
        """
        Return the top n most common skill buckets.
        """
        return sorted(
            bucket_by_tools_required.keys(),
            key=lambda x: len(bucket_by_tools_required[x]),
            reverse=True,
        )[:n]

    def get_top_n_skills_with_highest_error_rate(
        self,
        bucket_by_tools_required: dict[
            tuple[str, ...], list[CodeGenerationCompletedTaskInstance]
        ],
        n: int,
    ) -> list[tuple[str, ...]]:
        """
        Return the top n skills with the highest error rate.
        Error rate is defined as the number of incorrect predictions divided by total predictions.
        """
        error_rates: dict[tuple[str, ...], float] = {}
        for skill, instances in bucket_by_tools_required.items():
            num_incorrect = sum(1 for instance in instances if instance.was_correct)
            error_rate = num_incorrect / len(instances)
            error_rates[skill] = error_rate

        return sorted(error_rates.keys(), key=lambda x: error_rates[x], reverse=True)[
            :n
        ]

    def get_n_skills_with_error_rates_above_threshold(
        self,
        bucket_by_tools_required: dict[
            tuple[str, ...], list[CodeGenerationCompletedTaskInstance]
        ],
        n: int,
        threshold: float,
    ) -> list[tuple[str, ...]]:
        """
        Return a random sample of skills with accuracy below a threshold.
        """
        error_rates: dict[tuple[str, ...], float] = {}
        for skill, instances in bucket_by_tools_required.items():
            num_incorrect = sum(1 for instance in instances if instance.was_correct)
            error_rate = num_incorrect / len(instances)
            error_rates[skill] = error_rate

        skills_below_threshold = [
            skill for skill, rate in error_rates.items() if rate <= threshold
        ]

        for skill in skills_below_threshold:
            logger.info(
                f"Skill={skill} has error rate={error_rates[skill]} with {len(bucket_by_tools_required[skill])} instances < {threshold}",
            )

        return random.sample(
            skills_below_threshold, min(n, len(skills_below_threshold))
        )

    def generate_training_data(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> Sequence[CodeGenerationTrainingDatum]:
        skill_buckets = self.bucket_task_instances_into_skills_by_tools_required(
            completed_task_instances
        )
        logger.info(f"Identified a total of {len(skill_buckets)} skills.")
        if self.max_skills_to_use_during_generation:
            if self.sampling_method == "top_n_most_common":
                skills_to_generate_for = self.get_top_n_most_common_skill_buckets(
                    skill_buckets,
                    self.max_skills_to_use_during_generation,
                )
            elif self.sampling_method == "top_n_with_highest_error_rate":
                skills_to_generate_for = self.get_top_n_skills_with_highest_error_rate(
                    skill_buckets,
                    self.max_skills_to_use_during_generation,
                )
            elif self.sampling_method == "randomly_sample_imperfect_skills":
                skills_to_generate_for = (
                    self.get_n_skills_with_error_rates_above_threshold(
                        skill_buckets,
                        self.max_skills_to_use_during_generation,
                        self.imperfect_sampling_threshold,
                    )
                )
        else:
            skills_to_generate_for = list(skill_buckets.keys())

        for skill in skills_to_generate_for:
            logger.info(
                f"Generating training data for skill={skill} with num_instances={len(skill_buckets[skill])}",
            )

        training_data: list[CodeGenerationTrainingDatum] = []
        iterator = tqdm(
            skills_to_generate_for, desc="Generating training data", unit="skill"
        )
        for skill in iterator:
            task_instances_for_skill = skill_buckets[skill]
            training_data.extend(
                self.generate_data_for_skill(skill, task_instances_for_skill)
            )
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
