import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Collection, Iterable, Literal, Optional, Sequence, cast

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
    MathDataHypothesis,
    MathPredictorInterface,
    MathSkillDiscoveryInterface,
    MathDataGenerationAgent,
    MathTrainingDatum,
    implements,
)
from dataenvgym.gym.tasks.math.MATH.scoring import render_solution_for_scoring
from dataenvgym.utils import PydanticJSONLinesWriter

DEFAULT_TEMPLATE = jinja2.Template(
    """
You are an experienced math educator and your task is to create training data for improving a model's skills in solving math problems, especially under the category of "{{ skill_category }}".

Here are examples of mistakes the model made while solving problems requiring "{{ skill_category }}".
The model was given a math problem and responded incorrectly.
{% for math_task_error in math_task_errors %}
- Problem: {{ math_task_error.task_instance.instruction }}
  - Model Response: {{ math_task_error.predictor_response }}
  - Correct Response: {{ math_task_error.task_instance.ground_truth_label }}
{% endfor %}

You will propose hypotheses about what training data the model needs to improve its skills under "{{ skill_category }}".
For certain skills, the model may not have made any mistakes. In that case, propose hypotheses that will help the model improve on harder examples of the skill.

The training data you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "inferred_weak_skill": A concise description of the skill under "{{ skill_category }}" that the model is weak at, and what kind of (problem, response) data will help the model improve.
- "problem": The math problem you want the model to solve. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "chain_of_thought": A step-by-step explanation of how the model should solve the problem. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "final_answer": The final answer to the problem as a LaTeX string. For example '17' or '\\frac{1}{2} or `\\matrix{1 & 2 \\cr 3 & 4}`. Do not write a sentence here, just the answer.

Produce {{ num_hypotheses }} hypotheses.
For each hypothesis and weak skill, produce {{ num_data_specs }} specifications for training data.
"""
)


class Verbalizer:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        math_task_errors: Sequence[CompletedMathTaskInstance],
        num_data_specs: int,
        skill_category: str,
        num_hypotheses: int,
    ) -> str:
        return self.template.render(
            math_task_errors=math_task_errors,
            num_data_specs=num_data_specs,
            skill_category=skill_category,
            num_hypotheses=num_hypotheses,
            trim_blocks=True,
            lstrip_blocks=True,
        )


class MathTrainingDatumWithSkillCategory(MathTrainingDatum):
    skill_category: str


class DataGenerationAgent:
    def __init__(
        self,
        skill_discovery_module: MathSkillDiscoveryInterface,
        verbalizer_for_skill_category: Verbalizer = Verbalizer(),
        logging_folder: Optional[Path] = None,
        data_specs_per_hypothesis: int = 5,
        hypotheses_per_skill_category: int = 5,
        generate_data_only_for_errors: bool = True,
        resume_from_checkpoint: bool = False,
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
        self.verbalizer_for_skill_category = verbalizer_for_skill_category
        self.hypotheses_per_skill_category = hypotheses_per_skill_category
        self.data_specs_per_hypothesis = data_specs_per_hypothesis
        self.generation_index = 0
        self.logging_folder = logging_folder
        self.skill_discovery_module = skill_discovery_module
        self.generate_data_only_for_errors = generate_data_only_for_errors

        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_path_to_data_hypotheses(logging_folder: Path, generation_index: int):
        return (
            logging_folder / f"data_for_generation_generation_{generation_index}.jsonl"
        )

    def log_data_hypotheses(
        self, data: MathDataHypothesis, generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[MathDataHypothesis] = (
                PydanticJSONLinesWriter(
                    self.get_path_to_data_hypotheses(
                        self.logging_folder, generation_index
                    )
                )
            )
            writer(data)

    def render_prompt(
        self,
        math_task_errors: Sequence[CompletedMathTaskInstance],
        num_hypotheses: int,
        num_data_specs: int,
        skill_category: str,
    ) -> str:
        prompt = self.verbalizer_for_skill_category(
            math_task_errors=math_task_errors,
            num_data_specs=num_data_specs,
            num_hypotheses=num_hypotheses,
            skill_category=skill_category,
        )
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_hypothesis_from_llm(self, prompt: str) -> Iterable[MathDataHypothesis]:
        data_hypotheses = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[MathDataHypothesis],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_hypotheses = cast(Iterable[MathDataHypothesis], data_hypotheses)
        return data_hypotheses

    def generate_training_data_for_skill_category(
        self,
        skill_category: str,
        math_task_errors: Sequence[CompletedMathTaskInstance],
        num_hypotheses: int,
        num_data_specs: int,
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
        prompt = self.render_prompt(
            math_task_errors=math_task_errors,
            num_data_specs=num_data_specs,
            num_hypotheses=num_hypotheses,
            skill_category=skill_category,
        )
        try:
            data_hypotheses = self.get_data_hypothesis_from_llm(prompt)
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get data hypotheses for skill category {skill_category}."
            )
            return []
        else:
            for data_hypothesis in data_hypotheses:
                self.log_data_hypotheses(data_hypothesis, self.generation_index)

            training_data: list[MathTrainingDatumWithSkillCategory] = []

            data_specs = list(itertools.chain(*[_.data_specs for _ in data_hypotheses]))
            logger.info(
                f"Generated {len(data_specs)} data specifications for skill category {skill_category}."
            )

            for data_spec in data_specs:
                training_datum = MathTrainingDatumWithSkillCategory(
                    ulid=ULID(),
                    instruction=data_spec.problem,
                    # It may be better to make this configurable. The data specification contains
                    # the chain of thought and final answer separately, but for training we need to
                    # combine them into a single target that also matches the format expected
                    # by the evaluation code.
                    response=render_solution_for_scoring(
                        chain_of_thought=data_spec.chain_of_thought,
                        final_answer=data_spec.final_answer,
                    ),
                    skill_category=skill_category,
                )
                training_data.append(training_datum)

            return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CompletedMathTaskInstance]
    ) -> Sequence[MathTrainingDatum]:

        if self.generate_data_only_for_errors:
            errors = [_ for _ in completed_task_instances if not _.was_correct]
        else:
            errors = completed_task_instances
            logger.warning(
                "Generating training data for all task instances, "
                "not just the ones with errors."
            )

        logger.info(f"Found {len(errors)} errors to generate training data for.")

        errors_by_skill_category: dict[str, list[CompletedMathTaskInstance]] = (
            defaultdict(list)
        )
        for error in errors:
            skill_category = (
                self.skill_discovery_module.get_skill_category_name_for_task_instance(
                    error.task_instance
                )
            )
            errors_by_skill_category[skill_category].append(error)

        for skill_category, errors in errors_by_skill_category.items():
            logger.info(
                f"Found {len(errors)} errors for skill category {skill_category}."
            )

        training_data: Sequence[MathTrainingDatum] = []
        for skill_category in tqdm(
            errors_by_skill_category, desc="Generating training data"
        ):
            math_task_errors = errors_by_skill_category[skill_category]
            training_data_for_skill_category = (
                self.generate_training_data_for_skill_category(
                    skill_category=skill_category,
                    math_task_errors=math_task_errors,
                    num_hypotheses=self.hypotheses_per_skill_category,
                    num_data_specs=self.data_specs_per_hypothesis,
                )
            )
            training_data.extend(training_data_for_skill_category)

        logger.info(f"Generated {len(training_data)} training data instances.")

        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedMathTaskInstance],
        predictor: MathPredictorInterface,
    ) -> Sequence[MathTrainingDatum]:
        generated_training_data = self.generate_training_data(completed_task_instances)
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(MathDataGenerationAgent)(DataGenerationAgent)
