import os
import random
from pathlib import Path
from typing import Collection, Iterable, Optional, Sequence, cast, Literal

import instructor
import jinja2
from loguru import logger
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from dataenvgym.gym.domain_models import (
    CodeGenerationCompletedTaskInstance,
    CodeGenerationDataSpec,
    CodeGenerationPredictorInterface,
    CodeGenerationSkillDiscoveryInterface,
    CodeGenerationDataGenerationAgent,
    CodeGenerationTrainingDatum,
    implements,
)
from dataenvgym.gym.trainable_predictors.code.openai_predictor import (
    answer_code_generation_question,
)
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import render_data_spec
from dataenvgym.utils import PydanticJSONLinesWriter
from tqdm.auto import tqdm

DEFAULT_TEMPLATE = jinja2.Template(
    """You are an expert Python engineer and competitive programming tutor.
You are helping a junior engineer improve their coding skills.
You are focusing on problems requiring skills in the category of "{{ skill_category }}".

{% if completed_task_instances %}
# Junior Engineer's Coding Exam Results
Here are examples of code the junior engineer wrote while solving problems in this category.
{% for completed_task_instance in completed_task_instances %}
Problem: 
{{ completed_task_instance.task_instance.instruction }}
{% if completed_task_instance.task_instance.starter_code %}
Starter Code: 
{{ completed_task_instance.task_instance.starter_code }}
{% endif %}
Junior Engineer's Code:
{{ completed_task_instance.predictor_response }}
Passed: {{ completed_task_instance.was_correct }}
{% if completed_task_instance.task_instance.solution %}
The correct code is:
{{ completed_task_instance.task_instance.solution }}
{% endif %}
{% endfor %}
{% endif %}

You will propose a new set of problems that require applying skills in the category of "{{ skill_category }}".
The problems you propose should be such that solving them will help the junior engineer improve their skills in the category of "{{ skill_category }}".

Here are some guidelines:
- the problems should be similar to coding problems on platforms like LeetCode, Codeforces, etc.
- the problems should require applying skills in "{{ skill_category }}"
- only propose problems that YOU KNOW the solution to. This is CRITICAL.

# Output Format
For each problem, you need to include the following:
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


class CodingProblemProposal(BaseModel):
    index: int
    instruction: str
    starter_code: str | None = None

    def to_code_generation_data_spec(self, solution: str) -> CodeGenerationDataSpec:
        return CodeGenerationDataSpec(
            instruction=self.instruction,
            solution=solution,
            starter_code=self.starter_code,
        )


class CodeGenerationDataSpecWithSkillCategory(CodeGenerationDataSpec):
    skill_category: str


class CodeGenerationTrainingDatumWithSkillCategory(CodeGenerationTrainingDatum):
    skill_category: str


class VerbalizeSkillCategoryForDataCreation:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        completed_task_instances: Sequence[CodeGenerationCompletedTaskInstance],
        num_data_specs: int,
        skill_category: str,
        num_no_starter_code_problems: int,
    ) -> str:
        """
        Parameters:
        -----------
        - completed_task_instances: A list of CodeGenerationCompletedTaskInstance instances.
        - num_data_specs: The number of data specifications to propose.
        - skill_category: The skill category to propose problems for.
        - num_no_starter_code_problems: The number of problems that require reading from stdin/stdout
            rather than using starter code (filling in a function body, etc).

        Returns:
        --------
        A string containing the prompt for the LLM.
        """
        return self.template.render(
            completed_task_instances=completed_task_instances,
            num_data_specs=num_data_specs,
            skill_category=skill_category,
            num_no_starter_code_problems=num_no_starter_code_problems,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.StrictUndefined,
        )


class CodeGenerationDataHypothesis(BaseModel):
    inferred_weak_subskill: str
    problems: list[CodingProblemProposal]


class DataGenerationAgent:
    def __init__(
        self,
        skill_discovery_module: CodeGenerationSkillDiscoveryInterface,
        verbalizer_for_skill_category: VerbalizeSkillCategoryForDataCreation = VerbalizeSkillCategoryForDataCreation(),
        logging_folder: Optional[Path] = None,
        data_specs_per_skill_category: int = 5,
        generate_data_only_for_errors: bool = True,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
    ):
        """
        Parameters:
        -----------
        skill_discovery_module: Assigns each task instance to a skill category.
        """
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
        self.data_specs_per_skill_category = data_specs_per_skill_category
        self.generation_index = 0
        self.logging_folder = logging_folder
        self.skill_discovery_module = skill_discovery_module
        self.generate_data_only_for_errors = generate_data_only_for_errors

        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    def log_data_specs(
        self, data_specs: Sequence[CodeGenerationDataSpecWithSkillCategory]
    ) -> None:
        if self.logging_folder is None:
            return

        save_path = self.logging_folder / f"data_specs_{self.generation_index}.jsonl"
        writer = PydanticJSONLinesWriter(save_path)
        writer.write_batch(data_specs)

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_coding_problems_from_llm(
        self, prompt: str
    ) -> Iterable[CodingProblemProposal]:
        coding_problems = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[CodingProblemProposal],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        coding_problems = cast(Iterable[CodingProblemProposal], coding_problems)
        return coding_problems

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def solve_coding_problem(self, coding_problem: CodingProblemProposal) -> str:
        return answer_code_generation_question(
            client=self.client,
            question=coding_problem.instruction,
            starter_code=coding_problem.starter_code,
        )

    def generate_training_data_for_skill_category(
        self,
        skill_category: str,
        completions_for_skill_category: Sequence[CodeGenerationCompletedTaskInstance],
    ) -> Sequence[CodeGenerationTrainingDatumWithSkillCategory]:
        logger.info(f"Generating data for skill category {skill_category}")
        prompt = self.verbalizer_for_skill_category(
            completed_task_instances=completions_for_skill_category,
            num_data_specs=self.data_specs_per_skill_category,
            num_no_starter_code_problems=random.randint(
                0, self.data_specs_per_skill_category
            ),
            skill_category=skill_category,
        )
        try:
            data_specs: list[CodeGenerationDataSpecWithSkillCategory] = []
            coding_problems = list(self.get_coding_problems_from_llm(prompt))
            logger.info(f"Obtained {len(coding_problems)} coding problems")
            for coding_problem in tqdm(coding_problems, desc="Solving Coding Problems"):
                solution = self.solve_coding_problem(coding_problem)
                data_spec = coding_problem.to_code_generation_data_spec(solution)
                data_specs.append(
                    CodeGenerationDataSpecWithSkillCategory(
                        **data_spec.model_dump(), skill_category=skill_category
                    )
                )
            self.log_data_specs(data_specs)
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get data hypotheses for skill category {skill_category}."
            )
            return []
        else:
            training_data: list[CodeGenerationTrainingDatumWithSkillCategory] = []
            for data_spec in data_specs:
                training_datum = render_data_spec(data_spec=data_spec)
                training_data.append(
                    CodeGenerationTrainingDatumWithSkillCategory(
                        **training_datum.model_dump(), skill_category=skill_category
                    )
                )

            return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> Sequence[CodeGenerationTrainingDatum]:

        skill_categories = self.skill_discovery_module.get_skill_categories()

        skill_category_to_completed_task_instances: dict[
            str, list[CodeGenerationCompletedTaskInstance]
        ] = {skill_category: [] for skill_category in skill_categories}

        for completion in completed_task_instances:
            skill_category = (
                self.skill_discovery_module.get_skill_category_name_for_task_instance(
                    completion.task_instance
                )
            )
            skill_category_to_completed_task_instances[skill_category].append(
                completion
            )

        training_data: list[CodeGenerationTrainingDatumWithSkillCategory] = []
        for (
            skill_category,
            completions_in_skill_category,
        ) in skill_category_to_completed_task_instances.items():
            training_data_for_skill_category = (
                self.generate_training_data_for_skill_category(
                    skill_category=skill_category,
                    completions_for_skill_category=completions_in_skill_category,
                )
            )
            training_data.extend(training_data_for_skill_category)

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
