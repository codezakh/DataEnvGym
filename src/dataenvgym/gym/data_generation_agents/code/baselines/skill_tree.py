from dataenvgym.gym.data_generation_agents.skill_tree import (
    SkillTree,
    SubskillDataGenerationEngineInterface,
    SubskillProposalPolicy,
    Skill,
    Subskill,
    QualityCheckerInterface,
)
from pathlib import Path
from dataenvgym.gym.domain_models import (
    implements,
    CodeGenerationTaskInstance,
    CodeGenerationTrainingDataQualityCheck,
    CodeGenerationDataSpec,
    PredictorInterface,
    CodeGenerationTrainingDatum,
)
import jinja2
from typing import Literal, Optional, Sequence, Iterable, cast
from openai import AzureOpenAI
import instructor
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from dataenvgym.gym.domain_models import VqaDataSpec, VqaTrainingDatum
import os
from loguru import logger
from typing import Callable
from ulid import ULID
from tqdm import tqdm
from loguru import logger
from pydantic import BaseModel
import random
from dataenvgym.gym.data_generation_agents.skill_tree import QualityCheckerInterface
from .skill_list import (
    CodingProblemProposal,
    CodeGenerationDataSpecWithSkillCategory,
    CodeGenerationTrainingDatumWithSkillCategory,
)
from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchTask
from dataenvgym.gym.trainable_predictors.code.openai_predictor import (
    answer_code_generation_question,
)
from dataenvgym.utils import PydanticJSONLinesWriter
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import render_data_spec
from typing import Literal

PROPOSE_SUBSKILL_TEMPLATE = jinja2.Template(
    """
    You are an expert Python programmer and your task is to propose new subskills for improving a student's skills in solving problems that require the skill of "{{ skill_category }}".

    {% if subskills %}
    Here are the existing subskills under the category "{{ skill_category }}":
    {% for subskill in subskills %}
    - {{ subskill }}
    {% endfor %}
    {% endif %}

    {% if subskills %}
    Propose {{ num_new_subskills }} new subskills that are not already present in the list above. The new subskills should help the model improve its performance in the "{{ skill_category }}" category.
    {% else %}
    Propose {{ num_new_subskills }} new subskills. The new subskills should help the model improve its performance in the "{{ skill_category }}" category.
    {% endif %}

    Produce no more than {{ num_new_subskills }} subskills.
    Ensure each of the new subskills is unique and belongs to the category "{{ skill_category }}".
    """,
    undefined=jinja2.StrictUndefined,
)

GENERATE_DATA_FOR_SUBSKILL_TEMPLATE = jinja2.Template(
    """You are an expert Python engineer and competitive programming tutor.
You are helping a junior engineer improve their coding skills.

{% if lcb_examples %}
Here are representative examples of the kind of coding problems the junior engineer is facing.
{% for example in lcb_examples %}
Problem: 
{{ example.instruction }}
{% if example.starter_code %}
Starter Code: 
{{ example.starter_code }}
{% endif %}
{% if example.solution %}
Solution:
{{ example.solution }}
{% endif %}
{% endfor %}
{% endif %}

You are focusing on problems requiring skills in the category of "{{ subskill }}".

You will propose a new set of problems that require applying skills in the category of "{{ subskill }}".
The problems you propose should be such that solving them will help the junior engineer improve their skills in the category of "{{ subskill }}".

Here are some guidelines:
- the problems should be similar to coding problems on platforms like LeetCode, Codeforces, etc.
- the problems should require applying skills in "{{ subskill }}"
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
""",
    undefined=jinja2.StrictUndefined,
)


class LiveCodeBenchExample(BaseModel):
    instruction: str
    starter_code: str | None = None
    solution: str | None = None


class VerbalizeSubskillForDataCreation:
    def __init__(self, template: jinja2.Template = GENERATE_DATA_FOR_SUBSKILL_TEMPLATE):
        self.template = template

    def __call__(
        self,
        lcb_examples: Sequence[LiveCodeBenchExample],
        num_data_specs: int,
        subskill: Subskill,
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
            lcb_examples=lcb_examples,
            num_data_specs=num_data_specs,
            subskill=subskill,
            num_no_starter_code_problems=num_no_starter_code_problems,
            trim_blocks=True,
            lstrip_blocks=True,
        )


class OpenAiSubskillDataGenerationPolicy:
    def __init__(
        self,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
        template: jinja2.Template = GENERATE_DATA_FOR_SUBSKILL_TEMPLATE,
        num_examples: int = 3,
        verbalizer_for_subskill: VerbalizeSubskillForDataCreation = VerbalizeSubskillForDataCreation(),
        logging_folder: Optional[Path] = None,
    ):
        self.model = model
        self.num_examples = num_examples
        self.verbalizer_for_subskill = verbalizer_for_subskill
        self.logging_folder = logging_folder
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
        self.template = template
        self.model = model
        lcb_task_instances = LiveCodeBenchTask("val")
        self.lcb_examples = [
            LiveCodeBenchExample(
                instruction=task_instance.instruction,
                starter_code=task_instance.starter_code,
                solution=task_instance.solution,
            )
            for task_instance in lcb_task_instances.task_instances
        ]

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_coding_problems_from_llm(
        self, prompt: str
    ) -> Iterable[CodingProblemProposal]:
        data_specs = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[CodingProblemProposal],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_specs = cast(Iterable[CodingProblemProposal], data_specs)
        return data_specs

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

    def log_data_specs(
        self, data_specs: Sequence[CodeGenerationDataSpecWithSkillCategory]
    ):
        if self.logging_folder is None:
            return
        writer = PydanticJSONLinesWriter(
            self.logging_folder / "data_specs.jsonl",
        )
        writer.write_batch(data_specs)

    def generate_training_data_for_subskill(
        self,
        subskill: Subskill,
        data_budget: int,
    ) -> Sequence[CodeGenerationTrainingDatumWithSkillCategory]:
        logger.info(f"Generating data for skill category {subskill}")
        prompt = self.verbalizer_for_subskill(
            lcb_examples=random.sample(self.lcb_examples, self.num_examples),
            num_data_specs=data_budget,
            num_no_starter_code_problems=random.randint(0, data_budget),
            subskill=subskill,
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
                        **data_spec.model_dump(), skill_category=subskill
                    )
                )
            self.log_data_specs(data_specs)
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get data hypotheses for skill category {subskill}."
            )
            return []
        else:
            training_data: list[CodeGenerationTrainingDatumWithSkillCategory] = []
            for data_spec in data_specs:
                training_datum = render_data_spec(data_spec=data_spec)
                training_data.append(
                    CodeGenerationTrainingDatumWithSkillCategory(
                        **training_datum.model_dump(), skill_category=subskill
                    )
                )

            return training_data

    def __call__(
        self, subskill: Subskill, data_budget: int
    ) -> Sequence[CodeGenerationTrainingDatumWithSkillCategory]:
        if data_budget == 0:
            return []
        try:
            training_data = list(
                self.generate_training_data_for_subskill(subskill, data_budget)
            )
            if len(training_data) > data_budget:
                logger.warning(
                    f"LLM proposed {len(training_data)} data specs for subskill {subskill}, but we only need {data_budget}."
                )
                training_data = training_data[:data_budget]
            elif len(training_data) < data_budget:
                logger.warning(
                    f"LLM proposed {len(training_data)} data specs for subskill {subskill}, but we need {data_budget}."
                )
                num_missing_specs = data_budget - len(training_data)
                training_data.extend(
                    self.generate_training_data_for_subskill(
                        subskill, num_missing_specs
                    )
                )
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get data specs for subskill {subskill}."
            )
            return []

        return training_data


implements(SubskillDataGenerationEngineInterface)(OpenAiSubskillDataGenerationPolicy)


class StubCodeGenerationQualityChecker:
    def __call__(
        self,
        training_data: Sequence[CodeGenerationTrainingDatum],
        predictor: PredictorInterface[CodeGenerationTaskInstance],
    ) -> Sequence[CodeGenerationTrainingDataQualityCheck]:
        quality_checks = []
        for training_datum in training_data:
            quality_check = CodeGenerationTrainingDataQualityCheck(
                ulid=ULID(),
                training_datum_ulid=training_datum.ulid,
                student_accuracy=1.0,
            )
            quality_checks.append(quality_check)
        return quality_checks
