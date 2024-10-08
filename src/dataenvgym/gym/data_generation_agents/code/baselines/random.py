import os
import random
from pathlib import Path
from typing import Collection, Iterable, Optional, Sequence, cast

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
from tqdm.auto import tqdm

from dataenvgym.gym.domain_models import (
    CodeGenerationCompletedTaskInstance,
    CodeGenerationDataSpec,
    CodeGenerationPredictorInterface,
    CodeGenerationDataGenerationAgent,
    CodeGenerationTrainingDatum,
    implements,
)
from dataenvgym.gym.trainable_predictors.code.openai_predictor import (
    answer_code_generation_question,
)
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import render_data_spec
from dataenvgym.utils import PydanticJSONLinesWriter

DEFAULT_TEMPLATE = jinja2.Template(
    """You are an expert Python engineer and competitive programming tutor.
You are helping a junior engineer improve their coding skills.

You will propose a new set of problems.

Here are some guidelines:
- the problems should be similar to coding problems on platforms like LeetCode, Codeforces, etc.
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
            num_no_starter_code_problems=num_no_starter_code_problems,
            trim_blocks=True,
            lstrip_blocks=True,
        )


class CodeGenerationDataHypothesis(BaseModel):
    inferred_weak_subskill: str
    problems: list[CodingProblemProposal]


class DataGenerationAgent:
    def __init__(
        self,
        verbalizer_for_skill_category: VerbalizeSkillCategoryForDataCreation = VerbalizeSkillCategoryForDataCreation(),
        logging_folder: Optional[Path] = None,
        data_specs_per_llm_call: int = 5,
        num_training_data_per_invocation: int = 50,
    ):
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.verbalizer_for_skill_category = verbalizer_for_skill_category
        self.generation_index = 0
        self.logging_folder = logging_folder
        self.data_specs_per_llm_call = data_specs_per_llm_call
        self.num_training_data_per_invocation = num_training_data_per_invocation

        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    def log_data_specs(self, data_specs: Sequence[CodeGenerationDataSpec]) -> None:
        if self.logging_folder is None:
            return

        save_path = self.logging_folder / f"data_specs_{self.generation_index}.jsonl"
        writer = PydanticJSONLinesWriter(save_path)
        writer.write_batch(data_specs)

    def get_num_llm_calls_needed(self) -> int:
        return self.num_training_data_per_invocation // self.data_specs_per_llm_call

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_coding_problems_from_llm(
        self, prompt: str
    ) -> Iterable[CodingProblemProposal]:
        coding_problems = self.client.chat.completions.create(
            model="gpt-4o",
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

    def generate_training_data(self) -> Sequence[CodeGenerationTrainingDatum]:
        data_specs: list[CodeGenerationDataSpec] = []
        llm_calls_needed = self.get_num_llm_calls_needed()

        logger.info(
            f"Generating {self.num_training_data_per_invocation} training data specifications, requiring {llm_calls_needed} LLM calls."
        )

        for _ in tqdm(
            range(llm_calls_needed), desc="LLM calls", total=llm_calls_needed
        ):
            prompt = self.verbalizer_for_skill_category(
                completed_task_instances=[],  # We're not using this in the random baseline
                num_data_specs=self.data_specs_per_llm_call,
                num_no_starter_code_problems=random.randint(
                    0, self.data_specs_per_llm_call
                ),
            )
            try:
                coding_problems = list(self.get_coding_problems_from_llm(prompt))
                for coding_problem in coding_problems:
                    solution = self.solve_coding_problem(coding_problem)
                    data_spec = coding_problem.to_code_generation_data_spec(solution)
                    data_specs.append(data_spec)
            except RetryError:
                logger.opt(exception=True).error(
                    "A call to get data specs from the LLM failed."
                )
                continue

        logger.info(f"Generated {len(data_specs)} data specifications.")
        self.log_data_specs(data_specs)

        training_data = [render_data_spec(data_spec=spec) for spec in data_specs]
        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CodeGenerationCompletedTaskInstance],
        predictor: CodeGenerationPredictorInterface,
    ) -> Sequence[CodeGenerationTrainingDatum]:
        generated_training_data = self.generate_training_data()
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(CodeGenerationDataGenerationAgent)(DataGenerationAgent)
