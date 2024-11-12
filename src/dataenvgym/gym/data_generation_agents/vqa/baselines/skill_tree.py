import os
import random
from typing import Callable, Iterable, Literal, Optional, Sequence, cast

import instructor
import jinja2
from loguru import logger
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from ulid import ULID
import openai

from dataenvgym.gym.data_generation_agents.skill_tree import (
    QualityCheckerInterface,
    Subskill,
    SubskillDataGenerationEngineInterface,
)
from dataenvgym.gym.domain_models import (
    OpenEndedVqaTaskInstance,
    PredictorInterface,
    SerializableImage,
    VqaDataSpec,
    VqaTrainingDataQualityCheck,
    VqaTrainingDatum,
    implements,
)
from dataenvgym.gym.tasks.vqa.gqa import GqaRecord, load_gqa_split

from .open_ended import RandomImageGenerator

PROPOSE_SUBSKILL_TEMPLATE = jinja2.Template(
    """
    You are an expert educator and your task is to propose new subskills for improving a student's skills in solving problems that require the skill of "{{ skill_category }}".

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
    """
)

GENERATE_DATA_FOR_SUBSKILL_TEMPLATE = jinja2.Template(
    """You are a experienced machine learning engineer and your role is create training data for a model. 

Here are some examples of the style of question the model will be answering, and the correct response to the question:
{% for example in examples %}
- instruction: {{ example.instruction }}
- response: {{ example.ground_truth_label }}
{% endfor %}

We will focus on improving skills underneath the category of "{{ subskill }}".

You will propose hypotheses about what training data the model needs to improve its skills under "{{ subskill }}".
The hypotheses will contain specifications of the training data, and we will generate the data from those specifications, and then train the model on the data.

The training data you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "instruction": The instruction you want the model to respond to.
- "image": The description of an image the instruction is about. 
- "response": The correct response to the instruction. 

When crafting the training data, consider the following:
- the instructions should be similar in style, length, and complexity to the examples provided
- the images should be relevant to the instruction
- the responses should be similar in style, length, and complexity to the examples provided
- think about what knowledge the model might be missing that would help it answer the question correctly, and craft your training data to give it that knowledge
- each response should be a logically _correct_ response to the instruction in the context of the image description
- the training data should be diverse and help the model improve on "{{ subskill }}"

Produce {{ num_data_specs }} specifications for training data.
""",
    undefined=jinja2.StrictUndefined,
)


class VqaDataSpecWithIndex(VqaDataSpec):
    index: int


class VqaTrainingDatumWithSkillCategory(VqaTrainingDatum):
    skill_category: str


class GqaExample(BaseModel):
    instruction: str
    ground_truth_label: str


class DataGenerationAgent:
    def __init__(
        self,
        model: Literal[
            "gpt-4o", "gpt-4o-mini", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        ] = "gpt-4o",
        template: jinja2.Template = GENERATE_DATA_FOR_SUBSKILL_TEMPLATE,
        text_to_image_fn: Callable[[str], SerializableImage] = RandomImageGenerator(),
        num_examples: int = 10,
    ):
        self.text_to_image_fn = text_to_image_fn
        self.model = model
        self.num_examples = num_examples
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
        elif self.model == "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo":
            client = openai.OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ["TOGETHER_API_KEY"],
            )
            self.client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
        else:
            raise ValueError(f"Unknown model: {self.model}")
        self.template = template
        self.model = model
        # Just a way to ensure we get one example per question type.
        balanced_sample: dict[str, GqaRecord] = {}
        for record in load_gqa_split("val"):
            balanced_sample[record["question_type"]] = record
        self.gqa_examples = [
            GqaExample(
                instruction=record["question"],
                ground_truth_label=record["answer"],
            )
            for record in balanced_sample.values()
        ]

    def render_prompt(
        self,
        subskill: Subskill,
        data_budget: int,
        already_generated_data: Optional[Sequence[VqaDataSpec]] = None,
    ) -> str:
        return self.template.render(
            subskill=subskill,
            num_data_specs=data_budget,
            already_generated_data=already_generated_data,
            trim_blocks=True,
            lstrip_blocks=True,
            examples=random.sample(self.gqa_examples, self.num_examples),
        )

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_specs_from_llm(self, prompt: str) -> Iterable[VqaDataSpecWithIndex]:
        data_specs = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[VqaDataSpecWithIndex],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_specs = cast(Iterable[VqaDataSpecWithIndex], data_specs)
        return data_specs

    def render_data_spec_to_training_datum(
        self, data_spec: VqaDataSpecWithIndex, subskill: Subskill
    ) -> VqaTrainingDatumWithSkillCategory:
        return VqaTrainingDatumWithSkillCategory(
            ulid=ULID(),
            instruction=data_spec.instruction,
            image=self.text_to_image_fn(data_spec.image_description),
            response=data_spec.response,
            skill_category=str(subskill),
        )

    def __call__(
        self, subskill: Subskill, data_budget: int
    ) -> Sequence[VqaTrainingDatumWithSkillCategory]:
        if data_budget == 0:
            return []
        prompt = self.render_prompt(subskill, data_budget)
        try:
            data_specs = list(self.get_data_specs_from_llm(prompt))
            if len(data_specs) > data_budget:
                logger.warning(
                    f"LLM proposed {len(data_specs)} data specs for subskill {subskill}, but we only need {data_budget}."
                )
                data_specs = data_specs[:data_budget]
            elif len(data_specs) < data_budget:
                logger.warning(
                    f"LLM proposed {len(data_specs)} data specs for subskill {subskill}, but we need {data_budget}."
                )
                num_missing_specs = data_budget - len(data_specs)
                retry_prompt = self.render_prompt(
                    subskill, num_missing_specs, already_generated_data=data_specs
                )
                data_specs.extend(self.get_data_specs_from_llm(retry_prompt))
                data_specs = data_specs[:data_budget]
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get data specs for subskill {subskill}."
            )
            return []

        training_data = [
            self.render_data_spec_to_training_datum(data_spec, subskill)
            for data_spec in tqdm(
                data_specs, desc="Rendering data specs to training data"
            )
        ]
        return training_data


implements(SubskillDataGenerationEngineInterface)(DataGenerationAgent)


class StubVqaQualityChecker:
    def __call__(
        self,
        training_data: Sequence[VqaTrainingDatum],
        predictor: PredictorInterface[OpenEndedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDataQualityCheck]:
        quality_checks = []
        for training_datum in training_data:
            quality_check = VqaTrainingDataQualityCheck(
                ulid=ULID(),
                training_datum_ulid=training_datum.ulid,
                qa_passed=True,  # Stub implementation always passes QA
                student_accuracy=None,  # Stub implementation doesn't calculate accuracy
            )
            quality_checks.append(quality_check)
        return quality_checks


implements(
    QualityCheckerInterface[
        VqaTrainingDatum, VqaTrainingDataQualityCheck, OpenEndedVqaTaskInstance
    ]
)(StubVqaQualityChecker)
