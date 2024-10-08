import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Collection, Iterable, Literal, Optional, Sequence, cast

import instructor
import jinja2
from loguru import logger
from openai import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm
from ulid import ULID

from dataenvgym.gym.domain_models import (
    CompletedVqaTaskInstance,
    OpenEndedVqaTaskInstance,
    PredictorInterface,
    SerializableImage,
    VqaDataGenerationAgent,
    VqaDataHypothesis,
    VqaSkillDiscoveryInterface,
    VqaTrainingDatum,
    implements,
)
from dataenvgym.utils import PydanticJSONLinesWriter

from .open_ended import RandomImageGenerator

DEFAULT_TEMPLATE = jinja2.Template(
    """
You are a experienced machine learning engineer and your role is create training data for a model. 
We will focus on improving skills underneath the category of "{{ skill_category }}".

Here are examples of mistakes the model made when trying to answer questions requiring "{{ skill_category }}".
The model was given an instruction about an image and responded incorrectly.
The instruction required a skill under "{{ skill_category }}" to answer correctly.
{% for vqa_task_error in vqa_task_errors %}
- Instruction: {{ vqa_task_error.task_instance.instruction }}
  - Model Response: {{ vqa_task_error.predictor_response }}
  - Correct Response: {{ vqa_task_error.task_instance.ground_truth_label }}
{% endfor %}

You will propose hypotheses about what training data the model needs to improve its skills under "{{ skill_category }}".
The hypotheses will contain specifications of the training data, and we will generate the data from those specifications, and then train the model on the data.
For certain skills, the model may not have made any mistakes.
In that case, propose hypotheses that will help the model improve on harder examples of the skill.

The training data you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "inferred_weak_skill": Produce this first to give yourself a chance to think. A concise to-the-point description of a skill under "{{ skill_category }}" that the model is missing or weak at and what kind of (instruction, image, response) data will help the model learn the skill.
- "instruction": The instruction you want the model to respond to.
- "image": The description of an image the instruction is about. 
- "response": The correct response to the instruction. 

When crafting the training data, consider the following:
- the instructions should be similar in style, length, and complexity to the examples provided
- the images should be relevant to the instruction
- the responses should be similar in style, length, and complexity to the examples provided
- think about what knowledge the model might be missing that would help it answer the question correctly, and craft your training data to give it that knowledge
- each response should be a logically _correct_ response to the instruction in the context of the image description
- the training data should be diverse and help the model improve on as many skills under "{{ skill_category }}" as possible

Produce {{ num_hypotheses }} hypotheses.
For each hypothesis and weak skill, produce {{ num_data_specs }} specifications for training data.
"""
)


class Verbalizer:
    def __init__(self, template: jinja2.Template = DEFAULT_TEMPLATE):
        self.template = template

    def __call__(
        self,
        vqa_task_errors: Sequence[CompletedVqaTaskInstance],
        num_data_specs: int,
        skill_category: str,
        num_hypotheses: int,
    ) -> str:
        return self.template.render(
            vqa_task_errors=vqa_task_errors,
            num_data_specs=num_data_specs,
            skill_category=skill_category,
            num_hypotheses=num_hypotheses,
            trim_blocks=True,
            lstrip_blocks=True,
        )


class VqaTrainingDatumWithSkillCategory(VqaTrainingDatum):
    skill_category: str


class DataGenerationAgent:
    def __init__(
        self,
        skill_discovery_module: VqaSkillDiscoveryInterface,
        # TODO: Make this a protocol and move the logic to the verbalizer instances.
        # It should just take all the attributes available and generate the prompt.
        verbalizer_for_skill_category: Verbalizer = Verbalizer(),
        text_to_image_fn: Callable[[str], SerializableImage] = RandomImageGenerator(),
        logging_folder: Optional[Path] = None,
        data_specs_per_hypothesis: int = 5,
        hypotheses_per_skill_category: int = 5,
        generate_data_only_for_errors: bool = True,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
    ):
        # TODO: Pass this in as a dependency.
        # self.client = instructor.from_openai(OpenAI())
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
        self.text_to_image_fn = text_to_image_fn
        self.generation_index = 0
        self.logging_folder = logging_folder
        self.skill_discovery_module = skill_discovery_module
        self.generate_data_only_for_errors = generate_data_only_for_errors

        # Make sure the logging folder exists, create it if it doesn't.
        if self.logging_folder:
            self.logging_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_path_to_data_hypotheses(logging_folder: Path, generation_index: int):
        return (
            logging_folder / f"data_for_generation_generation_{generation_index}.jsonl"
        )

    def log_data_hypotheses(
        self, data: VqaDataHypothesis, generation_index: int
    ) -> None:
        if self.logging_folder:
            writer: PydanticJSONLinesWriter[VqaDataHypothesis] = (
                PydanticJSONLinesWriter(
                    self.get_path_to_data_hypotheses(
                        self.logging_folder, generation_index
                    )
                )
            )
            writer(data)

    def generate_image_from_text(self, text: str) -> SerializableImage:
        return self.text_to_image_fn(text)

    def render_prompt(
        self,
        vqa_task_errors: Sequence[CompletedVqaTaskInstance],
        num_hypotheses: int,
        num_data_specs: int,
        skill_category: str,
    ) -> str:
        prompt = self.verbalizer_for_skill_category(
            vqa_task_errors=vqa_task_errors,
            num_data_specs=num_data_specs,
            num_hypotheses=num_hypotheses,
            skill_category=skill_category,
        )
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_hypothesis_from_llm(self, prompt: str) -> Iterable[VqaDataHypothesis]:
        data_hypotheses = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[VqaDataHypothesis],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_hypotheses = cast(Iterable[VqaDataHypothesis], data_hypotheses)
        return data_hypotheses

    def generate_training_data_for_skill_category(
        self,
        skill_category: str,
        vqa_task_errors: Sequence[CompletedVqaTaskInstance],
        num_hypotheses: int,
        num_data_specs: int,
    ) -> Sequence[VqaTrainingDatumWithSkillCategory]:
        prompt = self.render_prompt(
            vqa_task_errors=vqa_task_errors,
            num_data_specs=num_data_specs,
            num_hypotheses=num_hypotheses,
            skill_category=skill_category,
        )
        data_hypotheses = self.get_data_hypothesis_from_llm(prompt)
        for data_hypothesis in data_hypotheses:
            self.log_data_hypotheses(data_hypothesis, self.generation_index)

        training_data: list[VqaTrainingDatumWithSkillCategory] = []

        data_specs = list(itertools.chain(*[_.data_specs for _ in data_hypotheses]))

        for data_spec in tqdm(data_specs):
            image = self.generate_image_from_text(data_spec.image_description)
            training_datum = VqaTrainingDatumWithSkillCategory(
                ulid=ULID(),
                instruction=data_spec.instruction,
                image=image,
                response=data_spec.response,
                skill_category=skill_category,
            )
            training_data.append(training_datum)

        return training_data

    def generate_training_data(
        self, completed_task_instances: Collection[CompletedVqaTaskInstance]
    ) -> Sequence[VqaTrainingDatum]:

        if self.generate_data_only_for_errors:
            errors = [_ for _ in completed_task_instances if not _.was_correct]
        else:
            # TODO: This is very confusing but is the least amount of work. We call these errors,
            # but they are not errors. They are all the task instances. We should rename this variable.
            # This also requires renaming the verbalizer's signature.
            errors = completed_task_instances
            logger.warning(
                "Generating training data for all task instances, "
                "not just the ones with errors. The logs will mention 'errors' but "
                " this is misleading."
            )

        logger.info(f"Found {len(errors)} errors to generate training data for.")

        # First we will group each error by the skill category it belongs to.
        errors_by_skill_category: dict[str, list[CompletedVqaTaskInstance]] = (
            defaultdict(list)
        )
        for error in errors:
            skill_category = (
                self.skill_discovery_module.get_skill_category_name_for_task_instance(
                    error.task_instance
                )
            )
            errors_by_skill_category[skill_category].append(error)
        # Log the # of errors by skill category.
        for skill_category, errors in errors_by_skill_category.items():
            logger.info(
                f"Found {len(errors)} errors for skill category {skill_category}."
            )

        # We are really generating VqaTrainingDatumWithSkillCategory instances, but the calling
        # code does not care about the skill category, so we type check them as the base class.
        training_data: Sequence[VqaTrainingDatum] = []
        # Next, we will generate training data for each skill category.
        for skill_category in errors_by_skill_category:
            vqa_task_errors = errors_by_skill_category[skill_category]
            training_data_for_skill_category = (
                self.generate_training_data_for_skill_category(
                    skill_category=skill_category,
                    vqa_task_errors=vqa_task_errors,
                    num_hypotheses=self.hypotheses_per_skill_category,
                    num_data_specs=self.data_specs_per_hypothesis,
                )
            )
            training_data.extend(training_data_for_skill_category)

        logger.info(f"Generated {len(training_data)} training data instances.")

        return training_data

    def __call__(
        self,
        completed_task_instances: Collection[CompletedVqaTaskInstance],
        predictor: PredictorInterface[OpenEndedVqaTaskInstance],
    ) -> Sequence[VqaTrainingDatum]:
        generated_training_data = self.generate_training_data(completed_task_instances)
        return generated_training_data

    def step(self) -> None:
        self.generation_index += 1


implements(VqaDataGenerationAgent)(DataGenerationAgent)
