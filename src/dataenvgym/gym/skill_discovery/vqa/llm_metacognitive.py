"""
This file implements skill discovery from the paper "Metacognitive Capabilities of 
LLMs: An Exploration in Mathematical Problem Solving"
"""

import instructor
from pydantic import BaseModel
from openai import AzureOpenAI
import os
import jinja2
from typing import Iterable, Collection
from tqdm.auto import tqdm
from ...domain_models import (
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    VqaSkillDiscoveryInterface,
    implements,
)
from loguru import logger
from pathlib import Path
import json
from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from dataenvgym.utils import (
    PydanticSqliteKeyValueStore,
    PydanticJSONLinesReader,
    PydanticJSONLinesWriter,
)
import re
from typing import cast


class Skill(BaseModel):
    name: str
    reason_for_skill: str


class InstructionWithSkill(BaseModel):
    instance_id: str
    instruction: str
    skill: Skill


class InstructionWithSkillCategoryName(BaseModel):
    instance_id: str
    instruction: str
    skill_category_descriptive_name: str


class SkillCategory(BaseModel):
    category_id: int
    descriptive_name: str
    skills: list[str]


SkillCategories = Iterable[SkillCategory]


QUESTION_TO_SKILL_TEMPLATE = jinja2.Template(
    """Consider this question about {{ topic }}.
The question tests the {{ topic }} skills of a {{ student_description }}.
Label this question with a specific skill that would be required to solve the question. 
You should be able to use the skill as a dictionary key in python. 
The skill name should be lower case letters only. 
The skill name should be very descriptive and you may use multiple words to describe the skills required in the question. 
If you do use multiple words per question, then join them by an underscore.

Question: {{question}}
"""
)


SKILL_TO_CATEGORIES_TEMPLATE = jinja2.Template(
    """Here is a list of skills required by a {{ student_description }} to solve questions about {{ topic }}:
{% for skill in skills %}
- {{- skill -}}
{% endfor %}

Reduce the number of unique skills by grouping similar skills into categories and give a descriptive name to each category.
When choosing categories, consider the following:
- The categories should be mutually exclusive.
- The categories should be collectively exhaustive.
- The categories should be descriptive of the skills they contain.

When designing the skill categories, keep in mind that we want to use the skill categories to guide training data collection to improve a {{ student_descriptor }}'s performance on {{ topic }} tasks.
Design the skill categories so that collecting data for each category will help improve the model's performance on the underlying skills.

{% if num_categories is not none %}
Group the skills into at least {{ num_categories }} categories.
{% endif %}
{% if no_function_calling %}
Respond with a list of lines, where each line is a category name followed by a colon and a list of representative skills for that category.
Here is a Python template demonstrating the expected format:
```python3
template = "{index}. {skill_category}: {representative_skills}
```
Here are examples:
1. Bird Identification: identifying_birds, recognizing_birds, bird_species
2. Welding: welding_steel, welding_aluminum, welding_titanium 

A skill category may encompass as many skill as you think are appropriate, but only list up to 3 representative skills.
Produce plain text, do not wrap the your response in backticks or triple backticks.
Write nothing but the lines that follow the template.
{% endif %}
"""
)

LABEL_QUESTIONS_WITH_SKILL_CATEGORIES_PROMPT = jinja2.Template(
    """Your task is to identify the skill required to solve a question that tests {{ topic }}.
Here is a list of possible skills required by the question: 
{% for skill in skills %}
- {{- skill -}}
{% endfor %}
Label the question with one skill from the list, and provide a reason for your choice.

You must ALWAYS choose a skill from the list of skills provided.

Question: {{question}}
"""
)


class VqaSkillDiscoveryResult(BaseModel):
    labeled_instructions: list[InstructionWithSkill]
    skill_categories: list[SkillCategory]
    instance_id_to_skill_category: dict[str, SkillCategory]


class VqaSkillDiscoverer:
    def __init__(
        self,
        question_to_skill_template: jinja2.Template = QUESTION_TO_SKILL_TEMPLATE,
        skill_to_categories_template: jinja2.Template = SKILL_TO_CATEGORIES_TEMPLATE,
        label_questions_with_skill_categories_prompt: jinja2.Template = LABEL_QUESTIONS_WITH_SKILL_CATEGORIES_PROMPT,
        num_skill_categories: int | None = None,
        topic: str = "image understanding",
        student_description: str = "multimodal language model",
        checkpoint_path: Path = Path("llm_skill_discovery_checkpoint"),
        resume_from_checkpoint: bool = False,
        reduce_token_usage_during_aggregation: bool = False,
        max_workers: int = 4,
    ):
        """
        Parameters
        ----------
        question_to_skill_template : jinja2.Template
            Template for asking the LLM to label a question with a skill.
        skill_to_categories_template : jinja2.Template
            Template for asking the LLM to aggregate skills into categories.
        label_questions_with_skill_categories_prompt : jinja2.Template
            Template for asking the LLM to label a question with a skill category.
        num_skill_categories : int, optional
            Number of skill categories to group the skills into. Leave as None to let the LLM decide.
        topic : str
            The topic of the questions that the LLM will be asked to label.
        student_description : str
            A description of the student that the LLM will be asked to label questions for.
        """
        self.reduce_token_usage_during_aggregation = (
            reduce_token_usage_during_aggregation
        )
        self.question_to_skill_template = question_to_skill_template
        self.skill_to_categories_template = skill_to_categories_template
        self.label_questions_with_skill_categories_prompt = (
            label_questions_with_skill_categories_prompt
        )
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY_GPT4O_MINI"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT4O_MINI"],
                api_version="2023-03-15-preview",
            )
        )
        self.num_skill_categories = num_skill_categories

        self.topic = topic
        self.student_description = student_description

        self.discovery_result: VqaSkillDiscoveryResult | None = None

        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Store each instruction labeled with the fine-grained skill that was assigned
        # to it by the LLM.
        self.label_task_instances_with_skill_checkpoint = (
            self.checkpoint_path / "labeled_instructions.sqlite"
        )
        # Store the aggregated skill categories determined by clustering the fine-grained skills.
        self.aggregate_skills_into_categories_checkpoint = (
            self.checkpoint_path / "aggregate_skills_into_categories.jsonl"
        )

        # Store the mapping from instance ids to skill categories.
        self.map_instance_ids_to_skill_categories_checkpoint = (
            self.checkpoint_path / "map_instance_ids_to_skill_categories.sqlite"
        )

        self.resume_from_checkpoint = resume_from_checkpoint
        self.max_workers = max_workers

    def set_precomputed_skills(self, result_path: Path | str) -> None:
        with open(result_path, "r") as f:
            result = VqaSkillDiscoveryResult.model_validate(json.load(f))

        self.discovery_result = result

    def save_precomputed_skills(self, result_path: Path | str) -> None:
        if self.discovery_result is None:
            raise ValueError("No discovery result to save.")

        if not isinstance(result_path, Path):
            result_path = Path(result_path)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(self.discovery_result.model_dump(), f)

    def _label_single_task_instance(
        self,
        task_instance: OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance,
    ) -> InstructionWithSkill:
        """Helper method to label a single task instance with a skill."""

        question = task_instance.instruction
        prompt = self.question_to_skill_template.render(
            question=question,
            topic=self.topic,
            student_description=self.student_description,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        skill: Skill = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_model=Skill,  # type: ignore
        )
        instruction_with_skill = InstructionWithSkill(
            instance_id=task_instance.instance_id,
            instruction=task_instance.instruction,
            skill=skill,
        )
        return instruction_with_skill

    def label_task_instances_with_skill(
        self,
        task_instances: (
            Collection[OpenEndedVqaTaskInstance]
            | Collection[MultipleChoiceVqaTaskInstance]
        ),
    ) -> list[InstructionWithSkill]:
        cache = PydanticSqliteKeyValueStore(
            InstructionWithSkill,
            f"sqlite:///{self.label_task_instances_with_skill_checkpoint}",
        )

        labeled_instructions: list[InstructionWithSkill] = []

        if self.resume_from_checkpoint:
            for task_instance in task_instances:
                if task_instance.instance_id in cache:
                    labeled_instructions.append(cache[task_instance.instance_id])
            logger.info(
                "Resumed from checkpoint, found {} labeled instructions in cache",
                len(labeled_instructions),
            )

        task_instances_needing_labels = [
            task_instance
            for task_instance in task_instances
            if task_instance.instance_id not in cache
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._label_single_task_instance, task): task
                for task in task_instances_needing_labels
            }

            for future in tqdm(
                as_completed(future_to_task), total=len(task_instances_needing_labels)
            ):
                try:
                    labeled_instruction = future.result()
                    labeled_instructions.append(labeled_instruction)
                    cache[labeled_instruction.instance_id] = labeled_instruction
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Task {task.instance_id} generated an exception: {e}")
                    raise e

        return labeled_instructions

    def aggregate_skills_into_categories(
        self, skills: list[str], num_categories: int | None = None
    ) -> SkillCategories:
        # This is an all-or-nothing operation, so we will checkpoint the entire
        # aggregation step and the results. There is no partial aggregation we can
        # recover from here. If a checkpoint exists, we will load it and return.
        if (
            self.resume_from_checkpoint
            and self.aggregate_skills_into_categories_checkpoint.exists()
        ):
            logger.info("Resuming from checkpoint")
            skill_categories: SkillCategories = list(
                PydanticJSONLinesReader[SkillCategory](
                    self.aggregate_skills_into_categories_checkpoint,
                    SkillCategory,
                )()
            )
            logger.info(
                "Loaded {} skill categories from checkpoint", len(skill_categories)
            )
            return skill_categories

        else:
            # Clear the checkpoint file so that we don't accidentally
            # resume from a previous run and we checkpoint _only_ the new
            # data in case the checkpoint file is not empty.
            if self.aggregate_skills_into_categories_checkpoint.exists():
                self.aggregate_skills_into_categories_checkpoint.unlink(missing_ok=True)

        checkpoint_writer = PydanticJSONLinesWriter[SkillCategory](
            self.aggregate_skills_into_categories_checkpoint
        )

        if self.reduce_token_usage_during_aggregation:
            prompt = self.skill_to_categories_template.render(
                skills=set(skills),
                topic=self.topic,
                student_description=self.student_description,
                num_categories=num_categories,
                no_function_calling=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = response.choices[0].message.content
            assert response_text, "No response text from LLM"
            skill_categories = []
            for line in response_text.split("\n"):
                if not line.strip():
                    continue
                # Match id. Descriptive name: skill1, skill2, skill3
                # Don't be sensitive to whitespace at the beginning or end of the line.
                # Don't be sensitive to whitespace around the colon or commas.
                match = re.match(r"\s*(\d+)\.\s*(.+)\s*:\s*(.+)\s*", line)
                assert match, f"Could not parse line: {line}"
                category_id, descriptive_name, skills_for_category = match.groups()
                skills_for_category = [
                    _.strip() for _ in skills_for_category.split(",")
                ]
                skill_categories.append(
                    SkillCategory(
                        category_id=int(category_id),
                        descriptive_name=descriptive_name,
                        skills=skills_for_category,
                    )
                )
            checkpoint_writer.write_batch(skill_categories)
            return skill_categories

        else:
            prompt = self.skill_to_categories_template.render(
                skills=skills,
                topic=self.topic,
                student_description=self.student_description,
                num_categories=num_categories,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            skill_categories: SkillCategories = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_model=SkillCategories,  # type: ignore
            )
            skill_categories = cast(list[SkillCategory], skill_categories)
            checkpoint_writer.write_batch(skill_categories)
            return skill_categories

    def _map_single_instruction_to_category(
        self,
        instruction: InstructionWithSkill,
        skill_categories: list[SkillCategory],
    ) -> tuple[str, SkillCategory]:
        """Helper method to map a single instruction to a skill category."""
        skill = instruction.skill.name
        for category in skill_categories:
            if skill in category.skills:
                return instruction.instance_id, category

        # This skill was not assigned to any category
        with logging_redirect_tqdm():
            logger.warning(
                "Skill {} was not assigned to any category, asking the LLM to assign it to a category.",
                skill,
            )
        category_name = (
            self.ask_llm_skill_category_of_instruction_without_discovery_result(
                instruction.instruction, list(skill_categories)
            )
        )
        category = next(
            category
            for category in skill_categories
            if category.descriptive_name == category_name
        )
        return instruction.instance_id, category

    def map_instance_ids_to_skill_categories(
        self,
        labeled_instructions: list[InstructionWithSkill],
        skill_categories: SkillCategories,
    ) -> dict[str, SkillCategory]:
        instance_id_to_skill_category: dict[str, SkillCategory] = {}

        cache = PydanticSqliteKeyValueStore(
            InstructionWithSkillCategoryName,
            f"sqlite:///{self.map_instance_ids_to_skill_categories_checkpoint}",
        )

        if self.resume_from_checkpoint:
            category_name_to_skill_category: dict[str, SkillCategory] = {
                category.descriptive_name: category for category in skill_categories
            }
            instructions_assigned_skill_categories = list(cache.values())
            logger.info(
                "Loaded {} instructions from checkpoint",
                len(instructions_assigned_skill_categories),
            )
            for instruction in instructions_assigned_skill_categories:
                try:
                    instance_id_to_skill_category[instruction.instance_id] = (
                        category_name_to_skill_category[
                            instruction.skill_category_descriptive_name
                        ]
                    )
                except KeyError:
                    # This happens in the case when there is a mismatch between
                    # the checkpoint and the current skill categories. If this happens,
                    # you have to delete one of the checkpoint files and re-run the skill
                    # discovery.
                    raise KeyError(
                        f"Could not find skill category {instruction.skill_category_descriptive_name} in the current skill categories."
                        " This happens in the case when there is a mismatch between"
                        " the checkpoint and the current skill categories. If this happens,"
                        " you have to delete one of the checkpoint files and re-run the skill"
                        " discovery. A instruction was labeled with a skill category that we are not aware of."
                    )

        skill_categories_list = list(skill_categories)

        instructions_needing_mapping = [
            instruction
            for instruction in labeled_instructions
            if instruction.instance_id not in cache
        ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            map_fn = partial(
                self._map_single_instruction_to_category,
                skill_categories=skill_categories_list,
            )
            future_to_instruction = {
                executor.submit(map_fn, instruction): instruction
                for instruction in instructions_needing_mapping
            }

            for future in tqdm(
                as_completed(future_to_instruction),
                total=len(instructions_needing_mapping),
            ):
                instruction = future_to_instruction[future]
                try:
                    instance_id, category = future.result()
                    instance_id_to_skill_category[instance_id] = category
                    cache[instruction.instance_id] = InstructionWithSkillCategoryName(
                        instance_id=instruction.instance_id,
                        instruction=instruction.instruction,
                        skill_category_descriptive_name=category.descriptive_name,
                    )
                    print(len(cache), instruction.instance_id)
                except Exception as e:
                    logger.error(
                        f"Instruction {instruction.instance_id} generated an exception: {e}"
                    )
                    raise e

        return instance_id_to_skill_category

    def discover_skills(
        self, task_instances: Collection[OpenEndedVqaTaskInstance]
    ) -> None:
        labeled_instructions = self.label_task_instances_with_skill(task_instances)
        skills = [instruction.skill.name for instruction in labeled_instructions]
        skill_categories = self.aggregate_skills_into_categories(
            skills, num_categories=self.num_skill_categories
        )

        labeled_instructions = labeled_instructions
        skill_categories = skill_categories
        instance_id_to_skill_category = self.map_instance_ids_to_skill_categories(
            labeled_instructions, skill_categories
        )

        self.discovery_result = VqaSkillDiscoveryResult(
            labeled_instructions=labeled_instructions,
            skill_categories=list(skill_categories),
            instance_id_to_skill_category=instance_id_to_skill_category,
        )

    def recompute_skill_discovery_with_new_num_categories(
        self, num_categories: int | None
    ) -> None:
        assert (
            self.discovery_result is not None
        ), "Skill discovery has not been run yet."

        # Get a list of all skills we have come up with so far.
        skills = [
            instruction.skill.name
            for instruction in self.discovery_result.labeled_instructions
        ]

        # Aggregate those skills into num_categories categories.
        skill_categories = self.aggregate_skills_into_categories(
            skills, num_categories=num_categories
        )

        logger.info(
            "Was asked for {} categories, produced {} categories",
            num_categories,
            len(list(skill_categories)),
        )

        # Map each task instance to a skill category.
        instance_id_to_skill_category = self.map_instance_ids_to_skill_categories(
            self.discovery_result.labeled_instructions, skill_categories
        )

        self.discovery_result = VqaSkillDiscoveryResult(
            labeled_instructions=self.discovery_result.labeled_instructions,
            skill_categories=list(skill_categories),
            instance_id_to_skill_category=instance_id_to_skill_category,
        )

    def ask_llm_skill_category_of_instruction_without_discovery_result(
        self, instruction: str, skill_categories: list[SkillCategory]
    ) -> str:
        prompt = self.label_questions_with_skill_categories_prompt.render(
            question=instruction,
            topic=self.topic,
            skills=[category.descriptive_name for category in skill_categories],
            trim_blocks=True,
            lstrip_blocks=True,
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        llm_inferred_skill = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,  # type: ignore
            response_model=Skill,
        )

        # If the skill is not in the list of categories, ask the LLM to try again.
        for _ in range(3):
            if llm_inferred_skill.name in [
                category.descriptive_name for category in skill_categories
            ]:
                return llm_inferred_skill.name

            logger.warning(
                "LLM choose a skill {} that is not in the list of categories {}, asking again.",
                llm_inferred_skill.name,
                [category.descriptive_name for category in skill_categories],
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Do not choose a skill that is not in the list of categories, like {}. Try again.".format(
                        llm_inferred_skill.name
                    ),
                },
            )
            llm_inferred_skill = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,  # type: ignore
                response_model=Skill,
            )

        raise ValueError(
            f"LLM did not choose a skill from the list of categories after 3 attempts. Chose {llm_inferred_skill.name}"
        )

    def ask_llm_skill_category_of_instruction(self, instruction: str) -> str:
        assert (
            self.discovery_result is not None
        ), "Skill discovery has not been run yet."

        return self.ask_llm_skill_category_of_instruction_without_discovery_result(
            instruction, self.discovery_result.skill_categories
        )

    def get_skill_category_name_for_task_instance(
        self, task_instance: OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
    ) -> str:
        assert (
            self.discovery_result is not None
        ), "Skill discovery has not been run yet."
        try:
            return self.discovery_result.instance_id_to_skill_category[
                task_instance.instance_id
            ].descriptive_name
        except KeyError:
            logger.warning(
                "Task instance {} not found in skill discovery result, asking the LLM to assign it to a category.",
                task_instance.instance_id,
            )
            return self.ask_llm_skill_category_of_instruction(task_instance.instruction)


implements(VqaSkillDiscoveryInterface)(VqaSkillDiscoverer)
