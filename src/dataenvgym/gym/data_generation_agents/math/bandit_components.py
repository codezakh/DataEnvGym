from dataenvgym.gym.domain_models import implements, MathDataSpec
from .baselines.skill_list import MathTrainingDatumWithSkillCategory
from .bandit_data_strategy import (
    SkillTree,
    PartialSkillTree,
    Subskill,
    Skill,
    SkillExperience,
    Action,
    Explore,
    Exploit,
    ActionPolicy,
    SubskillProposalPolicy,
)
from openai import AzureOpenAI
import instructor
import jinja2
from typing import Sequence
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from typing import Iterable, cast, Optional
import os
from pydantic import BaseModel, model_validator
from ulid import ULID
from dataenvgym.gym.tasks.math.MATH.scoring import render_solution_for_scoring
from loguru import logger
from typing import Literal, Callable
from typing_extensions import Self
from collections import defaultdict

DEFAULT_SUBSKILL_TEMPLATE = jinja2.Template(
    """
    You are an experienced math educator and your task is to propose new subskills for improving a model's skills in solving math problems under the category of "{{ skill_category }}".

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


class SubskillProposal(BaseModel):
    name: str
    index: int


class ProposeSubskillsWithAzureOpenAI:
    def __init__(self, template: jinja2.Template = DEFAULT_SUBSKILL_TEMPLATE):
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.template = DEFAULT_SUBSKILL_TEMPLATE

    def render_prompt(
        self, skill: Skill, subskills: list[Subskill], num_new_subskills: int
    ) -> str:
        return self.template.render(
            skill_category=skill,
            subskills=subskills,
            num_new_subskills=num_new_subskills,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_subskills_from_llm(self, prompt: str) -> Iterable[Subskill]:
        subskills = self.client.chat.completions.create(
            model="gpt-4o",
            response_model=Iterable[SubskillProposal],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        subskills = cast(Iterable[SubskillProposal], subskills)
        return [Subskill(subskill.name) for subskill in subskills]

    def __call__(
        self, skill_tree: SkillTree | PartialSkillTree, num_new_subskills: int
    ) -> Sequence[Subskill]:
        prompt = self.render_prompt(
            skill_tree.skill, skill_tree.subskills, num_new_subskills
        )
        try:
            new_subskills = list(self.get_subskills_from_llm(prompt))
            if len(new_subskills) > num_new_subskills:
                logger.warning(
                    f"LLM proposed {len(new_subskills)} new subskills for skill category {skill_tree.skill}, but we only need {num_new_subskills}."
                )
                new_subskills = new_subskills[:num_new_subskills]
            elif len(new_subskills) < num_new_subskills:
                logger.warning(
                    f"LLM proposed {len(new_subskills)} new subskills for skill category {skill_tree.skill}, but we need {num_new_subskills}."
                )
                num_missing_subskills = num_new_subskills - len(new_subskills)
                retry_prompt = self.render_prompt(
                    skill_tree.skill,
                    list(new_subskills) + list(skill_tree.subskills),
                    num_missing_subskills,
                )
                new_subskills.extend(self.get_subskills_from_llm(retry_prompt))
                # Chop the list to the correct size
                new_subskills = new_subskills[:num_new_subskills]
        except RetryError:
            logger.opt(exception=True).error(
                f"Failed to get new subskills for skill category {skill_tree.skill}."
            )
            return []
        return list(new_subskills)


class ProposeSubskillsFromFixedList:
    def __init__(self, subskills_dict: dict[Skill, list[Subskill]]):
        self.subskills_dict = subskills_dict
        self.returned_subskills: dict[Skill, set[Subskill]] = defaultdict(set)

    def __call__(
        self, skill_tree: SkillTree | PartialSkillTree, num_new_subskills: int
    ) -> Sequence[Subskill]:
        skill = skill_tree.skill
        available_subskills = self.subskills_dict.get(skill, [])
        new_subskills = []

        for subskill in available_subskills:
            if subskill not in self.returned_subskills[skill]:
                new_subskills.append(subskill)
                self.returned_subskills[skill].add(subskill)
                if len(new_subskills) == num_new_subskills:
                    break

        return new_subskills


implements(SubskillProposalPolicy)(ProposeSubskillsFromFixedList)


# To allow the LLM to count; otherwise it may have problems generating
# a specific number of outputs.
class MathDataSpecWithIndex(MathDataSpec):
    index: int


DEFAULT_SUBSKILL_DATA_TEMPLATE = jinja2.Template(
    """
You are an experienced math educator and your task is to create math problems for improving a student's skills in solving math problems.

{% if already_generated_data %}
Here are some problems that you have already written:
{% for data in already_generated_data %}
- Problem: {{ data.problem }}
  - Chain of Thought: {{ data.chain_of_thought }}
  - Final Answer: {{ data.final_answer }}
{% endfor %}
{% endif %}

Each problem should improve the student's ability to solve problems under the category of "{{ subskill }}".
Each problem should require the student to know the concept of "{{ subskill }}".
The problems you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "problem": The math problem you want the model to solve. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "chain_of_thought": A step-by-step explanation of how to solve the problem. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "final_answer": The final answer to the problem as a LaTeX string. For example '17' or '\\frac{1}{2} or `\\matrix{1 & 2 \\cr 3 & 4}`. Do not write a sentence here, just the answer.

Propose {{ data_budget }} new problems.
"""
)

LLAMA_3_FORMAT_TEMPLATE = jinja2.Template(
    """
You are an experienced math educator and your task is to create math problems for improving a student's skills in solving math problems.

{% if already_generated_data %}
Here are some problems that you have already written:
{% for data in already_generated_data %}
- Problem: {{ data.problem }}
  - Chain of Thought: {{ data.chain_of_thought }}
  - Final Answer: {{ data.final_answer }}
{% endfor %}
{% endif %}

Each problem should improve the student's ability to solve problems requiring "{{ subskill }}".
Each problem should require the student to know the concept of "{{ subskill }}".
The problems you produce must be valid JSON using the provided schema. 
Here are descriptions of the fields in the schema:
- "problem": The math problem you want the model to solve. Ensure this is valid LaTeX that is properly escaped for representation as a string in Python.
- "chain_of_thought": A step-by-step explanation of how to solve the problem. Format this as follows:
    - For simple problems (2 steps or fewer), provide a brief explanation in one or two sentences.
    - For complex problems (3 steps or more):
        Use this step-by-step format:

        ## Step 1: [Concise description]
        [Brief explanation and calculations]

        ## Step 2: [Concise description]
        [Brief explanation and calculations]
- "final_answer": The final answer to the problem as a LaTeX string. For example '17' or '\\frac{1}{2} or `\\matrix{1 & 2 \\cr 3 & 4}`. Do not write a sentence here, just the answer.

Propose {{ data_budget }} new problems.
"""
)


class GenerateSubskillDataWithAzureOpenAI:
    def __init__(
        self,
        model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
        template: jinja2.Template = DEFAULT_SUBSKILL_DATA_TEMPLATE,
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
        self.template = template
        self.model = model

    def render_prompt(
        self,
        subskill: Subskill,
        data_budget: int,
        already_generated_data: Optional[Sequence[MathDataSpec]] = None,
    ) -> str:
        return self.template.render(
            subskill=subskill,
            data_budget=data_budget,
            already_generated_data=already_generated_data,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
    )
    def get_data_specs_from_llm(self, prompt: str) -> Iterable[MathDataSpec]:
        data_specs = self.client.chat.completions.create(
            model=self.model,
            response_model=Iterable[MathDataSpecWithIndex],  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        data_specs = cast(Iterable[MathDataSpecWithIndex], data_specs)
        return data_specs

    def __call__(
        self, subskill: Subskill, data_budget: int
    ) -> Sequence[MathTrainingDatumWithSkillCategory]:
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
            MathTrainingDatumWithSkillCategory(
                ulid=ULID(),
                instruction=data_spec.problem,
                response=render_solution_for_scoring(
                    chain_of_thought=data_spec.chain_of_thought,
                    final_answer=data_spec.final_answer,
                ),
                # This should be the skill rather than the subskill, but we
                # don't have the skill tree available here.
                skill_category=str(subskill),
            )
            for data_spec in data_specs
        ]
        return training_data


class ConfigurableActionPolicy:
    def __init__(
        self,
        exploit_threshold: float = 0.01,
        data_allocation_increase_on_exploit: int = 10,
        num_new_skills: int = 5,
        data_allocation_for_new_skills: int = 10,
    ):
        self.exploit_threshold = exploit_threshold
        self.data_allocation_increase_on_exploit = data_allocation_increase_on_exploit
        self.num_new_skills = num_new_skills
        self.data_allocation_for_new_skills = data_allocation_for_new_skills

    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        last_experience = history[-1]
        performance_change = last_experience.reward

        if performance_change > self.exploit_threshold:
            data_allocation_delta = {
                subskill: self.data_allocation_increase_on_exploit
                for subskill in last_experience.state.skill_tree.subskills
            }
            return Exploit(data_allocation_delta=data_allocation_delta)
        else:
            return Explore(
                num_new_skills=self.num_new_skills,
                data_allocation_for_new_skills=self.data_allocation_for_new_skills,
            )


implements(ActionPolicy)(ConfigurableActionPolicy)

DEFAULT_ACTION_POLICY_TEMPLATE = jinja2.Template(
    """
    You are an experienced data scientist and your task is to decide the next action to take to improve the performance of a model.

    {% if history %}
    Here is the history of your actions and the effect they had on the model's performance.
    {% for experience in history %}
    - Skill: {{ experience.skill }}
      - Past Performance: {{ experience.state.past_performance }}
      - Reward: {{ experience.reward }}
      - Action Taken: {{ experience.action }}
      - Skill Tree:
        - Subskills: 
          {% for subskill in experience.state.skill_tree.subskills %}
          - {{ subskill }}: {{ experience.state.skill_tree.data_allocation[subskill] }} data points allocated
          {% endfor %}
        - Total Data Allocated: {{ experience.state.skill_tree.total_data_allocated }}
    {% endfor %}
    Here is an explanation of the data structure above:
    - "Skill": The high-level skill we are trying to improve.
    - "Past Performance": The performance of the model on the skill at the start of the experience.
    - "Reward": The change in performance of the model on the skill as a result of the action taken.
        - A positive reward indicates that the action taken improved the model's performance.
        - A negative reward indicates that the action taken hurt the model's performance.
    - "Action Taken": The action that was taken.
    - "Subskills": The subskills of the skill.
    - "Data Allocation": The amount of data allocated to each subskill.
    {% endif %}

    The skill tree will guide a swarm of workers to create training data.
    You should aim to maximize the reward while minimizing the amount of data allocated.
    You will need to identify which subskills are worth allocating data to, and which subskills can be ignored.
    You will need to identify whether you need to create new subskills, or if the set of subskills you have already created is sufficient.
    You will need to identify how much data to allocate to each subskill.
    You will have multiple rounds of actions to take over the course of many experiences.
    Pay careful attention to the history of actions and rewards to identify patterns and make informed decisions.

    The actions you can take are:
    - "Exploit": Increase the data allocation for existing subskills. Specify the data allocation increase for each subskill.
    - "Explore": Propose new subskills and allocate data for them. Specify the number of new skills to propose and the data allocation for each new skill.

    Based on the above history, decide the next action to take for each skill to improve the model's performance. 
    IMPORTANT: You must always take an action.
    To achieve more complex results, you can combine multiple actions over the course of many experiences.
    Here are some patterns you can use:
    - To start fresh and ignore all previous subskills:
        - emit an "Exploit" action with a data allocation delta that sets all subskill data allocations to zero.
    - To prune out a set of subskills:
        - emit an "Exploit" action with a data allocation delta that sets the data allocation for subskills to be ignored to 0. 
    - To bring back a set of subskills:
        - emit an "Exploit" action with a positive data allocation for each subskill to be brought back.
    - To create new subskills:
        - emit an "Explore" action with a data allocation for each new subskill.

    Choose the next action to take to improve the skill.
    Always explain your reasoning for the action you choose to take.
    When choosing "Explore":
    - always set `num_new_skills` and `data_allocation_for_new_skill` to non-None values.
    When choosing "Exploit":
    - always set `data_allocation_delta` to a non-None value.
    - include _all_ subskills in the data allocation delta, even if some subskills have not changed.
    """,
    undefined=jinja2.StrictUndefined,
)


class SimpleAction(BaseModel):
    action_type: Literal["explore"] | Literal["exploit"]
    reasoning: str
    data_allocation_delta: Optional[dict[str, int]] = None
    num_new_skills: Optional[int] = None
    data_allocation_for_new_skills: Optional[int] = None

    @model_validator(mode="after")
    def check_action_fields(self) -> Self:
        action_type = self.action_type
        if action_type == "explore":
            if (
                self.num_new_skills is None
                or self.data_allocation_for_new_skills is None
            ):
                raise ValueError(
                    'num_new_skills and data_allocation_for_new_skills must not be None when action_type is "explore"'
                )
        elif action_type == "exploit":
            if self.data_allocation_delta is None:
                raise ValueError(
                    'data_allocation_delta must not be None when action_type is "exploit"'
                )
        return self

    def emit_action(self) -> Action:
        if self.action_type == "explore":
            assert self.num_new_skills is not None
            assert self.data_allocation_for_new_skills is not None
            return Explore(
                num_new_skills=self.num_new_skills,
                data_allocation_for_new_skills=self.data_allocation_for_new_skills,
                reasoning=self.reasoning,
            )
        elif self.action_type == "exploit":
            assert self.data_allocation_delta is not None
            data_allocation_delta = {
                Subskill(k): v for k, v in self.data_allocation_delta.items()
            }
            return Exploit(
                data_allocation_delta=data_allocation_delta,
                reasoning=self.reasoning,
            )
        else:
            raise ValueError(f"Invalid action type: {self.action_type}")

    def validate_against_history(self, history: Sequence[SkillExperience]):
        # If there is a data allocation delta, check that is includes all subskills
        # in the skill tree.
        last_skill_tree = history[-1].state.skill_tree
        if self.action_type == "exploit":
            assert self.data_allocation_delta is not None
            for subskill in last_skill_tree.subskills:
                if subskill not in self.data_allocation_delta:
                    logger.warning(
                        f'data_allocation_delta does not include subskill "{subskill}". Setting to 0.'
                    )
                    self.data_allocation_delta[subskill] = 0


# The DEFAULT_ACTION_POLICY_TEMPLATE is used for the train-retrain loop in which the
# action policy is asked to make decisions based on how the performance of the model
# changed after training. However, in the accumulate-train loop, we don't train the model.
# Instead, we use the performance of the student on the training data as a reward signal.

ACCUMULATE_TRAIN_ACTION_POLICY_TEMPLATE = jinja2.Template(
    """
    You are an experienced data scientist and your task is to identify the weakest skills of a model.
    You are operating in a loop where you will grow and rebalance a skill tree that represents your knowledge of the model's abilities.

    {% if history %}
    Here is the history of your actions.
    {% for experience in history %}
    - Skill: {{ experience.skill }}
      - Validation Accuracy: {{ experience.state.past_performance }}
      - Training Accuracy: {{ experience.state.training_accuracy }}
      - Action Taken: {{ experience.action }}
      - Skill Tree:
        - Subskills: 
          {% for subskill in experience.state.skill_tree.subskills %}
          - {{ subskill }}:
            - Data Allocated: {{ experience.state.skill_tree.data_allocation[subskill] }}
            - Performance on Training Data: {{ experience.state.skill_tree.perf_on_training_data[subskill] }}
          {% endfor %}
        - Total Data Allocated: {{ experience.state.skill_tree.total_data_allocated }}
    {% endfor %}
    Here is an explanation of the data structure above:
    - "Skill": The high-level skill we are trying to improve.
    - "Validation Accuracy": The performance of the model on the skill at the start of the experience. 
    - "Training Accuracy": The weighted average performance of the model on the training data for all subskills.
    - "Action Taken": The action that was taken.
    - "Subskills": The subskills of the skill.
    - "Data Allocated": The amount of data allocated to each subskill.
    - "Performance on Training Data": The performance of the model on the training data for each subskill.
    {% endif %}

    The skill tree will guide a swarm of workers to create training data.
    Your primary goal is to identify subskills where the model's performance is weakest (i.e., subskills with the lowest training accuracy) and allocate more data to those subskills.
    You will need to decide whether to create new subskills or if the existing set of subskills is sufficient.
    You will need to determine how much data to allocate to each subskill, focusing on those where the model shows the poorest performance.
    You will have multiple rounds of actions to take over the course of many experiences.
    Pay careful attention to the history of actions and performance on training data to identify patterns and make informed decisions.

    The actions you can take are:
    - "Exploit": Increase the data allocation for existing subskills. Specify the data allocation increase for each subskill, prioritizing those with lower performance.
    - "Explore": Propose new subskills and allocate data for them. Specify the number of new skills to propose and the data allocation for each new skill.

    IMPORTANT: You cannot allocate more than 30 points to a single subskill in a single action. If you allocate more than 30, it may fail!

    Based on the above history, decide the next action to take for each skill to improve the model's performance. 
    IMPORTANT: You must always take an action.
    To achieve more complex results, you can combine multiple actions over the course of many experiences.
    Here are some patterns you can use:
    - To focus on the weakest subskills:
        - emit an "Exploit" action with a higher data allocation delta for subskills with lower performance on training data.
    - To prune out well-performing subskills:
        - emit an "Exploit" action with a data allocation delta that reduces or sets to 0 the allocation for high-performing subskills.
    - To create new subskills (if existing ones are not capturing all aspects of the skill):
        - emit an "Explore" action with a data allocation for each new subskill.

    Choose the next action to take to improve the skill, focusing on addressing the weakest areas of the model's performance.
    Always explain your reasoning for the action you choose to take.
    When choosing "Explore":
    - always set `num_new_skills` and `data_allocation_for_new_skill` to non-None values.
    - you cannot allocate more than 30 data points to a single subskill.
    When choosing "Exploit":
    - always set `data_allocation_delta` to a non-None value.
    - include _all_ subskills in the data allocation delta, even if some subskills have not changed.
    - prioritize increasing data allocation for subskills with lower performance on training data.
    - you cannot allocate more than 30 data points to a single subskill.

    Guidelines:
    - validation accuracy will not change over the course of a single experience
    - training accuracy will change over the course of a single experience
    - you want to find subskills for which the training accuracy is lowest and allocate data to them
    - to generate more than 30 data points to a subskill, exploit it over multiple rounds
    - the end goal is to obtain a training dataset which addresses the weakest skills of the model efficiently
    - efficiency in this case means wasting as little data as possible on subskills that are already strong
    - DO NOT allocate more than 30 data points to a single subskill in a single action
    - your goal is to make the training accuracy as LOW as possible; this means you have found training data that is likely to improve the model
        - you can fulfill this goal by finding subskills where the training accuracy is lowest
    - it is better to go broad rather than deep; explore instead of allocating more than 30 data to a single subskill
    - try to focus on a set of subskills at a time; don't allocate budget to all subskills continuously in each iteration
    """,
    undefined=jinja2.StrictUndefined,
)


def accumulate_train_prompt_formatter(history: Sequence[SkillExperience]) -> str:
    return ACCUMULATE_TRAIN_ACTION_POLICY_TEMPLATE.render(
        history=history,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )


class AzureOpenAIActionPolicy:
    def __init__(
        self,
        prompt_formatter: Optional[Callable[[Sequence[SkillExperience]], str]] = None,
    ):
        self.client = instructor.patch(
            AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2023-03-15-preview",
            )
        )
        self.prompt_formatter = prompt_formatter

    def make_prompt(self, history: Sequence[SkillExperience]) -> str:
        if self.prompt_formatter is None:
            return DEFAULT_ACTION_POLICY_TEMPLATE.render(
                history=history, trim_blocks=True, lstrip_blocks=True
            )
        else:
            return self.prompt_formatter(history)

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(8),
    )
    def __call__(self, history: Sequence[SkillExperience]) -> Action:
        prompt = self.make_prompt(history)
        action = self.client.chat.completions.create(
            model="gpt-4o",
            response_model=SimpleAction,  # type: ignore
            messages=[{"role": "user", "content": prompt}],
        )
        action = cast(SimpleAction, action)
        action.validate_against_history(history)
        return action.emit_action()
