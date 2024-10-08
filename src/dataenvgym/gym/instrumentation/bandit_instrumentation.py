from typing import TypedDict, Sequence, Mapping, Optional, Any
from pathlib import Path
import json
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    Skill,
    Subskill,
)
from loguru import logger
from pydantic import BaseModel
from collections import Counter
import seaborn as sns
import pandas as pd


class SubskillDistribution(BaseModel):
    counts: dict[Subskill, int]
    percentages: dict[Subskill, float]


class SkillDataDistribution(BaseModel):
    specified: dict[Skill, SubskillDistribution]
    actual: dict[Skill, SubskillDistribution]


def get_final_data_distribution(checkpoint_path: Path) -> SkillDataDistribution:
    """
    Calculates the final data distribution across skills and subskills from a checkpoint file.
    This version uses the LenientJsonExperienceCheckpointer for more flexible JSON parsing.

    Args:
        checkpoint_path (Path): The path to the checkpoint file containing past experiences.

    Returns:
        SkillDataDistribution: A Pydantic model with 'specified' and 'actual' distributions.

    This function expects the checkpoint file to contain a sequence of experiences, where each experience
    is a mapping of skill names to SkillExperienceDict objects.
    """
    # Initialize the LenientJsonExperienceCheckpointer
    checkpointer = LenientJsonExperienceCheckpointer(checkpoint_path)

    # Load the checkpoint
    past_experiences, _ = checkpointer.load_checkpoint()

    if not past_experiences:
        return SkillDataDistribution(specified={}, actual={})

    # Initialize the final data distributions
    specified_distribution: dict[Skill, SubskillDistribution] = {}
    actual_distribution: dict[Skill, SubskillDistribution] = {}

    # Iterate through all experiences in the history
    for experience in past_experiences:
        for skill_str, skill_experience in experience.items():
            skill = Skill(skill_str)
            skill_tree = skill_experience["state"]["skill_tree"]

            if skill not in specified_distribution:
                specified_distribution[skill] = SubskillDistribution(
                    counts={}, percentages={}
                )
            if skill not in actual_distribution:
                actual_distribution[skill] = SubskillDistribution(
                    counts={}, percentages={}
                )

            # Sum up the specified data allocation
            for subskill_str, allocation in skill_tree["data_allocation"].items():
                subskill = Subskill(subskill_str)
                if subskill not in specified_distribution[skill].counts:
                    specified_distribution[skill].counts[subskill] = 0
                specified_distribution[skill].counts[subskill] += allocation

            # Count the actual training data
            for subskill_str, training_data in skill_tree["training_data"].items():
                subskill = Subskill(subskill_str)
                if subskill not in actual_distribution[skill].counts:
                    actual_distribution[skill].counts[subskill] = 0
                actual_distribution[skill].counts[subskill] += len(training_data)

    # Calculate percentages for each skill in both distributions
    for distribution in [specified_distribution, actual_distribution]:
        for skill, skill_distribution in distribution.items():
            total_allocation = sum(skill_distribution.counts.values())
            skill_distribution.percentages = {
                subskill: count / total_allocation if total_allocation > 0 else 0
                for subskill, count in skill_distribution.counts.items()
            }

    return SkillDataDistribution(
        specified=specified_distribution, actual=actual_distribution
    )


def get_data_distribution_history(checkpoint_path: Path) -> list[dict[Skill, int]]:
    """
    Returns a list of dictionaries holding the cumulative amount of training data produced for each skill.
    Each element i in the list is a dictionary that is the sum of all the data produced for each skill from
    element 0 to element i.

    Args:
        checkpoint_path (Path): The path to the checkpoint file containing past experiences.

    Returns:
        list[dict[Skill, int]]: A list of dictionaries with cumulative training data counts for each skill.
    """
    # Initialize the LenientJsonExperienceCheckpointer
    checkpointer = LenientJsonExperienceCheckpointer(checkpoint_path)

    # Load the checkpoint
    past_experiences, _ = checkpointer.load_checkpoint()

    if not past_experiences:
        return []

    # Initialize the cumulative data distribution history
    cumulative_history: list[dict[Skill, int]] = []
    cumulative_counts: dict[Skill, int] = {}

    # Iterate through all experiences in the history
    for experience in past_experiences:
        current_counts: dict[Skill, int] = cumulative_counts.copy()
        for skill_str, skill_experience in experience.items():
            skill = Skill(skill_str)
            skill_tree = skill_experience["state"]["skill_tree"]

            # Count the actual training data
            for subskill_str, training_data in skill_tree["training_data"].items():
                if skill not in current_counts:
                    current_counts[skill] = 0
                current_counts[skill] += len(training_data)

        cumulative_counts = current_counts
        cumulative_history.append(cumulative_counts.copy())

    return cumulative_history


def plot_data_distribution_history(
    checkpoint_path: Path, col_wrap: int = 4
) -> Optional[sns.FacetGrid]:
    """
    Plots the data distribution history using Seaborn.

    Args:
        checkpoint_path (Path): The path to the checkpoint file containing past experiences.
        col_wrap (int): Number of columns to wrap the facets. Default is 4.

    Returns:
        sns.FacetGrid: The Seaborn FacetGrid object for further customization.
    """
    # Get the data distribution history
    history = get_data_distribution_history(checkpoint_path)

    if not history:
        print("No data to plot.")
        return None

    # Convert the history to a DataFrame for plotting
    data = []
    for iteration, counts in enumerate(history):
        for skill, count in counts.items():
            data.append(
                {"Iteration": iteration, "Skill": skill, "Cumulative Data": count}
            )

    df = pd.DataFrame(data)

    # Plot using Seaborn
    g = sns.FacetGrid(df, col="Skill", col_wrap=col_wrap, sharey=False)
    g.map(sns.lineplot, "Iteration", "Cumulative Data")

    g.set_axis_labels("Iteration", "Cumulative Data")
    g.set_titles(col_template="{col_name}")

    return g


class SkillTreeDict(TypedDict):
    skill: str
    subskills: list[str]
    data_allocation: dict[str, int]
    perf_on_training_data: dict[str, Optional[float]]


class SkillStateDict(TypedDict):
    skill_tree: SkillTreeDict
    past_performance: float


class SkillExperienceDict(TypedDict):
    state: SkillStateDict
    reward: float
    action: dict[str, Any]


class LenientJsonExperienceCheckpointer:
    """
    A checkpointer that reads JSON experience data without using Pydantic models.

    This class is useful when the structure of SkillExperience or related models
    has changed over time, making older checkpoint data incompatible with current
    Pydantic models. It provides a more flexible way to read and load checkpoint
    data, allowing for backwards compatibility with older data formats.
    """

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.current_experiences_path = output_path / "current_experiences.jsonl"
        self.past_experiences_path = output_path / "past_experiences.jsonl"

    def load_json(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            with file_path.open("r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def load_past_experiences(self) -> Sequence[Mapping[str, dict[str, Any]]]:
        # The experiences are written as a jsonl file where each line is a json encoded
        # dictionary from skill -> skill experience.
        experiences = self.load_json(self.past_experiences_path)
        logger.info(
            f"Loaded {len(experiences)} past experiences from {self.past_experiences_path}"
        )
        return experiences

    def load_current_experiences(self) -> Optional[Mapping[str, dict[str, Any]]]:
        experiences = self.load_json(self.current_experiences_path)
        if not experiences:
            return None
        # Return the most recent experience
        return experiences[-1]

    def load_checkpoint(
        self,
    ) -> tuple[
        Sequence[Mapping[str, dict[str, Any]]], Optional[Mapping[str, dict[str, Any]]]
    ]:
        past_experiences = self.load_past_experiences()
        current_experiences = self.load_current_experiences()
        return past_experiences, current_experiences

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(output_path={self.output_path})"


def visualize_skill_experience(skill_experience: SkillExperienceDict) -> str:
    """
    Visualize a single SkillExperience object as a tree structure.

    This function expects a dictionary representation of a SkillExperience,
    with the following structure:
    {
        "state": {
            "skill_tree": {
                "skill": str,
                "subskills": list[str],
                "data_allocation": dict[str, int],
                "perf_on_training_data": dict[str, Optional[float]]
            },
            "past_performance": float,
            "training_accuracy": Optional[float]
        },
        "reward": float,
        "action": dict[str, Any]
    }
    """
    state = skill_experience["state"]
    skill_tree = state["skill_tree"]
    tree_str = f"Skill: {skill_tree['skill']}\n"
    tree_str += "  Subskills:\n"
    for subskill in skill_tree["subskills"]:
        tree_str += f"    - {subskill}\n"
        # Older versions of SkillTree didn't have perf_on_training_data
        perf = skill_tree.get("perf_on_training_data", {}).get(subskill)
        perf_str = f"{perf}" if perf is not None else "N/A"
        tree_str += f"      Training Accuracy: {perf_str}\n"
    tree_str += "  Data Allocation:\n"
    for subskill, allocation in skill_tree["data_allocation"].items():
        tree_str += f"    - {subskill}: {allocation}\n"
    tree_str += f"  Past Performance: {state['past_performance']}\n"
    tree_str += f"  Reward: {skill_experience['reward']}\n"
    tree_str += f"  Action: {skill_experience['action']}\n"
    return tree_str


def visualize_checkpoint(
    past_experiences: Sequence[Mapping[str, SkillExperienceDict]],
    current_experiences: Optional[Mapping[str, SkillExperienceDict]] = None,
) -> str:
    """
    Visualize the checkpoint data by generating a text-based tree structure for each skill.

    This function expects:
    - past_experiences: A sequence of mappings, where each mapping represents a generation
      and maps skill names (str) to SkillExperienceDict objects.
    - current_experiences: An optional mapping of skill names (str) to SkillExperienceDict
      objects representing the current (incomplete) experiences.

    The structure of SkillExperienceDict is the same as described in visualize_skill_experience.
    """
    output = ""
    for i, experience in enumerate(past_experiences):
        output += f"Generation {i + 1}:\n\n"
        for skill, skill_experience in experience.items():
            output += visualize_skill_experience(skill_experience)
            output += "\n" + "-" * 50 + "\n\n"

    if current_experiences:
        output += "Current Experiences:\n\n"
        for skill, partial_skill_experience in current_experiences.items():
            output += visualize_skill_experience(partial_skill_experience)
            output += "\n" + "-" * 50 + "\n\n"

    return output


def map_experiences(
    experiences: Sequence[Mapping[str, dict[str, Any]]]
) -> Sequence[Mapping[str, SkillExperienceDict]]:
    """
    Convert a sequence of raw experience dictionaries to SkillExperienceDict format.

    This function is used to transform the output of LenientJsonExperienceCheckpointer
    into the format expected by visualize_checkpoint for past experiences.

    Args:
        experiences: A sequence of mappings from skill names to raw experience dictionaries.

    Returns:
        A sequence of mappings from skill names to SkillExperienceDict objects.
    """
    return [
        {
            skill: SkillExperienceDict(
                state=exp["state"], reward=exp["reward"], action=exp["action"]
            )
            for skill, exp in generation.items()
        }
        for generation in experiences
    ]


def map_current_experience(
    experience: Optional[Mapping[str, dict[str, Any]]]
) -> Optional[Mapping[str, SkillExperienceDict]]:
    """
    Convert a single raw current experience dictionary to SkillExperienceDict format.

    This function is used to transform the output of LenientJsonExperienceCheckpointer
    into the format expected by visualize_checkpoint for current experiences.

    Args:
        experience: An optional mapping from skill names to raw experience dictionaries.

    Returns:
        An optional mapping from skill names to SkillExperienceDict objects,
        or None if the input is None.
    """
    if experience is None:
        return None
    return {
        skill: SkillExperienceDict(
            state=exp["state"], reward=exp["reward"], action=exp["action"]
        )
        for skill, exp in experience.items()
    }


def get_action_distribution(checkpoint_path: Path) -> dict[Skill, dict[str, float]]:
    """
    Calculates the distribution of explore/exploit actions for each skill from a checkpoint file.

    Args:
        checkpoint_path (Path): The path to the checkpoint file containing past experiences.

    Returns:
        dict[Skill, dict[str, float]]: A dictionary mapping each skill to its action distribution.
    """
    checkpointer = LenientJsonExperienceCheckpointer(checkpoint_path)
    past_experiences, _ = checkpointer.load_checkpoint()

    action_counts: dict[Skill, Counter] = {}

    for experience in past_experiences:
        for skill_str, skill_experience in experience.items():
            skill = Skill(skill_str)
            action = skill_experience["action"]["action_type"]

            if skill not in action_counts:
                action_counts[skill] = Counter()

            action_counts[skill][action] += 1

    action_distribution: dict[Skill, dict[str, float]] = {}

    for skill, counts in action_counts.items():
        total = sum(counts.values())
        distribution = {action: count / total for action, count in counts.items()}
        action_distribution[skill] = distribution

    return action_distribution
