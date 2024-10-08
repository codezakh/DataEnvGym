from pathlib import Path

from dataenvgym.gym.data_generation_agents.code.baselines.skill_tree import (
    OpenAiSubskillDataGenerationPolicy,
)
from dataenvgym.gym.data_generation_agents.skill_tree import Subskill


def test_generate_data_for_subskill(tmp_path: Path):
    generate_data_for_subskill = OpenAiSubskillDataGenerationPolicy(
        logging_folder=tmp_path,
    )

    training_data = generate_data_for_subskill(
        subskill=Subskill("Backtracking in Dynamic Programming"),
        data_budget=10,
    )

    assert len(training_data) == 10
