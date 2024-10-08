from dataenvgym.gym.data_generation_agents.vqa.baselines.skill_tree import (
    DataGenerationAgent,
)
from pathlib import Path
from dataenvgym.gym.data_generation_agents.skill_tree import Subskill


def test_skill_tree_baseline(tmp_path: Path):
    data_generator = DataGenerationAgent()
    training_data = data_generator(Subskill("Material Identification"), 5)
    assert len(training_data) == 5
