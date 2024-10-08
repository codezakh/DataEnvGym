from dataenvgym.gym.data_generation_agents.code.baselines.random import DataGenerationAgent
from pathlib import Path


def test_random_baseline_data_strategy(tmp_path: Path):
    data_strategy = DataGenerationAgent(
        logging_folder=tmp_path,
        data_specs_per_llm_call=3,
        num_training_data_per_invocation=6,
    )
    training_data = data_strategy.generate_training_data()
    assert len(training_data) == 6
