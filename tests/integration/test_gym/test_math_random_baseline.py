from dataenvgym.gym.data_generation_agents.math.seeded_baseline import DataStrategy
from pathlib import Path


def test_random_baseline(tmp_path: Path):
    data_strategy = DataStrategy(
        logging_folder=tmp_path,
        data_specs_per_llm_call=3,
        num_training_data_per_invocation=6,
    )

    accumulated_data = []
    for _ in range(2):
        data = data_strategy.generate_training_data()
        accumulated_data.extend(data)

    import ipdb

    ipdb.set_trace()

    assert len(accumulated_data) == 12
