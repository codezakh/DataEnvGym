from dataenvgym.gym.data_selection_agents.math_random_selection import (
    RandomSelectionDataGenerationAgent,
)


def test_random_selection_data_generation_agent():
    agent = RandomSelectionDataGenerationAgent(num_training_data_per_invocation=10)
    data = agent.generate_training_data([])
    assert len(data) == 10
