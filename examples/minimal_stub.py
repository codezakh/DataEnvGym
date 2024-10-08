from pathlib import Path
from typing import Sequence
from collections import Counter

import ray
from dataenvgym.gym.data_generation_agents.math.baselines import open_ended
from dataenvgym.gym.environments.base_environment import MathEnvironment
from dataenvgym.gym.episode_runner import IoProvider, run_episode
from dataenvgym.gym.tasks.math.MATH import task as MATH
from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathTrainingDatum,
)

num_gpus = 4
ray.init(num_gpus=num_gpus)
experiment_dir = Path("workspace/minimal_example")

task = MATH.MATHTask(split="val_balanced_subset_50")


class ConstantAnswerTrainablePredictor:
    def __init__(self):
        self.constant_answer = "42"  # Default answer

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return [self.constant_answer] * len(task_instances)

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        # Your training code here #
        return

    def save(self, path: Path) -> None:
        # Your save code here #
        return


trainable_predictor = ConstantAnswerTrainablePredictor()

data_generation_agent = open_ended.DataGenerationAgent(
    logging_folder=experiment_dir / "data_strategy_outputs",
    data_specs_per_llm_call=10,
    num_training_data_per_invocation=120,
)

environment = MathEnvironment(
    validation_tasks=[task],
    trainable_predictor=trainable_predictor,
)

performance_history = run_episode(
    environment=environment,
    data_generation_agent=data_generation_agent,
    io_provider=IoProvider(experiment_dir),
    stop_at_num_training_data=1000,
    num_iterations=10,
)
