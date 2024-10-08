from pathlib import Path

import ray
from dataenvgym.gym.data_generation_agents.math.baselines import open_ended
from dataenvgym.gym.environments.base_environment import MathEnvironment
from dataenvgym.gym.episode_runner import IoProvider, run_episode
from dataenvgym.gym.tasks.math.MATH import task as MATH
from dataenvgym.gym.trainable_predictors.math import local_llm
from vllm import SamplingParams


num_gpus = 4  # Set to the number of available GPUs.
ray.init(num_gpus=num_gpus)
experiment_dir = Path("workspace/minimal_example")  # The output folder.

task = MATH.MATHTask(split="val_balanced_subset_50")  # The task to improve.

model_name_or_path = "google/gemma-2-2b-it"  # The student model.
trainable_predictor = local_llm.ParallelLlmTrainablePredictor(
    local_llm.ParallelLlmPredictor(
        sampling_params=SamplingParams(temperature=0.0, max_tokens=350),
        prompt_formatter_for_base_model=MATH.prepare_few_shot_prompt,
        model_name_or_path=model_name_or_path,
        num_workers=1,
    ),
    local_llm.LlamaFactoryTrainer(
        working_directory=experiment_dir / "llama_factory",
        cuda_visible_devices=None,
        overrides=[
            f"model_name_or_path={model_name_or_path}",
            "template=gemma",
        ],
    ),
)

# The data generation agent that will try to improve the student model.
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
