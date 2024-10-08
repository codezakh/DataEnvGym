# DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback
[Zaid Khan](https://zaidkhan.me/), [Elias Stengel-Eskin](https://esteng.github.io/), [Jaemin Cho](https://j-min.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

[Project Page](https://dataenvgym.github.io/)

`DataEnvGym` provides building blocks and patterns for creating and evaluating data generation agents and environments.

Useful features:
- Modular environments that support multimodal, math, and code generation tasks
- Baseline implementations of data generation agents for each of the above tasks
- Fully typed codebase
- Integration with vLLM and Ray for fast, parallel inference
- Integration with Llama-Factory for training

# Getting Started

## Installation

```shell
git clone https://github.com/codezakh/dataenvgym.git && cd dataenvgym
conda create -n dataenvgym python=3.10
conda activate dataenvgym
pip install -r requirements.txt
pip install -e src/external/LLaMA-Factory --config-settings editable_mode=compat
pip install -e . --config-settings editable_mode=compat
```
This will install `dataenvgym` as a Python module, so you can do `import dataenvgym`.

The flag `--config-settings editable_mode=compat` is required for type-checking due to a change in the way editable installs are handled in recent versions of setuptools.
If you don't care about type-checking, you can ignore the flag.

## API Keys
If using Azure OpenAI, set the following environment variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

If using OpenAI, set the following environment variable:
- `OPENAI_API_KEY`

## Datasets
Datasets used for the paper will be automatically downloaded by HuggingFace Datasets.

## A Minimal Example
Here is a simplified example that uses each high-level component of `DataEnvGym` to run an episode in which a data generation agent tries to improve a `gemma-2-2b-it` student model on the MATH dataset.

This example should be immediately runnable after installing the requirements. 

```python
from pathlib import Path

import ray
# We will use an agent designed for open-ended data generation for MATH.
from dataenvgym.gym.data_generation_agents.math.baselines import open_ended
from dataenvgym.gym.environments.base_environment import MathEnvironment
# Some utilities required to run an episode of data generation.
from dataenvgym.gym.episode_runner import IoProvider, run_episode
from dataenvgym.gym.tasks.math.MATH import task as MATH
# This is the class that manages the student model for us.
from dataenvgym.gym.trainable_predictors.math import local_llm
from vllm import SamplingParams

# Set this to the number of GPUs you have available.
num_gpus = 4
ray.init(num_gpus=num_gpus)
# The folder experiment outputs will be written to.
experiment_dir = Path("workspace/minimal_example")

# That task we will try to improve performance on.
task = MATH.MATHTask(split="val_balanced_subset_50")

# The student model we will try to improve.
model_name_or_path = "google/gemma-2-2b-it"
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
```
See `docs/components.md` for more details on the API.

# Examples
- GQA
    - Open-Ended Environment: `examples/gqa/open_ended_environment.py`
    - Skill-List Environment: `examples/gqa/skill_list_environment.py`
    - Skill-Tree Environment: `examples/gqa/skill_tree_environment.py`
- MATH
    - Open-Ended Environment: `examples/math/open_ended_environment.py`
    - Skill-List Environment: `examples/math/skill_list_environment.py`
    - Skill-Tree Environment: `examples/math/skill_tree_environment.py`
- LiveCodeBench
    - Open-Ended Environment: `examples/livecodebench/open_ended_environment.py`
    - Skill-List Environment: `examples/livecodebench/skill_list_environment.py`
    - Skill-Tree Environment: `examples/livecodebench/skill_tree_environment.py`

## Running Examples
Run the examples from the repository root. Set `CUDA_VISIBLE_DEVICES` to the GPUs you want to use and make sure to set `num_gpus` in `ray.init()` to the number of GPUs you have available.

```shell
CUDA_VISIBLE_DEVICES=<...> python examples/math/open_ended_environment.py
```

# Feature Requests
If there's anything you'd like to see added, please open an issue. PRs are welcome!