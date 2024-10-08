import ray
from dataenvgym.experiment_utils import (
    make_output_dir_for_run,
    write_current_commit_hash_to_file,
)
from dataenvgym.gym import accumulate_train_loop
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    DataGenerationAgent,
)
from dataenvgym.gym.skill_discovery.llm_metacognitive import MathSkillDiscoverer
from dataenvgym.gym.tasks.math.MATH.task import MATHTask, prepare_few_shot_prompt
from dataenvgym.gym.trainable_predictors.math.local_llm import (
    LlamaFactoryTrainer,
    ParallelLlmTrainablePredictor,
    ParallelLlmPredictor,
)
from loguru import logger
from vllm import SamplingParams
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    target_effective_batch_size = 16
    per_device_batch_size = 1
    num_gpus = 8
    gradient_accumulation_steps = target_effective_batch_size // (
        per_device_batch_size * num_gpus
    )
    logger.info(
        f"Target effective batch size: {target_effective_batch_size}, per-device batch size: {per_device_batch_size}, num-gpus: {num_gpus}, gradient accumulation steps: {gradient_accumulation_steps}"
    )
    ray.init(num_gpus=num_gpus)

    experiment_dir = Path("workspace/math_skill_list_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    io_provider = accumulate_train_loop.IoProvider(experiment_dir)

    validation_tasks = [MATHTask(split="val_balanced_subset_10")]
    test_tasks = [MATHTask(split="test_all")]

    predictor = ParallelLlmPredictor(
        sampling_params=SamplingParams(temperature=0.0, max_tokens=350),
        prompt_formatter_for_base_model=prepare_few_shot_prompt,
        model_name_or_path="google/gemma-2-2b-it",
        num_workers=num_gpus,
        always_apply_chat_template=True,
    )
    trainer = LlamaFactoryTrainer(
        working_directory=experiment_dir / "llama_factory",
        cuda_visible_devices=None,  # This should be None when using Ray.
        overrides=[
            "model_name_or_path=google/gemma-2-2b-it",
            "template=gemma",
            f"gradient_accumulation_steps={gradient_accumulation_steps}",
            f"per_device_train_batch_size={per_device_batch_size}",
        ],
    )

    trainable_predictor = ParallelLlmTrainablePredictor(predictor, trainer)

    discovered_skills_path = Path(
        hf_hub_download(
            repo_id="codezakh/dataenvbench-discovered-skills",
            filename="experiments_106_math_skills.json",
            repo_type="dataset",
        )
    )
    skill_discovery_module = MathSkillDiscoverer(use_as_offline_labeler=True)
    skill_discovery_module.set_precomputed_skills(discovered_skills_path)
    training_data_production_strategy = DataGenerationAgent(
        logging_folder=experiment_dir / "data_strategy_outputs",
        skill_discovery_module=skill_discovery_module,
        data_specs_per_hypothesis=1,
        hypotheses_per_skill_category=10,
        generate_data_only_for_errors=False,
        model="gpt-4o-mini",
    )

    accumulate_train_loop.run_generic_accumulation_train_loop(
        validation_tasks=validation_tasks,
        test_tasks=test_tasks,
        training_data_production_strategy=training_data_production_strategy,
        trainable_predictor=trainable_predictor,
        io_provider=io_provider,
        accumulation_iterations_per_cycle=1,
        num_cycles=3,
    )


if __name__ == "__main__":
    main()
