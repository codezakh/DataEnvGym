import ray
import rich
from dataenvgym.experiment_utils import (
    make_output_dir_for_run,
    write_current_commit_hash_to_file,
)
from dataenvgym.gym import accumulate_train_loop
from dataenvgym.gym.data_generation_agents.math.bandit_components import (
    GenerateSubskillDataWithAzureOpenAI,
    ProposeSubskillsWithAzureOpenAI,
)
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    Explore,
    JsonExperienceCheckpointer,
    MathBanditDataStrategy,
    Skill,
    Subskill,
)
from dataenvgym.gym.data_generation_agents.math.handwritten_policies import (
    AlternatingExploreExploitPolicy,
)
from dataenvgym.gym.data_generation_agents.math.baselines.skill_list import (
    MathTrainingDatumWithSkillCategory,
)
from dataenvgym.gym.domain_models import MathTrainingDatumQualityCheck
from dataenvgym.gym.quality_checking.math.minimal import MathTrainingDataQualityChecker
from dataenvgym.gym.skill_discovery.llm_metacognitive import (
    SkillCategory,
    MathSkillDiscoverer,
)
from dataenvgym.gym.tasks.math.MATH.task import MATHTask, prepare_few_shot_prompt
from dataenvgym.gym.trainable_predictors.math.local_llm import (
    LlamaFactoryTrainer,
    ParallelLlmTrainablePredictor,
    ParallelLlmPredictor,
)
from dataenvgym.utils import (
    JSONLinesKeyValueCache,
    PydanticJSONLinesWriter,
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
    experiment_dir = Path("workspace/math_skill_tree_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    discovered_skills_path = Path(
        hf_hub_download(
            repo_id="codezakh/dataenvbench-discovered-skills",
            filename="experiments_106_math_skills.json",
            repo_type="dataset",
        )
    )
    skill_discovery_module = MathSkillDiscoverer()
    skill_discovery_module.set_precomputed_skills(discovered_skills_path)
    io_provider = accumulate_train_loop.IoProvider(experiment_dir)

    # Get the current commit hash and write it to a file
    write_current_commit_hash_to_file(experiment_dir)

    # Initialize the JsonExperienceCheckpointer
    experience_checkpointer = JsonExperienceCheckpointer(
        output_path=experiment_dir / "checkpoints"
    )

    # Initialize the JSONLinesKeyValueCache
    training_data_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "training_data_cache.jsonl",
        model=MathTrainingDatumWithSkillCategory,
    )

    # Initialize the quality checker and a cache for it.
    # This is necessary for the accumulate-train loop; it is not
    # meaningful to run the loop without a real quality checker.
    quality_check_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "quality_check_cache.jsonl",
        model=MathTrainingDatumQualityCheck,
    )
    quality_checker = MathTrainingDataQualityChecker()

    validation_tasks = [
        MATHTask(split="val_balanced_subset_50"),
    ]
    test_tasks = [
        MATHTask(split="test_all"),
    ]

    # For Gemma-2b-it, we need to set always_apply_chat_template=True.
    # For Llama3-8b, you can leave it False.
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

    # We just use the standard components here.
    subskill_data_generator = GenerateSubskillDataWithAzureOpenAI(model="gpt-4o-mini")

    action_policy = AlternatingExploreExploitPolicy(
        explore_action=Explore(num_new_skills=2, data_allocation_for_new_skills=30),
        max_data_for_skill=200,
    )

    training_data_production_strategy = MathBanditDataStrategy(
        skill_discovery_module=skill_discovery_module,
        logging_folder=experiment_dir / "data_strategy_outputs",
        subskill_data_generator=subskill_data_generator,
        action_policy=action_policy,
        subskill_proposal_policy=ProposeSubskillsWithAzureOpenAI(),
        experience_checkpointer=experience_checkpointer,
        training_data_cache=training_data_cache,
        initial_explore_action=Explore(
            num_new_skills=2, data_allocation_for_new_skills=30
        ),
        quality_check_cache=quality_check_cache,
        training_data_quality_checker=quality_checker,
    )

    final_val_reports, final_test_reports = (
        accumulate_train_loop.run_generic_accumulation_train_loop(
            validation_tasks=validation_tasks,
            test_tasks=test_tasks,
            training_data_production_strategy=training_data_production_strategy,
            trainable_predictor=trainable_predictor,
            num_cycles=10,
            accumulation_iterations_per_cycle=2,
            io_provider=io_provider,
        )
    )

    # Write final validation performance report
    with open(experiment_dir / "val_final_performance.json", "w") as f:
        f.write(final_val_reports[0].model_dump_json())

    rich.print(final_val_reports[0])
    logger.info("Validation performance report generated")

    # Write final test performance report
    with open(experiment_dir / "test_final_performance.json", "w") as f:
        f.write(final_test_reports[0].model_dump_json())

    rich.print(final_test_reports[0])
    logger.info("Test performance report generated")

    # Evaluate on test_all.
    test_all = MATHTask(split="test_all")
    completed_task_instances = test_all.evaluate(trainable_predictor)
    writer = PydanticJSONLinesWriter(
        file_path=experiment_dir / "test_all_completed_task_instances.jsonl",
    )
    writer.write_batch(completed_task_instances)
    performance_report = test_all.generate_performance_report(
        completed_task_instances=completed_task_instances
    )

    with open(experiment_dir / "test_all_performance.json", "w") as f:
        f.write(performance_report.model_dump_json())

    rich.print(performance_report)
    logger.info("Test all performance report generated")


if __name__ == "__main__":
    main()
