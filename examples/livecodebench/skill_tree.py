import rich
from dataenvgym.gym.data_generation_agents.math.bandit_components import (
    ProposeSubskillsWithAzureOpenAI,
)
from dataenvgym.gym.data_generation_agents.math.bandit_data_strategy import (
    JsonExperienceCheckpointer,
)
from dataenvgym.gym.data_generation_agents.math.handwritten_policies import (
    AlternatingExploreExploitPolicy,
)
from dataenvgym.gym.data_generation_agents.skill_tree import (
    CodeGenerationDataGenerationAgent,
    Explore,
)
from dataenvgym.utils import (
    JSONLinesKeyValueCache,
    PydanticJSONLinesWriter,
)
from dataenvgym.gym.domain_models import (
    CodeGenerationTrainingDatum,
    CodeGenerationTrainingDataQualityCheck,
)
from loguru import logger
from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchTask
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import (
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG,
    LLAMA3_8B_INSTRUCT_TRAINER_CONFIG,
    SftCodeGenerationTrainer,
    ParallelVllmCodeGenerationPredictor,
    CodeGenerationTrainablePredictor,
)
from dataenvgym.gym.skill_discovery.llm_metacognitive import (
    CodeGenerationSkillDiscoverer,
)
from pathlib import Path
from dataenvgym.gym.data_generation_agents.code.baselines.skill_tree import (
    OpenAiSubskillDataGenerationPolicy,
    PROPOSE_SUBSKILL_TEMPLATE,
    StubCodeGenerationQualityChecker,
)
import ray
from huggingface_hub import hf_hub_download
from dataenvgym.gym.environments.base_environment import CodeGenerationEnvironment
from dataenvgym.gym import episode_runner
import pandas as pd


def main():
    num_gpus = 8
    ray.init(num_gpus=num_gpus)

    workspace_path = Path("workspace")
    experiment_dir = workspace_path / "livecodebench_skill_tree_example"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    val_task = LiveCodeBenchTask(split="val")

    discovered_skills_path = Path(
        hf_hub_download(
            repo_id="codezakh/dataenvbench-discovered-skills",
            filename="experiments_111_livecodebench_skills.json",
            repo_type="dataset",
        )
    )
    skill_discovery_module = CodeGenerationSkillDiscoverer()
    skill_discovery_module.set_precomputed_skills(discovered_skills_path)

    predictor_config = LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG.set_with_gpu_count(num_gpus)
    predictor_config.sampling_params.min_tokens = 10
    predictor = ParallelVllmCodeGenerationPredictor(config=predictor_config)
    trainer_config = LLAMA3_8B_INSTRUCT_TRAINER_CONFIG.set_with_gpu_count(
        num_gpus
    ).set_working_directory(experiment_dir)
    trainer_config.overrides = []
    trainer = SftCodeGenerationTrainer(
        config=trainer_config,
    )

    trainable_predictor = CodeGenerationTrainablePredictor(
        predictor=predictor,
        trainer=trainer,
    )
    test_task = LiveCodeBenchTask(split="test")
    io_provider = episode_runner.IoProvider(experiment_dir)

    # Initialize the JsonExperienceCheckpointer
    experience_checkpointer = JsonExperienceCheckpointer(
        output_path=experiment_dir / "checkpoints"
    )

    # Initialize the JSONLinesKeyValueCache
    training_data_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "training_data_cache.jsonl",
        model=CodeGenerationTrainingDatum,
    )

    quality_check_cache = JSONLinesKeyValueCache(
        file_path=experiment_dir / "quality_check_cache.jsonl",
        model=CodeGenerationTrainingDataQualityCheck,
    )
    quality_checker = StubCodeGenerationQualityChecker()

    logging_folder = experiment_dir / "data_strategy_outputs"
    logging_folder.mkdir(parents=True, exist_ok=True)
    subskill_data_generator = OpenAiSubskillDataGenerationPolicy(
        logging_folder=logging_folder,
    )

    action_policy = AlternatingExploreExploitPolicy(
        explore_action=Explore(num_new_skills=2, data_allocation_for_new_skills=30),
        max_data_for_skill=200,
    )

    subskill_proposer = ProposeSubskillsWithAzureOpenAI(
        template=PROPOSE_SUBSKILL_TEMPLATE
    )

    data_generation_agent = CodeGenerationDataGenerationAgent(
        skill_discovery_module=skill_discovery_module,
        subskill_data_generation_engine=subskill_data_generator,
        action_policy=action_policy,
        subskill_proposal_policy=subskill_proposer,
        experience_checkpointer=experience_checkpointer,
        training_data_cache=training_data_cache,
        initial_explore_action=Explore(
            num_new_skills=2, data_allocation_for_new_skills=10
        ),
        quality_check_cache=quality_check_cache,
        training_data_quality_checker=quality_checker,
    )

    environment = CodeGenerationEnvironment(
        validation_tasks=[val_task],
        trainable_predictor=trainable_predictor,
    )

    performance_history = episode_runner.run_episode(
        environment=environment,
        data_generation_agent=data_generation_agent,
        io_provider=io_provider,
        num_iterations=10,
    )

    # Write the performance history to a CSV.
    rows = []
    for iteration_idx, performance_reports in performance_history.items():
        # There will be one performance report for each task. Since we only passed in
        # one task, there will be one performance report.
        row = {
            "iteration_idx": iteration_idx,
            "task_name": performance_reports[0].task_name,
            "overall_performance": performance_reports[0].overall_performance,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(experiment_dir / "performance_history.csv", index=False)

    # Get the iteration index of the best performing model on the validation set.
    best_iteration_idx = df["overall_performance"].idxmax()
    logger.info(f"Best iteration idx: {best_iteration_idx}")

    # Evaluate the best performing model on the test set.
    assert isinstance(best_iteration_idx, int)
    best_model_save_path = io_provider.get_trainable_predictor_save_path(
        episode_runner.IterationMetadata(iteration=best_iteration_idx)
    )
    trainable_predictor.predictor.load_adapter(str(best_model_save_path))

    test_task = LiveCodeBenchTask(split="test")
    completed_task_instances = test_task.evaluate(trainable_predictor)
    writer = PydanticJSONLinesWriter(
        file_path=experiment_dir / "test_completed_task_instances.jsonl",
    )
    writer.write_batch(completed_task_instances)
    performance_report = test_task.generate_performance_report(
        completed_task_instances=completed_task_instances
    )

    with open(experiment_dir / "test_performance.json", "w") as f:
        f.write(performance_report.model_dump_json())

    rich.print(performance_report)


if __name__ == "__main__":
    main()
