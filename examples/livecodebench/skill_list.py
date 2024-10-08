from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchTask
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import (
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG,
    LLAMA3_8B_INSTRUCT_TRAINER_CONFIG,
    SftCodeGenerationTrainer,
    ParallelVllmCodeGenerationPredictor,
    CodeGenerationTrainablePredictor,
)
from dataenvgym.gym.data_generation_agents.code.baselines.skill_list import (
    DataGenerationAgent,
)
from dataenvgym.gym.accumulate_train_loop import (
    run_generic_accumulation_train_loop,
    IoProvider,
)
from dataenvgym.gym.skill_discovery.llm_metacognitive import (
    CodeGenerationSkillDiscoverer,
)
from pathlib import Path
from dataenvgym.utils import PydanticJSONLinesWriter
from huggingface_hub import hf_hub_download


def main():
    num_gpus = 8
    experiment_dir = Path("workspace/livecodebench_skill_list_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    discovered_skills_path = Path(
        hf_hub_download(
            repo_id="codezakh/dataenvbench-discovered-skills",
            filename="experiments_111_livecodebench_skills.json",
            repo_type="dataset",
        )
    )
    skill_discoverer = CodeGenerationSkillDiscoverer()
    skill_discoverer.set_precomputed_skills(discovered_skills_path)

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

    val_task = LiveCodeBenchTask(split="val")
    test_task = LiveCodeBenchTask(split="test")

    data_strategy = DataGenerationAgent(
        skill_discovery_module=skill_discoverer,
        data_specs_per_skill_category=5,
        logging_folder=experiment_dir / "data_strategy_outputs",
    )

    io_provider = IoProvider(experiment_dir=experiment_dir)

    val_history, test_history = run_generic_accumulation_train_loop(
        validation_tasks=[val_task],
        test_tasks=[test_task],
        trainable_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        io_provider=io_provider,
        accumulation_iterations_per_cycle=1,
        num_cycles=20,
    )

    writer = PydanticJSONLinesWriter(experiment_dir / "val_performance_history.jsonl")
    writer.write_batch(val_history)

    writer = PydanticJSONLinesWriter(experiment_dir / "test_performance_history.jsonl")
    writer.write_batch(test_history)


if __name__ == "__main__":
    main()
