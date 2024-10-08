from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchTask
from dataenvgym.gym.trainable_predictors.code.vllm_predictor import (
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG,
    LLAMA3_8B_INSTRUCT_TRAINER_CONFIG,
    SftCodeGenerationTrainer,
    ParallelVllmCodeGenerationPredictor,
    CodeGenerationTrainablePredictor,
)
from dataenvgym.gym.data_generation_agents.code.baselines.open_ended import (
    DataGenerationAgent,
)
from dataenvgym.gym.closed_loop import run_code_generation_closed_loop, IoProvider
from dataenvgym.gym.domain_models import (
    CodeGenerationTrainingDatum,
    CodeGenerationCompletedTaskInstance,
)
from pathlib import Path


def main():
    num_gpus = 8
    experiment_dir = Path("workspace/livecodebench_open_ended_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)

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
        datum_to_generate_per_error=5,
        logging_folder=experiment_dir / "data_strategy_outputs",
    )

    io_provider = IoProvider[
        CodeGenerationCompletedTaskInstance, CodeGenerationTrainingDatum
    ](experiment_dir=experiment_dir)

    run_code_generation_closed_loop(
        validation_code_tasks=[val_task],
        test_code_tasks=[test_task],
        trainable_code_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        io_provider=io_provider,
        num_iterations=5,
    )


if __name__ == "__main__":
    main()
