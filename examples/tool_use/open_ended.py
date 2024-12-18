import warnings
from pathlib import Path

from dataenvgym.gym.accumulate_train_loop import (
    IoProvider,
    run_generic_accumulation_train_loop,
)
from dataenvgym.gym.data_generation_agents.tool_use.baselines.open_ended import (
    DataGenerationAgent,
)
from dataenvgym.gym.tasks.tool_use.mnms.task import MnmsSplit, MnmsTask
from dataenvgym.gym.trainable_predictors.tool_use.vllm_predictor import (
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG,
    LLAMA3_8B_INSTRUCT_TRAINER_CONFIG,
    CodeGenerationTrainablePredictor,
    ParallelVllmCodeGenerationPredictor,
    SftCodeGenerationTrainer,
)
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def main():
    num_gpus = 8
    experiment_dir = Path("workspace/tool_use_open_ended_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    predictor_config = LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG.set_with_gpu_count(num_gpus)
    predictor = ParallelVllmCodeGenerationPredictor(config=predictor_config)
    trainer_config = LLAMA3_8B_INSTRUCT_TRAINER_CONFIG.set_with_gpu_count(
        num_gpus
    ).set_working_directory(experiment_dir)
    trainer_config.overrides = ["cutoff_len=1600"]
    trainer = SftCodeGenerationTrainer(
        config=trainer_config,
    )

    trainable_predictor = CodeGenerationTrainablePredictor(
        predictor=predictor,
        trainer=trainer,
    )

    val_task = MnmsTask(split=MnmsSplit.VAL)
    test_task = MnmsTask(split=MnmsSplit.TEST)

    data_strategy = DataGenerationAgent(
        datum_to_generate_per_error=2,
        logging_folder=experiment_dir / "data_strategy_outputs",
        data_generation_per_invocation_limit=60,
    )

    io_provider = IoProvider(experiment_dir=experiment_dir)

    run_generic_accumulation_train_loop(
        validation_tasks=[val_task],
        test_tasks=[test_task],
        trainable_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        io_provider=io_provider,
        num_cycles=10,
        accumulation_iterations_per_cycle=1,
    )


if __name__ == "__main__":
    main()
