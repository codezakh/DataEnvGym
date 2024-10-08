import ray
import rich
from dataenvgym.gym import accumulate_train_loop
from dataenvgym.gym.data_generation_agents.math.baselines.open_ended import (
    DataGenerationAgent,
)
from dataenvgym.gym.tasks.math.MATH.task import MATHTask, prepare_few_shot_prompt
from dataenvgym.gym.trainable_predictors.math.local_llm import (
    LlamaFactoryTrainer,
    ParallelLlmTrainablePredictor,
    ParallelLlmPredictor,
)
from dataenvgym.utils import (
    PydanticJSONLinesWriter,
)
from loguru import logger
from vllm import SamplingParams
import pandas as pd
from pathlib import Path


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
    experiment_dir = Path("workspace/math_open_ended_example")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    io_provider = accumulate_train_loop.IoProvider(experiment_dir)

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

    training_data_production_strategy = DataGenerationAgent(
        logging_folder=experiment_dir / "data_strategy_outputs",
        data_specs_per_llm_call=10,
        num_training_data_per_invocation=120,
        model="gpt-4o-mini",
    )

    final_val_reports, final_test_reports = (
        accumulate_train_loop.run_generic_accumulation_train_loop(
            validation_tasks=validation_tasks,
            test_tasks=test_tasks,
            training_data_production_strategy=training_data_production_strategy,
            trainable_predictor=trainable_predictor,
            num_cycles=10,
            accumulation_iterations_per_cycle=1,
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

    # Write the performance history as a human-readable table.
    performance_history = [
        {
            "cycle": i,
            "validation_performance": test_report.overall_performance,
            "test_performance": val_report.overall_performance,
        }
        for i, (test_report, val_report) in enumerate(
            zip(final_test_reports, final_val_reports)
        )
    ]
    df = pd.DataFrame(performance_history)
    df.to_csv(experiment_dir / "performance_history.csv", index=False)


if __name__ == "__main__":
    main()
