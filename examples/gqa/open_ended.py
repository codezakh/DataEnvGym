from pathlib import Path

import ray
from dataenvgym.gym import accumulate_train_loop
from dataenvgym.gym.data_generation_agents.vqa.baselines.open_ended import (
    DataGenerationAgent,
    SdxlTurboText2ImagePipelineWrapper,
)
from dataenvgym.gym.tasks.vqa.gqa import GqaTask
from dataenvgym.gym.trainable_predictors.vqa.paligemma import (
    PaliGemmaTrainablePredictor,
)


def main():
    experiment_dir = Path("workspace/gqa_open_ended_example")
    io_provider = accumulate_train_loop.IoProvider(experiment_dir)

    validation_tasks = [GqaTask(split="val")]
    test_tasks = [GqaTask(split="testdev")]
    ray.init(ignore_reinit_error=True)
    trainer_output_dir = experiment_dir / "trainer_output_dir"
    trainable_vqa_predictor = PaliGemmaTrainablePredictor(
        num_train_epochs=10, trainer_output_dir=trainer_output_dir
    )
    training_data_production_strategy = DataGenerationAgent(
        text_to_image_fn=SdxlTurboText2ImagePipelineWrapper(device="cuda:1"),
        logging_folder=experiment_dir / "data_strategy_outputs",
        model="gpt-4o",
        datum_to_generate_per_error=2,
    )

    final_val_reports, final_test_reports = (
        accumulate_train_loop.run_generic_accumulation_train_loop(
            validation_tasks=validation_tasks,
            test_tasks=test_tasks,
            training_data_production_strategy=training_data_production_strategy,
            trainable_predictor=trainable_vqa_predictor,
            num_cycles=5,
            accumulation_iterations_per_cycle=1,
            io_provider=io_provider,
        )
    )

    # Write the final validation report to a file
    with open(experiment_dir / "val_final_performance.json", "w") as f:
        f.write(final_val_reports[0].model_dump_json())

    # Write the final test report to a file
    with open(experiment_dir / "test_final_performance.json", "w") as f:
        f.write(final_test_reports[0].model_dump_json())


if __name__ == "__main__":
    main()
