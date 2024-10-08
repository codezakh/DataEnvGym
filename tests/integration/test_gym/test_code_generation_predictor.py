from dataenvgym.gym.trainable_predictors.code.vllm_predictor import (
    ParallelVllmCodeGenerationPredictor,
    GEMMA2_2B_INSTRUCT_INFERENCE_CONFIG,
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG,
)
from dataenvgym.gym.tasks.code.livecodebench_task import LiveCodeBenchTask
import rich


def test_gemma_2b_it_predictor():
    GEMMA2_2B_INSTRUCT_INFERENCE_CONFIG.num_workers = 2
    predictor = ParallelVllmCodeGenerationPredictor(GEMMA2_2B_INSTRUCT_INFERENCE_CONFIG)
    task = LiveCodeBenchTask(split="debug")
    completed_task_instances = task.evaluate(predictor)
    performance_report = task.generate_performance_report(completed_task_instances)
    rich.print(performance_report)


def test_llama_3_8b_predictor():
    LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG.num_workers = 2
    predictor = ParallelVllmCodeGenerationPredictor(LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG)
    task = LiveCodeBenchTask(split="debug")
    completed_task_instances = task.evaluate(predictor)
    performance_report = task.generate_performance_report(completed_task_instances)
    rich.print(performance_report)
