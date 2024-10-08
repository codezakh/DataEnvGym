from dataenvgym.gym.tasks.code.livecodebench_task import (
    LiveCodeBenchTask,
    LiveCodeBenchEvaluationManager,
)
from dataenvgym.gym.trainable_predictors.code.openai_predictor import (
    OpenAICodeGenerationPredictor,
    format_prompt,
)
from dataenvgym.gym.domain_models import CodeGenerationTaskInstance
from dataenvgym.gym.tasks.code.livecodebench.runner.codegen_evaluator import (
    CodeGenerationEvaluationParameters,
    CodeGenerationProblem,
    CodeGenerationScorable,
)
from pathlib import Path
import sh
import sys


def test_formatting_prompt():
    task_instance = CodeGenerationTaskInstance(
        task_name="test",
        instance_id="test",
        instruction="instruction",
        starter_code="starter_code",
    )
    print(format_prompt(task_instance))


def test_debug_task():
    task = LiveCodeBenchTask(split="debug")
    predictor = OpenAICodeGenerationPredictor()
    completed_task_instances = task.evaluate(predictor)
    performance_report = task.generate_performance_report(completed_task_instances)
    # Assert that the overall performance is greater than 0.3.
    # This is a bit flakey, but it's good enough for now.
    assert performance_report.overall_performance >= 0.3


def test_single_problem():
    task = LiveCodeBenchTask(split="single")
    predictor = OpenAICodeGenerationPredictor()
    completed_task_instances = task.evaluate(predictor)
    performance_report = task.generate_performance_report(completed_task_instances)


class TestEvaluationManager:
    @staticmethod
    def test_evaluation_manager():
        evaluation_manager = LiveCodeBenchEvaluationManager(split="debug")
        sh.python(  # type: ignore[attr-defined]
            evaluation_manager.path_to_cli_script,
            "--help",
            _out=sys.stdout,
            _err=sys.stderr,
        )
