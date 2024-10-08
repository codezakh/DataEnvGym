from dataenvgym.gym.trainable_predictors.math.local_llm import LlamaPredictor
from dataenvgym.gym.tasks.math.MATH.task import MATHTask
from dataenvgym.gym.domain_models import MathTaskInstance
from dataenvgym.utils import PydanticJSONLinesWriter
from pathlib import Path


def test_predicting():
    math_task = MATHTask(split="debug")
    task_instances = math_task.task_instances

    predictor = LlamaPredictor()
    predictor.predict(task_instances)


def test_trainer_can_fix_errors(tmp_path: Path):
    pass
