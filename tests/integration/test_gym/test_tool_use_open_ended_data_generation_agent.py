from dataenvgym.gym.data_generation_agents.tool_use.baselines.open_ended import (
    DataGenerationAgent,
)
from dataenvgym.gym.domain_models import (
    CodeGenerationTaskInstance,
    CodeGenerationCompletedTaskInstance,
)
from ulid import ULID
from dataenvgym.gym.tasks.tool_use.mnms.task import MnmsTask, MnmsSplit
from typing import Collection


class StubPredictorInterface:
    def predict(
        self, task_instances: Collection[CodeGenerationTaskInstance]
    ) -> list[str]:
        return [""] * len(task_instances)


def test_data_generation_agent():
    task = MnmsTask(split=MnmsSplit.VAL)
    agent = DataGenerationAgent(datum_to_generate_per_error=5)

    task_instance = task.task_instances[0]

    completed_task_instance = CodeGenerationCompletedTaskInstance(
        ulid=ULID(),
        task_instance=task_instance,
        predictor_response=r"```python\n# ... I don't know what to do```",
        was_correct=False,
    )

    training_data = agent([completed_task_instance], StubPredictorInterface())

    assert len(training_data) == 5
