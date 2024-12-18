from dataenvgym.gym.tasks.tool_use.mnms.evaluator import get_code_labels
import pytest
from dataenvgym.gym.tasks.tool_use.mnms.task import MnmsTask, MnmsSplit
from dataenvgym.gym.domain_models import CodeGenerationCompletedTaskInstance

from dataenvgym.gym.data_generation_agents.tool_use.baselines.skill_list import (
    DataGenerationAgent,
)
from ulid import ULID


def test_get_code_labels():
    code = """def solve()
    output0 = image_captioning(image="2327921.jpg")
    output1 = text_summarization(text=output0['text'])
    output2 = question_answering(text=output1['text'], question="What is the main dessert mentioned in the caption?")
    return output2
    """
    labels = get_code_labels(code)
    assert labels == [
        "image_captioning",
        "text_summarization",
        "question_answering",
    ]


@pytest.mark.xfail(reason="Order of labels is not consistent")
def test_get_code_labels_order_is_consistent():
    code_a = """def solve()
    output0 = image_captioning(image="2327921.jpg")
    output1 = text_summarization(text=output0['text'])
    output2 = question_answering(text=output1['text'], question="What is the main dessert mentioned in the caption?")
    return output2
    """

    code_b = """def solve()
    output1 = text_summarization(text=output0['text'])
    output0 = image_captioning(image="2327921.jpg")
    output2 = question_answering(text=output1['text'], question="What is the main dessert mentioned in the caption?")
    return output2
    """

    labels_a = get_code_labels(code_a)
    labels_b = get_code_labels(code_b)
    assert labels_a == labels_b


class TestSkillListBaseline:
    def test_data_generation_agent(self):
        task = MnmsTask(split=MnmsSplit.VAL)
        task_instance = task.task_instances[0]
        completed_task_instance = CodeGenerationCompletedTaskInstance(
            ulid=ULID(),
            task_instance=task_instance,
            predictor_response="",
            was_correct=False,
        )

        agent = DataGenerationAgent(
            datum_to_generate_per_skill=3,
        )

        training_data = agent.generate_training_data(
            completed_task_instances=[completed_task_instance],
        )

        assert len(training_data) == 3
