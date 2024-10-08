from dataenvgym.gym.tasks.vqa.gqa import GqaTask
from dataenvgym.gym.open_loop import PredictConstantAnswerTrainablePredictor


def test_gqa_task():
    task = GqaTask(split="val")
    predictor = PredictConstantAnswerTrainablePredictor(constant_answer="foo")

    completed_instances = task.evaluate(predictor)
    performance_report = task.generate_performance_report(completed_instances)

    assert performance_report.overall_performance == 0.0
