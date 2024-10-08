from dataenvgym.gym.open_loop import (
    PredictColorVqaTask,
    TrainingDataOfSpecificColorProductionStrategy,
    PredictMostCommonResponseTrainablePredictor,
    PreferenceTrainingDataOfSpecificColorProductionStrategy,
    PreferencePredictMostCommonResponseTrainablePredictor,
)
from dataenvgym.gym.closed_loop import (
    single_iteration,
    IterationMetadata,
    run_closed_loop,
    IoProvider,
    run_closed_preference_loop,
    single_preference_iteration,
)
from unittest.mock import MagicMock


def test_single_iteration_of_closed_loop(tmp_path):
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PredictMostCommonResponseTrainablePredictor()
    data_strategy = TrainingDataOfSpecificColorProductionStrategy(color="red")
    io_provider = IoProvider(tmp_path)

    performance_reports = single_iteration(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        iteration_metadata=IterationMetadata(iteration=1),
        io_provider=io_provider,
    )

    assert len(performance_reports) == 1
    performance_report = next(iter(performance_reports))
    assert performance_report.overall_performance == 1.0


def test_entire_closed_loop_with_stubs(tmp_path):
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PredictMostCommonResponseTrainablePredictor()
    data_strategy = TrainingDataOfSpecificColorProductionStrategy(color="red")
    io_provider = IoProvider(tmp_path)

    performance_reports = run_closed_loop(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        num_iterations=10,
        io_provider=io_provider,
    )

    assert len(performance_reports) == 10
    # TODO: Change type annotation on the return type of run_closed_loop
    # to Sequence
    performance_report = next(iter(performance_reports[0]))
    assert performance_report.overall_performance == 1.0


def test_closed_loop_steps_data_strategy(tmp_path, monkeypatch):
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PredictMostCommonResponseTrainablePredictor()
    data_strategy = TrainingDataOfSpecificColorProductionStrategy(color="red")
    io_provider = IoProvider(tmp_path)

    # Patch the .step method of the data strategy with monkeypatch and a MagicMock
    # to ensure that it is called num_iterations times.

    step_mock = MagicMock()
    monkeypatch.setattr(data_strategy, "step", step_mock)

    run_closed_loop(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        num_iterations=10,
        io_provider=io_provider,
    )

    assert step_mock.call_count == 10


def test_single_iteration_of_closed_preference_loop(tmp_path):
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PreferencePredictMostCommonResponseTrainablePredictor()
    data_strategy = PreferenceTrainingDataOfSpecificColorProductionStrategy(color="red")
    io_provider = IoProvider(tmp_path)

    performance_reports = single_preference_iteration(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        iteration_metadata=IterationMetadata(iteration=1),
        io_provider=io_provider,
    )

    assert len(performance_reports) == 1
    performance_report = next(iter(performance_reports))
    assert performance_report.overall_performance == 1.0


def test_entire_closed_preference_loop_with_stubs(tmp_path):
    predict_red_task = PredictColorVqaTask(color="red")
    trainable_predictor = PreferencePredictMostCommonResponseTrainablePredictor()
    data_strategy = PreferenceTrainingDataOfSpecificColorProductionStrategy(color="red")
    io_provider = IoProvider(tmp_path)

    performance_reports = run_closed_preference_loop(
        validation_vqa_tasks=[predict_red_task],
        test_vqa_tasks=[predict_red_task],
        trainable_vqa_predictor=trainable_predictor,
        training_data_production_strategy=data_strategy,
        num_iterations=10,
        io_provider=io_provider,
    )

    assert len(performance_reports) == 10
