from dataenvgym.gym.tasks.tool_use.mnms.evaluator import (
    MNMEvaluator,
    MnmScorable,
    MnmsRecordRaw,
)
from datasets import load_dataset
from typing import cast, Sequence
from dataenvgym.gym.tasks.tool_use.mnms.metrics import PRFMetrics
import ast
from dataenvgym.gym.tasks.tool_use.mnms.task import (
    get_split,
    MnmsSplit,
    load_mnms_human_verified_filtered,
    MnmsRecord,
    check_which_tools_missing,
    MnmsTool,
    MnmsTask,
)
from dataenvgym.gym.tasks.tool_use.mnms.constants import (
    DEMO_EXAMPLES,
    CODE_DEMO_EXAMPLES,
)
import json
from dataenvgym.gym.tasks.tool_use.mnms.prompts import (
    load_tool_descriptions,
    make_single_turn_inference_prompt_from_task_instance,
)
from dataenvgym.gym.domain_models import CodeGenerationTaskInstance
import tiktoken


class TestPrfMetrics:
    def test_macro_precision_recall_f1_when_not_all_categories_present(self):
        categories = ["a", "b", "c"]
        gt_labels = ["a"]
        pred_labels = ["a"]

        metrics = PRFMetrics(categories=categories, average="macro")

        metrics.update(gt_labels, pred_labels)

        metrics_result = metrics.compute()

        assert metrics_result["precision"] == 100.0
        assert metrics_result["recall"] == 100.0
        assert metrics_result["f1"] == 100.0

    def test_micro_precision_recall_f1_when_not_all_categories_present(self):
        categories = ["a", "b", "c"]
        gt_labels = ["a"]
        pred_labels = ["a"]

        metrics = PRFMetrics(categories=categories, average="micro")

        metrics.update(gt_labels, pred_labels)

        metrics_result = metrics.compute()

        assert metrics_result["precision"] == 100.0
        assert metrics_result["recall"] == 100.0
        assert metrics_result["f1"] == 100.0


class TestMnmEvaluator:
    def test_evaluate_all(self):
        ds = load_dataset(
            "zixianma/mnms",
            split="test_human_verified_filtered",
            revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
        )

        ds = cast(Sequence[MnmsRecordRaw], ds)

        ground_truth_data: list[MnmsRecordRaw] = []
        for record in ds:
            ground_truth_data.append(record)

        scorables: list[MnmScorable] = []
        for record in ground_truth_data:
            scorables.append(
                MnmScorable(
                    id=record["id"],
                    prediction=record["code_str"],
                )
            )

        evaluator = MNMEvaluator(
            gt_data=ground_truth_data,
            pred_data=scorables,
            plan_format="code",
        )

        df, invalid_predictions = evaluator.evaluate()

        assert len(invalid_predictions) == 0

    def test_evaluate_single_example_correct(self):
        ds = load_dataset(
            "zixianma/mnms",
            split="test_human_verified_filtered",
            revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
        )

        ds = cast(Sequence[MnmsRecordRaw], ds)

        ground_truth = ds[0]

        scorable = MnmScorable(
            id=ground_truth["id"],
            prediction=ground_truth["code_str"],
        )

        computed_metrics, invalid_predictions = MNMEvaluator.evaluate_single_instance(
            gt_sample=ground_truth, pred_sample=scorable, plan_format="code"
        )

        assert len(invalid_predictions) == 0
        assert computed_metrics["tool_macro"]["precision"] == 100.0
        assert computed_metrics["tool_macro"]["recall"] == 100.0
        assert computed_metrics["tool_macro"]["f1"] == 100.0

    def test_evaluate_single_example_incorrect(self) -> None:
        ds = load_dataset(
            "zixianma/mnms",
            split="test_human_verified_filtered",
            revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
        )

        ds = cast(Sequence[MnmsRecordRaw], ds)

        ground_truth = ds[0]

        incorrect_prediction = """
    def solve():
        output0 = image_classification(image)
        return output0
    """

        scorable = MnmScorable(
            id=ground_truth["id"],
            prediction=incorrect_prediction,
        )

        computed_metrics, invalid_predictions = MNMEvaluator.evaluate_single_instance(
            gt_sample=ground_truth, pred_sample=scorable, plan_format="code"
        )

        assert len(invalid_predictions) == 0
        assert computed_metrics["tool_macro"]["precision"] == 0.0
        assert computed_metrics["tool_macro"]["recall"] == 0.0
        assert computed_metrics["tool_macro"]["f1"] == 0.0

    def test_single_example_partially_correct(self) -> None:
        ds = load_dataset(
            "zixianma/mnms",
            split="test_human_verified_filtered",
            revision="da313260161c982eb2004bb15761d7aa2e03eb4f",
        )

        ds = cast(Sequence[MnmsRecordRaw], ds)

        # Find a record which has more than 1 step.
        for record in ds:
            plan_str = ast.literal_eval(record["plan_str"])
            if len(plan_str) > 1:
                break

        ground_truth = record

        # Delete the last 3 lines of the code.
        # The last 3 lines of the code look like:
        # output<number> = ...
        # output = ...
        # return output
        # This will make the prediction partially correct by removing the last step.

        partial_prediction = "\n".join(ground_truth["code_str"].split("\n")[:-3])

        scorable = MnmScorable(
            id=ground_truth["id"],
            prediction=partial_prediction,
        )

        computed_metrics, invalid_predictions = MNMEvaluator.evaluate_single_instance(
            gt_sample=ground_truth, pred_sample=scorable, plan_format="code"
        )

        assert len(invalid_predictions) == 0
        assert computed_metrics["tool_macro"]["precision"] == 50.00
        assert computed_metrics["tool_macro"]["recall"] == 50.00
        assert computed_metrics["tool_macro"]["f1"] == 50.00


def test_in_context_code_examples_correct() -> None:
    ground_truth = [
        MnmsRecordRaw(
            id=_["id"],
            user_request=_["user_request"],
            plan_str=json.dumps(_["nodes"]),
            code_str="",
            alt_plans_str="[]",
        )
        for _ in DEMO_EXAMPLES
    ]

    scorables = [
        MnmScorable(
            id=_["id"],
            prediction=_["prediction"],
        )
        for _ in CODE_DEMO_EXAMPLES
    ]

    evaluator = MNMEvaluator(
        gt_data=ground_truth,
        pred_data=scorables,
        plan_format="code",
    )

    scores, invalid_predictions = evaluator.evaluate()

    assert len(invalid_predictions) == 0
    assert scores["tool_macro"]["precision"] == 100.0
    assert scores["tool_macro"]["recall"] == 100.0
    assert scores["tool_macro"]["f1"] == 100.0


class AlwaysCorrectPredictor:
    def predict(
        self, task_instances: Sequence[CodeGenerationTaskInstance]
    ) -> list[str]:
        predictions: list[str] = []
        for task_instance in task_instances:
            assert task_instance.solution is not None
            predictions.append(task_instance.solution)
        return predictions


class RightHalfTheTimePredictor:
    def predict(
        self, task_instances: Sequence[CodeGenerationTaskInstance]
    ) -> list[str]:
        predictions: list[str] = []
        for idx, task_instance in enumerate(task_instances):
            assert task_instance.solution is not None
            if idx % 2 == 0:
                predictions.append(task_instance.solution)
            else:
                predictions.append("a bad prediction")
        return predictions


class TestMnmsTask:
    def test_check_which_tools_missing(self) -> None:
        records = MnmsRecord.sequence_from_ds(load_mnms_human_verified_filtered())
        missing_tools = check_which_tools_missing(records)
        assert len(missing_tools) == 0

    def test_split_covers_all_tools(self) -> None:
        test_set = MnmsRecord.sequence_from_ds(get_split(MnmsSplit.TEST))
        print(f"Length of test set: {len(test_set)}")
        assert check_which_tools_missing(test_set) == set()

    def test_validation_split_is_entirely_correct(self) -> None:
        validation_set = get_split(MnmsSplit.VAL)

        scorables = [
            MnmScorable(
                id=record["id"],
                prediction=record["code_str"],
            )
            for record in validation_set
        ]

        evaluator = MNMEvaluator(
            gt_data=validation_set,
            pred_data=scorables,
            plan_format="code",
        )

        scores, invalid_predictions = evaluator.evaluate()

        assert len(invalid_predictions) == 0
        assert scores["tool_macro"]["precision"] == 100.0
        assert scores["tool_macro"]["recall"] == 100.0
        assert scores["tool_macro"]["f1"] == 100.0

    def test_always_correct_predictor(self) -> None:
        predictor = AlwaysCorrectPredictor()
        task = MnmsTask(split=MnmsSplit.TEST)

        completed_task_instances = task.evaluate(predictor)

        assert all(
            completed_task_instance.was_correct
            for completed_task_instance in completed_task_instances
        )

        performance_report = task.generate_performance_report(completed_task_instances)

        assert performance_report.overall_performance == 100.0
        for slice_performance in performance_report.slices:
            if "macro" in slice_performance.slice_name:
                assert slice_performance.metric_value == 100.0

    def test_right_half_the_time_predictor(self) -> None:
        predictor = RightHalfTheTimePredictor()
        task = MnmsTask(split=MnmsSplit.TEST)

        completed_task_instances = task.evaluate(predictor)

        assert (
            sum(
                completed_task_instance.was_correct
                for completed_task_instance in completed_task_instances
            )
            == len(completed_task_instances) / 2
        )

        performance_report = task.generate_performance_report(completed_task_instances)

        assert performance_report.overall_performance == 50.0


class TestPrompts:
    def test_make_prompt_single_turn_answer(self) -> None:
        tool_descriptions = load_tool_descriptions()
        assert "BEGIN PREAMBLE" not in tool_descriptions
        for tool in MnmsTool:
            assert tool.value in tool_descriptions

    def test_make_prompt_single_turn_answer_renders(self) -> None:
        instruction = "Generate an image, caption the image, and classify sentiment of the caption."
        task_instance = CodeGenerationTaskInstance(
            task_name="mnms",
            instance_id="123",
            instruction=instruction,
        )
        prompt = make_single_turn_inference_prompt_from_task_instance(task_instance)
        print(prompt)
        # Use tiktoken to count the number of tokens in the prompt
        encoding = tiktoken.encoding_for_model("gpt-4o")
        num_tokens = len(encoding.encode(prompt))
        print(f"Number of tokens: {num_tokens}")
        assert instruction in prompt
