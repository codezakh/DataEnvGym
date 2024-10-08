from typing import Literal, Collection
from dataenvgym.gym.domain_models import (
    CodeGenerationTaskInstance,
    CodeGenerationTaskInterface,
    CodeGenerationPredictorInterface,
    CodeGenerationCompletedTaskInstance,
    TaskPerformanceReport,
    TaskSlicePerformance,
    PredictorInterface,
)
from pydantic import BaseModel
from .livecodebench.benchmarks import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)
from .livecodebench.runner.codegen_evaluator import (
    score_code_generation,
    CodeGenerationScorable,
    CodeGenerationEvaluationParameters,
)
from datasets import load_dataset
from ulid import ULID
from loguru import logger
from .livecodebench.benchmarks.code_generation import Difficulty
from pathlib import Path
from .livecodebench.runner.codegen_evaluator import (
    LcbDataDictWithEvaluation,
    PassAtKMetrics,
)
import tempfile
from typing_extensions import Self
from argparse import ArgumentParser
import json
import sys
import sh

LiveCodeBenchSplitChoices = Literal["val", "test", "debug", "single"]


def load_split(split: LiveCodeBenchSplitChoices) -> list[CodeGenerationProblem]:
    v1 = load_code_generation_dataset("release_v1")
    v2 = load_code_generation_dataset("release_v2")

    if split == "val":
        v1_qids = set([x.question_id for x in v1])
        validation_set = [x for x in v2 if x.question_id not in v1_qids]
        return validation_set
    elif split == "test":
        return v1
    elif split == "debug":
        # Return 10 problems from the validation set.
        v1_qids = set([x.question_id for x in v1])
        validation_set = [x for x in v2 if x.question_id not in v1_qids]
        return validation_set[:10]
    elif split == "single":
        # Return a single problem from the validation set.
        v1_qids = set([x.question_id for x in v1])
        validation_set = [x for x in v2 if x.question_id not in v1_qids]
        return validation_set[:1]
    else:
        raise ValueError(f"Invalid split: {split}")


def load_solutions() -> dict[str, str]:
    with open(
        "workspace/notebooks__013_get_ground_truth_solutions_for_lcb/solution_for_each_qid.json",
        "r",
    ) as f:
        solutions = json.load(f)

    return {qid: _["passing_solution"] for qid, _ in solutions.items()}


def code_generation_problem_to_task_instance(
    problem: CodeGenerationProblem, solution: str | None = None
) -> CodeGenerationTaskInstance:
    return CodeGenerationTaskInstance(
        task_name="LiveCodeBench",
        instance_id=problem.question_id,
        instruction=problem.question_content,
        starter_code=problem.starter_code,
        solution=solution,
    )


class CodeGenerationEvaluationRequest(BaseModel):
    scorables: list[CodeGenerationScorable]
    split: LiveCodeBenchSplitChoices

    def to_evaluation_parameters(self) -> CodeGenerationEvaluationParameters:
        problems = load_split(self.split)
        sorted_problems = sorted(problems, key=lambda x: x.question_id)
        qid_to_scorable = {
            scorable["question_id"]: scorable for scorable in self.scorables
        }
        sorted_scorables = [qid_to_scorable[p.question_id] for p in sorted_problems]
        return CodeGenerationEvaluationParameters(
            scorables=sorted_scorables,
            problems=sorted_problems,
        )


class LiveCodeBenchEvaluationManager:
    path_to_cli_script: Path = (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "commands"
        / "run_livecodebench_evaluation.py"
    )
    input_arg_name: str = "path_to_request"
    output_arg_name: str = "path_to_output"

    def __init__(
        self, split: LiveCodeBenchSplitChoices, use_temporary_directory: bool = True
    ):
        self.split: LiveCodeBenchSplitChoices = split
        self.use_temporary_directory = use_temporary_directory

    def serialize_for_evaluation(
        self,
        problems: list[CodeGenerationProblem],
        predictions: list[str],
        path_to_serialize_request: str | Path,
    ) -> None:
        scorables: list[CodeGenerationScorable] = []
        for problem, prediction in zip(problems, predictions):
            scorables.append(
                CodeGenerationScorable(
                    question_id=problem.question_id,
                    code_list=[prediction],
                )
            )

        evaluation_request = CodeGenerationEvaluationRequest(
            scorables=scorables,
            split=self.split,
        )

        with open(path_to_serialize_request, "w") as f:
            f.write(evaluation_request.model_dump_json())

    @classmethod
    def build_from_serialized_request(
        cls, path_to_serialized_request: str | Path
    ) -> tuple[Self, CodeGenerationEvaluationRequest]:
        with open(path_to_serialized_request, "r") as f:
            request = CodeGenerationEvaluationRequest.model_validate_json(f.read())

        return cls(request.split), request

    def run_evaluation_from_request(
        self, request: CodeGenerationEvaluationRequest
    ) -> tuple[list[LcbDataDictWithEvaluation], PassAtKMetrics]:
        evaluation_params = request.to_evaluation_parameters()
        return score_code_generation(evaluation_params)

    @staticmethod
    def load_evaluation_results(
        path_to_serialized_results: str | Path,
    ) -> list[LcbDataDictWithEvaluation]:
        with open(path_to_serialized_results, "r") as f:
            return [json.loads(line) for line in f.readlines()]

    def serialize_evaluation_results(
        self,
        results: list[LcbDataDictWithEvaluation],
        path_to_serialize_results: str | Path,
    ) -> None:
        with open(path_to_serialize_results, "w") as f:
            for result in results:
                result["contest_date"] = result["contest_date"].isoformat()  # type: ignore
                f.write(json.dumps(result) + "\n")

    def run_evaluation_from_cli(
        self, path_to_request: str | Path, path_to_output: str | Path
    ) -> None:
        sh.python(  # type: ignore[attr-defined]
            self.path_to_cli_script,
            path_to_request,
            path_to_output,
            _out=sys.stdout,
            _err=sys.stderr,
        )

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            cls.input_arg_name,
            type=Path,
            help="Path to the request file.",
        )
        parser.add_argument(
            cls.output_arg_name,
            type=Path,
            help="Path to the output file.",
        )
        return parser

    @classmethod
    def cli_entrypoint(cls) -> None:
        parser = cls.get_parser()
        args = parser.parse_args()
        manager, request = cls.build_from_serialized_request(args.path_to_request)
        results, _ = manager.run_evaluation_from_request(request)
        manager.serialize_evaluation_results(results, args.path_to_output)

    def run_evaluation_from_problems_and_predictions(
        self,
        problems: list[CodeGenerationProblem],
        predictions: list[str],
        working_directory: Path,
    ) -> list[LcbDataDictWithEvaluation]:
        path_to_request = working_directory / "request.json"
        path_to_output = working_directory / "output.jsonl"
        self.serialize_for_evaluation(problems, predictions, path_to_request)
        self.run_evaluation_from_cli(str(path_to_request), str(path_to_output))
        return self.load_evaluation_results(path_to_output)

    def __call__(
        self, problems: list[CodeGenerationProblem], predictions: list[str]
    ) -> list[LcbDataDictWithEvaluation]:
        # Get a temporary directory, use a context manager to ensure it is cleaned up.
        if self.use_temporary_directory:
            with tempfile.TemporaryDirectory() as temp_dir:
                # path_to_request = Path(temp_dir) / "request.json"
                # path_to_output = Path(temp_dir) / "output.jsonl"
                # self.serialize_for_evaluation(problems, predictions, path_to_request)
                # path_to_request = str(path_to_request)
                # path_to_output = str(path_to_output)
                # self.run_evaluation_from_cli(path_to_request, path_to_output)
                # return self.load_evaluation_results(path_to_output)
                return self.run_evaluation_from_problems_and_predictions(
                    problems, predictions, Path(temp_dir)
                )
        else:
            working_directory = Path(f"{self.__class__.__name__}_workspace")
            working_directory.mkdir(parents=True, exist_ok=True)
            return self.run_evaluation_from_problems_and_predictions(
                problems, predictions, working_directory
            )


class LiveCodeBenchTask:
    def __init__(
        self,
        split: LiveCodeBenchSplitChoices,
        test_case_parallelism: int = 1,
        debug_scoring: bool = False,
    ):
        """
        A class representing the LiveCodeBench task for code generation evaluation.

        This class handles loading, evaluating, and reporting performance on the
        LiveCodeBench dataset for code generation tasks.

        Parameters
        ----------
        split : LiveCodeBenchSplitChoices
            The dataset split to use ('val', 'test', or 'debug').
        test_case_parallelism : int, optional
            The number of parallel processes to use for test case evaluation (default is 4).
            This needs to be >= 1 otherwise downstream code will throw an error.
        """

        self.split: LiveCodeBenchSplitChoices = split
        self.problems = sorted(load_split(split), key=lambda x: x.question_id)
        self.solutions = load_solutions()
        self.task_instances = [
            code_generation_problem_to_task_instance(
                p, self.solutions.get(p.question_id)
            )
            for p in self.problems
        ]
        # Log the number of questions with solutions.
        num_questions_with_solutions = sum(
            1 for p in self.problems if p.question_id in self.solutions
        )
        logger.info(
            f"Loaded {len(self.problems)} problems, {num_questions_with_solutions} of which had solutions."
        )
        self.question_id_to_problem = {p.question_id: p for p in self.problems}
        self.test_case_parallelism = test_case_parallelism
        self.debug_scoring = debug_scoring

    def evaluate(
        self, predictor: PredictorInterface[CodeGenerationTaskInstance]
    ) -> Collection[CodeGenerationCompletedTaskInstance]:
        predictions = predictor.predict(self.task_instances)
        scorables: list[CodeGenerationScorable] = []
        for problem, prediction in zip(self.problems, predictions):
            scorables.append(
                CodeGenerationScorable(
                    question_id=problem.question_id,
                    # To save time, we only use the first prediction so
                    # only pass@1 is computed.
                    code_list=[prediction],
                )
            )

        params = CodeGenerationEvaluationParameters(
            scorables=scorables,
            problems=self.problems,
            num_process_evaluate=self.test_case_parallelism,
        )
        # results, pass_at_k_metrics = score_code_generation(params)
        evaluation_manager = LiveCodeBenchEvaluationManager(
            self.split, use_temporary_directory=not self.debug_scoring
        )
        results = evaluation_manager(self.problems, predictions)

        completed_task_instances = [
            CodeGenerationCompletedTaskInstance(
                ulid=ULID(),
                task_instance=self.task_instances[i],
                predictor_response=results[i]["code_list"][0],
                was_correct=results[i]["pass@1"] > 0,
            )
            for i in range(len(self.problems))
        ]

        return completed_task_instances

    def generate_performance_report(
        self, completed_task_instances: Collection[CodeGenerationCompletedTaskInstance]
    ) -> TaskPerformanceReport:
        accuracy = sum(
            [
                x.was_correct
                for x in completed_task_instances
                if x.was_correct is not None
            ]
        ) / len(completed_task_instances)

        # Calculate performance for each difficulty slice
        difficulty_slices = {}
        for difficulty in Difficulty:
            instances = [
                x
                for x in completed_task_instances
                if self.question_id_to_problem[x.task_instance.instance_id].difficulty
                is difficulty
            ]
            if instances:
                slice_accuracy = sum(
                    [x.was_correct for x in instances if x.was_correct]
                ) / len(instances)
                difficulty_slices[difficulty.value] = TaskSlicePerformance(
                    slice_name="difficulty",
                    slice_relname=difficulty.value,
                    metric_name="accuracy",
                    metric_value=slice_accuracy,
                    count=len(instances),
                )

        slices = list(difficulty_slices.values())

        return TaskPerformanceReport(
            task_name="LiveCodeBench",
            overall_performance=accuracy,
            slices=slices,
        )
