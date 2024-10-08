from ..benchmarks import load_code_generation_dataset, CodeGenerationProblem
from ..benchmarks.code_generation import LcbDataDictWithEvaluation, LcbDataDict
from typing import Optional
from pydantic import BaseModel
from typing_extensions import TypedDict
from ..runner.scenario_router import (
    sort_and_extract_save_results,
    get_metrics,
)
import os
import json

from .parser import get_args
from ..utils.scenarios import Scenario
from ..utils.path_utils import get_output_path
from ..evaluation import extract_instance_results, codegen_metrics
from ..evaluation.pass_k_utils import estimate_pass_at_k
from .scenario_router import (
    build_prompt_benchmark,
    sort_and_extract_save_results,
    get_metrics,
)
from datetime import datetime
from icecream import ic


def build_benchmark(release_version: str = "release_v1") -> list[CodeGenerationProblem]:
    dataset = load_code_generation_dataset(release_version)
    return dataset


def build_validation_set() -> list[CodeGenerationProblem]:
    v1 = load_code_generation_dataset("release_v1")
    v2 = load_code_generation_dataset("release_v2")

    # Validation set is the set of problems that are in v2 but not v1.
    v1_qids = set([x.question_id for x in v1])
    validation_set = [x for x in v2 if x.question_id not in v1_qids]
    import ipdb

    ipdb.set_trace()
    return validation_set


class CodeGenerationScorable(TypedDict):
    question_id: str
    code_list: list[str]


class CodeGenerationEvaluationParameters(BaseModel):
    # release_version: str = "release_v1"
    scorables: list[CodeGenerationScorable]
    problems: list[CodeGenerationProblem]
    num_process_evaluate: int = 8
    timeout: int = 5
    scenario: Scenario = Scenario.codegeneration


class PassAtKMetrics(BaseModel):
    pass_at_k: dict[int, float]
    easy_pass_at_k: dict[int, float]
    medium_pass_at_k: dict[int, float]
    hard_pass_at_k: dict[int, float]
    pass_1: float
    easy_pass_1: Optional[float]
    medium_pass_1: Optional[float]
    hard_pass_1: Optional[float]


def score_code_generation(
    params: CodeGenerationEvaluationParameters,
) -> tuple[list[LcbDataDictWithEvaluation], PassAtKMetrics]:
    benchmark: list[CodeGenerationProblem] = params.problems

    solutions_sorted_by_qid = [
        _["code_list"] for _ in sorted(params.scorables, key=lambda x: x["question_id"])
    ]

    save_results = [
        instance.insert_output(solutions, solutions)
        for instance, solutions in zip(benchmark, solutions_sorted_by_qid)
    ]

    save_results = sorted(save_results, key=lambda x: x["question_id"])
    combined_results = [
        (save_result_instance["output_list"], save_result_instance["code_list"])
        for save_result_instance in save_results
    ]

    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=params.num_process_evaluate,
        timeout=params.timeout,
    )

    graded = extract_instance_results(metrics[1])

    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    pass_at_k_metrics = compute_scores(save_eval_results)

    return save_eval_results, pass_at_k_metrics


def compute_scores(
    results: list[LcbDataDictWithEvaluation],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platform: Optional[str] = None,
) -> PassAtKMetrics:
    for res in results:
        res["contest_date"] = datetime.fromisoformat(res["contest_date"])

    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        results = [result for result in results if start_date <= result["contest_date"]]

    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        results = [result for result in results if result["contest_date"] <= end_date]

    if platform is not None:
        results = [result for result in results if result["platform"] == platform]

    totals = [len(x["graded_list"]) for x in results]
    corrects = [sum(x["graded_list"]) for x in results]

    easy_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    easy_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "easy"
    ]
    med_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "medium"
    ]
    hard_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "hard"
    ]

    pass_at_k = {}
    easy_pass_at_k = {}
    medium_pass_at_k = {}
    hard_pass_at_k = {}

    for k in [1, 5, 10, 25, 50, 100, 150, 200]:
        pass_at_k[k] = estimate_pass_at_k(totals, corrects, k).mean()
        easy_pass_at_k[k] = estimate_pass_at_k(easy_totals, easy_corrects, k).mean()
        medium_pass_at_k[k] = estimate_pass_at_k(med_totals, med_corrects, k).mean()
        hard_pass_at_k[k] = estimate_pass_at_k(hard_totals, hard_corrects, k).mean()

    pass_1_list = [result["pass@1"] for result in results]
    pass_1 = sum(pass_1_list) / len(pass_1_list)

    easy_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "easy"
    ]
    easy_pass_1 = (
        sum(easy_pass_1_list) / len(easy_pass_1_list) if easy_pass_1_list else None
    )

    medium_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "medium"
    ]
    medium_pass_1 = (
        sum(medium_pass_1_list) / len(medium_pass_1_list)
        if medium_pass_1_list
        else None
    )

    hard_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "hard"
    ]
    hard_pass_1 = (
        sum(hard_pass_1_list) / len(hard_pass_1_list) if hard_pass_1_list else None
    )

    return PassAtKMetrics(
        pass_at_k=pass_at_k,
        easy_pass_at_k=easy_pass_at_k,
        medium_pass_at_k=medium_pass_at_k,
        hard_pass_at_k=hard_pass_at_k,
        pass_1=pass_1,
        easy_pass_1=easy_pass_1,
        medium_pass_1=medium_pass_1,
        hard_pass_1=hard_pass_1,
    )


if __name__ == "__main__":
    # python3 -m envgenpp.gym.tasks.code.livecodebench.runner.codegen_evaluator
    with open("workspace/lcb_scorables.json", "r") as f:
        scorables: list[CodeGenerationScorable] = json.load(f)

    problems = build_benchmark()

    # Truncate the number of programs to only 1 for each scorable.
    scorables = [
        CodeGenerationScorable(
            question_id=scorable["question_id"], code_list=[scorable["code_list"][0]]
        )
        for scorable in scorables
    ]

    # Truncate both to only 100 problems.
    problems = problems[:100]
    scorables = scorables[:100]

    params = CodeGenerationEvaluationParameters(scorables=scorables, problems=problems)
    score_code_generation(params)
