from typing import Callable, Union, cast

from icecream import ic
from pydantic import BaseModel

from ..benchmarks import (
    CodeExecutionProblem,
    TestOutputPredictionProblem,
    load_code_generation_dataset,
    CodeGenerationProblem,
)
from ..lm_styles import LanguageModelStore
from ..prompts import (
    format_prompt_generation,
)
from ..runner.azure_openai_runner import OpenAiRunner
from .codegen_evaluator import (
    score_code_generation,
    CodeGenerationScorable,
    CodeGenerationEvaluationParameters,
)
from ..utils.extraction_utils import extract_code

# BenchMarkType = list[CodeGenerationProblem | TestOutputPredictionProblem]
BenchMarkType = list[
    Union[CodeGenerationProblem, CodeExecutionProblem, TestOutputPredictionProblem]
]


def build_prompt_benchmark(
    release_version: str,
) -> tuple[
    list[CodeExecutionProblem]
    | list[CodeGenerationProblem]
    | list[TestOutputPredictionProblem],
    Callable,
]:
    benchmark = load_code_generation_dataset(release_version)
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    format_prompt = format_prompt_generation
    return benchmark, format_prompt


# def build_runner(args, model: LanguageModel):
#     if model.model_style == LMStyle.OpenAIChat:
#         from .oai_runner import OpenAIRunner

#         return OpenAIRunner(args, model)
#     # if model.model_style == LMStyle.Gemini:
#     #     from lcb_runner.runner.gemini_runner import GeminiRunner

#     #     return GeminiRunner(args, model)
#     # if model.model_style == LMStyle.Claude3:
#     #     from lcb_runner.runner.claude3_runner import Claude3Runner

#     #     return Claude3Runner(args, model)
#     # if model.model_style == LMStyle.Claude:
#     #     from lcb_runner.runner.claude_runner import ClaudeRunner

#     #     return ClaudeRunner(args, model)
#     # if model.model_style == LMStyle.MistralWeb:
#     #     from lcb_runner.runner.mistral_runner import MistralRunner

#     #     return MistralRunner(args, model)
#     # if model.model_style == LMStyle.CohereCommand:
#     #     from lcb_runner.runner.cohere_runner import CohereRunner

#     #     return CohereRunner(args, model)
#     # if model.model_style == LMStyle.DeepSeekAPI:
#     #     from lcb_runner.runner.deepseek_runner import DeepSeekRunner

#     #     return DeepSeekRunner(args, model)
#     elif model.model_style in []:
#         raise NotImplementedError(
#             f"Runner for language model style {model.model_style} not implemented yet"
#         )
#     else:
#         from .vllm_runner import VLLMRunner

#         return VLLMRunner(args, model)


class RunnerParams(BaseModel):
    model: str
    release_version: str
    debug: bool
    multiprocess: int = 10


def main(params: RunnerParams):
    model = LanguageModelStore[params.model]
    benchmark, format_prompt = build_prompt_benchmark(params.release_version)
    if params.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:5]

    benchmark = cast(list[CodeGenerationProblem], benchmark)

    runner = OpenAiRunner(model, multiprocess=params.multiprocess)
    results: list[list[str]] = runner.run_main(benchmark, format_prompt)

    combined_results = [
        (
            outputs_list,
            [extract_code(output, model.model_style) for output in outputs_list],
        )
        for outputs_list in results
    ]

    save_results = [
        instance.insert_output(output_list=outputs_list, code_list=extracted_list)
        for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)
    ]

    scorables = [
        CodeGenerationScorable(question_id=_["question_id"], code_list=_["code_list"])
        for _ in save_results
    ]

    evaluation_params = CodeGenerationEvaluationParameters(
        scorables=scorables,
        problems=benchmark,
    )

    score_code_generation(evaluation_params)


if __name__ == "__main__":
    params = RunnerParams(
        model="gpt-4o-mini-2024-07-18",
        release_version="release_v1",
        debug=True,
        multiprocess=0,
    )
    main(params)
