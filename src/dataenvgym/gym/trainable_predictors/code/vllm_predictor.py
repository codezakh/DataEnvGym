from ..common import (
    ParallelVllmPredictorConfig,
    ParallelVllmPredictor,
    LlamaFactoryWrapper,
    LlamaFactoryWrapperConfig,
    ParallelSftTrainablePredictor,
)
from ...domain_models import (
    CodeGenerationTaskInstance,
    CodeGenerationPredictorInterface,
    CodeGenerationTrainingDatum,
    CodeGenerationDataSpec,
    implements,
    CodeGenerationTrainerInterface,
    TrainableCodeGenerationPredictorInterface,
)
from dataenvgym.gym.tasks.code.livecodebench.prompts.code_generation import (
    format_prompt_generation,
    PromptConstants,
)
from typing_extensions import TypeAlias
from typing import Sequence
import jinja2
from typing import Optional
from vllm import SamplingParams
from dataenvgym.gym.tasks.code.livecodebench.lm_styles import LMStyle
from dataenvgym.gym.tasks.code.livecodebench.utils.extraction_utils import extract_code
from functools import partial
from pathlib import Path
import numpy as np


# NOTE: This prompt formatting is copied from the official livecodebench repository.
# LCB defines a prompt style for every model, but emits the formatted prompt as
# conversation. We need instruction-response pairs since we are doing instruct tuning.
def get_inference_prompt_from_instruction_and_starter_code(
    instruction: str, starter_code: Optional[str]
) -> str:
    prompt = f"### Question:\n{instruction}\n\n"
    if starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_inference_prompt_from_task_instance(
    task_instance: CodeGenerationTaskInstance,
) -> str:
    return get_inference_prompt_from_instruction_and_starter_code(
        task_instance.instruction, task_instance.starter_code
    )


def render_data_spec(data_spec: CodeGenerationDataSpec) -> CodeGenerationTrainingDatum:
    """
    There are some constraints on the data spec for this to work properly.

    The `solution` should be formatted in a way that extract_code can parse it. The model
    will be trained to output the solution, so it needs to be parseable by extract_code.
    You can look at get_inference_prompt_from_instruction_and_starter_code to see what the
    expected output format is.
    """
    inference_prompt = get_inference_prompt_from_instruction_and_starter_code(
        data_spec.instruction, data_spec.starter_code
    )
    return CodeGenerationTrainingDatum(
        ulid=data_spec.ulid,
        instruction=inference_prompt,
        response=data_spec.solution,
    )


def convert_task_instances_to_training_data(
    task_instances: Sequence[CodeGenerationTaskInstance],
) -> Sequence[CodeGenerationTrainingDatum]:
    """
    This is a debugging utility so we can train on the test / val set and confirm that the
    training is in fact working.
    """
    data_specs: list[CodeGenerationDataSpec] = []
    for task_instance in task_instances:
        if task_instance.solution is not None:
            data_specs.append(
                CodeGenerationDataSpec(
                    instruction=task_instance.instruction,
                    starter_code=task_instance.starter_code,
                    solution=task_instance.solution,
                )
            )
    return [render_data_spec(data_spec) for data_spec in data_specs]


def extract_code_from_prediction(prediction: str) -> str:
    return extract_code(prediction, LMStyle.LLaMa3)


def get_token_count_summaries_for_training_data(
    training_data: Sequence[CodeGenerationTrainingDatum],
) -> dict[str, dict[str, int | float]]:
    from transformers import AutoTokenizer

    # Use this as the "default" tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    response_token_counts: list[int] = []
    instruction_token_counts: list[int] = []
    total_token_counts: list[int] = []

    for datum in training_data:
        response_tokens = len(tokenizer.encode(datum.response))
        instruction_tokens = len(tokenizer.encode(datum.instruction))
        total_tokens = response_tokens + instruction_tokens

        response_token_counts.append(response_tokens)
        instruction_token_counts.append(instruction_tokens)
        total_token_counts.append(total_tokens)

    def get_summary(counts: list[int]) -> dict[str, int | float]:
        return {
            "Min": int(np.min(counts)),
            "Max": int(np.max(counts)),
            "25th percentile": float(np.percentile(counts, 25)),
            "50th percentile": float(np.percentile(counts, 50)),
            "75th percentile": float(np.percentile(counts, 75)),
            "90th percentile": float(np.percentile(counts, 90)),
            "95th percentile": float(np.percentile(counts, 95)),
            "99th percentile": float(np.percentile(counts, 99)),
        }

    token_count_summary = {
        "Response": get_summary(response_token_counts),
        "Instruction": get_summary(instruction_token_counts),
        "Total": get_summary(total_token_counts),
    }

    return token_count_summary


LLAMA3_8B_INSTRUCT_INFERENCE_CONFIG = ParallelVllmPredictorConfig(
    task_instance_type=CodeGenerationTaskInstance,
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    num_workers=8,
    prompt_formatter=get_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2000),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=get_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)

GEMMA2_2B_INSTRUCT_INFERENCE_CONFIG = ParallelVllmPredictorConfig(
    task_instance_type=CodeGenerationTaskInstance,
    model_name_or_path="google/gemma-2-2b-it",
    num_workers=8,
    prompt_formatter=get_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2000),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=get_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)

QWEN2_1_5B_INSTRUCT_INFERENCE_CONFIG = ParallelVllmPredictorConfig(
    task_instance_type=CodeGenerationTaskInstance,
    model_name_or_path="Qwen/Qwen2-1.5B-Instruct",
    num_workers=8,
    prompt_formatter=get_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2000),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=get_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)

QWEN2_7B_INSTRUCT_INFERENCE_CONFIG = ParallelVllmPredictorConfig(
    task_instance_type=CodeGenerationTaskInstance,
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    num_workers=8,
    prompt_formatter=get_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2000),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=get_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)

GEMMA2_2B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("gemma2_2b_instruct_trainer"),
    model_name_or_path="google/gemma-2-2b-it",
    template="gemma",
)

LLAMA3_8B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("llama3_8b_instruct_trainer"),
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    template="llama3",
)

QWEN2_1_5B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("qwen2_1_5b_instruct_trainer"),
    model_name_or_path="Qwen/Qwen2-1.5B-Instruct",
    template="qwen",
)

QWEN2_7B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("qwen2_7b_instruct_trainer"),
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    template="qwen",
)


ParallelVllmCodeGenerationPredictor = ParallelVllmPredictor[CodeGenerationTaskInstance]
# This is just for type checking, it doesn't do anything at runtime.
implements(CodeGenerationPredictorInterface)(ParallelVllmCodeGenerationPredictor)


SftCodeGenerationTrainer = LlamaFactoryWrapper[CodeGenerationTrainingDatum]
# Also just for type checking, it doesn't do anything at runtime.
implements(CodeGenerationTrainerInterface)(SftCodeGenerationTrainer)

CodeGenerationTrainablePredictor = ParallelSftTrainablePredictor[
    CodeGenerationTrainingDatum, CodeGenerationTaskInstance
]

implements(TrainableCodeGenerationPredictorInterface)(CodeGenerationTrainablePredictor)
