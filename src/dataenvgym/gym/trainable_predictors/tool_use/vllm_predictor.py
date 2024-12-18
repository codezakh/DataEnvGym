from dataenvgym.utils import extract_code_in_markdown_backticks
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
from typing import Sequence
from typing import Optional
from vllm import SamplingParams
from pathlib import Path
import numpy as np
from dataenvgym.gym.tasks.tool_use.mnms.prompts import (
    make_single_turn_inference_prompt_from_task_instance,
    make_single_turn_inference_prompt_from_instruction,
)
from functools import partial


def render_data_spec(data_spec: CodeGenerationDataSpec) -> CodeGenerationTrainingDatum:
    """
    Formats a data specification into training data.

    Uses the same prompt as the single turn inference prompt so that there is
    continuity between training and inference.
    """
    inference_prompt = make_single_turn_inference_prompt_from_instruction(
        data_spec.instruction
    )
    return CodeGenerationTrainingDatum(
        ulid=data_spec.ulid,
        instruction=inference_prompt,
        response=f"```python\n{data_spec.solution}\n```",
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
    return extract_code_in_markdown_backticks(prediction)


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
    # Don't include in-context examples in the inference prompt for the trained model.
    prompt_formatter=make_single_turn_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=make_single_turn_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)

GEMMA2_2B_INSTRUCT_INFERENCE_CONFIG = ParallelVllmPredictorConfig(
    task_instance_type=CodeGenerationTaskInstance,
    model_name_or_path="google/gemma-2-2b-it",
    num_workers=8,
    # Don't include in-context examples in the inference prompt for the trained model.
    prompt_formatter=make_single_turn_inference_prompt_from_task_instance,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=256),
    always_apply_chat_template=True,
    prompt_formatter_for_base_model=make_single_turn_inference_prompt_from_task_instance,
    postprocess_prediction=extract_code_from_prediction,
)


GEMMA2_2B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("gemma2_2b_instruct_trainer"),
    model_name_or_path="google/gemma-2-2b-it",
    template="gemma",
    overrides=["cutoff_len=1250"],
)

LLAMA3_8B_INSTRUCT_TRAINER_CONFIG = LlamaFactoryWrapperConfig(
    working_directory=Path("llama3_8b_instruct_trainer"),
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    template="llama3",
    overrides=["cutoff_len=1250"],
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
