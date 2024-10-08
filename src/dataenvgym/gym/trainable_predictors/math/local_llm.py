import ray
from loguru import logger
from loguru._logger import Logger
from transformers import LlamaForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch
from typing import cast
from dataenvgym.gym.domain_models import (
    MathTaskInstance,
    MathTrainingDatum,
    implements,
    MathPredictorInterface,
    MathTrainerInterface,
    TrainableMathPredictorInterface,
)
from dataenvgym.utils import PydanticJSONLinesReader
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Sequence
from dataenvgym.gym.tasks.math.MATH.task import prepare_math_prompt
from tqdm.auto import tqdm
from typing import Optional
from dataenvgym.llama_factory_utils import (
    format_records_for_llama_factory_sft,
    generate_llama_factory_cli_args,
    run_training_with_llama_factory,
)
from pathlib import Path
import shutil
from typing import Callable
import tempfile
from typing import Sequence, Type, Optional, Callable

import ray

from dataenvgym.gym.domain_models import MathTaskInstance, MathPredictorInterface
from dataenvgym.utils import PydanticJSONLinesReader, PydanticJSONLinesWriter
from typing import TypeGuard, Any, TypeVar
from dataclasses import dataclass
from pydantic import BaseModel
from ray.experimental.tqdm_ray import tqdm as ray_tqdm


class PredictionItem(BaseModel):
    instance_id: str
    prediction: str


@dataclass
class RayMathPredictionWorkerConfig:
    model_name_or_path: str
    prompt_formatter: Optional[Callable[[MathTaskInstance], str]]
    sampling_params: SamplingParams
    always_apply_chat_template: bool
    prompt_formatter_for_base_model: Optional[Callable[[MathTaskInstance], str]]
    batch_size: int
    max_model_len: int = 4096


@ray.remote(num_gpus=1)
class RayMathPredictionWorker:
    def __init__(
        self,
        config: RayMathPredictionWorkerConfig,
        worker_index: int,
    ):
        self.model_name_or_path = config.model_name_or_path
        self.prompt_formatter = config.prompt_formatter
        self.sampling_params = config.sampling_params
        self.always_apply_chat_template = config.always_apply_chat_template
        self.prompt_formatter_for_base_model = config.prompt_formatter_for_base_model
        self.batch_size = config.batch_size
        self.max_model_len = config.max_model_len
        self.llm = None
        self.tokenizer = None
        self.lora_request: Optional[LoRARequest] = None
        self.worker_index = worker_index

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.llm = LLM(
            model=self.model_name_or_path,
            enable_lora=True,
            max_model_len=self.max_model_len,
        )

    def load_adapter(self, adapter_path: str):
        logger.info(f"Loading adapter from {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.lora_request = LoRARequest(
            lora_name=adapter_path,
            lora_int_id=1,
            lora_local_path=adapter_path,
        )

    def format_task_instance_into_prompt(self, task_instance: MathTaskInstance) -> str:
        assert self.tokenizer is not None

        # We assume that if we have a LoRA request, we are using a fine-tuned model
        # otherwise we are using a base model.
        using_base_model = self.lora_request is None
        # If we are using a base model we will use a prompt formatter if a prompt
        # formatter is specified for it, otherwise we will use the instruction as the
        # prompt.
        have_base_model_prompt_formatter = (
            self.prompt_formatter_for_base_model is not None
        )
        # If we are using a fine-tuned model we will use a prompt formatter if a
        # prompt formatter is specified for it, otherwise we will use the instruction
        # as the prompt.
        have_finetuned_model_prompt_formatter = self.prompt_formatter is not None

        if using_base_model and have_base_model_prompt_formatter:
            assert self.prompt_formatter_for_base_model is not None
            prompt = self.prompt_formatter_for_base_model(task_instance)
        elif using_base_model and not have_base_model_prompt_formatter:
            prompt = task_instance.instruction
        elif not using_base_model and have_finetuned_model_prompt_formatter:
            assert self.prompt_formatter is not None
            prompt = self.prompt_formatter(task_instance)
        elif not using_base_model and not have_finetuned_model_prompt_formatter:
            prompt = task_instance.instruction

        # Now we decide whether to apply the chat template.
        # We always apply a chat template if we have a LoRA request.
        # We also apply a chat template if self.always_apply_chat_template is True.
        if self.lora_request or self.always_apply_chat_template:
            # HACK: If we FT a base model, the chat template is not available,
            # even though we do in fact have the chat template.
            if (
                self.tokenizer.chat_template is None
                and "gemma" in self.model_name_or_path
            ):
                self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            elif (
                self.tokenizer.chat_template is None
                and "qwen" in self.model_name_or_path.lower()
                and "math" in self.model_name_or_path.lower()
            ):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-Math-1.5B-Instruct"
                )
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            )
        return prompt

    def predict(self, task_instances_path: str, output_path: str):
        if self.llm is None or self.tokenizer is None:
            self._load_model()

        assert self.llm is not None
        assert self.tokenizer is not None

        task_instances = list(
            PydanticJSONLinesReader(task_instances_path, MathTaskInstance)()
        )

        logger.info(
            f"Loaded {len(task_instances)} task instances from {task_instances_path}"
        )

        prompts = [
            self.format_task_instance_into_prompt(task_instance)
            for task_instance in task_instances
        ]

        predictions = []
        for i in ray_tqdm(range(0, len(prompts), self.batch_size), desc="Predicting"):
            batch_prompts = prompts[i : i + self.batch_size]
            request_outputs = self.llm.generate(
                batch_prompts,
                self.sampling_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )
            for _ in request_outputs:
                for output in _.outputs:
                    predictions.append(output.text)

        logger.info(f"Writing {len(predictions)} predictions to {output_path}")

        PydanticJSONLinesWriter(output_path).write_batch(
            [
                PredictionItem(
                    instance_id=task_instance.instance_id, prediction=prediction
                )
                for task_instance, prediction in zip(task_instances, predictions)
            ]
        )

    def __repr__(self):
        return f"RayMathPredictionWorker(worker_index={self.worker_index})"


class ParallelLlmPredictor(MathPredictorInterface):
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        num_workers: int = 4,
        prompt_formatter: Optional[Callable[[MathTaskInstance], str]] = None,
        sampling_params: SamplingParams = SamplingParams(
            temperature=0.8, max_tokens=350
        ),
        always_apply_chat_template: bool = False,
        prompt_formatter_for_base_model: Optional[
            Callable[[MathTaskInstance], str]
        ] = None,
        batch_size: int = 4,
        max_model_len: int = 4096,
    ):
        self.model_name_or_path = model_name_or_path
        self.prompt_formatter = prompt_formatter
        self.sampling_params = sampling_params
        self.always_apply_chat_template = always_apply_chat_template
        self.prompt_formatter_for_base_model = prompt_formatter_for_base_model
        self.batch_size = batch_size
        # We use a config object to pass parameters because the ray wrapper breaks
        # type checking, so we type check here and then pass in the config object.
        self.worker_config = RayMathPredictionWorkerConfig(
            model_name_or_path=model_name_or_path,
            prompt_formatter=prompt_formatter,
            sampling_params=sampling_params,
            always_apply_chat_template=always_apply_chat_template,
            prompt_formatter_for_base_model=prompt_formatter_for_base_model,
            batch_size=batch_size,
            max_model_len=max_model_len,
        )
        self.num_workers = num_workers
        self.workers = None
        self._create_workers_if_needed()

    def _create_workers_if_needed(self):
        if self.workers is None:
            logger.info(f"Creating {self.num_workers} workers")
            self.workers = [
                RayMathPredictionWorker.remote(
                    config=self.worker_config, worker_index=i  # type: ignore
                )
                for i in range(self.num_workers)
            ]
        else:
            logger.warning("Workers already created, skipping creation.")

    def load_adapter(self, adapter_path: str):
        self._create_workers_if_needed()

        assert self.workers is not None

        ray.get(
            [
                worker.load_adapter.remote(adapter_path=adapter_path)
                for worker in self.workers
            ]
        )

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        self._create_workers_if_needed()
        assert self.workers is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            input_paths = []
            output_paths = []
            worker_chunks = [[] for _ in range(self.num_workers)]
            for i, task_instance in enumerate(task_instances):
                worker_chunks[i % self.num_workers].append(task_instance)

            for i, chunk in enumerate(worker_chunks):
                input_path = f"{tmpdir}/input_{i}.jsonl"
                output_path = f"{tmpdir}/output_{i}.jsonl"
                PydanticJSONLinesWriter(input_path).write_batch(chunk)
                input_paths.append(input_path)
                output_paths.append(output_path)

            # Print some stats about the distribution of instances across workers.
            for i, worker_chunk in enumerate(worker_chunks):
                logger.info(
                    f"Worker {i} has {len(worker_chunk)} instances",
                )

            ray.get(
                [
                    worker.predict.remote(input_path, output_path)
                    for worker, input_path, output_path, worker_chunk in zip(
                        self.workers, input_paths, output_paths, worker_chunks
                    )
                    if len(worker_chunk) > 0
                ]
            )

            predictions_by_id: dict[str, str] = {}
            for output_path in output_paths:
                reader = PydanticJSONLinesReader(output_path, model=PredictionItem)
                predictions = list(reader())
                logger.info(f"Read {len(predictions)} predictions from {output_path}")
                for prediction in predictions:
                    predictions_by_id[prediction.instance_id] = prediction.prediction

            assert len(predictions_by_id) == len(task_instances)

            # Put the predictions in the same order as the instances.
            return [
                predictions_by_id[task_instance.instance_id]
                for task_instance in task_instances
            ]

    def reset(self):
        if self.workers is not None:
            logger.info(f"Destroying {len(self.workers)} workers.")
            for worker in self.workers:
                ray.kill(worker)
            self.workers = None
        else:
            logger.warning("Workers not created, skipping reset.")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Will always take GPU 0!
class LlamaPredictor:
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_formatter: Optional[Callable[[MathTaskInstance], str]] = None,
        sampling_params: SamplingParams = SamplingParams(
            temperature=0.8, max_tokens=350
        ),
        always_apply_chat_template: bool = False,
        prompt_formatter_for_base_model: Optional[
            Callable[[MathTaskInstance], str]
        ] = None,
        batch_size: int = 4,
    ):
        self.model_name_or_path = model_name_or_path
        self.llm = None
        self.tokenizer = None
        self.sampling_params = sampling_params
        self.batch_size = batch_size
        self.prompt_formatter = prompt_formatter
        self.prompt_formatter_for_untrained_model = prompt_formatter_for_base_model
        # If set, will be provided to vLLM as a LoRARequest.
        self.lora_request: Optional[LoRARequest] = None
        self.always_apply_chat_template = always_apply_chat_template

    def _load_model(self):
        self.tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(self.model_name_or_path),
        )
        # max_model_len needs to be set ~90k on an A6000 for the 8B model, ow
        # it will run out of space due to the large possible context of llama3-8b
        # https://github.com/vllm-project/vllm/issues/6689#issuecomment-2272027846
        self.llm = LLM(
            model=self.model_name_or_path, enable_lora=True, max_model_len=4096
        )

    def load_adapter(self, adapter_path: str):
        logger.info(f"Loading adapter from {adapter_path}")
        # Finetuning may add new tokens to the tokenizer, so we need to reload it
        self.tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(adapter_path),
        )
        # This will be supplied to vLLM.
        self.lora_request = LoRARequest(
            lora_name=adapter_path,
            lora_int_id=1,
            lora_local_path=adapter_path,
        )

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        if self.llm is None or self.tokenizer is None:
            self._load_model()

        assert self.llm is not None
        assert self.tokenizer is not None

        predictions = []
        for i in tqdm(
            range(0, len(task_instances), self.batch_size), desc="Predicting"
        ):
            # TODO: Move this processing out of the loop so we can log what
            # decisions are being made without spamming the logs.
            batch = task_instances[i : i + self.batch_size]
            if self.prompt_formatter:
                prompts = [
                    self.prompt_formatter(task_instance) for task_instance in batch
                ]
            elif (
                self.prompt_formatter_for_untrained_model and self.lora_request is None
            ):
                prompts = [
                    self.prompt_formatter_for_untrained_model(task_instance)
                    for task_instance in batch
                ]
            else:
                prompts = [_.instruction for _ in batch]

            prompts = cast(list[str], prompts)

            if self.lora_request or self.always_apply_chat_template:
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        tokenize=False,
                    )
                    for prompt in prompts
                ]
                prompts = cast(list[str], prompts)

            request_outputs = self.llm.generate(
                prompts,
                self.sampling_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )
            for _ in request_outputs:
                for output in _.outputs:
                    predictions.append(output.text)

        return predictions


implements(MathPredictorInterface)(LlamaPredictor)


class LlamaFactoryTrainer:
    def __init__(
        self,
        working_directory: Path,
        cuda_visible_devices: Optional[list[int]] = None,
        overrides: Optional[list[str]] = None,
    ) -> None:
        self.working_directory = working_directory
        self.dataset_dir = working_directory / "llama_factory_dataset_dir"
        self.output_dir = working_directory / "llama_factory_output_dir"
        self.config_output_path = working_directory / "llama_factory_config.yaml"
        self.cuda_visible_devices = cuda_visible_devices
        self.overrides = overrides or []
        for _ in self.overrides:
            assert not _.startswith("+dataset_dir=")
            assert not _.startswith("output_dir=")
            assert not _.startswith("dataset=")

    def get_weights_path(self) -> Path:
        return self.output_dir

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        instruction_key = "instruction"
        response_key = "response"

        # Check that the instruction and response keys are present in the training data.
        # Don't error out if there's no data.
        for _ in training_data:
            assert instruction_key in _.model_dump()
            assert response_key in _.model_dump()
            break

        sft_spec, _ = format_records_for_llama_factory_sft(
            [_.model_dump() for _ in training_data],
            llama_factory_dataset_dir=str(self.dataset_dir),
            instruction_key=instruction_key,
            response_key=response_key,
            overwrite=True,
        )

        generate_llama_factory_cli_args(
            overrides=[
                f"+dataset_dir={self.dataset_dir}",
                f"output_dir={self.output_dir}",
                f"dataset={sft_spec.dataset_name}",
            ]
            + self.overrides,
            output_path=self.config_output_path,
        )

        run_training_with_llama_factory(
            str(self.config_output_path), self.cuda_visible_devices
        )


implements(MathTrainerInterface)(LlamaFactoryTrainer)


class LlamaTrainablePredictor:
    def __init__(
        self, llama_predictor: LlamaPredictor, llama_trainer: LlamaFactoryTrainer
    ) -> None:
        self.llama_predictor = llama_predictor
        self.llama_trainer = llama_trainer

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        self.llama_trainer.train(training_data)
        self.llama_predictor.load_adapter(str(self.llama_trainer.get_weights_path()))

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return self.llama_predictor.predict(task_instances)

    def save(self, path: Path) -> None:
        # The adapter is saved in the llama_trainer, so we don't need to save anything here.
        # Instead, we copy the output dir to the specified path.
        shutil.copytree(self.llama_trainer.get_weights_path(), path)


# Note: When using this, do not set the cuda_visible_devices flag on the trainer.
# Set it outside the python process.
class ParallelLlmTrainablePredictor:
    def __init__(
        self, llama_predictor: ParallelLlmPredictor, llama_trainer: LlamaFactoryTrainer
    ):
        self.llama_predictor = llama_predictor
        self.llama_trainer = llama_trainer

    def train(self, training_data: Sequence[MathTrainingDatum]) -> None:
        # Before any training, we have to make sure the llama predictor is not
        # use any GPU memory so the trainer can run at full parallelism.
        self.llama_predictor.reset()
        # At this point, the predictor will have no memory allocated.
        # Memory will only be allocated when we call .predict() on the predictor
        # next.
        self.llama_trainer.train(training_data)
        # The trainer exits after training, so all memory is free again.
        self.llama_predictor.load_adapter(str(self.llama_trainer.get_weights_path()))
        # At this point, this is _still_ no memory allocated until the first call
        # to .predict().

    def predict(self, task_instances: Sequence[MathTaskInstance]) -> list[str]:
        return self.llama_predictor.predict(task_instances)

    def save(self, path: Path) -> None:
        # The adapter is saved in the llama_trainer, so we don't need to save anything here.
        # Instead, we copy the output dir to the specified path.
        shutil.copytree(self.llama_trainer.get_weights_path(), path)


implements(TrainableMathPredictorInterface)(LlamaTrainablePredictor)
