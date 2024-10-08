import tempfile
from pathlib import Path
from typing import Collection, Sequence, cast
import os
from dataenvgym.gym.domain_models import (
    TrainableVqaPredictorInterface,
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    VqaTrainingDatum,
    VqaPreferenceTrainingDatum,
    PreferenceTrainableVqaPredictorInterface,
    implements,
)
from dataenvgym.utils import PydanticJSONLinesWriter, PydanticJSONLinesReader
from dataenvgym.gym.serializers import VqaTaskInstanceSerializer
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from tqdm.auto import tqdm
from loguru import logger
import torch
import ray
from datasets import Dataset  # type: ignore[import-untyped]
from peft.tuners.lora.config import LoraConfig  # type: ignore[import-untyped]
from peft import get_peft_model
import shutil
from trl import DPOTrainer, DPOConfig

# See notebook 007 for how we determined this.
LORA_TARGET_MODULES = [
    "v_proj",
    "linear",
    "out_proj",
    "q_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
    "k_proj",
    "o_proj",
    "fc1",
    "fc2",
]


@ray.remote(num_gpus=1)
class RayPaliGemmaPredictor:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None

    def load_adapter(self, adapter_path: str):
        if self.model is None:
            self._load_model(self.model_name_or_path)
        assert self.model is not None, "Model is not loaded"
        self.model.load_adapter(adapter_path)
        logger.info("Loaded adapter from {}", adapter_path)

    def _load_model(self, model_name_or_path: str):
        logger.info("Loading PaliGemma model for prediction")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        logger.info("PaliGemma model loaded")
        self.processor = cast(
            PaliGemmaProcessor,
            PaliGemmaProcessor.from_pretrained(self.model_name_or_path),
        )
        self.model.eval()

    def predict(self, task_instances_path: str) -> list[str]:
        if self.model is None or self.processor is None:
            self._load_model(self.model_name_or_path)

        task_instances = VqaTaskInstanceSerializer.deserialize(
            Path(task_instances_path)
        )

        assert self.processor is not None, "Processor is not loaded"
        assert self.model is not None, "Model is not loaded"

        decoded_responses = []
        for task_instance in tqdm(task_instances, desc="Predicting", unit="instance"):
            with torch.no_grad():
                model_inputs = self.processor(
                    images=task_instance.image,
                    text=task_instance.instruction,
                    return_tensors="pt",
                )
                model_inputs.to(self.model.device)
                outputs = self.model.generate(**model_inputs, max_new_tokens=100)
                input_len = model_inputs["input_ids"].shape[-1]
                generated_tokens = outputs[0][input_len:]
                decoded_response = self.processor.decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_responses.append(decoded_response)
        return decoded_responses


@ray.remote(num_gpus=1)
class RayPaliGemmaTrainer:
    def __init__(
        self, model_name_or_path: str, device: str, trainer_output_dir: str | Path
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.trainer_output_dir = trainer_output_dir
        self.model = None
        self.processor = None
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
            modules_to_save=None,
        )
        self.peft_model = None

    def _load_model(self):
        logger.info("Loading PaliGemma model for training")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.peft_model = get_peft_model(self.model, self.peft_config)
        logger.info("PaliGemma model loaded")
        self.processor = cast(
            PaliGemmaProcessor,
            PaliGemmaProcessor.from_pretrained(self.model_name_or_path),
        )
        # for param in self.model.vision_tower.parameters():
        #     param.requires_grad = False
        # for param in self.model.multi_modal_projector.parameters():
        #     param.requires_grad = False

    def _collate_fn(self, examples):
        if self.processor is None:
            raise ValueError("Processor is not loaded")

        texts = ["answer " + example["instruction"] for example in examples]
        labels = [example["response"] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]
        tokens = self.processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
            tokenize_newline_separately=False,
        )
        tokens = tokens.to(torch.bfloat16).to(self.model.device)  # type: ignore
        return tokens

    # We pass in a path to the training data rather than hand it a list of training data
    # as Python objects because we expect to invoke this as a Ray remote function and
    # we cannot pass in massive Python objects — this uses up too much memory and causes
    # problems on the UNC NLP servers due to space issues.
    def train(self, training_data_path: str, num_train_epochs: int = 100) -> str:
        if self.model is None or self.processor is None:
            self._load_model()

        assert self.peft_model is not None, "PEFT model is not loaded"

        logger.info("Loading training data from {}", training_data_path)
        training_data = list(
            PydanticJSONLinesReader(training_data_path, VqaTrainingDatum)()
        )
        logger.info("Loaded {} training data instances", len(training_data))

        ds = Dataset.from_dict(
            {
                "instruction": [datum.instruction for datum in training_data],
                "response": [datum.response for datum in training_data],
                "image": [datum.image.pil_image for datum in training_data],
            }
        )

        args = TrainingArguments(
            num_train_epochs=num_train_epochs,
            remove_unused_columns=False,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=False,
            save_total_limit=1,
            output_dir=str(self.trainer_output_dir),
            bf16=True,
            report_to="none",
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=self.peft_model,
            train_dataset=ds,
            data_collator=self._collate_fn,
            args=args,
        )

        trainer.train()

        # Glob the trainer output dir and return the last checkpoint
        last_checkpoint = sorted(
            Path(self.trainer_output_dir).glob("checkpoint-*"), reverse=True
        )[0]
        return os.path.join(self.trainer_output_dir, last_checkpoint.name)


@ray.remote(num_gpus=1)
class RayPaliGemmaDpoTrainer:
    def __init__(
        self, model_name_or_path: str, device: str, trainer_output_dir: str | Path
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.trainer_output_dir = trainer_output_dir
        self.model = None
        self.processor = None
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
            modules_to_save=None,
        )

    def _load_model(self):
        logger.info("Loading PaliGemma model for training")
        # We only load the model here, _not_ turn it into a PEFT model.
        # We let the DPOTrainer handle turning it into a PEFT model.
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.processor = cast(
            PaliGemmaProcessor,
            PaliGemmaProcessor.from_pretrained(self.model_name_or_path),
        )

    def train(self, training_data_path: str, num_train_epochs: int = 100) -> str:
        if self.model is None or self.processor is None:
            self._load_model()

        assert self.model is not None, "Model is not loaded"
        assert self.processor is not None, "Processor is not loaded"

        logger.info("Loading training data from {}", training_data_path)
        training_data = list(
            PydanticJSONLinesReader(training_data_path, VqaPreferenceTrainingDatum)()
        )

        # These are the keys expected by the DPO Trainer in the Dataset.
        ds = Dataset.from_dict(
            {
                "prompt": [datum.instruction for datum in training_data],
                "chosen": [datum.chosen_response for datum in training_data],
                "rejected": [datum.rejected_response for datum in training_data],
                "images": [datum.image.pil_image for datum in training_data],
            }
        )

        # TODO: Instead of passing in `PeftConfig`, we should instantiate the model ourselves
        # and wrap it in a `PeftModel`. Then we can pass in the wrapped model as the `ref_model`.
        # When you pass in a non-peft model and a `PeftConfig`, the `DpoTrainer` will merge and unload
        # the LoRA adapters. The main issue with this is that checkpointing becomes _very_ slow.
        args = DPOConfig(
            num_train_epochs=num_train_epochs,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=32,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=False,
            save_total_limit=1,
            output_dir=str(self.trainer_output_dir),
            bf16=True,
            report_to="none",
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )

        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            train_dataset=ds,
            tokenizer=self.processor,  # type: ignore
            args=args,
            callbacks=None,
        )

        trainer.train()

        logger.info("Training completed")
        # Glob the trainer output dir and return the last checkpoint
        last_checkpoint = sorted(
            Path(self.trainer_output_dir).glob("checkpoint-*"), reverse=True
        )[0]
        return os.path.join(self.trainer_output_dir, last_checkpoint.name)


class PaliGemmaTrainablePredictor:
    def __init__(
        self,
        model_name_or_path: str = "google/paligemma-3b-pt-224",
        device: str = "cuda:0",
        trainer_output_dir: str | Path = "paligemma_dpo_trainer_output_dir",
        num_train_epochs: int = 100,
    ):
        self.model_name_or_path = model_name_or_path
        self.trainer_output_dir = trainer_output_dir
        self.predictor_actor = RayPaliGemmaPredictor.remote(model_name_or_path)
        self.device = device
        self.trainer_actor = RayPaliGemmaTrainer.remote(
            model_name_or_path,
            device,
            trainer_output_dir,
        )
        self.checkpoint_paths: list[str] = []
        self.num_train_epochs = num_train_epochs

    def reset_trainer(self):
        ray.kill(self.trainer_actor)
        self.trainer_actor = RayPaliGemmaTrainer.remote(
            self.model_name_or_path,
            self.device,
            self.trainer_output_dir,
        )

    def predict(
        self,
        task_instances: Sequence[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_instances_path = Path(tmpdir)
            VqaTaskInstanceSerializer.serialize(task_instances, task_instances_path)
            return ray.get(
                self.predictor_actor.predict.remote(str(task_instances_path))  # type: ignore
            )

    def train(self, training_data: Sequence[VqaTrainingDatum]) -> None:
        # Start from the base model to avoid repeatedly training the same model.
        self.reset_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            training_data_path = Path(tmpdir) / "training_data.jsonl"
            writer = PydanticJSONLinesWriter(training_data_path)
            writer.write_batch(training_data)
            last_checkpoint_path = ray.get(
                self.trainer_actor.train.remote(  # type: ignore
                    str(training_data_path), num_train_epochs=self.num_train_epochs
                )
            )
            logger.info(
                f"Training completed. Last checkpoint saved at {last_checkpoint_path}"
            )
            self.checkpoint_paths.append(last_checkpoint_path)

        # Now we kill the predictor and load the newly trained adapter.
        ray.kill(self.predictor_actor)

        self.predictor_actor = RayPaliGemmaPredictor.remote(
            model_name_or_path=self.model_name_or_path  # type: ignore
        )
        ray.get(
            self.predictor_actor.load_adapter.remote(adapter_path=last_checkpoint_path)
        )

    def save(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)

        # Copy the directory at self.last_checkpoint_path to the target path.
        shutil.copytree(self.trainer_output_dir, path)


implements(TrainableVqaPredictorInterface)(PaliGemmaTrainablePredictor)


class PaliGemmaPreferenceTrainablePredictor(PreferenceTrainableVqaPredictorInterface):
    def __init__(
        self,
        model_name_or_path: str = "google/paligemma-3b-pt-224",
        device: str = "cuda:0",
        trainer_output_dir: str | Path = "paligemma_dpo_trainer_output_dir",
        num_train_epochs: int = 100,
    ):
        self.model_name_or_path = model_name_or_path
        self.trainer_output_dir = trainer_output_dir
        self.predictor_actor = RayPaliGemmaPredictor.remote(model_name_or_path)
        self.trainer_actor = RayPaliGemmaDpoTrainer.remote(
            model_name_or_path,
            device,
            trainer_output_dir,
        )
        self.checkpoint_paths: list[str] = []
        self.num_train_epochs = num_train_epochs

    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_instances_path = Path(tmpdir)
            VqaTaskInstanceSerializer.serialize(task_instances, task_instances_path)
            return ray.get(
                self.predictor_actor.predict.remote(str(task_instances_path))  # type: ignore
            )

    def train_preference(
        self, training_data: Sequence[VqaPreferenceTrainingDatum]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            training_data_path = Path(tmpdir) / "training_data.jsonl"
            writer = PydanticJSONLinesWriter(training_data_path)
            writer.write_batch(training_data)
            last_checkpoint_path = ray.get(
                self.trainer_actor.train.remote(  # type: ignore
                    str(training_data_path), num_train_epochs=self.num_train_epochs
                )
            )
            logger.info(
                f"Training completed. Last checkpoint saved at {last_checkpoint_path}"
            )
            self.checkpoint_paths.append(last_checkpoint_path)

        logger.info("Killing the predictor and loading the newly trained adapter")
        ray.kill(self.predictor_actor)
        logger.info("Predictor actor killed")
        # ray.kill(self.trainer_actor)

        self.predictor_actor = RayPaliGemmaPredictor.remote(
            model_name_or_path=self.model_name_or_path  # type: ignore
        )
        # The DPOTrainer saves the _merged_ model, not the LoRA adapter, so we have to
        # load the _actual_ model here, not the adapter. Unfortunately.
        ray.get(
            self.predictor_actor._load_model.remote(
                model_name_or_path=last_checkpoint_path
            )
        )

    def save(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        logger.info("Saving the trained model to {}", path)
        # Copy the directory at self.last_checkpoint_path to the target path.
        shutil.copytree(self.trainer_output_dir, path)
        logger.info("Copied last checkpoint at {} to {}", self.trainer_output_dir, path)
