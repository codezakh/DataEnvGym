from dataenvgym.gym.domain_models import (
    TrainableVqaPredictorInterface,
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
    VqaTrainingDatum,
)
from torch.optim import Adam
from typing import Collection, Sequence, cast
from transformers import AutoProcessor, BlipForQuestionAnswering
from tqdm.auto import tqdm
from loguru import logger
from transformers import get_scheduler
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Protocol
from pathlib import Path


class BlipTrainablePredictor(TrainableVqaPredictorInterface):
    def __init__(
        self, model_name_or_path: str = "Salesforce/blip-vqa-base", device: str = "cpu"
    ):
        logger.info("Loading BLIP model")
        self.model = cast(
            BlipForQuestionAnswering,
            BlipForQuestionAnswering.from_pretrained(
                model_name_or_path, device_map=device
            ),
        )
        logger.info("BLIP model loaded")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.training_batch_size = 4

    def predict(
        self,
        task_instances: Collection[
            OpenEndedVqaTaskInstance | MultipleChoiceVqaTaskInstance
        ],
    ) -> list[str]:
        self.model.eval()
        decoded_responses = []
        for task_instance in tqdm(task_instances, desc="Predicting", unit="instance"):
            with torch.no_grad():
                model_inputs = self.processor(
                    images=task_instance.image,
                    text=task_instance.instruction,
                    return_tensors="pt",
                )
                model_inputs.to(self.model.device)
                outputs = self.model.generate(**model_inputs)
                decoded_response = self.processor.decode(
                    outputs[0], skip_special_tokens=True
                )
                decoded_responses.append(decoded_response)
        return decoded_responses

    # TODO: We should probably switch this to use the Huggingface Trainer
    # at some point, but for now we will do it the old school simple way
    # just to get it working.

    def _create_batches(
        self, training_data: Sequence[VqaTrainingDatum], batch_size: int
    ) -> Sequence[Sequence[VqaTrainingDatum]]:
        batches = []
        for i in range(0, len(training_data), batch_size):
            batches.append(training_data[i : i + batch_size])
        return batches

    # def train(self, training_data: Sequence[VqaTrainingDatum]) -> None:
    #     optimizer = Adam(self.model.parameters(), lr=1e-5)
    #     self.model.train()
    #     batches = self._create_batches(training_data, self.training_batch_size)

    #     for batch in tqdm(batches, desc="Training", unit="batch"):
    #         images = [data.image.pil_image for data in batch]
    #         texts = [data.instruction for data in batch]
    #         responses = [data.response for data in batch]

    #         inputs = self.processor(
    #             images=images, text=texts, return_tensors="pt", padding=True
    #         )
    #         labels = self.processor(
    #             text=responses, return_tensors="pt", padding=True
    #         ).input_ids

    #         inputs.to(self.model.device)
    #         labels.to(self.model.device)

    #         inputs["labels"] = labels
    #         outputs = self.model(**inputs)

    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         optimizer.zero_grad()

    #         print(f"Loss: {loss.item()}")

    # def train(self, training_data: Sequence[VqaTrainingDatum]) -> None:
    #     optimizer = Adam(self.model.parameters(), lr=1e-5)
    #     lr_scheduler = get_scheduler(
    #         "linear",
    #         optimizer=optimizer,
    #         num_warmup_steps=0,
    #         num_training_steps=len(training_data) // self.training_batch_size,
    #     )

    #     self.model.train()
    #     batches = self._create_batches(training_data, self.training_batch_size)

    #     for batch in tqdm(batches, desc="Training", unit="batch"):
    #         images = [data.image.pil_image for data in batch]
    #         texts = [data.instruction for data in batch]
    #         responses = [data.response for data in batch]

    #         inputs = self.processor(
    #             images=images, text=texts, return_tensors="pt", padding=True
    #         )
    #         labels = self.processor(
    #             text=responses, return_tensors="pt", padding=True
    #         ).input_ids

    #         inputs.to(self.model.device)
    #         labels.to(self.model.device)

    #         inputs["labels"] = labels
    #         outputs = self.model(**inputs)

    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()

    #         print(f"Loss: {loss.item()}")
    def train(self, training_data: Sequence[VqaTrainingDatum]) -> None:
        num_epochs = 2
        optimizer = Adam(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        total_steps = (len(training_data) // self.training_batch_size) * num_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

        self.model.train()

        for epoch in range(num_epochs):
            batches = self._create_batches(training_data, self.training_batch_size)
            epoch_loss = 0.0

            progress_bar = tqdm(
                batches, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"
            )
            for batch in progress_bar:
                images = [data.image.pil_image for data in batch]
                texts = [data.instruction for data in batch]
                responses = [data.response for data in batch]

                inputs = self.processor(
                    images=images, text=texts, return_tensors="pt", padding=True
                )
                labels = self.processor(
                    text=responses, return_tensors="pt", padding=True
                ).input_ids

                inputs.to(self.model.device)
                labels.to(self.model.device)

                inputs["labels"] = labels
                outputs = self.model(**inputs)

                loss = outputs.loss
                loss.backward()

                clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # Gradient clipping
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            average_epoch_loss = epoch_loss / len(batches)
            print(
                f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_epoch_loss:.4f}"
            )

    def save(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.model.save_pretrained(path)
