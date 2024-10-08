from typing import Collection, Union
import json
from pathlib import Path
from .domain_models import (
    OpenEndedVqaTaskInstance,
    MultipleChoiceVqaTaskInstance,
)
from PIL import Image


class VqaTaskInstanceSerializer:
    @staticmethod
    def serialize(
        instances: Collection[
            Union[OpenEndedVqaTaskInstance, MultipleChoiceVqaTaskInstance]
        ],
        save_dir: Path,
    ):
        images_dir = save_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        records_path = save_dir / "records.jsonl"

        with records_path.open("w") as f:
            for instance in instances:
                record = instance.model_dump()
                image_filename = f"{instance.instance_id}.png"
                image_path = images_dir / image_filename
                instance.image.save(image_path)
                record["image_path"] = str(image_path.relative_to(save_dir))
                json.dump(record, f)
                f.write("\n")

    @staticmethod
    def deserialize(
        load_dir: Path,
    ) -> Collection[Union[OpenEndedVqaTaskInstance, MultipleChoiceVqaTaskInstance]]:
        records_path = load_dir / "records.jsonl"
        instances: Collection[
            Union[OpenEndedVqaTaskInstance, MultipleChoiceVqaTaskInstance]
        ] = []

        with records_path.open("r") as f:
            for line in f:
                record = json.loads(line)
                image_path = load_dir / record["image_path"]
                record["image"] = Image.open(image_path)
                del record["image_path"]
                if record["vqa_task_type"] == "open_ended":
                    instances.append(OpenEndedVqaTaskInstance.model_validate(record))  # type: ignore
                elif record["vqa_task_type"] == "multiple_choice":
                    instances.append(  # type: ignore
                        MultipleChoiceVqaTaskInstance.model_validate(record)
                    )

        return instances
