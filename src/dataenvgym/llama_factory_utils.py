from typing import Literal, Optional, Collection
import json
import os
from pydantic import BaseModel
from loguru import logger
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import sh
import sys
from typing import Sequence


class AlpacaFormatExpectedKeysPerRecord(BaseModel):
    prompt: str = "instruction"
    response: str = "output"

    def adapt_record_to_spec(
        self, record: dict, prompt_key: str, response_key: str
    ) -> dict:
        """
        Convert a record to match the expected format of the llama-factory finetuning spec.

        Parameters
        ----------
        record : dict
            A record from the JSONL file.
        prompt_key : str
            Key in the JSON record that contains the instruction.
        response_key : str
            Key in the JSON record that contains the response.

        Returns
        -------
        dict
            The record in the expected format.
        """
        try:
            new_record = {
                self.prompt: record[prompt_key],
                self.response: record[response_key],
            }
        except KeyError as e:
            raise KeyError(
                f"Record has keys {record.keys()} but expected keys "
                f"{self.prompt} -> {prompt_key}, {self.response} -> {response_key} "
                f"could not access {e}"
            )
        return new_record


DEFAULT_DATASET_FILE_NAME = "custom_sft_dataset.jsonl"


class LlamaFactoryMinimalSftSpec(BaseModel):
    file_name: str = DEFAULT_DATASET_FILE_NAME
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    ranking: bool = False
    load_from: Literal["hf_hub", "ms_hub", "script", "file"] = "file"
    columns: AlpacaFormatExpectedKeysPerRecord = AlpacaFormatExpectedKeysPerRecord()

    @property
    def dataset_name(self) -> str:
        """
        The dataset name llama-factory will refer to this dataset as.

        Returns
        -------
        str
            The dataset name.
        """
        return os.path.splitext(self.file_name)[0]

    @property
    def llama_factory_dataset_info_entry(self) -> dict[str, dict]:
        """
        Generate the dataset_info.json entry for the llama-factory dataset.

        Returns
        -------
        dict
            The dataset_info.json entry.
        """
        dataset_name = self.dataset_name
        return {dataset_name: self.model_dump(exclude={"dataset_name"})}

    def get_path_to_dataset_records(self, llama_factory_dataset_dir: str) -> str:
        """
        Get the path to the dataset records.

        Parameters
        ----------
        llama_factory_dataset_dir : str
            Path to the directory where llama-factory will look for the dataset_info.json file.

        Returns
        -------
        str
            Path to the dataset records.
        """
        return os.path.join(llama_factory_dataset_dir, self.file_name)


def format_records_for_llama_factory_sft(
    jsonl_records_path_or_collection: str | Collection[dict],
    llama_factory_dataset_dir: str,
    instruction_key: str,
    response_key: str,
    overwrite: bool = False,
) -> tuple[LlamaFactoryMinimalSftSpec, str]:
    """
    Take a collection of JSONL records and prepare it for llama-factory finetuning.

    Parameters
    ----------
    jsonl_records_path_or_collection : str | Collection[dict]
        Path to the JSONL file containing the records or a collection of records.
    llama_factory_dataset_dir : str
        Path to the directory where llama-factory will look for the dataset_info.json file.
    instruction_key : str
        Key in the JSON record that contains the instruction.
    response_key : str
        Key in the JSON record that contains the response.
    overwrite : bool, optional
        Whether to overwrite the existing records in the dataset, by default False

    Returns
    -------
    tuple[LlamaFactoryMinimalSftSpec, str]
        The llama-factory finetuning spec and the path to the dataset records.
    """

    supervised_finetuning_spec = LlamaFactoryMinimalSftSpec()

    # Make llama-factory dataset directory if it doesn't exist
    os.makedirs(llama_factory_dataset_dir, exist_ok=True)

    with open(os.path.join(llama_factory_dataset_dir, "dataset_info.json"), "w") as f:
        json.dump(supervised_finetuning_spec.llama_factory_dataset_info_entry, f)

    logger.info(
        f"Dataset info written to {llama_factory_dataset_dir}/dataset_info.json\n"
        f"{supervised_finetuning_spec.llama_factory_dataset_info_entry}"
    )

    if isinstance(jsonl_records_path_or_collection, str):
        with open(jsonl_records_path_or_collection, "r") as f:
            records = [json.loads(line) for line in f]
    else:
        records = jsonl_records_path_or_collection

    logger.info(
        f"Loaded {len(records)} records from.",
    )

    adapted_records = [
        supervised_finetuning_spec.columns.adapt_record_to_spec(
            record, instruction_key, response_key
        )
        for record in records
    ]

    if overwrite:
        with open(
            records_output_path := supervised_finetuning_spec.get_path_to_dataset_records(
                llama_factory_dataset_dir
            ),
            "w",
        ) as f:
            for record in adapted_records:
                json.dump(record, f)
                f.write("\n")
    else:
        # Open in append mode to avoid overwriting existing records
        with open(
            records_output_path := supervised_finetuning_spec.get_path_to_dataset_records(
                llama_factory_dataset_dir
            ),
            "a",
        ) as f:
            for record in adapted_records:
                json.dump(record, f)
                f.write("\n")

    logger.info(
        f"Adapted records written to {records_output_path}\n"
        f"Example record:\n{adapted_records[0]}"
    )

    return supervised_finetuning_spec, records_output_path


def generate_llama_factory_cli_args(
    config_name: str = "llama3_lora_sft",
    config_path: str = "./configs/llama_factory",
    overrides: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    if overrides is None:
        overrides = []

    # Important to use it as a context manager. If not done this way, will
    # prevent or throw errors or other uses of hydra throughout the application.
    with hydra.initialize(
        version_base=None,
        config_path=config_path,
    ):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    if output_path:
        with open(output_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
    else:
        print(OmegaConf.to_yaml(config))

    assert config.output_dir is not None, "Output directory must be specified"
    return str(config.output_dir)


def run_training_with_llama_factory(
    llama_factory_args_path: str, cuda_visible_devices: Optional[Sequence[int]] = None
) -> None:

    # By default, llama-factory will use HF's trainer, which uses all available GPUs.
    # If you want to restrict the GPUs used by llama-factory, you have to set CUDA_VISIBLE_DEVICES,
    # but this only works if it is set _before_ the process is started.
    if cuda_visible_devices is not None:
        cuda_visible_devices_value = ",".join(map(str, cuda_visible_devices))
        _env = os.environ.copy()
        _env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_value
    else:
        _env = None
    sh.llamafactory_cli.train(llama_factory_args_path, _out=sys.stdout, _err=sys.stderr, _env=_env)  # type: ignore[attr-defined]
