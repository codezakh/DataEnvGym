import json
from pathlib import Path
from typing import (
    Collection,
    Generator,
    Generic,
    Iterator,
    MutableMapping,
    Optional,
    Type,
    TypeVar,
)

from pydantic import BaseModel
from tqdm.auto import tqdm
from ulid import ULID


class JsonlIoHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def append_dict(self, data: dict) -> None:
        """Appends a dictionary as a new line in the JSONL file."""
        with open(self.file_path, "a") as f:
            json_str = json.dumps(data)
            f.write(json_str + "\n")

    def read_all(self, progress: Optional[bool] = False) -> list[dict]:
        """Reads all dictionaries from the JSONL file. Optional progress indicator."""
        data = []
        with open(self.file_path, "r") as f:
            lines = f.readlines()
            if progress:
                lines = tqdm(lines, desc="Reading JSONL")  # type: ignore
            for line in lines:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
        return data

    def read_n(self, n: int) -> list[dict]:
        """Reads the first n dictionaries from the JSONL file."""
        data = []
        with open(self.file_path, "r") as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                json_obj = json.loads(line.strip())
                data.append(json_obj)
        return data


T = TypeVar("T", bound=BaseModel)


class PydanticJSONLinesWriter(Generic[T]):
    def __init__(self, file_path: str | Path):
        self.file_path = file_path

    def __call__(self, serializable: T, mode: str = "a") -> None:
        with open(self.file_path, mode) as f:
            f.write(serializable.model_dump_json() + "\n")

    def write_batch(self, serializables: Collection[T], mode: str = "a") -> None:
        with open(self.file_path, mode) as f:
            for serializable in serializables:
                f.write(serializable.model_dump_json() + "\n")


class PydanticJSONLinesReader(Generic[T]):
    def __init__(self, file_path: str | Path, model: Type[T]):
        self.file_path = file_path
        self.model = model

    def __call__(self) -> Generator[T, None, None]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield self.model.parse_raw(line)

    def __iter__(self) -> Generator[T, None, None]:
        return self()


class JSONLinesReader:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path

    def __call__(self) -> Generator[dict, None, None]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield json.loads(line)

    def __iter__(self) -> Generator[dict, None, None]:
        return self()


class KeyValue(BaseModel):
    key: ULID
    value: str

    def parse_as_model(self, model: Type[T]) -> T:
        return model.model_validate_json(self.value)


# TODO: Make a sqlite based version of this.
class JSONLinesKeyValueCache(MutableMapping[ULID, T], Generic[T]):
    """
    This class works like a dictionary, but the keys are ULID and the values
    are Pydantic models. The data is saved in a JSONL file.
    """

    def __init__(self, file_path: str | Path, model: Type[T], read_only: bool = False):
        self.file_path = file_path
        self.model = model
        self.data: dict[ULID, T] = {}
        self.read_only = read_only

        if Path(file_path).exists():
            reader = PydanticJSONLinesReader(file_path, KeyValue)
            for item in reader():
                self.data[item.key] = item.parse_as_model(self.model)

        self.writer = PydanticJSONLinesWriter[KeyValue](file_path)

    def __getitem__(self, key: ULID) -> T:
        return self.data[key]

    def __setitem__(self, key: ULID, value: T) -> None:
        if self.read_only:
            raise ValueError("Cache is read-only")
        self.data[key] = value
        self.writer(KeyValue(key=key, value=value.model_dump_json()))

    def __delitem__(self, key: ULID) -> None:
        if self.read_only:
            raise ValueError("Cache is read-only")
        del self.data[key]
        # Overwrite the file with the remaining data. This is
        # obviously very inefficient.
        self.writer.write_batch(
            [
                KeyValue(key=key, value=value.model_dump_json())
                for key, value in self.data.items()
            ],
            mode="w",
        )

    def __iter__(self) -> Iterator[ULID]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
