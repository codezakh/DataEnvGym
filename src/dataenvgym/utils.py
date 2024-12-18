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
    cast,
    get_args,
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


def extract_code_in_markdown_backticks(model_output: str) -> str:
    outputlines = model_output.split("\n")
    # Find the first line that starts with ```
    opening_backticks_idx = next(
        (i for i, line in enumerate(outputlines) if line.startswith("```")), None
    )
    if opening_backticks_idx is None:
        # We don't know what to do, so just return the whole thing.
        return "\n".join(outputlines)

    # Find the line that contains the closing code block.
    closing_backticks_idx = next(
        (
            i
            for i, line in enumerate(outputlines[opening_backticks_idx:])
            if line.endswith("```")
        ),
        None,
    )

    if closing_backticks_idx is None:
        # We don't know what to do, so just return the whole thing.
        return "\n".join(outputlines)

    # If there isn't any code between them, return the whole thing.
    if opening_backticks_idx + 1 >= closing_backticks_idx:
        return "\n".join(outputlines)

    return "\n".join(outputlines[opening_backticks_idx + 1 : closing_backticks_idx])


from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

Base = declarative_base()


class KeyValueModel(Base):
    __tablename__ = "key_value_store"
    key = Column(String, primary_key=True, nullable=False)
    value = Column(String, nullable=False)


class SqliteKeyValueStore:
    def __init__(self, db_url: str = "sqlite:///key_value_store.db"):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def set(self, key: str, value: str):
        session = self.Session()
        try:
            existing_entry = session.query(KeyValueModel).filter_by(key=key).first()
            if existing_entry:
                existing_entry.value = value  # type: ignore
            else:
                new_entry = KeyValueModel(key=key, value=value)
                session.add(new_entry)
            session.commit()
        finally:
            session.close()

    def get(self, key: str) -> Optional[str]:
        session = self.Session()
        try:
            entry = session.query(KeyValueModel).filter_by(key=key).first()
            return cast(str, entry.value) if entry else None
        finally:
            session.close()

    def delete(self, key: str) -> None:
        session = self.Session()
        try:
            session.query(KeyValueModel).filter_by(key=key).delete()
            session.commit()
        finally:
            session.close()

    def keys(self) -> list[str]:
        session = self.Session()
        try:
            return cast(
                list[str], [entry.key for entry in session.query(KeyValueModel).all()]
            )
        finally:
            session.close()

    def values(self) -> list[str]:
        session = self.Session()
        try:
            return cast(
                list[str], [entry.value for entry in session.query(KeyValueModel).all()]
            )
        finally:
            session.close()

    def items(self) -> list[tuple[str, str]]:
        session = self.Session()
        try:
            return cast(
                list[tuple[str, str]],
                [
                    (entry.key, entry.value)
                    for entry in session.query(KeyValueModel).all()
                ],
            )
        finally:
            session.close()


class PydanticSqliteKeyValueStore(MutableMapping[str, T], Generic[T]):
    def __init__(self, model: Type[T], db_url: str = "sqlite:///key_value_store.db"):
        self.db_url = db_url
        self.store = SqliteKeyValueStore(db_url)
        self.model = model

    def __getitem__(self, key: str) -> T:
        value = self.store.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found in store")
        return self.model.model_validate_json(value)

    def __setitem__(self, key: str, value: T) -> None:
        self.store.set(key, value.model_dump_json())

    def __delitem__(self, key: str) -> None:
        self.store.delete(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.keys())

    def __len__(self) -> int:
        return len(self.store.keys())
