from dataenvgym.utils import SqliteKeyValueStore, PydanticSqliteKeyValueStore
from pathlib import Path
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor


class TestSqliteKeyValueStore:
    def test_set_and_get(self, tmp_path: Path):
        db_path = "sqlite:///" + str(tmp_path / "test.db")
        store = SqliteKeyValueStore(db_url=db_path)
        store.set("key1", "value1")
        assert store.get("key1") == "value1"


class Animal(BaseModel):
    name: str
    age: int
    food: list[str]


class TestPydanticSqliteKeyValueStore:
    def test_set_and_get(self, tmp_path: Path):
        db_path = "sqlite:///" + str(tmp_path / "test.db")

        animal_a = Animal(name="dog", age=10, food=["dog food"])
        animal_b = Animal(name="cat", age=10, food=["cat food"])
        store = PydanticSqliteKeyValueStore(Animal, db_url=db_path)
        store["animal_a"] = animal_a
        store["animal_b"] = animal_b
        assert store["animal_a"] == animal_a
        assert store["animal_b"] == animal_b

    def test_get_and_set_multiple_threads(self, tmp_path: Path):
        db_path = "sqlite:///" + str(tmp_path / "test.db")
        store = PydanticSqliteKeyValueStore(Animal, db_url=db_path)

        animal_a = Animal(name="dog", age=10, food=["dog food"])
        animal_b = Animal(name="cat", age=10, food=["cat food"])

        with ThreadPoolExecutor() as executor:
            executor.map(
                store.__setitem__, ["animal_a", "animal_b"], [animal_a, animal_b]
            )

        retrieved: list[Animal] = []
        with ThreadPoolExecutor() as executor:
            retrieved = list(executor.map(store.__getitem__, ["animal_a", "animal_b"]))
        assert retrieved == [animal_a, animal_b]
