import time
import uuid
from dataclasses import dataclass

import lancedb
import pyarrow as pa

TABLE_NAME = "memories"

_MEMORY_COLUMN_DEFAULTS = {
    "type": "'other'",
}


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    type: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class MemoryHit:
    memory: Memory
    distance: float


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _ensure_columns(table, defaults: dict[str, str]) -> None:
    names = set(table.schema.names)
    additions = {name: expr for name, expr in defaults.items() if name not in names}
    if additions:
        table.add_columns(additions)


def _memories_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("type", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("created_at", pa.float64()),
            pa.field("updated_at", pa.float64()),
        ]
    )


def _memory_from_row(row: dict) -> Memory:
    return Memory(
        id=row["id"],
        text=row["text"],
        type=row["type"],
        created_at=_as_float(row.get("created_at")),
        updated_at=_as_float(row.get("updated_at")),
    )


class MemoryStore:
    """Local LanceDB-backed vector store for memories."""

    def __init__(self, db_path: str, dim: int) -> None:
        self._db = lancedb.connect(db_path)
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
            _ensure_columns(self._table, _MEMORY_COLUMN_DEFAULTS)
        else:
            self._table = self._db.create_table(TABLE_NAME, schema=_memories_schema(dim))

    def add(self, text: str, memory_type: str, vector: list[float]) -> Memory:
        now = time.time()
        memory = Memory(
            id=uuid.uuid4().hex, text=text, type=memory_type, created_at=now, updated_at=now
        )
        self._table.add([self._memory_row(memory, vector)])
        return memory

    def update(self, memory_id: str, text: str, memory_type: str, vector: list[float]) -> None:
        self._table.update(
            where=f"id = '{memory_id}'",
            values={
                "text": text,
                "type": memory_type,
                "vector": vector,
                "updated_at": time.time(),
            },
        )

    def delete(self, memory_id: str) -> None:
        self._table.delete(f"id = '{memory_id}'")

    def search(self, vector: list[float], limit: int = 5) -> list[MemoryHit]:
        if self._table.count_rows() == 0:
            return []
        rows = self._table.search(vector).limit(limit).to_list()
        return [
            MemoryHit(memory=_memory_from_row(row), distance=row["_distance"])
            for row in rows
        ]

    def list_all(self) -> list[Memory]:
        return [_memory_from_row(row) for row in self._memory_rows()]

    def _memory_rows(self) -> list[dict]:
        if self._table.count_rows() == 0:
            return []
        return self._table.to_arrow().to_pylist()

    def _memory_row(self, memory: Memory, vector: list[float]) -> dict:
        return {
            "id": memory.id,
            "text": memory.text,
            "type": memory.type,
            "vector": vector,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }
