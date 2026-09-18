import time
import uuid
from dataclasses import dataclass

import lancedb
import pyarrow as pa

TABLE_NAME = "memories"


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


class MemoryStore:
    """Local LanceDB-backed vector store for memories."""

    def __init__(self, db_path: str, dim: int) -> None:
        self._db = lancedb.connect(db_path)
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
            if "type" not in self._table.schema.names:
                self._table.add_columns({"type": "'other'"})
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("type", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("created_at", pa.float64()),
                    pa.field("updated_at", pa.float64()),
                ]
            )
            self._table = self._db.create_table(TABLE_NAME, schema=schema)

    def add(self, text: str, memory_type: str, vector: list[float]) -> Memory:
        now = time.time()
        memory = Memory(
            id=uuid.uuid4().hex, text=text, type=memory_type, created_at=now, updated_at=now
        )
        self._table.add(
            [
                {
                    "id": memory.id,
                    "text": memory.text,
                    "type": memory.type,
                    "vector": vector,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                }
            ]
        )
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
            MemoryHit(
                memory=Memory(
                    id=row["id"],
                    text=row["text"],
                    type=row["type"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                ),
                distance=row["_distance"],
            )
            for row in rows
        ]

    def list_all(self) -> list[Memory]:
        if self._table.count_rows() == 0:
            return []
        rows = self._table.to_arrow().to_pylist()
        return [
            Memory(
                id=row["id"],
                text=row["text"],
                type=row["type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
