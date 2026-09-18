import time
import uuid
from dataclasses import dataclass

import lancedb
import pyarrow as pa

TABLE_NAME = "memories"

_MEMORY_COLUMN_DEFAULTS = {
    "type": "'other'",
    "source_text": "''",
    "extracted_at": "0.0",
    "confidence": "0.0",
    "operation": "''",
    "operation_confidence": "0.0",
    "target_id": "''",
}


@dataclass(frozen=True)
class IngestMeta:
    source_text: str
    extracted_at: float
    confidence: float
    operation: str
    operation_confidence: float
    target_id: str | None = None


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    type: str
    created_at: float
    updated_at: float
    source_text: str = ""
    extracted_at: float = 0.0
    confidence: float = 0.0
    operation: str = ""
    operation_confidence: float = 0.0
    target_id: str | None = None


@dataclass(frozen=True)
class MemoryHit:
    memory: Memory
    distance: float


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _id_clause(memory_id: str) -> str:
    if not memory_id or any(char not in "0123456789abcdef" for char in memory_id):
        raise ValueError(f"Invalid memory id: {memory_id}")
    return f"id = '{memory_id}'"


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
            pa.field("source_text", pa.string()),
            pa.field("extracted_at", pa.float64()),
            pa.field("confidence", pa.float64()),
            pa.field("operation", pa.string()),
            pa.field("operation_confidence", pa.float64()),
            pa.field("target_id", pa.string()),
        ]
    )


def _memory_from_row(row: dict) -> Memory:
    return Memory(
        id=row["id"],
        text=row["text"],
        type=row["type"],
        created_at=_as_float(row.get("created_at")),
        updated_at=_as_float(row.get("updated_at")),
        source_text=str(row.get("source_text") or ""),
        extracted_at=_as_float(row.get("extracted_at"), _as_float(row.get("created_at"))),
        confidence=_as_float(row.get("confidence")),
        operation=str(row.get("operation") or ""),
        operation_confidence=_as_float(row.get("operation_confidence")),
        target_id=_optional_str(row.get("target_id")),
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

    def add(self, text: str, memory_type: str, vector: list[float], meta: IngestMeta) -> Memory:
        now = time.time()
        memory = Memory(
            id=uuid.uuid4().hex,
            text=text,
            type=memory_type,
            created_at=now,
            updated_at=now,
            source_text=meta.source_text,
            extracted_at=meta.extracted_at,
            confidence=meta.confidence,
            operation=meta.operation,
            operation_confidence=meta.operation_confidence,
            target_id=meta.target_id,
        )
        self._table.add([self._memory_row(memory, vector)])
        return memory

    def update(
        self,
        memory_id: str,
        text: str,
        memory_type: str,
        vector: list[float],
        meta: IngestMeta,
    ) -> Memory:
        existing = self.get(memory_id)
        if existing is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        now = time.time()
        self._table.update(
            where=_id_clause(existing.id),
            values={
                "text": text,
                "type": memory_type,
                "vector": vector,
                "updated_at": now,
                "source_text": meta.source_text,
                "extracted_at": meta.extracted_at,
                "confidence": meta.confidence,
                "operation": meta.operation,
                "operation_confidence": meta.operation_confidence,
                "target_id": meta.target_id or existing.id,
            },
        )
        updated = self.get(existing.id)
        if updated is None:
            raise RuntimeError(f"Memory {existing.id} vanished after update")
        return updated

    def delete(self, memory_id: str) -> None:
        memory = self.get(memory_id)
        if memory is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        self._table.delete(_id_clause(memory.id))

    def get(self, memory_id: str) -> Memory | None:
        rows = self._memory_rows()
        exact = [row for row in rows if row["id"] == memory_id]
        if exact:
            return _memory_from_row(exact[0])
        prefixed = [row for row in rows if row["id"].startswith(memory_id)]
        if len(prefixed) == 1:
            return _memory_from_row(prefixed[0])
        if len(prefixed) > 1:
            raise RuntimeError(f"Ambiguous memory id prefix: {memory_id}")
        return None

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
            "source_text": memory.source_text,
            "extracted_at": memory.extracted_at,
            "confidence": memory.confidence,
            "operation": memory.operation,
            "operation_confidence": memory.operation_confidence,
            "target_id": memory.target_id or "",
        }
