import time
import uuid
from dataclasses import dataclass

import lancedb
import pyarrow as pa

TABLE_NAME = "memories"
EVENTS_TABLE_NAME = "memory_events"

_MEMORY_COLUMN_DEFAULTS = {
    "type": "'other'",
    "source_text": "''",
    "extracted_at": "0.0",
    "confidence": "0.0",
    "operation": "''",
    "operation_confidence": "0.0",
    "target_id": "''",
    "previous_text": "''",
    "previous_type": "''",
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
    previous_text: str | None = None
    previous_type: str | None = None


@dataclass(frozen=True)
class MemoryEvent:
    id: str
    memory_id: str | None
    source_text: str
    candidate_text: str
    candidate_type: str
    extracted_at: float
    action_taken: str
    worth: float
    operation: str
    operation_confidence: float
    target_id: str | None
    previous_text: str | None
    previous_type: str | None
    created_at: float


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
            pa.field("previous_text", pa.string()),
            pa.field("previous_type", pa.string()),
        ]
    )


def _events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("memory_id", pa.string()),
            pa.field("source_text", pa.string()),
            pa.field("candidate_text", pa.string()),
            pa.field("candidate_type", pa.string()),
            pa.field("extracted_at", pa.float64()),
            pa.field("action_taken", pa.string()),
            pa.field("worth", pa.float64()),
            pa.field("operation", pa.string()),
            pa.field("operation_confidence", pa.float64()),
            pa.field("target_id", pa.string()),
            pa.field("previous_text", pa.string()),
            pa.field("previous_type", pa.string()),
            pa.field("created_at", pa.float64()),
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
        previous_text=_optional_str(row.get("previous_text")),
        previous_type=_optional_str(row.get("previous_type")),
    )


def _event_from_row(row: dict) -> MemoryEvent:
    return MemoryEvent(
        id=row["id"],
        memory_id=_optional_str(row.get("memory_id")),
        source_text=str(row.get("source_text") or ""),
        candidate_text=str(row.get("candidate_text") or ""),
        candidate_type=str(row.get("candidate_type") or ""),
        extracted_at=_as_float(row.get("extracted_at")),
        action_taken=str(row.get("action_taken") or ""),
        worth=_as_float(row.get("worth")),
        operation=str(row.get("operation") or ""),
        operation_confidence=_as_float(row.get("operation_confidence")),
        target_id=_optional_str(row.get("target_id")),
        previous_text=_optional_str(row.get("previous_text")),
        previous_type=_optional_str(row.get("previous_type")),
        created_at=_as_float(row.get("created_at")),
    )


class MemoryStore:
    """Local LanceDB-backed vector store for memories and gate audit events."""

    def __init__(self, db_path: str, dim: int) -> None:
        self._db = lancedb.connect(db_path)
        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
            _ensure_columns(self._table, _MEMORY_COLUMN_DEFAULTS)
        else:
            self._table = self._db.create_table(TABLE_NAME, schema=_memories_schema(dim))
        if EVENTS_TABLE_NAME in self._db.table_names():
            self._events = self._db.open_table(EVENTS_TABLE_NAME)
        else:
            self._events = self._db.create_table(EVENTS_TABLE_NAME, schema=_events_schema())

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
        self._append_event(
            memory_id=memory.id,
            candidate_text=text,
            candidate_type=memory_type,
            action_taken="added",
            meta=meta,
        )
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
                "previous_text": existing.text,
                "previous_type": existing.type,
            },
        )
        updated = self.get(existing.id)
        if updated is None:
            raise RuntimeError(f"Memory {existing.id} vanished after update")
        self._append_event(
            memory_id=updated.id,
            candidate_text=text,
            candidate_type=memory_type,
            action_taken="updated",
            meta=meta,
            previous_text=existing.text,
            previous_type=existing.type,
        )
        return updated

    def record_skip(self, candidate_text: str, candidate_type: str, meta: IngestMeta) -> None:
        self._append_event(
            memory_id=None,
            candidate_text=candidate_text,
            candidate_type=candidate_type,
            action_taken="skipped",
            meta=meta,
        )

    def revert(self, memory_id: str, restored_vector: list[float]) -> Memory:
        existing = self.get(memory_id)
        if existing is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        if not existing.previous_text:
            raise RuntimeError(f"Memory {existing.id} has no previous value to revert")
        restored_type = existing.previous_type or existing.type
        now = time.time()
        self._table.update(
            where=_id_clause(existing.id),
            values={
                "text": existing.previous_text,
                "type": restored_type,
                "vector": restored_vector,
                "updated_at": now,
                "operation": "revert",
                "previous_text": existing.text,
                "previous_type": existing.type,
            },
        )
        reverted = self.get(existing.id)
        if reverted is None:
            raise RuntimeError(f"Memory {existing.id} vanished after revert")
        self._append_event(
            memory_id=reverted.id,
            candidate_text=reverted.text,
            candidate_type=reverted.type,
            action_taken="reverted",
            meta=IngestMeta(
                source_text=existing.source_text,
                extracted_at=existing.extracted_at,
                confidence=existing.confidence,
                operation="revert",
                operation_confidence=1.0,
                target_id=existing.id,
            ),
            previous_text=existing.text,
            previous_type=existing.type,
        )
        return reverted

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

    def list_events(self, memory_id: str | None = None) -> list[MemoryEvent]:
        if self._events.count_rows() == 0:
            return []
        rows = self._events.to_arrow().to_pylist()
        events = [_event_from_row(row) for row in rows]
        if memory_id:
            resolved = self.get(memory_id)
            target = resolved.id if resolved else memory_id
            events = [
                event
                for event in events
                if event.memory_id == target or (event.target_id == target)
            ]
        events.sort(key=lambda event: event.created_at)
        return events

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
            "previous_text": memory.previous_text or "",
            "previous_type": memory.previous_type or "",
        }

    def _append_event(
        self,
        *,
        memory_id: str | None,
        candidate_text: str,
        candidate_type: str,
        action_taken: str,
        meta: IngestMeta,
        previous_text: str | None = None,
        previous_type: str | None = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            id=uuid.uuid4().hex,
            memory_id=memory_id,
            source_text=meta.source_text,
            candidate_text=candidate_text,
            candidate_type=candidate_type,
            extracted_at=meta.extracted_at,
            action_taken=action_taken,
            worth=meta.confidence,
            operation=meta.operation,
            operation_confidence=meta.operation_confidence,
            target_id=meta.target_id,
            previous_text=previous_text,
            previous_type=previous_type,
            created_at=time.time(),
        )
        self._events.add(
            [
                {
                    "id": event.id,
                    "memory_id": event.memory_id or "",
                    "source_text": event.source_text,
                    "candidate_text": event.candidate_text,
                    "candidate_type": event.candidate_type,
                    "extracted_at": event.extracted_at,
                    "action_taken": event.action_taken,
                    "worth": event.worth,
                    "operation": event.operation,
                    "operation_confidence": event.operation_confidence,
                    "target_id": event.target_id or "",
                    "previous_text": event.previous_text or "",
                    "previous_type": event.previous_type or "",
                    "created_at": event.created_at,
                }
            ]
        )
        return event
