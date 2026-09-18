import tempfile
import unittest
from types import SimpleNamespace

from atlas_jev.gate import GateDecision
from atlas_jev.llm import ExtractedMemory, MemoryType
from atlas_jev.pipeline import MemoryPipeline
from atlas_jev.store import IngestMeta, Memory, MemoryStore


def _meta(
    source_text: str,
    extracted_at: float,
    operation: str,
    *,
    target_id: str | None = None,
    conflict: float = 0.0,
) -> IngestMeta:
    return IngestMeta(
        source_text=source_text,
        extracted_at=extracted_at,
        confidence=0.8,
        extraction_confidence=0.9,
        operation=operation,
        operation_confidence=0.95,
        target_id=target_id,
        conflict=conflict,
    )


class _Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class _UpdateGate:
    def evaluate(self, candidate: str, memory_type: str, similar: list) -> GateDecision:
        return GateDecision(
            worth=0.9,
            operation="update",
            operation_confidence=0.9,
            target_id="existing-memory",
        )


class _RecordingStore:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.updated: list[str] = []

    def recall(self, vector: list[float], limit: int) -> list:
        return []

    def add(
        self,
        text: str,
        memory_type: str,
        vector: list[float],
        meta: IngestMeta,
    ) -> Memory:
        self.added.append(text)
        return Memory("new-memory", text, memory_type, 1.0, 1.0)

    def update(self, memory_id: str, *args) -> Memory:
        self.updated.append(memory_id)
        raise AssertionError("compatible updates must not overwrite existing memories")


class PipelineUpdateTests(unittest.TestCase):
    def test_compatible_update_is_stored_as_a_separate_atomic_fact(self) -> None:
        pipeline = object.__new__(MemoryPipeline)
        pipeline._settings = SimpleNamespace(
            recall_limit=5,
            recall_distance_threshold=1.0,
            worth_threshold=0.5,
        )
        pipeline._embedder = _Embedder()
        pipeline._gate = _UpdateGate()
        pipeline._store = _RecordingStore()
        candidate = ExtractedMemory(
            text="Pixel is a small black mutt.",
            type=MemoryType.RELATIONSHIP,
            confidence=0.9,
        )

        result = pipeline._process_candidate(candidate, "Pixel is small and black.", 10.0)

        self.assertEqual(result.action_taken, "added")
        self.assertEqual(pipeline._store.added, ["Pixel is a small black mutt."])
        self.assertEqual(pipeline._store.updated, [])


class StoreRevertTests(unittest.TestCase):
    def test_revert_restores_full_versions_and_walks_backward(self) -> None:
        with tempfile.TemporaryDirectory() as db_path:
            store = MemoryStore(db_path, 2)
            berlin = store.add(
                "The user lives in Berlin.",
                "fact",
                [1.0, 0.0],
                _meta("I live in Berlin.", 10.0, "add", conflict=0.1),
            )
            austin = store.update(
                berlin.id,
                "The user lives in Austin.",
                "fact",
                [0.0, 1.0],
                _meta(
                    "I moved to Austin.",
                    20.0,
                    "replace",
                    target_id=berlin.id,
                    conflict=0.8,
                ),
            )
            store.update(
                berlin.id,
                "The user lives in Paris.",
                "fact",
                [0.5, 0.5],
                _meta(
                    "I moved to Paris.",
                    30.0,
                    "replace",
                    target_id=berlin.id,
                    conflict=0.9,
                ),
            )

            restored_austin = store.revert(berlin.id, [9.0, 9.0])
            restored_berlin = store.revert(berlin.id, [9.0, 9.0])

            self.assertEqual(restored_austin, austin)
            self.assertEqual(restored_berlin, berlin)
            row = next(row for row in store._memory_rows() if row["id"] == berlin.id)
            self.assertEqual(row["vector"], [1.0, 0.0])
            self.assertEqual(
                [event.action_taken for event in store.list_events(berlin.id)],
                ["added", "updated", "updated", "reverted", "reverted"],
            )
            self.assertEqual(
                [event.operation for event in store.list_events(berlin.id)[-2:]],
                ["revert", "revert"],
            )
            with self.assertRaisesRegex(RuntimeError, "no previous value"):
                store.revert(berlin.id, [9.0, 9.0])
            self.assertEqual(store.get(berlin.id), berlin)


if __name__ == "__main__":
    unittest.main()
