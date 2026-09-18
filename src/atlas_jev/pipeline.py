import time
from dataclasses import dataclass, field

from atlas_jev.config import Settings, load_settings
from atlas_jev.embeddings import Embedder
from atlas_jev.gate import GateDecision, MemoryGate
from atlas_jev.llm import ExtractedMemory, LLMService
from atlas_jev.store import IngestMeta, Memory, MemoryEvent, MemoryHit, MemoryStore


@dataclass(frozen=True)
class CandidateResult:
    candidate: ExtractedMemory
    decision: GateDecision
    action_taken: str
    memory_id: str | None


@dataclass(frozen=True)
class IngestReport:
    source_text: str
    extracted_at: float
    results: list[CandidateResult] = field(default_factory=list)


class MemoryPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        api_key = self._settings.openrouter_api_key
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
        self._llm = LLMService(api_key, self._settings.openrouter_model)
        self._embedder = Embedder(self._settings.embedding_model)
        self._store = MemoryStore(self._settings.db_path, self._embedder.dim)
        self._gate = MemoryGate(api_key, self._settings.jev_model)

    def add(self, text: str) -> IngestReport:
        candidates = self._llm.extract_memories(text)
        extracted_at = time.time()
        report = IngestReport(source_text=text, extracted_at=extracted_at)
        for candidate in candidates:
            report.results.append(self._process_candidate(candidate, text, extracted_at))
        return report

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:
        [vector] = self._embedder.embed([query])
        return self._store.search(vector, limit=limit)

    def list_memories(self) -> list[Memory]:
        return self._store.list_all()

    def history(self, memory_id: str | None = None) -> list[MemoryEvent]:
        return self._store.list_events(memory_id)

    def revert(self, memory_id: str) -> Memory:
        memory = self._store.get(memory_id)
        if memory is None:
            raise RuntimeError(f"Memory {memory_id} not found")
        if not memory.previous_text:
            raise RuntimeError(f"Memory {memory.id} has no previous value to revert")
        [vector] = self._embedder.embed([memory.previous_text])
        return self._store.revert(memory.id, vector)

    def _process_candidate(
        self,
        candidate: ExtractedMemory,
        source_text: str,
        extracted_at: float,
    ) -> CandidateResult:
        [vector] = self._embedder.embed([candidate.text])
        similar = [
            hit
            for hit in self._store.search(vector, limit=self._settings.recall_limit)
            if hit.distance <= self._settings.recall_distance_threshold
        ]

        decision = self._gate.evaluate(candidate.text, candidate.type, similar)
        meta = _ingest_meta(source_text, extracted_at, decision)

        if decision.worth < self._settings.worth_threshold:
            self._store.record_skip(candidate.text, candidate.type, meta)
            return CandidateResult(candidate, decision, "skipped", None)

        if decision.operation == "update" and decision.target_id:
            updated = self._store.update(
                decision.target_id, candidate.text, candidate.type, vector, meta
            )
            return CandidateResult(candidate, decision, "updated", updated.id)

        if decision.operation == "skip":
            self._store.record_skip(candidate.text, candidate.type, meta)
            return CandidateResult(candidate, decision, "skipped", None)

        memory = self._store.add(candidate.text, candidate.type, vector, meta)
        return CandidateResult(candidate, decision, "added", memory.id)


def _ingest_meta(source_text: str, extracted_at: float, decision: GateDecision) -> IngestMeta:
    return IngestMeta(
        source_text=source_text,
        extracted_at=extracted_at,
        confidence=decision.worth,
        operation=decision.operation,
        operation_confidence=decision.operation_confidence,
        target_id=decision.target_id,
    )
