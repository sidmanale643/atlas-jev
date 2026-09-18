from dataclasses import dataclass, field

from jev_mem.config import Settings, load_settings
from jev_mem.embeddings import Embedder
from jev_mem.gate import GateDecision, MemoryGate
from jev_mem.llm import ExtractedMemory, LLMService
from jev_mem.store import Memory, MemoryHit, MemoryStore


@dataclass(frozen=True)
class CandidateResult:
    candidate: ExtractedMemory
    decision: GateDecision
    action_taken: str  # "added" | "updated" | "skipped"
    memory_id: str | None


@dataclass(frozen=True)
class IngestReport:
    source_text: str
    results: list[CandidateResult] = field(default_factory=list)


class MemoryPipeline:
    """Extract memories with an OpenRouter LLM, gate them with OpenRouter Jev, store in LanceDB."""

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
        report = IngestReport(source_text=text)
        for candidate in candidates:
            report.results.append(self._process_candidate(candidate))
        return report

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:
        [vector] = self._embedder.embed([query])
        return self._store.search(vector, limit=limit)

    def list_memories(self) -> list[Memory]:
        return self._store.list_all()

    def _process_candidate(self, candidate: ExtractedMemory) -> CandidateResult:
        [vector] = self._embedder.embed([candidate.text])
        similar = [
            hit
            for hit in self._store.search(vector, limit=self._settings.recall_limit)
            if hit.distance <= self._settings.recall_distance_threshold
        ]

        decision = self._gate.evaluate(candidate.text, candidate.type, similar)

        if decision.worth < self._settings.worth_threshold:
            return CandidateResult(candidate, decision, "skipped", None)

        if decision.operation == "update" and decision.target_id:
            self._store.update(decision.target_id, candidate.text, candidate.type, vector)
            return CandidateResult(candidate, decision, "updated", decision.target_id)

        if decision.operation == "skip":
            return CandidateResult(candidate, decision, "skipped", None)

        memory = self._store.add(candidate.text, candidate.type, vector)
        return CandidateResult(candidate, decision, "added", memory.id)
