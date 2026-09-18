from dataclasses import dataclass, replace

from typesafe_sdk import Choice, Noul, NoulCriteria

from atlas_jev.jev import DEFAULT_JEV_MODEL, JevClient
from atlas_jev.store import MemoryHit


@dataclass(frozen=True)
class GateDecision:
    worth: float
    operation: str
    operation_confidence: float
    target_id: str | None
    conflict: float = 0.0


class MemoryGate:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_JEV_MODEL,
        conflict_threshold: float = 0.7,
        relevance_threshold: float = 0.5,
    ) -> None:
        self._client = JevClient(api_key, model)
        self._conflict_threshold = conflict_threshold
        self._relevance_threshold = relevance_threshold

    def evaluate(self, candidate: str, memory_type: str, similar: list[MemoryHit]) -> GateDecision:
        state: dict = {
            "candidate_memory": candidate,
            "candidate_type": memory_type,
            "existing_memories": [hit.memory.text for hit in similar],
        }

        questions: dict = {
            "worth_remembering": Noul(
                instructions=(
                    "Is `candidate_memory` (classified as `candidate_type`) a "
                    "durable fact, preference, goal, decision, or relationship "
                    "worth keeping in long-term memory, rather than small talk "
                    "or a transient detail?"
                ),
            ),
            "operation": Choice(
                instructions=(
                    "Given `existing_memories`, what should the memory system do "
                    "with `candidate_memory`?"
                ),
                criteria={
                    "add": (
                        "No existing memory captures this fact. Compatible details "
                        "about the same subject are separate facts; store them as new."
                    ),
                    "update": (
                        "The candidate adds compatible detail to an existing fact. "
                        "Preserve the existing memory and store this detail separately."
                    ),
                    "skip": "An existing memory already captures this; do nothing.",
                },
            ),
        }

        if similar:
            questions["target_memory"] = Choice(
                instructions=(
                    "If `candidate_memory` updates an existing memory, which entry "
                    "in `existing_memories` does it refer to?"
                ),
                criteria={
                    **{
                        f"memory_{i}": hit.memory.text
                        for i, hit in enumerate(similar)
                    },
                    "none": "The candidate does not update any existing memory.",
                },
            )
            for i in range(len(similar)):
                questions[f"conflicts_{i}"] = Noul(
                    instructions=(
                        f"Does `candidate_memory` contradict `existing_memories[{i}]`? "
                        "They conflict if they cannot both be true about the same "
                        "subject at the same time. Compatible extra detail, a "
                        "restatement, or a different subject is not a conflict."
                    ),
                    criteria=NoulCriteria(
                        true=(
                            "The two statements cannot both be true about the same "
                            "subject at the same time."
                        ),
                        false=(
                            "Compatible extra detail, a restatement of the same fact, "
                            "a different subject, or no shared claim."
                        ),
                    ),
                )
            questions["resolution"] = Choice(
                instructions=(
                    "If `candidate_memory` contradicts an existing memory, how "
                    "should the store resolve it? Prefer the more recent, more "
                    "specific, or explicitly corrective statement."
                ),
                criteria={
                    "replace": (
                        "Overwrite the conflicting existing memory with the "
                        "candidate. The candidate is a correction, a move, or a "
                        "more recent statement of the same fact."
                    ),
                    "keep": (
                        "Leave the existing memory unchanged and drop the "
                        "candidate. The existing memory is still accurate."
                    ),
                },
            )

        answers = self._client.decide(state, questions)

        worth = float(answers["worth_remembering"]["noul"])
        operation_answer = answers["operation"]
        operation = operation_answer["choice"]
        operation_confidence = float(operation_answer.get("confidence") or 0.0)

        target_id: str | None = None
        if similar and operation == "update":
            target_answer = answers["target_memory"]
            if target_answer["choice"].startswith("memory_"):
                index = int(target_answer["choice"].removeprefix("memory_"))
                target_id = similar[index].memory.id

        conflict = 0.0
        if similar:
            conflict, conflict_target_id = max(
                (
                    float(answers[f"conflicts_{i}"]["noul"]),
                    hit.memory.id,
                )
                for i, hit in enumerate(similar)
            )
            if conflict >= self._conflict_threshold:
                resolution_answer = answers["resolution"]
                operation = resolution_answer["choice"]
                operation_confidence = float(resolution_answer.get("confidence") or 0.0)
                target_id = conflict_target_id

        return GateDecision(
            worth=worth,
            operation=operation,
            operation_confidence=operation_confidence,
            target_id=target_id,
            conflict=conflict,
        )

    def filter_relevant(self, query: str, hits: list[MemoryHit]) -> list[MemoryHit]:
        if not hits:
            return []
        state = {
            "query": query,
            "memories": [hit.memory.text for hit in hits],
        }
        questions = {
            f"relevant_{i}": Noul(
                instructions=(
                    f"Does `memories[{i}]` contain information that is useful "
                    "for answering `query`? Sharing a word or a nearby topic "
                    "is not enough."
                ),
                criteria=NoulCriteria(
                    true=(
                        "The memory is about the same subject, person, "
                        "preference, or fact the query is asking about, and "
                        "would help answer it."
                    ),
                    false=(
                        "The memory is a different subject, only shares a word "
                        "or topic neighborhood, or would not help answer the "
                        "query."
                    ),
                ),
            )
            for i in range(len(hits))
        }
        answers = self._client.decide(state, questions)
        kept = []
        for i, hit in enumerate(hits):
            relevance = float(answers[f"relevant_{i}"]["noul"])
            if relevance >= self._relevance_threshold:
                kept.append(replace(hit, relevance=relevance))
        return kept
