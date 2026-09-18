from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, TypeSafeClient

from jev_mem.store import MemoryHit


@dataclass(frozen=True)
class GateDecision:
    worth: float
    operation: str  # "add" | "update" | "skip"
    operation_confidence: float
    target_id: str | None


class MemoryGate:
    """Jev (TypeSafe System One) judgments over candidate memories.

    One request per candidate, with speculative fan-out: the operation and
    target questions are answered even when the candidate turns out not to be
    worth remembering, and the code simply ignores those answers.
    """

    def __init__(self, model: str | None = None) -> None:
        self._client = TypeSafeClient()
        self._model = model

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
                    "add": "No existing memory covers this; store it as new.",
                    "update": (
                        "An existing memory is about the same fact and should be "
                        "revised or merged with the candidate."
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

        kwargs = {"state": state, "questions": questions}
        if self._model:
            kwargs["model"] = self._model
        response = self._client.system_one(**kwargs)

        worth = response.answers["worth_remembering"].noul
        operation_answer = response.answers["operation"]

        target_id: str | None = None
        if similar and operation_answer.choice == "update":
            target_answer = response.answers["target_memory"]
            if target_answer.choice.startswith("memory_"):
                index = int(target_answer.choice.removeprefix("memory_"))
                target_id = similar[index].memory.id

        return GateDecision(
            worth=worth,
            operation=operation_answer.choice,
            operation_confidence=operation_answer.confidence,
            target_id=target_id,
        )
