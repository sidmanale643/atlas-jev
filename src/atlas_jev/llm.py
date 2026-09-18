from enum import StrEnum

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    DECISION = "decision"
    PLAN = "plan"
    CONSTRAINT = "constraint"
    OTHER = "other"


class ExtractedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        description="A single self-contained long-term memory, pronouns resolved."
    )
    type: MemoryType = Field(
        description=(
            "preference, fact, goal, relationship, decision, plan, constraint, or other."
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="How sure this is a durable, correctly typed memory, from 0 to 1.",
    )


class MemoryExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[ExtractedMemory] = Field(
        description="Discrete long-term memories. Empty if nothing is worth remembering."
    )


EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extractor for a personal assistant. From the user's text, \
extract discrete facts worth remembering long-term.

Rules:
- Each memory is a single self-contained statement, understandable without \
the original text.
- Resolve pronouns and relative references where possible.
- Do not extract small talk, greetings, or transient details.
- If nothing is worth remembering, return an empty list.
- Classify each memory with a type:
  - preference: likes, dislikes, and tastes (e.g. "hates cilantro")
  - fact: stable personal facts (e.g. "lives in Berlin", "dog is named Pixel")
  - goal: something the user wants to achieve
  - relationship: people, pets, or organizations in the user's life
  - decision: a choice the user has made
  - plan: an intended future action or event
  - constraint: a limitation or requirement the user must work within
  - other: worth remembering, but none of the above fit
- Set confidence from 0 to 1 for how sure you are that the statement is a \
durable, correctly typed memory grounded in the source text."""


class LLMService:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._model = model

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def extract_memories(self, text: str) -> list[ExtractedMemory]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_extraction",
                    "strict": True,
                    "schema": MemoryExtraction.model_json_schema(),
                },
            },
            extra_body={"provider": {"require_parameters": True}},
        )
        message = response.choices[0].message
        if message.refusal:
            raise RuntimeError(f"Memory extraction refused: {message.refusal}")
        content = message.content
        if not content:
            raise RuntimeError("Memory extraction returned no structured output")
        extraction = MemoryExtraction.model_validate_json(content)
        return [memory for memory in extraction.memories if memory.text.strip()]
