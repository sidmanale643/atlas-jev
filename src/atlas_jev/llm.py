from enum import StrEnum

from openai import OpenAI
from pydantic import BaseModel


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
    text: str
    type: MemoryType


class MemoryExtraction(BaseModel):
    memories: list[ExtractedMemory]


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
  - other: worth remembering, but none of the above fit"""


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
        parsed = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format=MemoryExtraction,
            temperature=0,
        )
        extraction = parsed.choices[0].message.parsed
        if extraction is None:
            return []
        return [m for m in extraction.memories if m.text.strip()]
