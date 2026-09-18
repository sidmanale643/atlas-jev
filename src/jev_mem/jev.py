from collections.abc import Mapping
from typing import Any

import httpx
from typesafe_sdk import Choice, Noul, Question, Score

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"


def encode_question(question: Question) -> dict[str, Any]:
    match question:
        case dict():
            return dict(question)
        case Noul(instructions=instructions, criteria=criteria):
            payload: dict[str, Any] = {"type": "noul", "instructions": instructions}
            if criteria:
                payload["criteria"] = dict(criteria)
            return payload
        case Choice(instructions=instructions, criteria=criteria):
            return {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(criteria),
            }
        case Score(instructions=instructions, criteria=criteria):
            return {
                "type": "score",
                "instructions": instructions,
                "criteria": list(criteria),
            }
        case _:
            raise TypeError(f"Unsupported question type: {type(question)!r}")


class JevClient:
    """TypeSafe Jev via OpenRouter's Decisions API."""

    def __init__(self, api_key: str, model: str = DEFAULT_JEV_MODEL) -> None:
        self._model = model
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "jev-mem",
            },
            timeout=30.0,
        )

    def decide(self, state: Any, questions: Mapping[str, Question]) -> dict[str, Any]:
        response = self._http.post(
            DECISIONS_URL,
            json={
                "model": self._model,
                "state": state,
                "questions": {name: encode_question(question) for name, question in questions.items()},
            },
        )
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            payload = response.text
        if response.status_code >= 400:
            error = payload.get("error", payload) if isinstance(payload, dict) else payload
            raise RuntimeError(f"OpenRouter Decisions API error {response.status_code}: {error}")
        if not isinstance(payload, dict) or "answers" not in payload:
            raise RuntimeError(f"OpenRouter Decisions API returned an unexpected body: {payload}")
        answers = payload["answers"]
        if not isinstance(answers, dict):
            raise RuntimeError(f"OpenRouter Decisions API returned invalid answers: {answers}")
        return answers

    def close(self) -> None:
        self._http.close()
