import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str
    typesafe_model: str | None
    db_path: str
    embedding_model: str
    recall_limit: int
    recall_distance_threshold: float
    worth_threshold: float


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        typesafe_model=os.environ.get("TYPESAFE_MODEL"),
        db_path=os.environ.get("JEV_MEM_DB", ".jev_mem/lancedb"),
        embedding_model=os.environ.get("JEV_MEM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        recall_limit=int(os.environ.get("JEV_MEM_RECALL_LIMIT", "3")),
        recall_distance_threshold=float(os.environ.get("JEV_MEM_RECALL_DISTANCE", "0.85")),
        worth_threshold=float(os.environ.get("JEV_MEM_WORTH_THRESHOLD", "0.5")),
    )
