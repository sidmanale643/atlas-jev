import os
from dataclasses import dataclass

from dotenv import load_dotenv

from atlas_jev.jev import DEFAULT_JEV_MODEL

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str
    jev_model: str
    db_path: str
    embedding_model: str
    recall_limit: int
    recall_distance_threshold: float
    worth_threshold: float
    conflict_threshold: float


def load_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        jev_model=os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL),
        db_path=os.environ.get("ATLAS_JEV_DB", ".atlas_jev/lancedb"),
        embedding_model=os.environ.get("ATLAS_JEV_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        recall_limit=int(os.environ.get("ATLAS_JEV_RECALL_LIMIT", "3")),
        recall_distance_threshold=float(os.environ.get("ATLAS_JEV_RECALL_DISTANCE", "0.85")),
        worth_threshold=float(os.environ.get("ATLAS_JEV_WORTH_THRESHOLD", "0.5")),
        conflict_threshold=float(os.environ.get("ATLAS_JEV_CONFLICT_THRESHOLD", "0.7")),
    )
