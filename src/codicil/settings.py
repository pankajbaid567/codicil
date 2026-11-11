"""Typed configuration from environment variables (prefix CODICIL_). Nothing else reads the environment."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODICIL_", env_file=".env", extra="ignore")

    index_dir: Path = Path("index/multilingual-e5-small")
    models_dir: Path = Path("data/models")
    embedder: str = "multilingual-e5-small"
    rerank: bool = True
    min_dense: float = Field(0.84, ge=0.0, le=1.0)  # abstain below this top cosine; tied to the embedder, fitted on dev (docs/results.md)
    min_rerank: float = -0.18  # English abstention on the cross-encoder score, chosen on the dev split only (docs/results.md)
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_s: float = Field(12.0, gt=0)
    llm_send_temperature: bool = True  # set false for providers that deprecate sampling parameters (Gemini 3.5+)
    llm_reasoning_effort: str = ""  # optional OpenAI-style `reasoning_effort` (low/medium/high); empty = provider default
    qdrant_url: str = ""  # empty = in-process file index
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "codicil"
    rate_limit_per_min: int = Field(30, ge=1)
    cors_origins: str = ""
    max_question_chars: int = Field(500, ge=20)
    cache_size: int = Field(256, ge=0)
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key.get_secret_value() and self.llm_model)
