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
    llm_provider: str = "openai"  # "openai" (any OpenAI-compatible endpoint) or "bedrock" (Amazon Bedrock Converse, credentials from the AWS chain)
    llm_region: str = ""  # bedrock only; empty = the AWS default region
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout_s: float = Field(12.0, gt=0)
    llm_send_temperature: bool = True  # set false for providers that deprecate sampling parameters (Gemini 3.5+)
    llm_reasoning_effort: str = ""  # optional OpenAI-style `reasoning_effort` (low/medium/high); empty = provider default
    qdrant_url: str = ""  # empty = in-process file index
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "codicil"
    freshness_file: Path = Path("data/corpus/freshness.json")  # written by scripts/refresh_corpus.py; shown in the UI when present
    s3vectors_bucket: str = ""  # empty = Amazon S3 Vectors off; credentials come from the AWS chain (instance role on EC2)
    s3vectors_index: str = ""  # empty = codicil-<first 12 chars of the index sha>
    s3vectors_region: str = ""  # empty = the AWS default region
    rate_limit_per_min: int = Field(30, ge=1)
    cors_origins: str = ""
    max_question_chars: int = Field(500, ge=20)
    cache_size: int = Field(256, ge=0)
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "bedrock":
            return bool(self.llm_model)
        return bool(self.llm_base_url and self.llm_api_key.get_secret_value() and self.llm_model)
