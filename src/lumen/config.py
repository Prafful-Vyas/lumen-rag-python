from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob is an env var (see .env.example). Defaults match a local, fully offline run:
    local Mongo + Redis, mock AI providers, brute-force vector search, files on disk."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = "mongodb://127.0.0.1:27017/lumen"
    redis_url: str = "redis://127.0.0.1:6379"
    jwt_secret: str = "dev-secret-change-me-before-any-real-deploy"
    jwt_ttl_days: int = 7
    port: int = 8000

    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024
    completion_model: str = "gpt-4o-mini"
    contextual_header_model: str = "gpt-4o-mini"

    vector_backend: Literal["local", "atlas"] = "local"
    storage_backend: Literal["local", "s3"] = "local"
    storage_root: str = "storage"

    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_s3_bucket: str | None = None
    aws_endpoint_url: str | None = None

    vector_candidate_limit: int = 40
    lexical_candidate_limit: int = 40
    rrf_k: int = 60
    rerank_top_k: int = 8
    context_token_budget: int = 6000
    refusal_score_threshold: float = 0.15
    near_duplicate_cosine_threshold: float = 0.95

    workspace_default_monthly_cap_usd: float = 50.0

    query_rate_limit_per_minute: int = 30
    max_upload_bytes: int = 100 * 1024 * 1024

    @property
    def using_real_ai_providers(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def mongo_db_name(self) -> str:
        tail = self.mongo_uri.rsplit("/", 1)[-1].split("?", 1)[0]
        return tail or "lumen"


@lru_cache
def get_settings() -> Settings:
    return Settings()
