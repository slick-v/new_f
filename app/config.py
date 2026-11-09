from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"
    database_url: str

    system_prompt: str = "You are a helpful assistant. Answer clearly and concisely."
    max_history: int = 20
    max_message_chars: int = 4000
    demo_api_keys: str = ""
    max_output_tokens: int = 1024

    app_env: Literal["dev", "prod"] = "dev"
    rate_limit_per_minute: int = 10

    # RAG
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 100
    search_top_k: int = 5
    max_upload_mb: int = 10

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def api_keys(self) -> set[str]:
        return {k.strip() for k in self.demo_api_keys.split(",") if k.strip()}


    @model_validator(mode="after")
    def require_keys_in_prod(self):
        if self.app_env == "prod" and not self.api_keys:
            raise ValueError("DEMO_API_KEYS must be set when APP_ENV=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()