from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr
    groq_model: str = "llama-3.3-70b-versatile"
    database_url: str

    system_prompt: str = "You are a helpful assistant. Answer clearly and concisely."
    max_history: int = 20
    max_message_chars: int = 4000
    demo_api_keys: str = ""

    @property
    def api_keys(self) -> set[str]:
        return {k.strip() for k in self.demo_api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()