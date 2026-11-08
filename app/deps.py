import time
from collections import defaultdict, deque

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_session
from app.llm import LLMService
from app.repository import ChatRepository

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_repository(session: SessionDep) -> ChatRepository:
    return ChatRepository(session)


@lru_cache
def get_llm() -> LLMService:
    return LLMService(get_settings())


RepoDep = Annotated[ChatRepository, Depends(get_repository)]
LLMDep = Annotated[LLMService, Depends(get_llm)]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# def require_api_key(settings: SettingsDep, api_key: str | None = Security(api_key_header)):
#     # No keys configured = open access, for local development
#     if settings.api_keys and api_key not in settings.api_keys:
#         raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str):
        now = time.monotonic()
        hits = self.hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = int(self.window - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests, slow down",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


def require_api_key(settings: SettingsDep, api_key: str | None = Security(api_key_header)) -> str:
    # No keys configured = open access, for local development
    if settings.api_keys and api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return api_key or "anonymous"


def rate_limit(
    api_key: Annotated[str, Depends(require_api_key)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
):
    limiter.check(api_key)