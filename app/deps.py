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
from app.repositories.chat import ChatRepository

from app.rag.embeddings import EmbeddingService
from app.rag.ingest import IngestService
from app.rag.retriever import Retriever
from app.repositories.documents import DocumentRepository




SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_repository(session: SessionDep) -> ChatRepository:
    return ChatRepository(session)


@lru_cache
def get_llm() -> LLMService:
    return LLMService(get_settings())


RepoDep = Annotated[ChatRepository, Depends(get_repository)]
LLMDep = Annotated[LLMService, Depends(get_llm)]


@lru_cache
def get_embedder() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model, settings.embedding_dim)


EmbedderDep = Annotated[EmbeddingService, Depends(get_embedder)]

def get_document_repository(session: SessionDep) -> DocumentRepository:
    return DocumentRepository(session)


DocumentRepoDep = Annotated[DocumentRepository, Depends(get_document_repository)]


def get_ingest_service(
    repo: DocumentRepoDep, embedder: EmbedderDep, settings: SettingsDep
) -> IngestService:
    return IngestService(repo, embedder, settings)


def get_retriever(repo: DocumentRepoDep, embedder: EmbedderDep, settings: SettingsDep) -> Retriever:
    return Retriever(repo, embedder, settings.search_top_k)


IngestDep = Annotated[IngestService, Depends(get_ingest_service)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
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