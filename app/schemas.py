import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class ChatRequest(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"message": "What is RAG?"}]}
    )
    message: str = Field(min_length=1, max_length=get_settings().max_message_chars)
    conversation_id: uuid.UUID | None = None


class UsageStats(BaseModel):
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int
    ttft_ms: int | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    stats: UsageStats


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    ttft_ms: int | None
    created_at: datetime


class ConversationTotals(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    avg_latency_ms: float | None


class ConversationResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageOut]
    totals: ConversationTotals



class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    num_pages: int
    num_chunks: int
    created_at: datetime


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"query": "What is the refund policy?"}]}
    )

    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[uuid.UUID] | None = None


class SearchResult(BaseModel):
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    page: int | None
    chunk_index: int
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    retrieval_ms: int