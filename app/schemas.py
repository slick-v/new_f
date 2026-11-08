from pydantic import BaseModel, Field

from app.config import MAX_MESSAGE_CHARS


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = None