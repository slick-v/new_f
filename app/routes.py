import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.database import SessionLocal
from app.deps import LLMDep, RepoDep, SettingsDep, rate_limit,require_api_key
from app.llm import LLMError, LLMRateLimitError
from app.repository import ChatRepository
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageOut,
    UsageStats,
)

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(rate_limit)])


async def resolve_conversation(repo: ChatRepository, conversation_id: uuid.UUID | None) -> uuid.UUID:
    if conversation_id is None:
        return await repo.create_conversation()
    if not await repo.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation_id


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, repo: RepoDep, llm: LLMDep, settings: SettingsDep):
    conversation_id = await resolve_conversation(repo, req.conversation_id)
    history = await repo.load_history(conversation_id, settings.max_history)
    messages = llm.build_messages(history, req.message)

    try:
        reply, stats = await llm.complete(messages)
    except LLMRateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit hit, try again shortly")
    except LLMError:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail="The model service failed, try again")

    await repo.save_exchange(conversation_id, req.message, reply, stats)
    return ChatResponse(conversation_id=conversation_id, reply=reply, stats=stats)


@router.post("/stream_chat")
async def stream_chat(req: ChatRequest, repo: RepoDep, llm: LLMDep, settings: SettingsDep):
    conversation_id = await resolve_conversation(repo, req.conversation_id)
    history = await repo.load_history(conversation_id, settings.max_history)
    messages = llm.build_messages(history, req.message)

    async def generate():
        reply_parts = []
        stats = None

        try:
            async for item in llm.stream(messages):
                if isinstance(item, UsageStats):
                    stats = item
                else:
                    reply_parts.append(item)
                    yield json.dumps({"type": "token", "content": item}) + "\n"
        except LLMRateLimitError:
            yield json.dumps({"type": "error", "detail": "Rate limit hit, try again shortly"}) + "\n"
            return
        except LLMError:
            logger.exception("LLM stream failed")
            yield json.dumps({"type": "error", "detail": "The model service failed, try again"}) + "\n"
            return

        # Depending on the FastAPI version, the request's injected session can be
        # closed before the stream body runs, so save with a fresh session
        async with SessionLocal() as session:
            await ChatRepository(session).save_exchange(
                conversation_id, req.message, "".join(reply_parts), stats
            )

        yield json.dumps({
            "type": "stats",
            "conversation_id": str(conversation_id),
            **stats.model_dump(),
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: uuid.UUID, repo: RepoDep):
    messages = await repo.get_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    totals = await repo.get_totals(conversation_id)
    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[MessageOut.model_validate(m) for m in messages],
        totals=totals,
    )