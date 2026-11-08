import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from groq import APIError, RateLimitError

from app import config, db
from app.auth import require_api_key
from app.llm import build_messages, client, usage_stats
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


def resolve_conversation(conversation_id: str | None) -> str:
    if conversation_id is None:
        return db.create_conversation()
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation_id


@router.post("/chat")
async def chat(req: ChatRequest):
    conversation_id = resolve_conversation(req.conversation_id)
    messages = build_messages(conversation_id, req.message)

    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(model=config.MODEL, messages=messages)
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit hit, try again shortly")
    except APIError:
        logger.exception("Groq API error")
        raise HTTPException(status_code=502, detail="The model service failed, try again")
    latency_ms = round((time.perf_counter() - start) * 1000)

    reply = response.choices[0].message.content
    stats = usage_stats(response.usage, latency_ms)
    db.save_exchange(conversation_id, req.message, reply, stats)

    return {"conversation_id": conversation_id, "reply": reply, "stats": stats}


@router.post("/stream_chat")
async def stream_chat(req: ChatRequest):
    conversation_id = resolve_conversation(req.conversation_id)
    messages = build_messages(conversation_id, req.message)

    async def generate():
        start = time.perf_counter()
        ttft_ms = None
        usage = None
        reply_parts = []

        try:
            stream = await client.chat.completions.create(
                model=config.MODEL, messages=messages, stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - start) * 1000)
                    text = chunk.choices[0].delta.content
                    reply_parts.append(text)
                    yield json.dumps({"type": "token", "content": text}) + "\n"

                # Groq puts token usage on the final chunk under x_groq
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq and getattr(x_groq, "usage", None):
                    usage = x_groq.usage
        except RateLimitError:
            yield json.dumps({"type": "error", "detail": "Rate limit hit, try again shortly"}) + "\n"
            return
        except APIError:
            logger.exception("Groq API error during stream")
            yield json.dumps({"type": "error", "detail": "The model service failed, try again"}) + "\n"
            return

        latency_ms = round((time.perf_counter() - start) * 1000)
        stats = usage_stats(usage, latency_ms, ttft_ms)
        db.save_exchange(conversation_id, req.message, "".join(reply_parts), stats)

        yield json.dumps({"type": "stats", "conversation_id": conversation_id, **stats}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation