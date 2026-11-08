from groq import AsyncGroq

from app import config, db

client = AsyncGroq(api_key=config.GROQ_API_KEY)


def build_messages(conversation_id: str, user_message: str) -> list[dict]:
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        *db.load_history(conversation_id),
        {"role": "user", "content": user_message},
    ]


def usage_stats(usage, latency_ms: int, ttft_ms: int | None = None) -> dict:
    return {
        "model": config.MODEL,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
    }