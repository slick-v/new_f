import time
from collections.abc import AsyncIterator

from groq import APIError, AsyncGroq, RateLimitError

from app.config import Settings
from app.schemas import UsageStats


class LLMError(Exception):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())

    def build_messages(self, history: list[dict], user_message: str) -> list[dict]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

    async def complete(self, messages: list[dict]) -> tuple[str, UsageStats]:
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.groq_model, messages=messages, max_tokens=self.settings.max_output_tokens,
            )
        except RateLimitError as e:
            raise LLMRateLimitError() from e
        except APIError as e:
            raise LLMError() from e

        latency_ms = round((time.perf_counter() - start) * 1000)
        return response.choices[0].message.content, self._stats(response.usage, latency_ms)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str | UsageStats]:
        """Yields text chunks, then one final UsageStats."""
        start = time.perf_counter()
        ttft_ms = None
        usage = None

        try:
            stream = await self.client.chat.completions.create(
                model=self.settings.groq_model, messages=messages, max_tokens=self.settings.max_output_tokens, stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - start) * 1000)
                    yield chunk.choices[0].delta.content

                # Groq puts token usage on the final chunk under x_groq
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq and getattr(x_groq, "usage", None):
                    usage = x_groq.usage
        except RateLimitError as e:
            raise LLMRateLimitError() from e
        except APIError as e:
            raise LLMError() from e

        latency_ms = round((time.perf_counter() - start) * 1000)
        yield self._stats(usage, latency_ms, ttft_ms)

    def _stats(self, usage, latency_ms: int, ttft_ms: int | None = None) -> UsageStats:
        return UsageStats(
            model=self.settings.groq_model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        )