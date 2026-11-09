from app.schemas import SearchResult

RAG_SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every fact with its source number in square brackets, like [2]. If an answer combines sources, cite each one, like [1][3].
- Some sources may be irrelevant to the question. Ignore them.
- If the sources don't contain the answer, reply: "I couldn't find that in the uploaded documents." Do not guess.
- Be concise."""


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        location = f"{r.filename}, page {r.page}" if r.page else r.filename
        blocks.append(f"[{i}] ({location})\n{r.content}")
    return "\n\n".join(blocks)


def build_user_message(question: str, results: list[SearchResult]) -> str:
    return f"Sources:\n\n{build_context(results)}\n\nQuestion: {question}"