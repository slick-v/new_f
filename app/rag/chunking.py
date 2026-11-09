import re
from dataclasses import dataclass

from app.rag.loaders import Page


# Try to split on the biggest natural boundary first: paragraph, line, sentence, word
SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


@dataclass
class TextChunk:
    index: int
    page: int | None
    content: str



def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[TextChunk]:
    # Chunk page by page so every chunk maps to exactly one page for citations
    chunks: list[TextChunk] = []
    for page in pages:
        for content in split_text(page.text, chunk_size, overlap):
            chunks.append(TextChunk(index=len(chunks), page=page.number, content=content))
    return chunks

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    units = _split_units(text, chunk_size, SEPARATORS)
    return _merge_units(units, chunk_size, overlap)


def _split_units(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # No boundary left (e.g. one giant word): hard cut
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    units = []
    for part in re.split(separators[0], text):
        units.extend(_split_units(part, chunk_size, separators[1:]))
    return units


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        if current and current_len + len(unit) + 1 > chunk_size:
            chunks.append(" ".join(current))

            # Carry the last few units into the next chunk as overlap
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            if carry_len + len(unit) > chunk_size:
                carry, carry_len = [], 0

            current, current_len = carry, carry_len

        current.append(unit)
        current_len += len(unit) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks