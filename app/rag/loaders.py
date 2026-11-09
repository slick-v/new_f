import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.rag.errors import IngestError

SUPPORTED_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


@dataclass
class Page:
    number: int | None  # None for files without pages (txt, md)
    text: str


def detect_content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        allowed = ", ".join(SUPPORTED_TYPES)
        raise IngestError(f"Unsupported file type '{ext}'. Allowed: {allowed}")
    return SUPPORTED_TYPES[ext]

def load_pages(filename: str, data: bytes) -> list[Page]:
    if Path(filename).suffix.lower() == ".pdf":
        return _load_pdf(data)
    return [Page(number=None, text=_clean(_decode(data)))]


def _load_pdf(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as e:
        raise IngestError("Could not read PDF (corrupt or password-protected)") from e

    boilerplate = _find_boilerplate(raw_pages)
    pages = []
    for number, raw in enumerate(raw_pages, start=1):
        lines = [line for line in raw.splitlines() if _line_key(line) not in boilerplate]
        text = _clean("\n".join(lines))
        if text:
            pages.append(Page(number=number, text=text))
    return pages


def _line_key(line: str) -> str:
    # Treat "Page 3 of 10" and "Page 4 of 10" as the same line
    return re.sub(r"\d+", "#", line.strip().lower())


def _find_boilerplate(raw_pages: list[str], min_share: float = 0.6) -> set[str]:
    """Lines that repeat on most pages: headers, footers, page numbers."""
    if len(raw_pages) < 3:
        return set()
    counts = Counter()
    for raw in raw_pages:
        counts.update({_line_key(line) for line in raw.splitlines() if line.strip()})
    threshold = len(raw_pages) * min_share
    return {key for key, n in counts.items() if n >= threshold}


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise IngestError("Text files must be UTF-8 encoded") from e


def _clean(text: str) -> str:
    text = text.replace("\x00", "")         # Postgres rejects NUL characters
    text = re.sub(r"[ \t]+", " ", text)     # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # keep paragraph breaks, drop the rest
    return text.strip()