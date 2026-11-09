import asyncio
import hashlib
import logging
import uuid

from app.config import Settings
from app.models import Chunk, Document
from app.rag.chunking import chunk_pages
from app.rag.embeddings import EmbeddingService
from app.rag.errors import DuplicateDocumentError, IngestError
from app.rag.loaders import detect_content_type, load_pages
from app.repositories.documents import DocumentRepository

logger = logging.getLogger(__name__)



class IngestService:
    def __init__(self, repo: DocumentRepository, embedder: EmbeddingService, settings: Settings):
        self.repo = repo
        self.embedder = embedder
        self.settings = settings

    async def ingest(self, filename: str, data: bytes) -> Document:
        content_type = detect_content_type(filename)

        sha256 = hashlib.sha256(data).hexdigest()
        existing = await self.repo.get_by_hash(sha256)
        if existing:
            raise DuplicateDocumentError(existing.id)

        pages = await asyncio.to_thread(load_pages, filename, data)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise IngestError("No extractable text found (scanned PDFs need OCR)")

        embeddings = await self.embedder.embed_passages([c.content for c in chunks])

        document = Document(
            id=uuid.uuid4(),
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            sha256=sha256,
            num_pages=len(pages),
            num_chunks=len(chunks),
        )
        rows = [
            Chunk(
                document_id=document.id,
                chunk_index=chunk.index,
                page=chunk.page,
                content=chunk.content,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        await self.repo.create_with_chunks(document, rows)

        logger.info("Ingested %s: %d pages, %d chunks", filename, len(pages), len(chunks))
        return document