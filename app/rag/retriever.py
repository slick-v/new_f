import uuid

from app.rag.embeddings import EmbeddingService
from app.repositories.documents import DocumentRepository
from app.schemas import SearchResult


class Retriever:
    def __init__(self, repo: DocumentRepository, embedder: EmbeddingService, default_top_k: int):
        self.repo = repo
        self.embedder = embedder
        self.default_top_k = default_top_k

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[SearchResult]:
        embedding = await self.embedder.embed_query(query)
        rows = await self.repo.search(embedding, top_k or self.default_top_k, document_ids)
        return [
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                page=row.page,
                chunk_index=row.chunk_index,
                content=row.content,
                score=round(1 - row.distance, 4),  # cosine similarity: 1 = identical
            )
            for row in rows
        ]