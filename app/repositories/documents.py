import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_hash(self, sha256: str) -> Document | None:
        return await self.session.scalar(select(Document).where(Document.sha256 == sha256))

    async def create_with_chunks(self, document: Document, chunks: list[Chunk]):
        # One transaction: either the document and all its chunks are saved, or nothing is
        self.session.add(document)
        await self.session.flush()  # INSERT the document first so the chunks' foreign key is valid
        self.session.add_all(chunks)
        await self.session.commit()

    async def delete(self, document_id: uuid.UUID) -> bool:
        # Chunks are removed by the database (ON DELETE CASCADE on chunks.document_id)
        result = await self.session.execute(delete(Document).where(Document.id == document_id))
        await self.session.commit()
        return result.rowcount > 0

    async def search(
        self,
        embedding: list[float],
        top_k: int,
        document_ids: list[uuid.UUID] | None = None,
    ):
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Chunk.chunk_index,
                Chunk.page,
                Chunk.content,
                distance,
            )
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance)
            .limit(top_k)
        )
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return (await self.session.execute(stmt)).all()