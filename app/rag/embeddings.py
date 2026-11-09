import asyncio

from fastembed import TextEmbedding


class EmbeddingService:
    def __init__(self, model_name: str, dim: int):
        self.model = TextEmbedding(model_name=model_name)
        self.dim = dim

        actual = len(self._embed_query("dimension check"))
        if actual != dim:
            raise RuntimeError(
                f"{model_name} produces {actual}-dim vectors but EMBEDDING_DIM={dim}"
            )


    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # Embedding is CPU-heavy; run it off the event loop so other requests aren't blocked
        return await asyncio.to_thread(self._embed_passages, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_query, text)

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def _embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()