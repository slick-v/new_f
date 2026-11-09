import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.api import chat, documents
from app.database import engine, init_db
from app.deps import SessionDep, get_embedder

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Load (and on first run, download) the embedding model now, not on the first request
    await asyncio.to_thread(get_embedder)
    yield
    await engine.dispose()


app = FastAPI(
    title="Groq Chat API",
    version="0.2.0",
    description="Chat API with streaming, Postgres history, and token/latency tracking.",
    lifespan=lifespan,
)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health")
async def health(session: SessionDep):
    pgvector = await session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    return {"status": "ok", "database": "ok", "pgvector": pgvector}