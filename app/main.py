import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.database import engine, init_db
from app.deps import SessionDep
from app.routes import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title="Groq Chat API",
    version="0.2.0",
    description="Chat API with streaming, Postgres history, and token/latency tracking.",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
async def health(session: SessionDep):
    pgvector = await session.scalar(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    )
    return {"status": "ok", "database": "ok", "pgvector": pgvector}