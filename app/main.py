import logging

from fastapi import FastAPI

from app import db
from app.routes import router

logging.basicConfig(level=logging.INFO)
db.init_db()

app = FastAPI(
    title="Groq Chat API",
    version="0.1.0",
    description="Chat API with streaming, persistent history, and token/latency tracking.",
)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}