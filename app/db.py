
import sqlite3
import uuid
from contextlib import contextmanager

from app.config import DB_PATH, MAX_HISTORY


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                model           TEXT,
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                total_tokens    INTEGER,
                latency_ms      INTEGER,
                ttft_ms         INTEGER,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id);
        """)


def create_conversation() -> str:
    conversation_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("INSERT INTO conversations (id) VALUES (?)", (conversation_id,))
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def load_history(conversation_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id = ?
               ORDER BY id DESC LIMIT ?""",
            (conversation_id, MAX_HISTORY),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_exchange(conversation_id: str, user_message: str, reply: str, stats: dict):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
            (conversation_id, user_message),
        )
        conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, model, input_tokens, output_tokens,
                total_tokens, latency_ms, ttft_ms)
               VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id, reply, stats["model"],
                stats["input_tokens"], stats["output_tokens"], stats["total_tokens"],
                stats["latency_ms"], stats.get("ttft_ms"),
            ),
        )


def get_conversation(conversation_id: str) -> dict | None:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        if not rows:
            return None
        totals = conn.execute(
            """SELECT COALESCE(SUM(input_tokens), 0)  AS input_tokens,
                      COALESCE(SUM(output_tokens), 0) AS output_tokens,
                      COALESCE(SUM(total_tokens), 0)  AS total_tokens,
                      ROUND(AVG(latency_ms))          AS avg_latency_ms
               FROM messages
               WHERE conversation_id = ? AND role = 'assistant'""",
            (conversation_id,),
        ).fetchone()
    return {"messages": [dict(r) for r in rows], "totals": dict(totals)}