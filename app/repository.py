import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message
from app.schemas import UsageStats


class ChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_conversation(self) -> uuid.UUID:
        conversation = Conversation()
        self.session.add(conversation)
        await self.session.commit()
        return conversation.id

    async def conversation_exists(self, conversation_id: uuid.UUID) -> bool:
        return await self.session.get(Conversation, conversation_id) is not None

    async def load_history(self, conversation_id: uuid.UUID, limit: int) -> list[dict]:
        stmt = (
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    async def save_exchange(
        self, conversation_id: uuid.UUID, user_message: str, reply: str, stats: UsageStats
    ):
        self.session.add_all([
            Message(conversation_id=conversation_id, role="user", content=user_message),
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=reply,
                **stats.model_dump(),
            ),
        ])
        await self.session.commit()

    async def get_messages(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list((await self.session.scalars(stmt)).all())

    async def get_totals(self, conversation_id: uuid.UUID) -> dict:
        stmt = select(
            func.coalesce(func.sum(Message.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(Message.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(Message.total_tokens), 0).label("total_tokens"),
            func.round(func.avg(Message.latency_ms)).label("avg_latency_ms"),
        ).where(
            Message.conversation_id == conversation_id,
            Message.role == "assistant",
        )
        return (await self.session.execute(stmt)).one()._asdict()