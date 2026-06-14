"""Agent-to-agent (A2A) routing: discovery + a store-and-route mailbox.

v1 is intentionally transport-light — a message is persisted addressed to a
target agent, which polls its inbox. Push delivery (webhooks/streaming) and
richer conversation semantics are on the roadmap.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from wald.models import Agent, AgentMessage


def resolve_agent(session: Session, ref: str) -> Agent | None:
    """Look up an agent by slug or by UUID string."""
    try:
        agent_id = uuid.UUID(ref)
        return session.get(Agent, agent_id)
    except ValueError:
        return session.scalar(select(Agent).where(Agent.slug == ref))


def find_agents_by_capability(session: Session, capability: str) -> list[Agent]:
    """Discovery: agents advertising a given capability."""
    stmt = select(Agent).where(Agent.capabilities.any(capability), Agent.status == "active")
    return list(session.scalars(stmt))


def send_message(
    session: Session,
    *,
    from_agent: Agent,
    to_agent: Agent,
    content: str,
    role: str = "request",
    thread_id: uuid.UUID | None = None,
) -> AgentMessage:
    msg = AgentMessage(
        thread_id=thread_id or uuid.uuid4(),
        from_agent_id=from_agent.id,
        to_agent_id=to_agent.id,
        role=role,
        content=content,
        status="queued",
    )
    session.add(msg)
    session.flush()
    return msg


def inbox(session: Session, agent: Agent, *, unread_only: bool = False) -> list[AgentMessage]:
    stmt = select(AgentMessage).where(AgentMessage.to_agent_id == agent.id)
    if unread_only:
        stmt = stmt.where(AgentMessage.status.in_(("queued", "delivered")))
    stmt = stmt.order_by(AgentMessage.created_at.desc())
    return list(session.scalars(stmt))


def thread(session: Session, thread_id: uuid.UUID) -> list[AgentMessage]:
    stmt = (
        select(AgentMessage)
        .where(AgentMessage.thread_id == thread_id)
        .order_by(AgentMessage.created_at.asc())
    )
    return list(session.scalars(stmt))
