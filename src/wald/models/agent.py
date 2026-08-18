"""Agent directory and agent-to-agent (A2A) messaging."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from wald.models.base import Base, Timestamps, UUIDPrimaryKey

# protocol: mcp | a2a | http
# status: active | offline | retired


class Agent(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "agent"

    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    capabilities: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    endpoint_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    protocol: Mapped[str] = mapped_column(String(32), default="mcp", index=True)
    auth: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    # Full descriptor (capabilities schema, contact, tags, etc.).
    agent_card: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    # SHA-256 of the agent's bearer token, never the token itself. The hub holds the means to
    # *check* an identity, not to present one, so a database dump leaks who exists rather than
    # how to impersonate them. Indexed because verification looks up by hash on every call.
    token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class AgentMessage(UUIDPrimaryKey, Timestamps, Base):
    """A single message routed between two agents (the A2A mailbox substrate).

    status: queued | delivered | read | failed
    role:   sender's role label, e.g. "request" | "response" | "notify"
    """

    __tablename__ = "agent_message"

    thread_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True, default=uuid.uuid4)
    from_agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent.id", ondelete="CASCADE"), index=True
    )
    to_agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32), default="request")
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
