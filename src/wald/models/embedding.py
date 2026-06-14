"""Unified embedding index across all pillars (powers semantic search + RAG)."""

from __future__ import annotations

import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from wald.config import get_settings
from wald.models.base import Base, Timestamps, UUIDPrimaryKey

_EMBED_DIM = get_settings().embed_dim

# source_type: wiki | resource | agent


class Embedding(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "embedding"

    source_type: Mapped[str] = mapped_column(String(32), index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBED_DIM))
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
