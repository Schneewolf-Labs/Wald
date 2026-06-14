"""Wiki: versioned markdown pages."""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from wald.models.base import Base, Timestamps, UUIDPrimaryKey


class WikiPage(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "wiki_page"

    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512))
    space: Mapped[str] = mapped_column(String(128), default="general", index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)

    revisions: Mapped[list["WikiPageRevision"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )


class WikiPageRevision(UUIDPrimaryKey, Timestamps, Base):
    """Immutable snapshot of a page at a given version."""

    __tablename__ = "wiki_page_revision"

    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("wiki_page.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)

    page: Mapped[WikiPage] = relationship(back_populates="revisions")
