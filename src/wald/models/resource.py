"""Resource directory: enterprise systems, datasources, and tools."""

from __future__ import annotations

from typing import Any

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from wald.models.base import Base, Timestamps, UUIDPrimaryKey

# kind: database | api | service | tool | dataset
# status: active | deprecated | planned


class Resource(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "resource"

    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(512))
    kind: Mapped[str] = mapped_column(String(64), default="service", index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    # How to reach it: hosts, endpoints, ports, params. Free-form JSON.
    connection: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    # How to authenticate: method + *references* to secrets (never raw secrets).
    auth: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    docs_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
