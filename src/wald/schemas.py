"""Pydantic request/response models for the REST + MCP surfaces."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Wiki -----------------------------------------------------------------
class WikiPageIn(BaseModel):
    slug: str
    title: str
    space: str = "general"
    content: str = ""
    tags: list[str] = Field(default_factory=list)


class WikiPageUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    author: str | None = None


class WikiPageOut(_ORM):
    id: uuid.UUID
    slug: str
    title: str
    space: str
    content: str
    tags: list[str]
    version: int
    created_at: datetime
    updated_at: datetime


# --- Resources ------------------------------------------------------------
class ResourceIn(BaseModel):
    slug: str
    name: str
    kind: str = "service"
    description: str = ""
    connection: dict[str, Any] = Field(default_factory=dict)
    auth: dict[str, Any] = Field(default_factory=dict)
    docs_url: str | None = None
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: str = "active"


class ResourceOut(_ORM):
    id: uuid.UUID
    slug: str
    name: str
    kind: str
    description: str
    connection: dict[str, Any]
    auth: dict[str, Any]
    docs_url: str | None
    owner: str | None
    tags: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


# --- Agents ---------------------------------------------------------------
class AgentIn(BaseModel):
    slug: str
    name: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    endpoint_url: str | None = None
    protocol: str = "mcp"
    auth: dict[str, Any] = Field(default_factory=dict)
    owner: str | None = None
    status: str = "active"
    agent_card: dict[str, Any] = Field(default_factory=dict)


class AgentOut(_ORM):
    id: uuid.UUID
    slug: str
    name: str
    description: str
    capabilities: list[str]
    endpoint_url: str | None
    protocol: str
    auth: dict[str, Any]
    owner: str | None
    status: str
    agent_card: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AgentMessageIn(BaseModel):
    from_agent: str  # slug or id
    to_agent: str  # slug or id
    content: str
    role: str = "request"
    thread_id: uuid.UUID | None = None


class AgentMessageOut(_ORM):
    id: uuid.UUID
    thread_id: uuid.UUID
    from_agent_id: uuid.UUID
    to_agent_id: uuid.UUID
    role: str
    content: str
    status: str
    created_at: datetime


# --- Search / RAG ---------------------------------------------------------
class SearchHit(BaseModel):
    source_type: str
    source_id: uuid.UUID
    title: str
    snippet: str
    score: float


class SearchResults(BaseModel):
    query: str
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str
    top_k: int = 6


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[SearchHit]
    synthesized: bool  # False in dev mode (retrieval only, no LLM)
