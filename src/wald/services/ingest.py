"""Ingestion: turn a domain object into embedded chunks in the unified index.

Called after writes to wiki / resource / agent so search and RAG stay current.
For the stub this runs inline; move it to a background worker before scaling.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from wald.models import Agent, Embedding, Resource, WikiPage
from wald.services.embeddings import get_embedding_provider

_CHUNK_CHARS = 1200


def _chunk(text: str, size: int = _CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # TODO: smarter, semantically-aware chunking (headings, sentences).
    return [text[i : i + size] for i in range(0, len(text), size)]


def _reindex(session: Session, source_type: str, source_id: uuid.UUID, title: str, body: str) -> int:
    """Replace all embeddings for one source with freshly computed ones."""
    session.execute(
        delete(Embedding).where(
            Embedding.source_type == source_type, Embedding.source_id == source_id
        )
    )
    chunks = _chunk(f"{title}\n\n{body}")
    if not chunks:
        return 0
    vectors = get_embedding_provider().embed(chunks)
    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        session.add(
            Embedding(
                source_type=source_type,
                source_id=source_id,
                chunk_index=i,
                content=chunk,
                embedding=vector,
                meta={"title": title},
            )
        )
    return len(chunks)


def index_wiki_page(session: Session, page: WikiPage) -> int:
    return _reindex(session, "wiki", page.id, page.title, page.content)


def index_resource(session: Session, resource: Resource) -> int:
    body = f"{resource.description}\nkind: {resource.kind}\ntags: {', '.join(resource.tags)}"
    return _reindex(session, "resource", resource.id, resource.name, body)


def index_agent(session: Session, agent: Agent) -> int:
    body = f"{agent.description}\ncapabilities: {', '.join(agent.capabilities)}"
    return _reindex(session, "agent", agent.id, agent.name, body)
