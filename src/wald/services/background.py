"""Background re-embedding, so a write does not block on an embedding API call.

Ingestion used to run inline: create a wiki page, and the HTTP response waits while every
chunk goes through the embedding provider. Tolerable with the local hash fallback,
seconds-per-write against a real provider -- and a provider outage turned into a *write*
outage, which is backwards: the write is the durable thing, the index is derived from it.

The shape here: the write path commits its transaction first, then hands
``(source_type, source_id)`` to a single worker thread. The worker opens its own session,
re-reads the object as committed, and rebuilds its chunks. Working from the id rather
than a snapshot means a task enqueued twice for the same object just redoes the same
idempotent delete-and-reinsert against current state -- there is no stale payload to
apply out of order.

One worker, deliberately: embedding calls dominate the cost and providers rate-limit, so
parallelism buys little, and a single thread serializes reindexes of the same object
without any locking.

The tradeoff is a window where a write has landed but search does not see it (or sees the
previous text). That is the right trade for an information hub -- the content directory
and the row are the truth, the index catches up. `WALD_BACKGROUND_INDEXING=false` restores
inline indexing for tests and for setups that want read-your-writes search; the seed
loader always indexes inline regardless, because its dry-run mode must count chunks and
roll everything back in one transaction.

A failed background reindex logs and leaves the previous chunks standing, so search
degrades to slightly-stale rather than to a half-indexed document.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy.orm import Session

from wald.config import get_settings
from wald.models import Agent, Resource, WikiPage
from wald.services import ingest

_log = logging.getLogger(__name__)

_INDEXERS: dict[str, tuple[type, Any]] = {
    "wiki": (WikiPage, ingest.index_wiki_page),
    "resource": (Resource, ingest.index_resource),
    "agent": (Agent, ingest.index_agent),
}

_executor: ThreadPoolExecutor | None = None


def _session_factory() -> Session:
    # Resolved at call time (and patchable in tests) rather than imported at module load,
    # so the worker binds to whatever engine the application is actually using.
    from wald.db import SessionLocal

    return SessionLocal()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wald-index")
    return _executor


def _reindex(source_type: str, source_id: uuid.UUID) -> None:
    model, indexer = _INDEXERS[source_type]
    try:
        with _session_factory() as session:
            obj = session.get(model, source_id)
            if obj is None:
                # Deleted between the write and the reindex; its chunks go when it goes
                # (or with the next rebuild), and there is nothing to embed.
                return
            indexer(session, obj)
            session.commit()
    except Exception:
        _log.exception("background reindex failed: %s %s", source_type, source_id)


def schedule(source_type: str, source_id: uuid.UUID) -> None:
    if source_type not in _INDEXERS:
        raise ValueError(f"unknown source_type '{source_type}'")
    _get_executor().submit(_reindex, source_type, source_id)


def finish_write(session: Session, source_type: str, obj: Any) -> None:
    """Commit a write and index it: inline when background indexing is off, else queued.

    Ordering is the point of this function existing. Inline, the chunks join the same
    transaction as the write. In the background, the commit must come *first* -- the
    worker reads through its own session, and a task scheduled before the commit would
    race it and could embed the previous version.
    """
    if not get_settings().background_indexing:
        _, indexer = _INDEXERS[source_type]
        indexer(session, obj)
        session.commit()
        return
    session.commit()
    schedule(source_type, obj.id)


def drain() -> None:
    """Run every queued reindex to completion. For shutdown paths and tests."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
