"""Background re-embedding: ordering, worker behavior, and the inline escape hatch."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from wald.config import get_settings
from wald.models import Embedding, WikiPage
from wald.services import background


def _chunks(session, source_id):
    return session.scalars(select(Embedding).where(Embedding.source_id == source_id)).all()


def test_inline_mode_indexes_before_returning(session):
    # WALD_BACKGROUND_INDEXING=false is the suite-wide default (conftest): the chunks must
    # be in the same transaction as the write.
    page = WikiPage(slug="inline-page", title="Inline", content="indexed synchronously")
    session.add(page)
    session.flush()
    background.finish_write(session, "wiki", page)
    assert _chunks(session, page.id)


def test_schedule_rejects_unknown_source_type():
    with pytest.raises(ValueError):
        background.schedule("mailbox", uuid.uuid4())


def test_worker_reindexes_committed_write(_engine, monkeypatch):
    """The full background path: commit, schedule, worker session, chunks appear.

    Runs on real committed rows (the worker's own session cannot see an uncommitted
    savepoint), so it cleans up after itself.
    """
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    monkeypatch.setattr(background, "_session_factory", factory)
    monkeypatch.setattr(
        background,
        "get_settings",
        lambda: get_settings().model_copy(update={"background_indexing": True}),
    )

    slug = f"bg-{uuid.uuid4().hex[:8]}"
    page_id = None
    try:
        with factory() as session:
            page = WikiPage(slug=slug, title="Background", content="indexed on the worker")
            session.add(page)
            session.flush()
            page_id = page.id
            background.finish_write(session, "wiki", page)
            # The write itself is already committed and visible...
            assert session.scalar(select(WikiPage).where(WikiPage.slug == slug)) is not None
        background.drain()
        with factory() as session:
            assert _chunks(session, page_id)
    finally:
        background.drain()
        with factory() as session:
            if page_id is not None:
                session.execute(delete(Embedding).where(Embedding.source_id == page_id))
            session.execute(delete(WikiPage).where(WikiPage.slug == slug))
            session.commit()


def test_worker_skips_a_row_deleted_before_it_ran(_engine, monkeypatch):
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    monkeypatch.setattr(background, "_session_factory", factory)

    # Never raises, never writes: the object is simply gone.
    background._reindex("wiki", uuid.uuid4())
