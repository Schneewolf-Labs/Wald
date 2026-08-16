"""Database engine, session factory, and schema bootstrap."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from wald.config import get_settings
from wald.models import Base

_settings = get_settings()

engine = create_engine(_settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_schema(target: Engine) -> None:
    """Create everything Wald needs in `target`. Idempotent.

    Takes an engine so the test suite builds its side database through exactly this code
    path. A fixture that hand-rolled its own schema would drift from the real one, and the
    first thing to drift is usually an index -- which does not fail a test, it just makes
    the query it was written for silently slow.

    TODO: replace with Alembic migrations before this leaves the prototype stage.
    """
    with target.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(target)
    with target.begin() as conn:
        # Functional GIN index matching the expression in services/search.py exactly; a
        # mismatch leaves the planner sequentially scanning every chunk. Declared here
        # rather than on the model because create_all skips tables that already exist,
        # indexes included, so an existing deployment would never acquire it.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_embedding_content_fts "
                "ON embedding USING GIN (to_tsvector('english', content))"
            )
        )


def init_db() -> None:
    """Create the schema in the configured database."""
    init_schema(engine)


def init_db_cli() -> None:
    """Console-script entry point (``wald-init-db``)."""
    init_db()
    print(f"Wald: schema initialized at {_settings.database_url}")
