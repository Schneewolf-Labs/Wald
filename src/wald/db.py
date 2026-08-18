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
        # create_all does not alter existing tables, so a hub that predates agent tokens
        # would never acquire the column and every verification would error on a missing
        # attribute. Same reasoning as the index above.
        conn.execute(text("ALTER TABLE agent ADD COLUMN IF NOT EXISTS token_hash VARCHAR(64)"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_token_hash ON agent (token_hash)")
        )
    check_embedding_dim(target)


class SchemaMismatch(Exception):
    """The stored schema cannot hold what the current configuration produces."""


def check_embedding_dim(target: Engine) -> None:
    """Fail if the vector column's width disagrees with ``embed_dim``.

    The width is fixed when the table is created, so changing `WALD_EMBED_DIM` -- or
    switching to a provider whose model has a different one -- leaves a table that cannot
    store the new vectors. Postgres reports that as `expected 2048 dimensions, not 1024`
    on the first insert, which names neither the setting nor the table and is genuinely
    confusing to land on midway through loading content.

    Checked at schema time so it surfaces before a long re-index rather than during one.
    """
    with target.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'embedding'::regclass AND attname = 'embedding' "
                "AND NOT attisdropped"
            )
        ).scalar()

    if not stored or not stored.startswith("vector("):
        return
    width = int(stored[len("vector(") : -1])
    configured = _settings.embed_dim
    if width != configured:
        raise SchemaMismatch(
            f"the embedding table stores {width}-dimensional vectors but WALD_EMBED_DIM "
            f"is {configured}. The column width is fixed at table creation, so either set "
            f"WALD_EMBED_DIM={width}, or drop the embedding table and re-run wald-init-db "
            "followed by wald-seed. Rebuilding is cheap: the content directory is the "
            "source of truth and the database is a derived index of it."
        )


def init_db() -> None:
    """Create the schema in the configured database."""
    init_schema(engine)


def init_db_cli() -> None:
    """Console-script entry point (``wald-init-db``)."""
    init_db()
    print(f"Wald: schema initialized at {_settings.database_url}")
