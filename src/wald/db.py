"""Database engine, session factory, and schema bootstrap."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
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


def init_db() -> None:
    """Enable pgvector and create all tables. Idempotent.

    TODO: replace with Alembic migrations before this leaves the prototype stage.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def init_db_cli() -> None:
    """Console-script entry point (``wald-init-db``)."""
    init_db()
    print(f"Wald: schema initialized at {_settings.database_url}")
