"""Shared fixtures.

Two rules.

**Tests never touch the real hub.** They run against a side database (`<db>_test`), created
on demand. Wald exists to hold an organization's private material, so a suite that wrote
into the configured database would be writing into exactly the data nobody can afford to
have clobbered -- and the quickstart tells people to load content into it.

**Tests that do not need Postgres never open a connection**, and tests that do skip
cleanly when it is absent, so `pytest` stays useful on a machine with no Docker.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError


@pytest.fixture(scope="session")
def _engine():
    from wald.config import get_settings

    url = make_url(get_settings().database_url)
    test_url = url.set(database=f"{url.database}_test")

    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT on a connection to
    # the configured database rather than the one being created.
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.exec_driver_sql(
                "SELECT 1 FROM pg_database WHERE datname = %s", (test_url.database,)
            ).scalar()
            if not exists:
                conn.exec_driver_sql(f'CREATE DATABASE "{test_url.database}"')
    except SQLAlchemyError as exc:
        pytest.skip(f"no database available: {type(exc).__name__}")
    finally:
        admin.dispose()

    # Built through the production path, so the suite cannot pass against a schema the
    # application would never produce.
    from wald.db import init_schema

    engine = create_engine(test_url, future=True)
    init_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(_engine):
    """A session on a transaction that is always rolled back.

    Every test therefore sees the same empty starting state regardless of order, which is
    what lets a test assert on the difference between "created" and "unchanged".
    """
    from sqlalchemy.orm import Session

    connection = _engine.connect()
    transaction = connection.begin()
    # join_transaction_mode keeps a commit() inside the code under test as a savepoint
    # release, so the outer rollback still discards everything.
    db = Session(bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()
