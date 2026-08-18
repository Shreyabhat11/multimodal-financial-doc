"""
Database engine/session setup.

`get_db` follows the standard FastAPI dependency-injection generator pattern (Phase
12 wires this in as a route dependency) — yield a session, ensure it's closed after
the request regardless of success/failure. `SessionLocal` is a sessionmaker, not a
single shared session: SQLAlchemy sessions are not thread-safe / concurrency-safe to
share across requests, so a new one is created per request/unit-of-work.
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.models.orm import Base


def create_db_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    # pool_pre_ping: checks connection liveness before handing it out from the pool,
    # avoiding "server has gone away" errors on long-lived worker processes where a
    # Postgres connection has been idle long enough for the DB (or a proxy like
    # pgbouncer) to have silently dropped it.
    engine = create_engine(settings.database_url, echo=settings.database_echo, pool_pre_ping=True)

    if settings.database_url.startswith("sqlite"):
        # SQLite-only tuning: WAL mode allows one writer and multiple readers to
        # operate concurrently without "database is locked" errors, which matters
        # here because a FastAPI request's own session and a BackgroundTasks-spawned
        # session can briefly overlap. This block is a no-op for Postgres (the
        # production target) — Postgres handles concurrent connections natively and
        # never takes this branch.
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


_engine = None
_SessionLocal = None


def _get_engine_and_sessionmaker():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_db_engine()
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine, _SessionLocal


def init_db() -> None:
    """Create all tables. Used for local development / tests only — production
    schema changes go through Alembic migrations (migrations/), never through this
    function, so that schema evolution is versioned and reviewable rather than
    implicit."""
    engine, _ = _get_engine_and_sessionmaker()
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: `db: Session = Depends(get_db)`.

    Commits on successful completion of the request, rolls back on any exception —
    without this, a route that flushes but never explicitly commits would have its
    writes silently discarded when the session closes, since SQLAlchemy's default
    behavior on close() without a prior commit() is an implicit rollback.
    """
    _, SessionLocal = _get_engine_and_sessionmaker()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope():
    """Context-manager form for use outside FastAPI request handlers (e.g. the
    LangGraph pipeline persisting results, background workers, scripts/seed.py)."""
    _, SessionLocal = _get_engine_and_sessionmaker()
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
