import contextlib
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .telemetry.db import name_engine

_settings = get_settings()
_engine_kwargs: dict[str, Any] = {
    "pool_pre_ping": True,
    "pool_recycle": _settings.db_pool_recycle,
    "future": True,
}
if not _settings.database_url.startswith("sqlite"):
    _engine_kwargs["pool_size"] = _settings.db_pool_size
    _engine_kwargs["max_overflow"] = _settings.db_max_overflow
engine = create_engine(_settings.database_url, **_engine_kwargs)
# MEDIADB (U8): each connection's application_name is the process's workload
# and run, once its entry point names them (app.telemetry.db)
name_engine(engine)
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        with contextlib.suppress(Exception):
            db.close()


@contextlib.contextmanager
def worker_session(
    session_factory: sessionmaker[Session] | None = None,
) -> Generator[Session, None, None]:
    """Context manager for worker tasks — commits on success, rolls back on error."""
    factory = session_factory or SessionLocal
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
