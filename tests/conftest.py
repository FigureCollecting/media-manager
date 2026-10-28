from __future__ import annotations

import os

# Set test environment BEFORE any app imports so Settings picks up overrides
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["ENVIRONMENT"] = "test"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["ALLOW_DEV_TOKENS"] = "true"

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
from app.auth import create_token
from app.config import Settings, get_settings
from app.db import get_db
from app.main import app
from app.models import Base
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.ext.compiler import compiles  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Register PostgreSQL type adaptors so SQLite can handle UUID/CITEXT columns
# ---------------------------------------------------------------------------


@compiles(UUID, "sqlite")  # type: ignore[misc]
def _compile_uuid_sqlite(type_: UUID, compiler: object, **kw: object) -> str:  # type: ignore[no-untyped-def]
    return "TEXT"


@compiles(CITEXT, "sqlite")  # type: ignore[misc]
def _compile_citext_sqlite(type_: CITEXT, compiler: object, **kw: object) -> str:  # type: ignore[no-untyped-def]
    return "TEXT"


# ---------------------------------------------------------------------------
# Database fixtures -- in-memory SQLite, recreated per test
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # SQLite doesn't enforce FK by default
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine) -> Generator[Session, None, None]:  # type: ignore[type-arg]
    TestSession = sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---------------------------------------------------------------------------
# Settings override
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        database_url="sqlite://",
        redis_url="redis://localhost:6379/15",
        s3_endpoint_url="http://localhost:9000",
        environment="test",
        jwt_secret="test-secret",
        allow_dev_tokens=True,
    )


# ---------------------------------------------------------------------------
# FastAPI TestClient with dependency overrides
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(db_session: Session, test_settings: Settings) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def _override_get_settings() -> Settings:
        return test_settings

    # Clear lru_cache so test settings are used
    get_settings.cache_clear()

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_settings] = _override_get_settings

    # Clear rate-limit state so tests don't bleed into each other
    from app.rate_limit import _buckets

    _buckets.clear()

    # Patch Celery tasks to be no-ops so they don't require a broker
    with (
        patch("app.workers.tasks.verify_and_register_object.delay", new=MagicMock()),
        patch("app.workers.tasks.create_transformed_version.delay", new=MagicMock()),
        patch("app.workers.tasks.generate_album_cover.delay", new=MagicMock()),
        patch("app.workers.tasks.ingest_gallery_images.delay", new=MagicMock()),
    ):
        yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_headers(test_settings: Settings) -> dict[str, str]:
    """Return Authorization header for a regular user."""
    token = create_token(
        test_settings,
        subject="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        tenant_id="11111111-2222-3333-4444-555555555555",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def service_headers(test_settings: Settings) -> dict[str, str]:
    """Return Authorization header for a service client with assets:read scope."""
    token = create_token(
        test_settings,
        subject="service:test-svc",
        scopes=["assets:read"],
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def make_auth_headers(test_settings: Settings):
    """Factory to create auth headers with custom claims."""

    def _make(
        *,
        subject: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        tenant_id: str | None = "11111111-2222-3333-4444-555555555555",
        scopes: list[str] | None = None,
    ) -> dict[str, str]:
        token = create_token(
            test_settings,
            subject=subject,
            tenant_id=tenant_id,
            scopes=scopes,
        )
        return {"Authorization": f"Bearer {token}"}

    return _make


@pytest.fixture()
def link_image_to_user(db_session: Session):
    """Create a UserImageLink so the test user owns the given image."""
    from app.models import UserImageLink

    def _link(
        image_id: int,
        user_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        tenant_id: str = "11111111-2222-3333-4444-555555555555",
    ) -> None:
        link = UserImageLink(
            user_id=user_id,
            tenant_id=tenant_id,
            image_id=image_id,
            current_version_id=None,
            role="owner",
        )
        db_session.add(link)
        db_session.flush()

    return _link


# ---------------------------------------------------------------------------
# LG-U11: a test that runs an entrypoint (a CLI main, an app's lifespan, the
# worker's setup_logging) routes Python's logging into the estate's line on
# stdout. Put the process's logging back after each test, so no later test
# finds those lines in its own captured output.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_process_logging() -> Generator[None, None, None]:
    import logging

    from app.telemetry import log as telemetry_log

    root = logging.getLogger()
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    saved = [(lg, list(lg.handlers), lg.level, lg.propagate, lg.disabled) for lg in
             (root, *(logging.getLogger(n) for n in names))]  # fmt: skip
    process = telemetry_log.use_logger(None)
    telemetry_log.use_logger(process)
    yield
    for lg, handlers, level, propagate, disabled in saved:
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate, lg.disabled = propagate, disabled
    telemetry_log.use_logger(process)
