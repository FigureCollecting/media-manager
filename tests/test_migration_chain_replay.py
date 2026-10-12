"""Full alembic migration chain replay tests.

Regression coverage for two migration-hygiene bugs that block a clean
fresh-namespace deploy:

  * alembic.ini's ``sqlalchemy.url = %(DATABASE_URL)s`` crashes ConfigParser
    interpolation before env.py ever runs (see test_alembic_config.py).
  * 0001_init previously built its schema via
    ``Base.metadata.create_all(bind)`` against the *live* app.models.Base,
    so replaying 0001 -> 0002 -> 0003 against an empty database made 0001
    create every current table (including figure_galleries and the matte
    columns), and 0002's explicit ``op.create_table("figure_galleries")``
    then collided with it.

0001 emits Postgres-only DDL (CREATE EXTENSION citext/pg_trgm, GIN trgm
indexes), so SQLite cannot run the real chain end to end. These tests spin
up a disposable postgres:16 container via the docker CLI for the duration
of the module; hosts without docker skip rather than fail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker is required to replay the real (Postgres-dialect) migration chain",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_alembic(*args: str, database_url: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DATABASE_URL": database_url}
    return subprocess.run(
        ["alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def pg_admin_url():
    """Start a disposable postgres:16 container; yield an admin connection URL."""
    name = f"im-migtest-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=postgres",
            "-p",
            "127.0.0.1::5432",
            "postgres:16",
        ],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start disposable postgres container: {started.stderr}")

    try:
        port_result = subprocess.run(
            ["docker", "port", name, "5432/tcp"],
            capture_output=True,
            text=True,
            check=True,
        )
        host_port = port_result.stdout.strip().rsplit(":", 1)[-1]
        admin_url = f"postgresql+psycopg2://postgres:postgres@127.0.0.1:{host_port}/postgres"

        engine = create_engine(admin_url)
        deadline = time.monotonic() + 30
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.5)
        engine.dispose()
        if last_error is not None:
            pytest.skip(f"postgres container never became ready: {last_error}")

        yield admin_url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


@pytest.fixture()
def empty_database_url(pg_admin_url: str):
    """Create a fresh, empty database for a single test and drop it after."""
    dbname = f"migtest_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{dbname}"'))
    admin_engine.dispose()

    url = pg_admin_url.rsplit("/", 1)[0] + f"/{dbname}"
    yield url

    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)'))
    admin_engine.dispose()


def test_full_chain_replays_on_empty_database(empty_database_url: str) -> None:
    """alembic upgrade head must succeed from a truly empty DB with no collisions."""
    result = _run_alembic("upgrade", "head", database_url=empty_database_url)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, f"upgrade head failed:\n{combined}"
    assert "already exists" not in combined


def test_full_chain_downgrade_and_upgrade_again(empty_database_url: str) -> None:
    """The chain must be reversible: downgrade to base, then upgrade again cleanly."""
    up1 = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up1.returncode == 0, up1.stdout + up1.stderr

    down = _run_alembic("downgrade", "base", database_url=empty_database_url)
    assert down.returncode == 0, f"downgrade base failed:\n{down.stdout + down.stderr}"

    up2 = _run_alembic("upgrade", "head", database_url=empty_database_url)
    combined = up2.stdout + up2.stderr
    assert up2.returncode == 0, f"second upgrade head failed:\n{combined}"
    assert "already exists" not in combined


def _schema_snapshot(engine) -> tuple[dict[str, object], dict[str, set[tuple[str, ...]]]]:
    """Reflect tables/columns/keys and, separately, indexes (excludes alembic's own table).

    Tables/columns/keys are returned as an exact-comparable snapshot. Indexes are
    returned separately (per table, as a set of sorted column-name tuples) because
    the migration chain adds two raw-SQL GIN trigram indexes on top of what
    Base.metadata.create_all() alone would produce -- see the caller.
    """
    insp = inspect(engine)
    tables: dict[str, object] = {}
    indexes: dict[str, set[tuple[str, ...]]] = {}
    for table_name in sorted(insp.get_table_names()):
        if table_name == "alembic_version":
            continue
        columns = {
            col["name"]: {"type": str(col["type"]), "nullable": col["nullable"]}
            for col in insp.get_columns(table_name)
        }
        pk = sorted(insp.get_pk_constraint(table_name).get("constrained_columns") or [])
        fks = sorted(
            (fk["constrained_columns"][0], fk["referred_table"], fk["referred_columns"][0])
            for fk in insp.get_foreign_keys(table_name)
        )
        uniques = sorted(
            tuple(sorted(uc["column_names"])) for uc in insp.get_unique_constraints(table_name)
        )
        checks = sorted(
            (cc["name"], cc["sqltext"]) for cc in insp.get_check_constraints(table_name)
        )
        tables[table_name] = {
            "columns": columns,
            "pk": pk,
            "fks": fks,
            "uniques": uniques,
            "checks": checks,
        }
        indexes[table_name] = {
            tuple(sorted(ix["column_names"])) for ix in insp.get_indexes(table_name)
        }
    return tables, indexes


def test_migrated_schema_matches_live_models(empty_database_url: str, pg_admin_url: str) -> None:
    """The migrated (0001..head) schema must exactly match today's app.models.Base."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr

    from app.models import Base

    live_dbname = f"migtest_live_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{live_dbname}"'))
    admin_engine.dispose()
    live_url = pg_admin_url.rsplit("/", 1)[0] + f"/{live_dbname}"

    live_engine = create_engine(live_url)
    try:
        with live_engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        Base.metadata.create_all(live_engine)

        migrated_engine = create_engine(empty_database_url)
        try:
            migrated_tables, migrated_indexes = _schema_snapshot(migrated_engine)
            live_tables, live_indexes = _schema_snapshot(live_engine)
            assert migrated_tables == live_tables

            # The migration chain additionally creates two raw-SQL GIN trigram
            # indexes on albums that are deliberately not modeled as SQLAlchemy
            # Index() objects, so live (create_all-only) indexes must be a subset
            # of the migrated ones, not an exact match.
            for table_name, live_table_indexes in live_indexes.items():
                assert live_table_indexes <= migrated_indexes[table_name], table_name
        finally:
            migrated_engine.dispose()

        with create_engine(empty_database_url).connect() as conn:
            gin_index_names = {
                row[0]
                for row in conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'albums'")
                )
            }
        assert {"idx_album_title_trgm", "idx_album_desc_trgm"} <= gin_index_names
    finally:
        live_engine.dispose()
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{live_dbname}" WITH (FORCE)'))
        admin_engine.dispose()
