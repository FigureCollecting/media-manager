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
from sqlalchemy.exc import OperationalError

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


#: The two raw-SQL GIN trigram indexes the migration chain adds on albums,
#: deliberately not modeled as SQLAlchemy Index() objects.
GIN_TRGM_INDEXES = {
    ("idx_album_desc_trgm", ("description",)),
    ("idx_album_title_trgm", ("title",)),
}


def _schema_snapshot(
    engine,
) -> tuple[dict[str, object], dict[str, set[tuple[str, tuple[str, ...]]]]]:
    """Reflect tables/columns/keys and, separately, indexes (excludes alembic's own table).

    A foreign key is compared with its ON DELETE rule: a migration that
    cascades differently from the model is drift too.

    Tables/columns/keys are returned as an exact-comparable snapshot. Indexes are
    returned separately (per table, as a set of (name, sorted column names))
    because the migration chain adds two raw-SQL GIN trigram indexes on top of
    what Base.metadata.create_all() alone would produce -- see the caller.
    """
    insp = inspect(engine)
    tables: dict[str, object] = {}
    indexes: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    for table_name in sorted(insp.get_table_names()):
        if table_name == "alembic_version":
            continue
        columns = {
            col["name"]: {"type": str(col["type"]), "nullable": col["nullable"]}
            for col in insp.get_columns(table_name)
        }
        pk = sorted(insp.get_pk_constraint(table_name).get("constrained_columns") or [])
        fks = sorted(
            (
                fk["constrained_columns"][0],
                fk["referred_table"],
                fk["referred_columns"][0],
                str(fk["options"].get("ondelete")),
            )
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
            (str(ix["name"]), tuple(sorted(ix["column_names"])))
            for ix in insp.get_indexes(table_name)
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

            # Every index exactly, by name and columns -- one a migration
            # creates that the model lacks is drift too -- except the two GIN
            # trigram indexes on albums, which only the migration chain makes.
            assert migrated_indexes["albums"] >= GIN_TRGM_INDEXES
            migrated_indexes["albums"] -= GIN_TRGM_INDEXES
            assert migrated_indexes == live_indexes
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


def test_downgrading_the_mask_lane_restores_its_predecessors_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0004's downgrade undoes all of 0004, the column it adds to images included."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    down = _run_alembic(
        "downgrade", "0003_add_matte_grounding_fields", database_url=empty_database_url
    )
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0003_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0003_add_matte_grounding_fields", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_downgrading_the_spine_report_ledger_restores_the_mask_lanes_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0005's downgrade (MM-REPORT-1) drops exactly the two ledger tables."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    down = _run_alembic("downgrade", "0004_add_mask_lane", database_url=empty_database_url)
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0004_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0004_add_mask_lane", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            tables = set(inspect(downgraded_engine).get_table_names())
            assert not tables & {"spine_report_runs", "spine_mask_withdrawals"}
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_downgrading_the_gallery_source_class_restores_the_ledgers_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0006's downgrade (GAL3) drops exactly figure_galleries.source_class."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    migrated = create_engine(empty_database_url)
    try:
        columns = {c["name"]: c for c in inspect(migrated).get_columns("figure_galleries")}
        assert columns["source_class"]["nullable"] is True
    finally:
        migrated.dispose()
    down = _run_alembic(
        "downgrade", "0005_add_spine_report_ledger", database_url=empty_database_url
    )
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0005_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0005_add_spine_report_ledger", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            names = {c["name"] for c in inspect(downgraded_engine).get_columns("figure_galleries")}
            assert "source_class" not in names
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_downgrading_the_display_copy_queue_restores_the_gallery_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0007's downgrade (SHOW1) drops exactly the two queue tables and the feed's walk."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    migrated = create_engine(empty_database_url)
    try:
        tables = set(inspect(migrated).get_table_names())
        assert {"display_originals", "display_depictions", "display_feed_walks"} <= tables
        fks = inspect(migrated).get_foreign_keys("display_depictions")
        assert [(f["referred_table"], f["options"].get("ondelete")) for f in fks] == [
            ("display_originals", "CASCADE")
        ]
    finally:
        migrated.dispose()
    down = _run_alembic(
        "downgrade", "0006_add_gallery_source_class", database_url=empty_database_url
    )
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0006_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0006_add_gallery_source_class", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            names = set(inspect(downgraded_engine).get_table_names())
            assert not names & {"display_originals", "display_depictions", "display_feed_walks"}
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


COPY_LOOKUP_INDEXES = {
    ("public_objects", "ix_public_objects_image_version_id"),
    ("public_object_images", "ix_public_object_images_image_id"),
}


def _copy_lookup_indexes(url: str) -> dict[tuple[str, str], bool]:
    """The 0008 indexes present in ``url``'s database, each with whether it is
    valid (a CONCURRENTLY build cut short leaves an invalid one behind)."""
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT t.relname, c.relname, i.indisvalid FROM pg_index i"
                    " JOIN pg_class c ON c.oid = i.indexrelid"
                    " JOIN pg_class t ON t.oid = i.indrelid"
                    " WHERE c.relname IN ('ix_public_objects_image_version_id',"
                    " 'ix_public_object_images_image_id')"
                )
            )
            return {(table, name): valid for table, name, valid in rows}
    finally:
        engine.dispose()


def test_downgrading_the_copy_lookup_indexes_restores_the_display_queues_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0008's downgrade (SHOW1-THROUGHPUT) drops exactly the two copy-lookup indexes."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    assert _copy_lookup_indexes(empty_database_url) == dict.fromkeys(COPY_LOOKUP_INDEXES, True)
    down = _run_alembic("downgrade", "0007_add_display_copy_queue", database_url=empty_database_url)
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0007_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0007_add_display_copy_queue", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            assert _copy_lookup_indexes(empty_database_url) == {}
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_the_copy_lookup_indexes_build_behind_a_long_transaction_under_the_jobs_lock_timeout(
    empty_database_url: str,
) -> None:
    """The migrate Job (fc-infra db/media-migrate-job.yaml) runs alembic with
    PGOPTIONS=-c lock_timeout=5s. A CONCURRENTLY build waits for every
    transaction that wrote its table to end (a seeder's runs 40-85 s), and
    that wait is a lock wait: under the timeout the build would fail and
    leave an invalid index. 0008 waits instead; its lock queues no reader or
    writer behind it."""
    up = _run_alembic("upgrade", "0007_add_display_copy_queue", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    watch = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    seeder = engine.connect()
    seeder.execute(text("LOCK TABLE public_objects, public_object_images IN ROW EXCLUSIVE MODE"))
    proc = subprocess.Popen(
        ["alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": empty_database_url, "PGOPTIONS": "-c lock_timeout=1s"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        with watch.connect() as conn:
            while proc.poll() is None and time.monotonic() < deadline:
                building = conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                        " AND query LIKE 'CREATE INDEX CONCURRENTLY%' AND wait_event_type = 'Lock'"
                    )
                ).scalar_one()
                if building:
                    break
                time.sleep(0.1)
        time.sleep(2)  # the seeder holds on for twice the lock timeout
        seeder.commit()
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        seeder.close()
        engine.dispose()
        watch.dispose()
    assert proc.returncode == 0, output
    assert _copy_lookup_indexes(empty_database_url) == dict.fromkeys(COPY_LOOKUP_INDEXES, True)


def test_a_copy_lookup_index_a_cut_short_build_left_invalid_is_rebuilt(
    empty_database_url: str,
) -> None:
    """A CONCURRENTLY build cut short (a lock timeout, the Job's deadline)
    leaves an INVALID index of that name: never read, still written on every
    insert. IF NOT EXISTS alone would keep it for good; 0008 rebuilds it."""
    up = _run_alembic("upgrade", "0007_add_display_copy_queue", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    builder = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as seeder, builder.connect() as conn:
            seeder.execute(text("LOCK TABLE public_objects IN ROW EXCLUSIVE MODE"))
            conn.execute(text("SET lock_timeout = '1s'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                conn.execute(
                    text(
                        "CREATE INDEX CONCURRENTLY ix_public_objects_image_version_id"
                        " ON public_objects (image_version_id)"
                    )
                )
            seeder.rollback()
    finally:
        engine.dispose()
        builder.dispose()
    assert _copy_lookup_indexes(empty_database_url) == {
        ("public_objects", "ix_public_objects_image_version_id"): False
    }
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    assert _copy_lookup_indexes(empty_database_url) == dict.fromkeys(COPY_LOOKUP_INDEXES, True)


def test_the_copy_lookup_indexes_render_as_offline_sql_outside_a_transaction(
    pg_admin_url: str,
) -> None:
    """``alembic upgrade --sql`` (no database: the catalog is not read) prints
    0008 as the Job would run it: the run so far committed, both builds
    CONCURRENTLY with no lock timeout, then a new transaction for the stamp."""
    out = _run_alembic(
        "upgrade", "0007_add_display_copy_queue:head", "--sql", database_url=pg_admin_url
    )
    assert out.returncode == 0, out.stdout + out.stderr
    sql = "\n".join(line for line in out.stdout.splitlines() if not line.startswith("--"))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    builds = statements.index("SET lock_timeout = 0")
    assert statements[builds - 1 : builds + 5] == [
        "COMMIT",
        "SET lock_timeout = 0",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_public_objects_image_version_id"
        " ON public_objects (image_version_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_public_object_images_image_id"
        " ON public_object_images (image_id)",
        "RESET lock_timeout",
        "BEGIN",
    ]


def test_the_copy_lookup_indexes_downgrade_as_offline_sql_outside_a_transaction(
    pg_admin_url: str,
) -> None:
    """``alembic downgrade --sql`` prints 0008's rollback as the Job would run
    it: both drops CONCURRENTLY outside a transaction, so a rollback never
    takes ACCESS EXCLUSIVE on the tables the display lane reads and writes."""
    out = _run_alembic(
        "downgrade",
        "0008_index_copy_lookups:0007_add_display_copy_queue",
        "--sql",
        database_url=pg_admin_url,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    sql = "\n".join(line for line in out.stdout.splitlines() if not line.startswith("--"))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    drops = statements.index("SET lock_timeout = 0")
    assert statements[drops - 1 : drops + 5] == [
        "COMMIT",
        "SET lock_timeout = 0",
        "DROP INDEX CONCURRENTLY IF EXISTS ix_public_object_images_image_id",
        "DROP INDEX CONCURRENTLY IF EXISTS ix_public_objects_image_version_id",
        "RESET lock_timeout",
        "BEGIN",
    ]


#: The index prod's figure_galleries got by hand on 2026-11-03 (MASKRATE), as
#: pg_indexes.indexdef reads it back; 0009 carries it.
GALLERY_IMAGE_INDEXDEF = (
    "CREATE INDEX ix_figure_galleries_image_id ON public.figure_galleries USING btree (image_id)"
)


def _gallery_image_index(url: str) -> tuple[int, bool, str] | None:
    """0009's index in ``url``'s database: its oid, whether it is valid and
    its definition; None while there is none."""
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT i.indexrelid::int, i.indisvalid, pg_get_indexdef(i.indexrelid)"
                    " FROM pg_index i WHERE i.indexrelid ="
                    " to_regclass('public.ix_figure_galleries_image_id')"
                )
            ).one_or_none()
            return None if row is None else (row[0], row[1], row[2])
    finally:
        engine.dispose()


def test_the_gallery_image_index_is_a_no_op_where_it_was_built_by_hand(
    empty_database_url: str,
) -> None:
    """MASKRATE: prod's index was built by hand, CONCURRENTLY, before 0009
    shipped. 0009 keeps that very index (same oid, valid, same definition)
    and only moves the version."""
    up = _run_alembic("upgrade", "0008_index_copy_lookups", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    builder = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    try:
        with builder.connect() as conn:
            conn.execute(
                text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_figure_galleries_image_id"
                    " ON public.figure_galleries USING btree (image_id)"
                )
            )
    finally:
        builder.dispose()
    by_hand = _gallery_image_index(empty_database_url)
    assert by_hand is not None and by_hand[1:] == (True, GALLERY_IMAGE_INDEXDEF)
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    assert _gallery_image_index(empty_database_url) == by_hand


def test_the_gallery_image_index_matches_the_one_built_by_hand(empty_database_url: str) -> None:
    """Where no one built it, 0009 builds the definition prod has."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    built = _gallery_image_index(empty_database_url)
    assert built is not None and built[1:] == (True, GALLERY_IMAGE_INDEXDEF)


def test_downgrading_the_gallery_image_index_restores_the_copy_lookups_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0009's downgrade (MASKRATE) drops exactly figure_galleries' image_id index."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    down = _run_alembic("downgrade", "0008_index_copy_lookups", database_url=empty_database_url)
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0008_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0008_index_copy_lookups", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            assert _gallery_image_index(empty_database_url) is None
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_a_gallery_image_index_a_cut_short_build_left_invalid_is_rebuilt(
    empty_database_url: str,
) -> None:
    """As 0008: a CONCURRENTLY build cut short leaves an INVALID index that
    IF NOT EXISTS alone would keep; 0009 drops and builds it again."""
    up = _run_alembic("upgrade", "0008_index_copy_lookups", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    builder = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as seeder, builder.connect() as conn:
            seeder.execute(text("LOCK TABLE figure_galleries IN ROW EXCLUSIVE MODE"))
            conn.execute(text("SET lock_timeout = '1s'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                conn.execute(
                    text(
                        "CREATE INDEX CONCURRENTLY ix_figure_galleries_image_id"
                        " ON figure_galleries (image_id)"
                    )
                )
            seeder.rollback()
    finally:
        engine.dispose()
        builder.dispose()
    invalid = _gallery_image_index(empty_database_url)
    assert invalid is not None and invalid[1] is False
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    rebuilt = _gallery_image_index(empty_database_url)
    assert rebuilt is not None and rebuilt[1:] == (True, GALLERY_IMAGE_INDEXDEF)


def test_the_gallery_image_index_renders_as_offline_sql_outside_a_transaction(
    pg_admin_url: str,
) -> None:
    """``alembic upgrade --sql`` prints 0009 as the migrate Job would run it:
    the run so far committed, the build CONCURRENTLY and IF NOT EXISTS with
    no lock timeout, then a new transaction for the stamp; and its rollback
    as a CONCURRENTLY drop."""
    for args, built in (
        (
            ("upgrade", "0008_index_copy_lookups:head"),
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS"
            " ix_figure_galleries_image_id ON figure_galleries (image_id)",
        ),
        (
            ("downgrade", "0009_index_gallery_image_id:0008_index_copy_lookups"),
            "DROP INDEX CONCURRENTLY IF EXISTS ix_figure_galleries_image_id",
        ),
    ):
        out = _run_alembic(*args, "--sql", database_url=pg_admin_url)
        assert out.returncode == 0, out.stdout + out.stderr
        sql = "\n".join(line for line in out.stdout.splitlines() if not line.startswith("--"))
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        at = statements.index("SET lock_timeout = 0")
        assert statements[at - 1 : at + 4] == [
            "COMMIT",
            "SET lock_timeout = 0",
            built,
            "RESET lock_timeout",
            "BEGIN",
        ]


def test_the_display_seeders_threads_claim_each_original_once_on_postgres(
    empty_database_url: str, tmp_path: Path
) -> None:
    """SHOW1: FOR UPDATE SKIP LOCKED is Postgres's; six threads, twelve
    originals, each seeded exactly once and every claim settled."""
    import datetime as dt
    import hashlib
    import io

    from app.models import DisplayDepiction, DisplayOriginal, FigureGallery, PublicObject
    from app.models import Image as ImageModel
    from app.object_store import LocalObjectStore
    from app.workers import display_copy as dc
    from app.workers.raw_source import ReadOnlyStore
    from PIL import Image
    from sqlalchemy import func, select
    from sqlalchemy.orm import sessionmaker

    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url, pool_size=8)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    raw = LocalObjectStore(tmp_path / "raw")
    at = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)
    with factory() as db:
        for n in range(12):
            buf = io.BytesIO()
            Image.new("RGB", (40, 60), (n * 20, 30, 90)).save(buf, format="PNG")
            data = buf.getvalue()
            sha = hashlib.sha256(data).hexdigest()
            key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
            raw.put(key, data, "image/png")
            db.add(
                DisplayOriginal(
                    origin_sha256=sha,
                    storage_key=key,
                    codec="identity",
                    first_seen_at=at - dt.timedelta(minutes=n),
                    state="pending",
                )
            )
            db.flush()
            db.add(
                DisplayDepiction(
                    product_image_id=str(uuid.UUID(int=n + 1)),
                    origin_sha256=sha,
                    product_id=str(uuid.UUID(int=100 + n)),
                    origin_url=f"https://img.amiami.com/{n}.png",
                    source_site="amiami",
                    first_seen_at=at,
                )
            )
        db.commit()
    lines: list[dict[str, object]] = []
    code = dc.run_seed(
        factory,
        ReadOnlyStore(raw),
        LocalObjectStore(tmp_path / "media"),
        dc.SeedConfig(threads=6, grants={"amiami": ["img.amiami.com"]}),
        emit=lines.append,
    )
    try:
        assert code == dc.EXIT_DONE, lines
        with factory() as db:
            rows = db.execute(select(DisplayOriginal)).scalars().all()
            assert {(r.state, r.attempts, r.claimed_by) for r in rows} == {("seeded", 0, None)}
            assert db.scalar(select(func.count()).select_from(FigureGallery)) == 12
            assert db.scalar(select(func.count()).select_from(ImageModel)) == 12
            assert db.scalar(select(func.count()).select_from(PublicObject)) == 12  # one copy each
    finally:
        engine.dispose()


def test_the_display_feed_reports_enqueues_and_purges_on_postgres(empty_database_url: str) -> None:
    """SHOW1: the feed's queries (the FAILED cap's count, the purge's
    database-side delete, requeue's prefix) as Postgres runs them."""
    import datetime as dt

    from app.models import DisplayDepiction, DisplayOriginal
    from app.spine import display_feed as feed
    from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
    from app.workers import display_copy as dc
    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

    from spine_fakes import serve, sha
    from test_display_feed import QueueSpine

    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    db = sessionmaker(bind=engine, expire_on_commit=False)()
    at = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)
    fake = QueueSpine()
    try:
        with serve(fake) as url:
            new = fake.depict_full(sha("a"), seen=at)
            fake.depict_full(sha("b"), seen=at - dt.timedelta(days=1))

            def connect() -> SpineMediaClient:
                return SpineMediaClient(open_channel(url), policy=RetryPolicy(max_attempts=1))

            lines: list[dict[str, object]] = []
            code = feed.run_sync(db, connect, feed.FeedConfig(), emit=lines.append, now=lambda: at)
            assert code == feed.EXIT_DONE, lines
            assert db.get(DisplayOriginal, sha("a")).first_seen_at == at
            failed = db.get(DisplayOriginal, sha("a"))
            failed.state, failed.reason, failed.updated_at = "failed", "undecodable", at
            db.commit()
            later = at + dt.timedelta(days=20)
            code = feed.run_sync(
                db, connect, feed.FeedConfig(max_new=0), emit=lines.append, now=lambda: at
            )
            assert code == feed.EXIT_DONE
            assert db.get(DisplayDepiction, new).outcome == "failed"
            code = feed.run_sync(
                db, connect, feed.FeedConfig(max_new=0), emit=lines.append, now=lambda: later
            )
            assert (
                db.get(DisplayDepiction, new) is None and db.get(DisplayOriginal, sha("a")) is None
            )
            assert db.get(DisplayOriginal, sha("b")).state == "pending"
            pending = db.get(DisplayOriginal, sha("b"))
            pending.state, pending.reason = "stuck", "gave_up:missing_raw"
            db.commit()
            assert (
                dc.requeue(db, state="stuck", reason_prefix="gave_up:", force=False, apply=True)
                == 1
            )
            assert dc.status(db)["originals"] == {"pending": 1}
            assert db.scalar(select(DisplayOriginal.attempts)) == 0
    finally:
        db.close()
        engine.dispose()


# --------------------------------------------------------------------------
# MEDIADB: 0010 drops two prefix indexes
# --------------------------------------------------------------------------

#: The two indexes 0010 drops: each is the leading column of a UNIQUE index
#: its readers move to (uq_image_version_no (image_id, version_no),
#: uq_figure_source_url (figure_id, source_url)) at the same buffer counts.
PREFIX_INDEXES = {
    ("image_versions", "ix_image_versions_image_id_visibility"),
    ("figure_galleries", "ix_figure_galleries_figure_id"),
}
#: What they leave the readers (never dropped: integrity, and the readers' index).
WIDER_INDEXES = {
    ("image_versions", "uq_image_version_no"),
    ("figure_galleries", "uq_figure_source_url"),
}


def _indexes(url: str, names: set[tuple[str, str]]) -> dict[tuple[str, str], bool]:
    """Those of ``names`` present in ``url``'s database, each with whether it is valid."""
    wanted = sorted(name for _table, name in names)
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT t.relname, c.relname, i.indisvalid FROM pg_index i"
                    " JOIN pg_class c ON c.oid = i.indexrelid"
                    " JOIN pg_class t ON t.oid = i.indrelid WHERE c.relname = ANY(:names)"
                ),
                {"names": wanted},
            )
            return {(table, name): valid for table, name, valid in rows}
    finally:
        engine.dispose()


def test_the_prefix_indexes_are_dropped_and_their_readers_keep_the_wider_ones(
    empty_database_url: str,
) -> None:
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    assert _indexes(empty_database_url, PREFIX_INDEXES) == {}
    assert _indexes(empty_database_url, WIDER_INDEXES) == dict.fromkeys(WIDER_INDEXES, True)


def test_downgrading_the_prefix_index_drops_restores_the_gallery_image_indexs_schema(
    empty_database_url: str, pg_admin_url: str
) -> None:
    """0010's downgrade builds exactly the two indexes again."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    down = _run_alembic("downgrade", "0009_index_gallery_image_id", database_url=empty_database_url)
    assert down.returncode == 0, down.stdout + down.stderr

    before_dbname = f"migtest_0009_{uuid.uuid4().hex[:8]}"
    admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{before_dbname}"'))
    admin_engine.dispose()
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{before_dbname}"
    try:
        only = _run_alembic("upgrade", "0009_index_gallery_image_id", database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded_engine = create_engine(empty_database_url)
        before_engine = create_engine(before_url)
        try:
            assert _indexes(empty_database_url, PREFIX_INDEXES) == dict.fromkeys(
                PREFIX_INDEXES, True
            )
            assert _schema_snapshot(downgraded_engine) == _schema_snapshot(before_engine)
        finally:
            downgraded_engine.dispose()
            before_engine.dispose()
    finally:
        admin_engine = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{before_dbname}" WITH (FORCE)'))
        admin_engine.dispose()


def test_the_prefix_index_drops_wait_behind_a_reading_transaction_under_the_jobs_lock_timeout(
    empty_database_url: str,
) -> None:
    """The verifier's probe: a transaction that only READ the tables, left
    idle (the seeder's before MEDIADB), blocks DROP INDEX CONCURRENTLY, and
    under the migrate Job's lock_timeout the drop would fail. 0010 waits for
    it with no lock timeout instead (its lock queues no read or write) and
    drops both once the reader commits."""
    up = _run_alembic("upgrade", "0009_index_gallery_image_id", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    watch = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    reader = engine.connect()
    reader.execute(text("SELECT count(*) FROM image_versions"))
    reader.execute(text("SELECT count(*) FROM figure_galleries"))
    proc = subprocess.Popen(
        ["alembic", "upgrade", "0010_drop_prefix_indexes"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": empty_database_url, "PGOPTIONS": "-c lock_timeout=1s"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        waited = 0
        with watch.connect() as conn:
            while proc.poll() is None and time.monotonic() < deadline:
                waited = conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                        " AND ltrim(query, E' \\n') LIKE 'DROP INDEX CONCURRENTLY%'"
                        " AND wait_event_type = 'Lock'"
                    )
                ).scalar_one()
                if waited:
                    break
                time.sleep(0.1)
        time.sleep(2)  # the reader holds on for twice the lock timeout
        reader.commit()
        output, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.close()
        engine.dispose()
        watch.dispose()
    assert waited, output  # it did wait behind the reader
    assert proc.returncode == 0, output
    assert _indexes(empty_database_url, PREFIX_INDEXES) == {}


def test_the_prefix_index_drops_render_as_offline_sql_outside_a_transaction(
    pg_admin_url: str,
) -> None:
    """``alembic upgrade --sql`` prints 0010 as the Job would run it: the run
    so far committed, both drops CONCURRENTLY with no lock or statement
    timeout, then a new transaction; its downgrade builds them the same way."""
    for args, ddl in (
        (
            ("upgrade", "0009_index_gallery_image_id:0010_drop_prefix_indexes"),
            [
                "DROP INDEX CONCURRENTLY IF EXISTS ix_image_versions_image_id_visibility",
                "DROP INDEX CONCURRENTLY IF EXISTS ix_figure_galleries_figure_id",
            ],
        ),
        (
            ("downgrade", "0010_drop_prefix_indexes:0009_index_gallery_image_id"),
            [
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_figure_galleries_figure_id"
                " ON figure_galleries (figure_id)",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_image_versions_image_id_visibility"
                " ON image_versions (image_id, visibility)",
            ],
        ),
    ):
        out = _run_alembic(*args, "--sql", database_url=pg_admin_url)
        assert out.returncode == 0, out.stdout + out.stderr
        sql = "\n".join(line for line in out.stdout.splitlines() if not line.startswith("--"))
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        at = statements.index("SET lock_timeout = 0")
        assert statements[at - 1 : at + 6] == [
            "COMMIT",
            "SET lock_timeout = 0",
            "SET statement_timeout = 0",
            *ddl,
            "RESET statement_timeout",
            "RESET lock_timeout",
        ]
        assert statements[at + 6] == "BEGIN"
