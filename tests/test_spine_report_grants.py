"""MM-REPORT-1: the replay runs under exactly the grants its Job's role gets
(fc-infra ``media-grants``; ~/tmp/mm-report-1/JOB.md section 2): SELECT on
three tables, COLUMN grants on ``images`` (id, sha256, deleted_at) and
``image_versions`` (id, image_id, deleted_at), and its own ledger. Postgres
checks column privileges on every column a query names, ``SELECT *`` inside
an EXISTS included, so this runs on a real postgres:16 (docker; skipped
without it), migrated to head."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from app.spine import replay as rp
from app.spine.media_client import SpineMediaClient, open_channel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from spine_fakes import FakeSpine, seed_copy, serve
from test_migration_chain_replay import (  # noqa: F401  (fixtures)
    _run_alembic,
    empty_database_url,
    pg_admin_url,
    pytestmark,
)

#: The Job role's grants, as JOB.md section 2 writes them (by name, fail closed).
GRANTS = """
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM media_report;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM media_report;
GRANT USAGE ON SCHEMA public TO media_report;
GRANT SELECT (id, sha256, deleted_at) ON images TO media_report;
GRANT SELECT (id, image_id, deleted_at) ON image_versions TO media_report;
GRANT SELECT ON public_objects, public_object_images, image_masks TO media_report;
GRANT SELECT, INSERT, UPDATE ON spine_report_runs TO media_report;
GRANT SELECT, INSERT, DELETE ON spine_mask_withdrawals TO media_report;
GRANT USAGE ON SEQUENCE spine_report_runs_id_seq TO media_report;
"""


@pytest.fixture()
def as_media_report(empty_database_url: str) -> Iterator[tuple[str, str]]:  # noqa: F811
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    owner = create_engine(empty_database_url, isolation_level="AUTOCOMMIT")
    with owner.connect() as conn:
        conn.execute(
            text(
                "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'media_report') "
                "THEN CREATE ROLE media_report LOGIN PASSWORD 'r'; END IF; END $$"
            )
        )
        for statement in filter(str.strip, GRANTS.split(";")):
            conn.execute(text(statement))
    owner.dispose()
    report_url = empty_database_url.replace("postgres:postgres@", "media_report:r@", 1)
    yield empty_database_url, report_url


def test_the_replay_reads_only_what_the_jobs_role_is_granted(
    as_media_report: tuple[str, str],
) -> None:
    owner_url, report_url = as_media_report
    owner = create_engine(owner_url)
    fake = FakeSpine()
    with Session(owner, expire_on_commit=False) as db:
        live = seed_copy(db, "pg-live", extra_originals=("pg-live2",))
        seed_copy(db, "pg-dead", deleted=True)  # the liveness EXISTS clauses run on rows
        withdrawn = seed_copy(db, "pg-w", masks=(("mask-v1", "withdrawn", "pg-mw"),))
    for origin in (live["origin"], live["others"][0], withdrawn["origin"]):
        fake.depict(origin)
    owner.dispose()

    report = create_engine(report_url)
    lines: list[dict[str, Any]] = []
    try:
        with serve(fake) as url, Session(report, expire_on_commit=False) as db:
            code = rp.run_replay(
                db,
                rp.ReplayConfig(),
                connect=lambda: SpineMediaClient(open_channel(url)),
                emit=lines.append,
            )
            assert code == rp.EXIT_DONE, lines
            batch = next(line for line in lines if line["event"] == "batch")
            assert batch["matched"] == 3 and batch["ReportMask"]["stored"] == 1
            dry = rp.run_replay(
                db, rp.ReplayConfig(dry_run=True), connect=lambda: None, emit=lines.append
            )
            assert dry == rp.EXIT_DONE
        with report.connect() as conn:
            for column, table in (("mime", "images"), ("storage_key", "image_versions")):
                with pytest.raises(ProgrammingError, match="permission denied"):
                    conn.execute(text(f"SELECT {column} FROM {table} LIMIT 1"))
                conn.rollback()
    finally:
        report.dispose()
