"""SHOW1: each half of the display-copy lane runs under exactly the grants
its CronJob's role gets (fc-infra ``media-grants``):

  sync  (mm-display-sync, role media_report): the replay's grants as they
        stand -- column grants on images and image_versions -- plus DML on
        the two queue tables and the feed's walk;
  seed  (mm-display-seed, role media_display): DML (no DELETE) on the five
        tables a seed writes and on the queue, SELECT on image_masks and
        mask_store_revocations (the held guard), USAGE on the three id
        sequences a seed draws from.

Postgres checks every column a query names, so this runs on a real
postgres:16 (docker; skipped without it), migrated to head."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from app.models import DisplayDepiction, DisplayOriginal, FigureGallery
from app.object_store import LocalObjectStore
from app.spine import display_feed as feed
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from app.workers import display_copy as dc
from app.workers.raw_source import ReadOnlyStore
from PIL import Image
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from spine_fakes import serve
from test_display_feed import QueueSpine
from test_migration_chain_replay import (  # noqa: F401  (fixtures)
    _run_alembic,
    empty_database_url,
    pg_admin_url,
    pytestmark,
)
from test_spine_report_grants import GRANTS as REPLAY_GRANTS

#: media_report's additions (fc-infra media-grants, SHOW1): the queue and the
#: feed's walk (display_feed_walks: where its lease cursor goes on), read and written.
SYNC_GRANTS = """
GRANT SELECT, INSERT, UPDATE, DELETE ON display_originals, display_depictions, display_feed_walks TO media_report;
"""

#: media_display's grants (fc-infra media-grants, SHOW1), by name, fail closed.
SEED_GRANTS = """
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM media_display;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM media_display;
GRANT USAGE ON SCHEMA public TO media_display;
GRANT SELECT, INSERT, UPDATE ON images, image_versions, public_objects, public_object_images, figure_galleries TO media_display;
GRANT SELECT, UPDATE ON display_originals TO media_display;
GRANT SELECT, UPDATE ON display_depictions TO media_display;
GRANT SELECT ON image_masks, mask_store_revocations TO media_display;
GRANT USAGE ON SEQUENCE images_id_seq, image_versions_id_seq, figure_galleries_id_seq TO media_display;
"""

T0 = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)


def _role(owner_url: str, role: str, grants: str) -> str:
    owner = create_engine(owner_url, isolation_level="AUTOCOMMIT")
    with owner.connect() as conn:
        conn.execute(
            text(
                f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
                f"THEN CREATE ROLE {role} LOGIN PASSWORD 'r'; END IF; END $$"
            )
        )
        for statement in filter(str.strip, grants.split(";")):
            conn.execute(text(statement))
    owner.dispose()
    return owner_url.replace("postgres:postgres@", f"{role}:r@", 1)


@pytest.fixture()
def roles(empty_database_url: str) -> Iterator[dict[str, str]]:  # noqa: F811
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    yield {
        "owner": empty_database_url,
        "report": _role(empty_database_url, "media_report", REPLAY_GRANTS + SYNC_GRANTS),
        "display": _role(empty_database_url, "media_display", SEED_GRANTS),
    }


def _png(n: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), (n * 30 % 256, 30, 90)).save(buf, format="PNG")
    return buf.getvalue()


def test_sync_then_seed_then_sync_each_under_its_own_role(
    roles: dict[str, str], tmp_path: Path
) -> None:
    raw = LocalObjectStore(tmp_path / "raw")
    fake = QueueSpine()
    pids = []
    for n in range(3):
        data = _png(n)
        origin = hashlib.sha256(data).hexdigest()
        key = f"raw-img/sha256/{origin[:2]}/{origin}.png"
        raw.put(key, data, "image/png")
        pids.append(
            fake.depict_full(
                origin,
                seen=T0 - dt.timedelta(minutes=n),
                key=key,
                site="amiami",
                url=f"https://img.amiami.com/{n}.png",
            )
        )
    report = create_engine(roles["report"])
    display = create_engine(roles["display"], pool_size=4)
    lines: list[dict[str, Any]] = []
    try:
        with serve(fake) as url:

            def connect() -> SpineMediaClient:
                return SpineMediaClient(open_channel(url), policy=RetryPolicy(max_attempts=1))

            with Session(report, expire_on_commit=False) as db:
                assert feed.run_sync(db, connect, feed.FeedConfig(), emit=lines.append) == 0, lines
            code = dc.run_seed(
                sessionmaker(bind=display, expire_on_commit=False),
                ReadOnlyStore(raw),
                LocalObjectStore(tmp_path / "media"),
                dc.SeedConfig(threads=2, grants={"amiami": ["img.amiami.com"]}),
                emit=lines.append,
            )
            assert code == dc.EXIT_DONE, lines
            with Session(report, expire_on_commit=False) as db:
                assert (
                    feed.run_sync(db, connect, feed.FeedConfig(max_new=0), emit=lines.append) == 0
                )
                # status and requeue, as an operator runs them under the sync's role
                assert dc.status(db)["originals"] == {"seeded": 3}
                assert (
                    dc.requeue(db, state="failed", reason_prefix=None, force=False, apply=True) == 0
                )
        assert {fake.depictions[p]["state"] for p in pids} == {"derived"}
        with Session(create_engine(roles["owner"])) as db:
            assert len(db.execute(select(FigureGallery)).all()) == 3
            assert {d.outcome for d in db.execute(select(DisplayDepiction)).scalars()} == {
                "derived"
            }
            assert {o.state for o in db.execute(select(DisplayOriginal)).scalars()} == {"seeded"}
        # what each role must NOT reach
        with display.connect() as conn:
            for statement in (
                "DELETE FROM images",
                "SELECT * FROM spine_report_runs",
                "INSERT INTO image_masks (source_sha256) VALUES ('x')",
                "SELECT * FROM service_clients",
                "DELETE FROM display_originals",
            ):
                with pytest.raises(ProgrammingError, match="permission denied"):
                    conn.execute(text(statement))
                conn.rollback()
        with report.connect() as conn:
            for statement in (
                "SELECT storage_key FROM image_versions",
                "INSERT INTO figure_galleries (figure_id) VALUES ('x')",
            ):
                with pytest.raises(ProgrammingError, match="permission denied"):
                    conn.execute(text(statement))
                conn.rollback()
    finally:
        report.dispose()
        display.dispose()


def test_the_held_guard_runs_under_the_seeders_role(roles: dict[str, str], tmp_path: Path) -> None:
    """The guard reads image_masks, mask_store_revocations and figure_galleries
    (app.publication.mask_refusal): an ungranted capture of a picture with a
    published MFC mask is held under media_display's grants too."""
    import importlib

    lane = importlib.import_module("test_display_copy")
    owner = create_engine(roles["owner"])
    raw = lane._Raw(tmp_path / "raw")
    media = LocalObjectStore(tmp_path / "media")
    with Session(owner, expire_on_commit=False) as db:
        from app.workers.mask_policy import ImageOrigin
        from app.workers.mask_seed import seed_clean_image

        data = lane._photo(90)
        clean = seed_clean_image(
            db,
            media,
            data,
            origins=[ImageOrigin(url=lane.MFC_ITEM.format(n=90), source="mfc")],
            figure_id=str(uuid.UUID(int=290)),
        )
        lane._mask(db, media, clean.sha256)
        with Image.open(io.BytesIO(data)) as im:  # the same picture, other bytes
            buf = io.BytesIO()
            im.save(buf, format="PNG", compress_level=1)
        sha, key = raw.put(buf.getvalue())
        lane._queue(
            db,
            sha,
            key,
            depictions=[
                (
                    str(uuid.UUID(int=90)),
                    str(uuid.UUID(int=190)),
                    "https://cdn.unlisted.example/p/90.jpg",
                    "unlisted",
                )
            ],
        )
    owner.dispose()
    display = create_engine(roles["display"])
    try:
        code = dc.run_seed(
            lambda: Session(display, expire_on_commit=False),
            ReadOnlyStore(raw.store),
            media,
            dc.SeedConfig(threads=0, grants=lane.GRANTS),
            emit=lambda e: None,
        )
        assert code == dc.EXIT_DONE
        with Session(display) as db:
            row = db.get(DisplayOriginal, sha)
            assert (row.state, row.reason) == ("held", "would_withdraw_mask:not_granted")
    finally:
        display.dispose()
