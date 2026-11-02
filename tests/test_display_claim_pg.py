"""SHOW1 on a real postgres:16 (the migration test's
disposable container, removed by name at module teardown): the claim's
FOR UPDATE SKIP LOCKED (a mutant without it survives every SQLite test),
the seed race between two seeders inside real transactions, a
depiction the feed commits inside the seed's settle, and what a sync that
races another says (postgres's message has a DETAIL line naming the row)."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from app.models import DisplayDepiction, DisplayOriginal, PublicObject
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.spine import copies
from app.spine import display_feed as feed
from app.workers import display_copy as dc
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image, seed_with_retry
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from spine_fakes import serve, sha
from test_display_feed import QueueSpine, _event, _sync
from test_migration_chain_replay import (  # noqa: F401  (fixtures)
    _run_alembic,
    empty_database_url,
    pg_admin_url,
    pytestmark,
)

T0 = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)


@pytest.fixture()
def factory(empty_database_url: str) -> Iterator[Any]:  # noqa: F811
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url, pool_size=4)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def _png(n: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), (n * 30 % 256, 30, 90)).save(buf, format="PNG")
    return buf.getvalue()


def test_pg1_a_claim_in_flight_is_skipped_never_claimed_twice(factory: Any) -> None:
    with factory() as db:
        for i, sha in enumerate(("a" * 64, "b" * 64)):
            db.add(
                DisplayOriginal(
                    origin_sha256=sha,
                    storage_key="k",
                    codec="identity",
                    first_seen_at=T0 - dt.timedelta(minutes=i),
                    state="pending",
                    attempts=0,
                )
            )
        db.commit()
    a = factory()
    # pod A's claim of the newest, its row lock held (between its SELECT and its COMMIT)
    a.execute(
        text("SELECT origin_sha256 FROM display_originals WHERE origin_sha256 = :s FOR UPDATE"),
        {"s": "a" * 64},
    )
    got: list[Any] = []
    b = factory()
    t = threading.Thread(
        target=lambda: got.append(dc.claim_next(b, owner="pod-b", claim_seconds=900, now=T0)),
        daemon=True,
    )
    t.start()
    t.join(8)
    blocked = t.is_alive()
    a.rollback()
    t.join(8)
    claimed = got[0].origin_sha256 if got and got[0] is not None else None
    assert not blocked and claimed == "b" * 64
    a.close()
    b.close()


@pytest.mark.parametrize("late", ["lane", "batch"])
def test_pg2_two_seeders_racing_one_picture(factory: Any, tmp_path: Path, late: str) -> None:
    """Both pass `image is None`, then both INSERT images(sha256); the LATE one
    blocks on the unique key and loses. The lane (seed_with_retry) recovers;
    the live batch's build (seed_clean_image, no retry) raises."""
    store = LocalObjectStore(tmp_path / "media")
    data = _png(7)
    sha = hashlib.sha256(data).hexdigest()
    gate = threading.Barrier(2, timeout=10)

    class Gated:
        def __init__(self, delay: float) -> None:
            self.first, self.delay = True, delay

        def get(self, key: str) -> bytes:
            return store.get(key)

        def put(self, key: str, body: bytes, content_type: str) -> None:
            store.put(key, body, content_type)
            if self.first:  # both seeders are past their `image is None` read
                self.first = False
                gate.wait()
                time.sleep(self.delay)

    outcomes: dict[str, str] = {}

    def run(name: str, seed: Any, url: str, delay: float) -> None:
        with factory() as db:
            try:
                seed(
                    db,
                    Gated(delay),
                    data,
                    origins=[ImageOrigin(url=url, source="amiami")],
                    figure_id=f"00000000-0000-0000-0000-00000000000{len(name)}",
                )
                outcomes[name] = "ok"
            except IntegrityError:
                outcomes[name] = "IntegrityError"

    lane = threading.Thread(
        target=run,
        args=("lane", seed_with_retry, "https://a.example/1.jpg", 0.5 if late == "lane" else 0.0),
    )
    batch = threading.Thread(
        target=run,
        args=(
            "batch",
            seed_clean_image,
            "https://a.example/2.jpg",
            0.5 if late == "batch" else 0.0,
        ),
    )
    lane.start()
    batch.start()
    lane.join(30)
    batch.join(30)
    with factory() as db:
        images = db.scalar(
            select(func.count()).select_from(ImageModel).where(ImageModel.sha256 == sha)
        )
        objects = db.scalar(select(func.count()).select_from(PublicObject))
    assert (images, objects) == (1, 1)
    assert outcomes == {"lane": "ok", "batch": "IntegrityError" if late == "batch" else "ok"}


class _NoSpine:
    """The feed's client where no report may be sent."""

    def report_derivative(self, message: Any) -> Any:
        raise AssertionError(f"nothing is reported here: {message}")


def test_pg3_a_depiction_committed_inside_the_settle_goes_back_to_the_seeder(
    factory: Any,
) -> None:
    """The settle reads the original's unlinked depictions, then commits
    ``seeded``. The feed (its own transaction; it read the original
    ``pending``, so left its state) commits D2 between the two: the original
    ends seeded with D2 never linked, and no claim takes a seeded original.
    The feed's next REPORT sends it back to the seeder."""
    origin = "c" * 64

    def depiction(pid: str, product: str) -> DisplayDepiction:
        return DisplayDepiction(
            product_image_id=pid,
            origin_sha256=origin,
            product_id=product,
            origin_url=f"https://img.amiami.example/{pid}.jpg",
            source_site="amiami",
            first_seen_at=T0,
        )

    d1, d2 = "00000000-0000-0000-0000-0000000000d1", "00000000-0000-0000-0000-0000000000d2"
    with factory() as db:
        db.add(
            DisplayOriginal(
                origin_sha256=origin,
                storage_key="k",
                codec="identity",
                first_seen_at=T0,
                state="pending",
                attempts=0,
            )
        )
        db.flush()
        db.add(depiction(d1, "00000000-0000-0000-0000-000000000001"))
        db.commit()
    seeder = factory()
    original = dc.claim_next(seeder, owner="pod-a", claim_seconds=900, now=T0)
    assert original is not None
    fired: list[str] = []

    def feed_commits_d2(conn: Any, cursor: Any, statement: str, *rest: Any) -> None:
        if (
            fired
            or "FROM display_depictions" not in statement
            or "linked_at IS NULL" not in statement
        ):
            return
        fired.append(statement)  # right after the settle's read: the feed queues D2
        with factory() as feed_db:
            feed_db.add(depiction(d2, "00000000-0000-0000-0000-000000000002"))
            feed_db.commit()

    engine = seeder.get_bind()
    event.listen(engine, "after_cursor_execute", feed_commits_d2)
    try:
        dc._settle(
            seeder,
            original,
            dc.SeedResult("seeded", look={}, linked=(d1,)),
            now=T0 + dt.timedelta(seconds=5),
            max_attempts=3,
            hold_until=T0 + dt.timedelta(hours=1),
        )
    finally:
        event.remove(engine, "after_cursor_execute", feed_commits_d2)
        seeder.close()
    assert fired
    with factory() as db:
        assert db.get(DisplayOriginal, origin).state == "seeded"  # the race happened
        assert db.get(DisplayDepiction, d2).linked_at is None
        counts = feed.report(
            db,
            _NoSpine(),  # type: ignore[arg-type]
            feed.FeedConfig(),
            lambda e: None,
            T0 + dt.timedelta(minutes=30),
        )
        assert db.get(DisplayOriginal, origin).state == "pending"
        assert counts["relink"] == 1
        claimed = dc.claim_next(
            db, owner="pod-b", claim_seconds=900, now=T0 + dt.timedelta(minutes=31)
        )
        assert claimed is not None and claimed.origin_sha256 == origin


def test_pg4_a_sync_racing_another_names_the_constraint_never_the_row(
    factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another sync commits the same bytes' original after this one read
    none: the insert clashes. Postgres puts the clashing row's key on a
    DETAIL line after the constraint; the stop names the constraint alone."""
    origin = sha("same-bytes")
    with factory() as other:
        other.add(
            DisplayOriginal(
                origin_sha256=origin,
                storage_key="k",
                codec="identity",
                first_seen_at=T0,
                state="pending",
                attempts=0,
            )
        )
        other.commit()
    fake = QueueSpine()
    fake.depict_full(origin, seen=T0)
    db = factory()
    real_get = db.get

    def read_before_their_commit(entity: Any, key: Any, **kw: Any) -> Any:
        return None if entity is DisplayOriginal else real_get(entity, key, **kw)

    monkeypatch.setattr(db, "get", read_before_their_commit)
    try:
        with serve(fake) as url:
            code, lines = _sync(db, url, lease_items=5)
    finally:
        monkeypatch.undo()
        db.close()
    assert code == feed.EXIT_STOPPED
    assert _event(lines, "stopped")["why"] == (
        "database: IntegrityError: another sync wrote the row first (two at once?): "
        'duplicate key value violates unique constraint "display_originals_pkey"'
    )


#: The two tables the copy lookups read once per original, and a filler of
#: copies, each minted from its own image and linked behind it (the prod shape),
#: so that a table walked whole is a real cost to the planner.
_LOOKED_UP = ("public_objects", "public_object_images")
#: The keys the lane looks a copy up by. An index that leads on any other
#: column bounds nothing here: on kind, say, every copy matches 'sanitized'.
_LOOKUP_KEYS = ("sha256", "image_version_id", "image_id")
_FILLER = """
INSERT INTO images (sha256, created_at)
  SELECT encode(sha256(convert_to('fill:' || g, 'UTF8')), 'hex'), now()
  FROM generate_series(1, 2000) g;
INSERT INTO image_versions
  (image_id, version_no, transform_spec, storage_key, visibility, age_rating, created_at)
  SELECT id, 2, '{}', 'd/' || sha256, 'public', 0, now() FROM images;
INSERT INTO public_objects (sha256, kind, mime, bytes, width, height, storage_key,
                            image_version_id, pixel_sha256, own_pixel_sha256, created_at)
  SELECT encode(sha256(convert_to('copy:' || id, 'UTF8')), 'hex'), 'sanitized', 'image/jpeg',
         1, 1, 1, 'copy/' || id, id, encode(sha256(convert_to('px:' || id, 'UTF8')), 'hex'),
         encode(sha256(convert_to('own:' || id, 'UTF8')), 'hex'), now()
  FROM image_versions;
INSERT INTO public_object_images (sha256, image_id)
  SELECT p.sha256, v.image_id FROM public_objects p JOIN image_versions v
  ON v.id = p.image_version_id;
"""


def _leading_column(conn: Any, index: str) -> str:
    return str(
        conn.exec_driver_sql(
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a"
            " ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]"
            " WHERE i.indexrelid = %(ix)s::regclass",
            {"ix": index},
        ).scalar_one()
    )


def _walked_whole(conn: Any, node: dict[str, Any], read: set[str]) -> list[str]:
    """Each read of a _LOOKED_UP table in the plan ``node`` that no index
    bounds by a _LOOKUP_KEYS column leading it: a Seq Scan, or an index walked
    whole (a condition on a later column of a primary key, or on a leading
    column that is no lookup key). ``read`` collects the tables the plan
    reads."""
    found: list[str] = []
    table = node.get("Relation Name")
    if table in _LOOKED_UP:
        read.add(table)
        probes = [node] if "Index Name" in node else []
        stack = [] if probes else list(node.get("Plans", []))
        while stack:  # a Bitmap Heap Scan reads through its Bitmap Index Scans
            child = stack.pop()
            probes += [child] if child["Node Type"] == "Bitmap Index Scan" else []
            stack += child.get("Plans", [])
        leads = [(_leading_column(conn, p["Index Name"]), p.get("Index Cond", "")) for p in probes]
        bounded = bool(leads) and all(
            column in _LOOKUP_KEYS and re.search(rf"\(\(?{column}\b", cond)
            for column, cond in leads
        )
        if not bounded:
            found.append(f"{node['Node Type']} on {table} {node.get('Index Name', '')}".strip())
    for child in node.get("Plans", []):
        found += _walked_whole(conn, child, read)
    return found


def test_pg5_the_copy_lookups_read_both_tables_by_index(factory: Any, tmp_path: Path) -> None:
    """SHOW1-THROUGHPUT: for every original, the feed's and the seeder's
    reported_copy reads public_objects by image_version_id (the copies its
    image minted) and, when it minted none, public_object_images by image_id
    (the copies it is linked behind); the seeder's _picture_copies reads
    public_object_images by image_id too. Each must be an index probe: in prod
    (1.33M rows) a table walked whole costs seconds per original. Seq scans
    are switched off, so a plan that still walks a table has no index to
    probe it by."""
    store = LocalObjectStore(tmp_path / "media")
    first = _png(5)
    tagged = io.BytesIO()  # the same picture, other bytes: a second capture
    info = PngInfo()
    info.add_text("Comment", "captured again")
    Image.open(io.BytesIO(first)).save(tagged, format="PNG", pnginfo=info)
    second = tagged.getvalue()
    origins = [hashlib.sha256(data).hexdigest() for data in (first, second)]
    with factory() as db:
        db.execute(text(_FILLER))
        db.commit()
        made = [
            seed_clean_image(
                db,
                store,
                data,
                origins=[ImageOrigin(url=f"https://a.example/{n}.png", source="amiami")],
                figure_id=f"00000000-0000-0000-0000-00000000005{n}",
            ).sha256
            for n, data in enumerate((first, second))
        ]
        db.commit()
        db.execute(text("ANALYZE"))
        db.commit()
    copy = made[0]
    assert made == [copy, copy]  # the second capture has no copy of its own

    statements: list[tuple[str, Any]] = []

    def record(*args: Any) -> None:
        statements.append((args[2], args[3]))  # (conn, cursor, statement, parameters, ...)

    with factory() as db:
        engine = db.get_bind()
        event.listen(engine, "before_cursor_execute", record)
        try:
            assert [copies.reported_copy(db, origin) for origin in origins] == [copy, copy]
            assert dc._picture_copies(db, {copy}, origins[1]) == {copy}
        finally:
            event.remove(engine, "before_cursor_execute", record)

    read: set[str] = set()
    walked: list[str] = []
    with factory() as db:
        conn = db.connection()
        conn.exec_driver_sql("SET LOCAL enable_seqscan = off")
        for statement, parameters in statements:
            if statement.lstrip().upper().startswith("SELECT"):
                explained = conn.exec_driver_sql(
                    "EXPLAIN (FORMAT JSON) " + statement, parameters
                ).scalar_one()
                walked += [
                    f"{w}: {' '.join(statement.split())}"
                    for w in _walked_whole(conn, explained[0]["Plan"], read)
                ]
    assert read == set(_LOOKED_UP)
    assert walked == []
