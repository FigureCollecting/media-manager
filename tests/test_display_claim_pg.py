"""SHOW1 on a real postgres:16 (the migration test's
disposable container, removed by name at module teardown): the claim's
FOR UPDATE SKIP LOCKED (a mutant without it survives every SQLite test),
the seed race between two seeders inside real transactions, and a
depiction the feed commits inside the seed's settle."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from app.models import DisplayDepiction, DisplayOriginal, PublicObject
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.spine import display_feed as feed
from app.workers import display_copy as dc
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image, seed_with_retry
from PIL import Image
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

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
