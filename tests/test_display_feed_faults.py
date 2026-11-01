# ruff: noqa: F811  (the spine fixture is imported from test_display_feed)
"""SHOW1 feed faults: the batch's copy of a row the lane left to it is
reported by the feed (the feed re-leases the row every run, so the replay never
can); a window full of never-reportable rows ends 1; the FAILED cap counts a day."""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from pathlib import Path
from typing import Any

from app.models import DisplayDepiction, DisplayOriginal
from app.object_store import LocalObjectStore
from app.spine import display_feed as feed
from app.spine.media_v1 import media_pb2
from app.workers import display_copy as dc
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image
from app.workers.raw_source import ReadOnlyStore
from sqlalchemy import select
from sqlalchemy.orm import Session

from spine_fakes import sha
from test_display_copy import _photo
from test_display_feed import T0, QueueSpine, _event, _sync, spine  # noqa: F401


def test_a_row_left_to_the_batch_is_reported_once_the_batch_seeds_it(
    db_session: Session, spine: tuple[QueueSpine, str], tmp_path: Any
) -> None:
    fake, url = spine
    data = _photo(5)
    origin = hashlib.sha256(data).hexdigest()
    pid = fake.depict_full(origin, seen=T0, url="https://img.store.example/5.jpg")
    assert _sync(db_session, url)[0] == feed.EXIT_DONE
    o = db_session.get(DisplayOriginal, origin)
    o.state, o.reason = "stuck", "too_large_for_lane:30000000"
    db_session.commit()
    fake.held.clear()
    code, lines = _sync(db_session, url)  # nothing in media yet: nothing sent
    assert fake.depictions[pid]["state"] == "captured"
    d = db_session.get(DisplayDepiction, pid)
    seed_clean_image(  # the GPU batch reaches it
        db_session,
        LocalObjectStore(tmp_path / "m"),
        data,
        origins=[ImageOrigin(url=d.origin_url, source=d.source_site)],
        figure_id=d.product_id,
    )
    fake.held.clear()
    code, lines = _sync(db_session, url)
    assert _event(lines, "reported")["batch_copy"] == 1
    assert fake.depictions[pid]["state"] == "derived"


def test_a_window_full_of_never_reportable_rows_ends_1_with_stalled(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    for i in range(10):
        fake.depict_full(sha(f"s{i}"), seen=T0 - dt.timedelta(minutes=i))
    fake.depict_full(sha("old"), seen=T0 - dt.timedelta(days=5))
    cfg = {"lease_items": 5, "max_rounds": 2}
    _sync(db_session, url, **cfg)
    for o in db_session.execute(select(DisplayOriginal)).scalars():
        o.state = "stuck"
    db_session.commit()
    fake.held.clear()
    code, lines = _sync(db_session, url, **cfg)
    assert code == feed.EXIT_REFUSED and _event(lines, "stalled")["already"] == 10


def test_failed_reports_are_capped_across_runs_not_per_run(
    db_session: Session, spine: tuple[QueueSpine, str], tmp_path: Path
) -> None:
    fake, url = spine
    raw = LocalObjectStore(tmp_path / "raw")
    media = LocalObjectStore(tmp_path / "media")
    n = 200
    for i in range(n):
        origin = sha(f"wrong{i}")
        fake.depict_full(origin, seen=T0 - dt.timedelta(seconds=i))
        # every object is a real image, but not the bytes the spine recorded
        raw.put(f"raw-img/sha256/{origin[:2]}/{origin}.jpg", _photo(i), "image/png")
    assert _sync(db_session, url)[0] == feed.EXIT_DONE  # queue all 200
    runs = []
    for _run in range(4):  # four hourly seeder runs, a feed run after each
        lines: list[dict[str, Any]] = []
        code = dc.run_seed(
            lambda: db_session,
            ReadOnlyStore(raw),
            media,
            dc.SeedConfig(threads=0),  # the CronJob's defaults: max_failed 25, 5 %
            emit=lines.append,
            now=lambda: T0,
        )
        fcode, flines = _sync(db_session, url, max_new=0)
        runs.append((code, next(e for e in flines if e["event"] == "reported")["failed"], fcode))
    sent = [
        r for r in fake.sent("ReportDerivative") if r.outcome == media_pb2.DERIVE_OUTCOME_FAILED
    ]
    assert len(sent) == 26  # the first run's; every later run holds them (exit 1)
    assert [r[2] for r in runs] == [0, 1, 1, 1]


def test_failed_reports_sent_today_alone_hold_nothing_and_say_nothing(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """Past the day's cap with no FAILED report waiting: nothing to hold, so
    no failed_held line (an alert on it would fire every run) and exit 0."""
    fake, url = spine
    origin = sha("sent")
    db_session.add(
        DisplayOriginal(
            origin_sha256=origin,
            storage_key=f"raw-img/sha256/{origin[:2]}/{origin}.jpg",
            codec="identity",
            first_seen_at=T0,
            state="failed",
            reason="undecodable",
        )
    )
    db_session.flush()
    db_session.add(
        DisplayDepiction(
            product_image_id=fake.depict_full(origin, seen=T0),
            origin_sha256=origin,
            product_id=str(uuid.UUID(int=5)),
            origin_url="https://x.example/a.jpg",
            source_site="amiami",
            first_seen_at=T0,
            reported_at=T0 - dt.timedelta(hours=1),
            outcome="failed",
        )
    )
    db_session.commit()
    code, lines = _sync(db_session, url, max_new=0, max_failed_reports=0)
    assert code == feed.EXIT_DONE
    assert [e for e in lines if e["event"] == "failed_held"] == []
