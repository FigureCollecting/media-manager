# ruff: noqa: F811  (the lane fixture is imported from test_display_copy)
"""SHOW1 seeder faults: poison captures past the header, transient bucket codes, a
depiction queued mid-seed, and an isolated claim's decode units."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.models import Base, DisplayDepiction, DisplayOriginal, FigureGallery
from app.object_store import LocalObjectStore
from app.workers import display_copy as dc
from app.workers.raw_source import ReadOnlyStore
from botocore.exceptions import ClientError
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import sessionmaker

from test_display_copy import (  # noqa: F401
    GRANTS,
    STORE,
    T0,
    _photo,
    _pid,
    _queue,
    _Raw,
    _seed_all,
    lane,
)


def _jpeg(n: int) -> bytes:
    rng = np.random.default_rng(n)
    arr = rng.integers(0, 255, (400, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_a_truncated_capture_is_failed_undecodable(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    whole = _jpeg(7)
    cut = whole[: len(whole) * 6 // 10]
    sha, key = lane["raw"].put(cut)
    _queue(db, sha, key, depictions=[(_pid(70), _pid(170), STORE.format(n=70), "amiami")])
    seen = []
    for _ in range(3):
        o = db.get(DisplayOriginal, sha)
        o.claimed_until = None  # the next run (after hold_until)
        db.commit()
        code, lines = _seed_all(lane)
        o = db.get(DisplayOriginal, sha)
        seen.append((code, o.state, o.reason, o.attempts))
    assert seen[0][:3] == (0, "failed", "undecodable")


def test_truncated_captures_do_not_stop_the_run(lane: dict[str, Any]) -> None:
    db = lane["db"]
    import datetime as dt

    for n in range(12):
        whole = _jpeg(100 + n)
        sha, key = lane["raw"].put(whole[: len(whole) // 2])
        _queue(
            db,
            sha,
            key,
            depictions=[(_pid(700 + n), _pid(800 + n), STORE.format(n=700 + n), "amiami")],
            seen=T0 + dt.timedelta(seconds=n),
        )
    good = _jpeg(999)
    sha, key = lane["raw"].put(good)
    _queue(
        db,
        sha,
        key,
        depictions=[(_pid(799), _pid(899), STORE.format(n=799), "amiami")],
        seen=T0 - dt.timedelta(days=1),
    )
    code, _lines = _seed_all(lane)
    assert code == dc.EXIT_DONE and db.get(DisplayOriginal, sha).state == "seeded"


def _ce(code: str, status: int) -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject"
    )


def test_transient_bucket_refusals_are_outages() -> None:
    for code, status in (("RequestTimeout", 400), ("TooManyRequests", 429), ("SlowDown", 503)):
        assert dc._raised("fetch", _ce(code, status)).kind == "outage", code
    assert dc._raised("fetch", _ce("AccessDenied", 403)).kind == "config"
    assert dc._raised("fetch", _ce("BadDigest", 400)).kind == "error"


def test_a_broken_decode_is_the_images_own() -> None:
    assert (
        dc._raised("seed", OSError("image file is truncated (12 bytes not processed)")).kind
        == "failed"
    )
    assert dc._raised("seed", OSError("[Errno 28] No space left on device")).kind == "error"
    assert dc._raised("seed", ConnectionResetError(104, "reset")).kind == "outage"


UNLISTED = "https://cdn.unlisted.example/p/{n}.jpg"


def test_a_depiction_queued_mid_seed_goes_back_to_pending(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(91))
    _queue(db, sha, key, depictions=[(_pid(91), _pid(191), STORE.format(n=91), "amiami")])

    class FeedMidSeed(LocalObjectStore):
        done = False

        def put(self, k: str, data: bytes, content_type: str) -> None:
            super().put(k, data, content_type)
            if not FeedMidSeed.done:  # the feed's _queue_one, committed while the seed runs
                FeedMidSeed.done = True
                db.add(
                    DisplayDepiction(
                        product_image_id=_pid(92),
                        origin_sha256=sha,
                        product_id=_pid(192),
                        origin_url=UNLISTED.format(n=92),
                        source_site="unlisted",
                        first_seen_at=T0,
                    )
                )

    code = dc.run_seed(
        lambda: db,
        ReadOnlyStore(lane["raw"].store),
        FeedMidSeed(lane["media"]._root),
        dc.SeedConfig(threads=0, grants=GRANTS),
        emit=lambda e: None,
        now=lambda: T0,
    )
    assert code == dc.EXIT_DONE
    late = db.get(DisplayDepiction, _pid(92))
    links = {(g.figure_id, g.source_url) for g in db.execute(select(FigureGallery)).scalars()}
    assert late.linked_at is None or (_pid(192), UNLISTED.format(n=92)) in links


def test_a_request_for_every_unit_goes_before_new_small_ones() -> None:
    """An isolated claim (every unit) waits only for the seeds in flight:
    a small request made while it waits is granted after it, never before."""
    units = dc._Units(6)
    held = units.acquire(1)
    order: list[str] = []
    whole = threading.Thread(
        target=lambda: (units.acquire(6), order.append("whole"), units.release(6)),
        daemon=True,
    )
    whole.start()
    while not getattr(units, "_whole_waiting", 1):
        time.sleep(0.001)
    small = threading.Thread(
        target=lambda: (units.acquire(1), order.append("small"), units.release(1)),
        daemon=True,
    )
    small.start()
    time.sleep(0.05)
    assert order == []  # the small one waits behind the whole one
    units.release(held)
    whole.join(5)
    small.join(5)
    assert order == ["whole", "small"]


def _run_with_a_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, error: Exception
) -> tuple[int, list[dict[str, Any]]]:
    """Four queued originals; ``where`` (the claim or the settle) raises
    ``error`` from its second call on."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'q.db'}", connect_args={"check_same_thread": False, "timeout": 30}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    raw = _Raw(tmp_path / "raw")
    with factory() as db:
        for n in range(4):
            sha, key = raw.put(_photo(n))
            _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    real = getattr(dc, where)
    calls = {"n": 0}

    def flaky(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise error
        return real(*a, **k)

    monkeypatch.setattr(dc, where, flaky)
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        factory,
        ReadOnlyStore(raw.store),
        LocalObjectStore(tmp_path / "m"),
        dc.SeedConfig(threads=2, grants=GRANTS),
        emit=lines.append,
        now=lambda: T0,
    )
    return code, lines


@pytest.mark.parametrize("where", ["claim_with_history", "_settle"])
def test_a_database_drop_at_the_claim_or_the_settle_stops_the_run_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    drop = OperationalError("UPDATE", {}, Exception("server closed the connection unexpectedly"))
    code, lines = _run_with_a_fault(tmp_path, monkeypatch, where, drop)
    assert code == dc.EXIT_STOPPED
    assert lines[-1]["event"] == "stopped" and lines[-1]["why"] == "database:OperationalError"


@pytest.mark.parametrize(
    ("error", "code", "why"),
    [
        (
            ProgrammingError("SELECT", {}, Exception("permission denied")),
            dc.EXIT_CONFIG,
            "database:ProgrammingError",
        ),
        (RuntimeError("a bug in the settle"), dc.EXIT_BREAKER, "lane:RuntimeError"),
    ],
)
def test_a_grant_or_a_bug_at_the_settle_stops_the_run_with_its_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, code: int, why: str
) -> None:
    got, lines = _run_with_a_fault(tmp_path, monkeypatch, "_settle", error)
    assert got == code
    assert lines[-1]["event"] == "stopped" and lines[-1]["why"] == why


def test_a_decoder_buffer_overrun_is_the_images_own() -> None:
    """Pillow's codec status -1 (the data runs past the image buffer)."""
    fault = OSError("buffer overrun when reading image file")
    assert dc._raised("seed", fault).kind == "failed"
    assert dc._raised("seed", OSError("out of memory when reading image file")).kind == "error"
