# ruff: noqa: F811  (the spine fixture is imported from test_display_feed)
"""SHOW1: the feed WALKS the spine's queue with a lease cursor (media.v1
0.12.0 LeasePendingImagesRequest.after) and holds what it leases briefly.

  * A run looks at the newest end first, goes on from where the last run's
    walk stopped once it meets rows it has queued already, and keeps its
    place in ``display_feed_walks``. A walk that reaches the end of the queue
    ends; the next run starts a new one at the newest end.
  * Every row of an answer must come after the cursor sent, in the total
    order asked; a token the cursor could not carry back verbatim, or a row
    before the cursor, is a deployment fault.
  * The hold is at most 120 s: a hold ends only by lapsing (no release RPC),
    so a row the feed will never report goes back to the replay quickly.

RT4 and RT5 are the SHOW1 red team's starvation findings."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import grpc
import pytest
from app.models import DisplayDepiction, DisplayOriginal
from app.spine import display_feed as feed
from app.spine.media_v1 import media_pb2
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spine_fakes import serve, sha
from test_display_feed import (
    INSTANT,
    T0,
    QueueSpine,
    _client,
    _event,
    _row,
    _Rows,
    _sync,
    spine,  # noqa: F401 -- fixture
)


def _queued(db: Session, pids: list[str]) -> int:
    return sum(1 for pid in pids if db.get(DisplayDepiction, pid) is not None)


def _afters(fake: QueueSpine) -> list[tuple[str, str] | None]:
    return [
        (r.after.first_seen_at, r.after.product_image_id) if r.HasField("after") else None
        for r in fake.sent("LeasePendingImages")
    ]


def _place(fake: QueueSpine, pid: str) -> tuple[str, str]:
    return (fake.depictions[pid]["seen"].strftime(INSTANT), pid)


def _walk(db: Session, order: str = "newest") -> tuple[str, str] | None:
    from app.models import DisplayFeedWalk

    db.expire_all()
    walk = db.get(DisplayFeedWalk, order)
    return (walk.after_first_seen_at, walk.after_product_image_id) if walk else None


def _stick(db: Session) -> None:
    for o in db.execute(select(DisplayOriginal)).scalars():
        o.state, o.reason = "stuck", "unsupported_codec:zstd"
    db.commit()


# --------------------------------------------------------------------------
# RT4 and RT5
# --------------------------------------------------------------------------


def test_rt4_never_reportable_rows_no_longer_starve_the_older_ones(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """The 10 newest originals end stuck and fill a run's whole window (2
    rounds x 5 rows); 5 older ones wait. The next run goes on from the walk's
    place and queues them; a walk that reaches the end starts again at the
    newest end."""
    fake, url = spine
    stuck = [fake.depict_full(sha(f"s{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(10)]
    older = [
        fake.depict_full(sha(f"o{i}"), seen=T0 - dt.timedelta(days=5, minutes=i)) for i in range(5)
    ]
    cfg = {"lease_items": 5, "max_rounds": 2}
    assert _sync(db_session, url, **cfg)[0] == feed.EXIT_DONE
    _stick(db_session)
    results = []
    for _run in range(3):
        fake.lapse(3600)  # the next hourly run: every hold of the last one has lapsed
        code, lines = _sync(db_session, url, **cfg)
        e = _event(lines, "enqueued")
        results.append((code, e["new_depictions"], e["already"], e["why"]))
    assert _queued(db_session, older) == 5
    assert results == [
        (0, 5, 5, "rounds"),  # the newest end (all queued already), then the walk: the older 5
        (0, 0, 5, "empty"),  # the walk reaches the end of the queue and ends
        (0, 0, 10, "rounds"),  # a new walk from the newest end
    ]
    assert _queued(db_session, stuck) == 10


def test_rt5_the_replay_leases_a_row_the_feed_leased_once_two_minutes_pass(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """A row the lane left to the GPU batch (stuck too_large_for_lane) is
    leased again by the feed's next walk. The hold is the feed's only claim on
    it and ends only by lapsing, so the feed asks for 120 s at most: the
    mm-report replay's origin-filtered lease gets the row two minutes after
    any feed run. (An hourly feed holding for 3600 s kept it held always.)"""
    fake, url = spine
    origin = sha("big")
    pid = fake.depict_full(origin, seen=T0)
    assert _sync(db_session, url)[0] == feed.EXIT_DONE
    o = db_session.get(DisplayOriginal, origin)
    o.state, o.reason = "stuck", "too_large_for_lane:30000000"
    db_session.commit()
    fake.lapse(3600)  # the next hourly feed run comes first
    _sync(db_session, url)
    assert pid in fake.held  # its walk leased the row again
    fake.lapse(120)
    out = _client(url).lease([bytes.fromhex(origin)], max_items=200, lease_owner="mm-report/x")
    assert [i.product_image_id for i in out.response.images] == [pid]
    assert {r.lease_seconds for r in fake.sent("LeasePendingImages")[:-1]} == {120}


# --------------------------------------------------------------------------
# the cursor
# --------------------------------------------------------------------------


def test_a_run_walks_on_after_the_last_row_it_was_handed_verbatim_ties_included(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """Rows that share a first_seen_at are ordered by product_image_id (by
    value: either case, mixed here, so ``B`` sorts between ``a`` and ``c``);
    the cursor carries both halves back as sent."""
    fake, url = spine
    spelt = [str(uuid.UUID(int=0xA)), str(uuid.UUID(int=0xB)).upper(), str(uuid.UUID(int=0xC))]
    tied = [fake.depict_full(sha(f"t{n}"), seen=T0, pid=pid) for n, pid in enumerate(spelt)]
    older = fake.depict_full(sha("older"), seen=T0 - dt.timedelta(hours=1))
    code, lines = _sync(db_session, url, lease_items=1)
    assert code == feed.EXIT_DONE
    token = T0.strftime(INSTANT)
    assert _afters(fake) == [
        None,
        (token, tied[2]),
        (token, tied[1]),
        (token, tied[0]),
        _place(fake, older),
    ]
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 4
    e = _event(lines, "enqueued")
    assert (e["why"], e["walk_done"], e["walk_after"]) == ("empty", True, None)
    assert _walk(db_session) is None  # a walk that reached the end of the queue is gone


def test_a_later_run_looks_at_the_newest_end_then_goes_on_from_the_walks_place(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    w = [fake.depict_full(sha(f"w{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(4)]
    _sync(db_session, url, lease_items=2, max_new=2)
    assert _walk(db_session) == _place(fake, w[1])
    fake.lapse(3600)
    first = len(fake.sent("LeasePendingImages"))
    code, lines = _sync(db_session, url, lease_items=2)
    assert code == feed.EXIT_DONE
    # the newest end (w0 and w1, queued already), then on after w1: w2 and w3
    assert _afters(fake)[first:] == [None, _place(fake, w[1]), _place(fake, w[3])]
    assert _queued(db_session, w) == 4
    assert _walk(db_session) is None


def test_a_head_pass_that_reaches_the_walks_place_carries_the_walk_on(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """New captures at the newest end are queued first; once the pass meets
    the walk's place it is the walk, and its place moves with it."""
    fake, url = spine
    w = [fake.depict_full(sha(f"w{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(4)]
    _sync(db_session, url, lease_items=2, max_new=2)  # the walk at w1
    fresh = fake.depict_full(sha("fresh"), seen=T0 + dt.timedelta(minutes=5))
    fake.lapse(3600)
    first = len(fake.sent("LeasePendingImages"))
    code, lines = _sync(db_session, url, lease_items=2, max_rounds=2)
    # fresh and w0 (new rows: go on), then w1 and w2: past the walk's place
    assert _afters(fake)[first:] == [None, _place(fake, w[0])]
    assert _queued(db_session, [fresh, *w[:3]]) == 4
    assert _walk(db_session) == _place(fake, w[2])
    assert _event(lines, "enqueued")["walk_after"] == _place(fake, w[2])[0]


def test_a_head_pass_with_nothing_new_that_reaches_the_walks_place_carries_it_on(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """Rows past the walk's place may be queued already (here the oldest-first
    walk queued them). An answer with nothing new that reaches the walk's
    place carries the walk on from its last row, never back to the place it
    passed."""
    fake, url = spine
    r = [fake.depict_full(sha(f"r{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(4)]
    _sync(db_session, url, lease_items=2, max_new=2)  # the newest walk at r1
    fake.lapse(3600)
    _sync(db_session, url, order="oldest", lease_items=2, max_new=2)  # r3 and r2 queued
    fake.lapse(3600)
    first = len(fake.sent("LeasePendingImages"))
    code, lines = _sync(db_session, url, lease_items=3)
    assert code == feed.EXIT_DONE and _event(lines, "enqueued")["new_depictions"] == 0
    # r0-r2, all queued already and past r1: on after r2, then r3, then the end
    assert _afters(fake)[first:] == [None, _place(fake, r[2]), _place(fake, r[3])]
    assert _walk(db_session) is None


def test_a_run_cut_off_among_new_rows_leaves_the_walk_there_not_past_the_rest(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """A walk is under way in the backlog when a burst of new captures larger
    than a run's window lands (2 rounds x 3 rows). The head pass moves the
    walk with it, so the next run goes on through the rest of the burst, not
    to the backlog walk's place (the rest would wait for that walk to wrap).
    The price: past the burst, the walk goes on through backlog rows queued
    already, a round per lease of them, never a row lost."""
    fake, url = spine
    backlog = [
        fake.depict_full(sha(f"b{i}"), seen=T0 - dt.timedelta(days=3, minutes=i)) for i in range(30)
    ]
    cfg = {"lease_items": 3, "max_rounds": 2}
    _sync(db_session, url, **cfg)
    assert _walk(db_session) == _place(fake, backlog[5])
    burst = [
        fake.depict_full(sha(f"n{i}"), seen=T0 + dt.timedelta(minutes=30, seconds=-i))
        for i in range(12)
    ]
    per_run = []
    for _run in range(3):
        fake.lapse(1800)  # the next half-hourly run: every hold of the last one has lapsed
        _sync(db_session, url, **cfg)
        per_run.append(_queued(db_session, burst))
    assert per_run == [6, 9, 12]
    assert _queued(db_session, backlog) == 6  # no backlog row went before a newer capture
    assert _walk(db_session) == _place(fake, burst[11])
    fake.lapse(1800)
    code, lines = _sync(db_session, url, **cfg)
    e = _event(lines, "enqueued")
    assert (code, e["new_depictions"], e["already"]) == (0, 0, 6)  # n0-n2, then b0-b2 again
    assert _walk(db_session) == _place(fake, backlog[2])


def test_a_head_pass_that_meets_ground_covered_puts_the_walk_back_before_it_goes_on(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """The head pass queues two new captures (the walk moves with it), then
    meets rows queued already: everything before the walk's place is covered,
    so the walk goes back there at once. A run that then stops (the spine
    down) keeps the deep place, never the head pass's."""
    fake, url = spine
    w = [fake.depict_full(sha(f"w{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(8)]
    _sync(db_session, url, lease_items=2, max_rounds=2)
    assert _walk(db_session) == _place(fake, w[3])
    fresh = [
        fake.depict_full(sha(f"f{i}"), seen=T0 + dt.timedelta(minutes=5 - i)) for i in range(2)
    ]
    fake.lapse(3600)
    fake.fail(
        "LeasePendingImages", None, None, grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAVAILABLE
    )
    first = len(fake.sent("LeasePendingImages"))
    code, _lines = _sync(db_session, url, lease_items=2)
    assert code == feed.EXIT_STOPPED
    # fresh (new: on), w0 and w1 (queued already: back to the walk's place), then down
    assert _afters(fake)[first : first + 3] == [None, _place(fake, fresh[1]), _place(fake, w[3])]
    assert _queued(db_session, fresh) == 2
    assert _walk(db_session) == _place(fake, w[3])


def test_a_run_that_neither_queues_nor_moves_its_walk_is_stalled_and_ends_1(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    k = [fake.depict_full(sha(f"k{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(4)]
    _sync(db_session, url, lease_items=2, max_new=2)  # the walk at k1
    fake.lapse(3600)
    lines: list[dict[str, Any]] = []
    ticks = iter([0.0, 0.0, 10_000.0])  # its budget spent after one round
    code = feed.run_sync(
        db_session,
        lambda: _client(url),
        feed.FeedConfig(lease_items=2),
        emit=lines.append,
        now=lambda: T0,
        monotonic=lambda: next(ticks),
    )
    # its one round went to the newest end, every row of it queued already
    assert code == feed.EXIT_REFUSED
    assert _event(lines, "stalled") == {
        "event": "stalled",
        "rounds": 1,
        "already": 2,
        "walk_after": _place(fake, k[1])[0],
    }
    assert _walk(db_session) == _place(fake, k[1])
    fake.lapse(3600)
    code, lines = _sync(db_session, url, lease_items=2, max_rounds=2)
    assert code == feed.EXIT_DONE and _event(lines, "enqueued")["new_depictions"] == 2
    assert not [e for e in lines if e["event"] == "stalled"]


def test_a_walk_that_moves_without_queueing_is_progress_not_a_stall(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    for i in range(4):
        fake.depict_full(sha(f"r{i}"), seen=T0 - dt.timedelta(minutes=i))
    _sync(db_session, url, lease_items=2)  # queues all four; the walk ends
    _stick(db_session)
    fake.lapse(3600)
    code, lines = _sync(db_session, url, lease_items=2, max_rounds=2)  # a new walk: all known
    assert code == feed.EXIT_DONE
    assert _event(lines, "enqueued")["new_depictions"] == 0
    assert not [e for e in lines if e["event"] == "stalled"]


def test_an_empty_queue_ends_the_walk(db_session: Session, spine: tuple[QueueSpine, str]) -> None:
    fake, url = spine
    for i in range(3):
        fake.depict_full(sha(f"e{i}"), seen=T0 - dt.timedelta(minutes=i))
    _sync(db_session, url, lease_items=1, max_new=1)
    assert _walk(db_session) is not None
    for d in fake.depictions.values():
        d["state"] = "derived"
    code, lines = _sync(db_session, url)
    assert code == feed.EXIT_DONE and _event(lines, "enqueued")["walk_done"] is True
    assert _walk(db_session) is None


def test_a_stopped_run_keeps_the_walks_place_and_the_next_one_goes_on_from_it(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    r = [fake.depict_full(sha(f"r{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(3)]
    fake.fail("LeasePendingImages", None, grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAVAILABLE)
    code, _lines = _sync(db_session, url, lease_items=1)
    assert code == feed.EXIT_STOPPED
    assert _walk(db_session) == _place(fake, r[0])  # each round's place is committed with it
    fake.lapse(3600)
    code, _lines = _sync(db_session, url, lease_items=1)
    assert code == feed.EXIT_DONE and _queued(db_session, r) == 3


def test_each_order_keeps_its_own_walk_and_oldest_walks_upward(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    r = [fake.depict_full(sha(f"r{i}"), seen=T0 - dt.timedelta(minutes=i)) for i in range(3)]
    _sync(db_session, url, lease_items=1, max_new=1)
    fake.lapse(3600)
    first = len(fake.sent("LeasePendingImages"))
    _sync(db_session, url, order="oldest", lease_items=1, max_rounds=2)
    assert _afters(fake)[first:] == [None, _place(fake, r[2])]
    assert _walk(db_session, "newest") == _place(fake, r[0])
    assert _walk(db_session, "oldest") == _place(fake, r[1])


def test_two_syncs_at_once_stop_the_later_one_cleanly_and_keep_the_others_walk(
    db_session: Session, spine: tuple[QueueSpine, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CronJob's concurrencyPolicy Forbid is the only guard; a Job made
    by hand from it bypasses it. Both read "no walk"; the other commits its
    walk first, so this one's commit of a new walk breaks the key. It stops,
    resumable (exit 3), naming the clash: never a traceback, never exit 1."""
    from app.models import DisplayFeedWalk

    fake, url = spine
    for i in range(4):
        fake.depict_full(sha(f"q{i}"), seen=T0 - dt.timedelta(minutes=i))
    theirs = ("2026-10-31T11:00:00.000000Z", str(uuid.UUID(int=1)))
    db_session.add(
        DisplayFeedWalk(
            lease_order="newest",
            after_first_seen_at=theirs[0],
            after_product_image_id=theirs[1],
            started_at=T0,
            updated_at=T0,
        )
    )
    db_session.commit()
    real_get = db_session.get

    def read_before_their_commit(entity: Any, key: Any, **kw: Any) -> Any:
        return None if entity is DisplayFeedWalk else real_get(entity, key, **kw)

    monkeypatch.setattr(db_session, "get", read_before_their_commit)
    code, lines = _sync(db_session, url, lease_items=2)
    monkeypatch.undo()
    assert code == feed.EXIT_STOPPED
    assert "another sync" in _event(lines, "stopped")["why"]
    assert _walk(db_session) == theirs
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 0


# --------------------------------------------------------------------------
# deployment faults
# --------------------------------------------------------------------------


def test_a_spine_that_ignores_the_cursor_is_a_deployment_fault(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """A 0.12.0 draft built before the cursor sends the new fields but starts
    at the newest end again: its first row is not after the cursor sent."""
    fake, url = spine
    fake.ignores_cursor = True
    for i in range(4):
        fake.depict_full(sha(f"i{i}"), seen=T0 - dt.timedelta(minutes=i))
    _sync(db_session, url, lease_items=1, max_rounds=2)  # the walk at i1
    fake.lapse(3600)
    before = db_session.scalar(select(func.count()).select_from(DisplayDepiction))
    code, lines = _sync(db_session, url, lease_items=1)
    assert code == feed.EXIT_STOPPED
    assert "after the cursor" in _event(lines, "stopped")["why"]
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == before


@pytest.mark.parametrize(
    "token",
    [
        "2026-10-31T12:00:00.000Z",  # three digits
        "2026-10-31T12:00:00Z",  # no fraction
        "2026-10-31T12:00:00.000000+00:00",  # an offset (Python's isoformat)
        "2026-10-31T24:00:00.000000Z",  # hour 24 (PostgreSQL reads it)
        "2026-10-31T12:00:60.000000Z",  # second 60 (PostgreSQL reads it)
        "2026-02-30T12:00:00.000000Z",  # no such day
        "\uff12\uff10\uff12\uff16-10-31T12:00:00.000000Z",  # fullwidth digits (a str \d matches them)
        "2026-10-31T12:0\u0660:00.000000Z",  # an Arabic-Indic digit (strptime's %M reads it)
    ],
)
def test_a_first_seen_at_a_cursor_could_not_carry_back_is_a_deployment_fault(
    db_session: Session, token: str
) -> None:
    fake = _Rows()
    fake.rows = [_row(sha("a"), first_seen_at=token)]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_STOPPED
    assert "0.12.0" in _event(lines, "stopped")["why"]
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 0


def test_a_row_with_first_seen_at_but_no_product_id_is_a_deployment_fault(
    db_session: Session,
) -> None:
    """A 0.12.0 server always sends both; one without the other is not
    skipped as a bad row but stops the run, as a server before 0.12.0 does."""
    fake = _Rows()
    fake.rows = [_row(sha("a"), product_id=None)]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_STOPPED
    assert "0.12.0" in _event(lines, "stopped")["why"]


def test_a_product_image_id_that_is_not_a_uuid_is_a_deployment_fault(
    db_session: Session,
) -> None:
    fake = _Rows()
    fake.rows = [_row(sha("a")), _row(sha("b"), product_image_id="not-a-uuid")]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_STOPPED
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 0


def test_tied_rows_out_of_their_uuid_order_are_a_deployment_fault(db_session: Session) -> None:
    class Unsorted(_Rows):
        def LeasePendingImages(self, request: Any, context: Any) -> Any:  # noqa: N802
            self._enter("LeasePendingImages", request, context)
            out, self.rows = self.rows, []
            return media_pb2.LeasePendingImagesResponse(images=out, lease_expires_at="x")

    fake = Unsorted()
    low, high = str(uuid.UUID(int=1)), str(uuid.UUID(int=2))
    fake.rows = [_row(sha("a"), product_image_id=low), _row(sha("b"), product_image_id=high)]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)  # newest first: high before low
    assert code == feed.EXIT_STOPPED
    assert "order" in _event(lines, "stopped")["why"]
