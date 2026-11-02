"""SHOW1: the display-copy lane's FEED (app.spine.display_feed) -- the two
steps that hold the spine identity and no bucket key.

  REPORT: every seeded original's unreported depictions get ReportDerivative
  with the copy the replay would report (re-checked against the media DB
  first); every failed original's get a FAILED report, held past a cap.
  ENQUEUE: LeasePendingImages{order, scope} (media.v1 0.12.0) fills the
  queue newest first, below a high-water mark; an answer from a server
  before 0.12.0 stops the run.

Driven against a stateful fake spine over real gRPC."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import grpc
import pytest
from app.models import DisplayDepiction, DisplayOriginal, PublicObject
from app.spine import display_feed as feed
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from app.spine.media_v1 import media_pb2
from app.workers.mask_seed import CLEAN_PIPELINE_VERSION
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from spine_fakes import AT, FakeSpine, seed_copy, serve, sha

T0 = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)
C = grpc.StatusCode


def _key(origin: str) -> str:
    return f"raw-img/sha256/{origin[:2]}/{origin}.jpg"


#: PendingImage.first_seen_at as a 0.12.0 server spells it (and a LeaseCursor must).
INSTANT = "%Y-%m-%dT%H:%M:%S.%fZ"
_TOKEN = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z")
_UUID = re.compile(r"[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}")


@dataclass
class QueueSpine(FakeSpine):
    """A FakeSpine that also leases the WHOLE queue (media.v1 0.12.0): by
    order, in a scope, after a cursor, with the four provenance fields, the
    order total ((first_seen_at, product_image_id)) and origin_url on a
    SHOWABLE lease only (A-3) -- or, ``old``, as a server before 0.12.0
    answers (no new fields; order, scope and cursor ignored), or,
    ``ignores_cursor``, as a 0.12.0 draft before the cursor did.

    HOLDS LAPSE on the server's own clock (``lapse``): a lease holds what it
    hands out for its lease_seconds (0: 300, at most 3600), and nothing else
    ends a hold but a report (there is no release RPC)."""

    old: bool = False
    ignores_cursor: bool = False
    clock: float = 0.0
    until: dict[str, float] = field(default_factory=dict)

    def lapse(self, seconds: float) -> None:
        """The server's clock moves on; every hold that has ended lapses."""
        self.clock += seconds
        for pid, end in list(self.until.items()):
            if end <= self.clock:
                self.held.discard(pid)
                del self.until[pid]

    def _hold(self, request: Any, images: Any) -> None:
        seconds = min(request.lease_seconds or 300, 3600)
        for image in images:
            self.until[image.product_image_id] = self.clock + seconds

    def depict_full(
        self,
        origin: str,
        *,
        seen: dt.datetime,
        showable: bool = True,
        url: str | None = "https://img.store.example/a.jpg",
        site: str = "amiami",
        product: str | None = None,
        codec: str = "identity",
        key: str | None = None,
        pid: str | None = None,
    ) -> str:
        pid = pid or self.depict(origin)
        self.depictions[pid] = self.depictions.get(pid) or {
            "origin": bytes.fromhex(origin),
            "state": "captured",
        }
        self.depictions[pid].update(
            seen=seen,
            showable=showable,
            url=url,
            site=site,
            product=product or str(uuid.uuid4()),
            codec=codec,
            key=key or _key(origin),
        )
        return pid

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        if request.origin_sha256:
            response = super().LeasePendingImages(request, context)
            self._hold(request, response.images)
            return response
        self._enter("LeasePendingImages", request, context)
        if request.order not in (0, 1, 2) or request.scope not in (0, 1, 2):
            context.abort(C.INVALID_ARGUMENT, "undeclared order or scope")
        after: tuple[str, str] | None = None
        if request.HasField("after") and not (self.old or self.ignores_cursor):
            cursor = request.after
            if not (
                _TOKEN.fullmatch(cursor.first_seen_at) and _UUID.fullmatch(cursor.product_image_id)
            ):
                context.abort(C.INVALID_ARGUMENT, "a LeaseCursor the contract refuses")
            after = (cursor.first_seen_at, cursor.product_image_id.lower())
        limit = min(request.max_items or 50, 200)
        rows = [
            (pid, d)
            for pid, d in self.depictions.items()
            if d["state"] == "captured"
            and pid not in self.held
            and (self.old or request.scope != 2 or (d.get("showable") and d.get("url")))
        ]
        newest = request.order == 2 and not self.old

        def place(row: tuple[str, dict[str, Any]]) -> tuple[str, str]:
            return (row[1]["seen"].strftime(INSTANT), row[0].lower())

        rows.sort(key=place, reverse=newest)
        if after is not None:
            rows = [r for r in rows if (place(r) < after if newest else place(r) > after)]
        out = []
        for pid, d in rows[:limit]:
            image = media_pb2.PendingImage(
                product_image_id=pid,
                origin_sha256=d["origin"],
                storage_key=d["key"],
                source_site=d["site"],
                role="gallery",
            )
            if not self.old:
                image.product_id = d["product"]
                image.codec = d["codec"]
                image.first_seen_at = d["seen"].strftime(INSTANT)
                if request.scope == 2 and d["url"] is not None:
                    image.origin_url = d["url"]
            out.append(image)
        self.held.update(i.product_image_id for i in out)
        self._hold(request, out)
        return media_pb2.LeasePendingImagesResponse(images=out, lease_expires_at="x")

    def ReportDerivative(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        if request.outcome == media_pb2.DERIVE_OUTCOME_FAILED:
            self._enter("ReportDerivative", request, context)
            d = self.depictions[request.product_image_id]
            d["state"] = "failed"
            self.held.discard(request.product_image_id)
            return media_pb2.DerivativeAck(stored=False, product_image_state="failed")
        return super().ReportDerivative(request, context)


@pytest.fixture()
def spine() -> Iterator[tuple[QueueSpine, str]]:
    fake = QueueSpine()
    with serve(fake) as url:
        yield fake, url


def _client(url: str, attempts: int = 2) -> SpineMediaClient:
    return SpineMediaClient(
        open_channel(url),
        policy=RetryPolicy(max_attempts=attempts, initial_backoff=0.01),
        sleep=lambda s: None,
    )


def _sync(
    db: Session, url: str, *, attempts: int = 2, **config: Any
) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    code = feed.run_sync(
        db,
        lambda: _client(url, attempts),
        feed.FeedConfig(**config),
        emit=lines.append,
        now=lambda: T0,
    )
    return code, lines


def _timed_sync(
    db: Session, url: str, monotonic: Callable[[], float], **config: Any
) -> tuple[int, list[dict[str, Any]]]:
    """A run on a fake ``monotonic`` clock."""
    lines: list[dict[str, Any]] = []
    code = feed.run_sync(
        db,
        lambda: _client(url),
        feed.FeedConfig(**config),
        emit=lines.append,
        now=lambda: T0,
        monotonic=monotonic,
    )
    return code, lines


def _event(lines: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(line for line in lines if line["event"] == name)


# --------------------------------------------------------------------------
# enqueue
# --------------------------------------------------------------------------


def test_the_lease_asks_for_newest_first_showable_and_queues_each_depiction(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    a, b = sha("a"), sha("b")
    old = fake.depict_full(a, seen=T0 - dt.timedelta(days=3))
    new = fake.depict_full(
        b, seen=T0, product=str(uuid.UUID(int=7)), site="mfc", url="https://x.example/b.jpg"
    )
    hidden = fake.depict_full(sha("c"), seen=T0, showable=False)

    code, lines = _sync(db_session, url)

    assert code == feed.EXIT_DONE
    first = fake.sent("LeasePendingImages")[0]
    assert (first.order, first.scope) == (
        media_pb2.LEASE_ORDER_NEWEST_FIRST,
        media_pb2.LEASE_SCOPE_SHOWABLE,
    )
    # a short hold: the feed reports none of these rows under it, and only a
    # lapse ends it (no release RPC), so another caller gets them back soon
    assert list(first.origin_sha256) == [] and first.lease_seconds == 120
    assert not first.HasField("after")  # a new walk starts at the newest end
    depictions = {
        d.product_image_id: d for d in db_session.execute(select(DisplayDepiction)).scalars()
    }
    assert set(depictions) == {old, new}  # never the hidden one: the spine's scope
    d = depictions[new]
    assert (d.product_id, d.origin_url, d.source_site, d.origin_sha256) == (
        str(uuid.UUID(int=7)),
        "https://x.example/b.jpg",
        "mfc",
        b,
    )
    original = db_session.get(DisplayOriginal, b)
    assert (original.state, original.storage_key, original.codec) == (
        "pending",
        _key(b),
        "identity",
    )
    assert _event(lines, "enqueued")["new_depictions"] == 2
    assert hidden not in depictions


def test_a_spine_before_0120_is_a_deployment_fault_that_stops_the_run(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    fake.old = True
    fake.depict_full(sha("a"), seen=T0)
    code, lines = _sync(db_session, url)
    assert code == feed.EXIT_STOPPED
    assert "0.12.0" in _event(lines, "stopped")["why"]
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 0


def test_an_answer_out_of_the_order_asked_stops_the_run(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    fake.depict_full(sha("a"), seen=T0 - dt.timedelta(days=1))
    fake.depict_full(sha("b"), seen=T0)
    code, lines = _sync(db_session, url, order="oldest")
    assert code == feed.EXIT_DONE  # asked oldest, answered oldest

    class OrderIgnoring(QueueSpine):
        """A server that ignores the order: oldest first, whatever was asked."""

        def LeasePendingImages(self, request: Any, context: Any) -> Any:  # noqa: N802
            request.order = media_pb2.LEASE_ORDER_OLDEST_FIRST
            return super().LeasePendingImages(request, context)

    fake2 = OrderIgnoring()
    with serve(fake2) as url2:
        fake2.depict_full(sha("c"), seen=T0 - dt.timedelta(days=1))
        fake2.depict_full(sha("d"), seen=T0)
        code, lines = _sync(db_session, url2)
    assert code == feed.EXIT_STOPPED
    assert "order" in _event(lines, "stopped")["why"]


def test_rows_the_lane_cannot_seed_are_skipped_and_counted(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    good = fake.depict_full(sha("a"), seen=T0)
    long_url = fake.depict_full(sha("c"), seen=T0, url="https://x.example/" + "a" * 2100)
    bad_key = fake.depict_full(sha("d"), seen=T0, key="elsewhere/" + sha("d"))
    wrong_key = fake.depict_full(sha("e"), seen=T0, key=_key(sha("f")))
    code, lines = _sync(db_session, url)
    assert code == feed.EXIT_DONE
    queued = set(db_session.execute(select(DisplayDepiction.product_image_id)).scalars())
    assert queued == {good}
    skipped = _event(lines, "enqueued")["skipped"]
    assert skipped == {"url_too_long": 1, "bad_storage_key": 2}
    assert {long_url, bad_key, wrong_key}.isdisjoint(queued)


def test_enqueue_stops_at_the_high_water_mark(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    for n in range(5):
        fake.depict_full(sha(f"n{n}"), seen=T0 - dt.timedelta(minutes=n))
    code, lines = _sync(db_session, url, high_water=2, lease_items=1)
    assert code == feed.EXIT_DONE
    assert db_session.scalar(select(func.count()).select_from(DisplayOriginal)) == 2
    # the newest two
    assert set(db_session.execute(select(DisplayOriginal.origin_sha256)).scalars()) == {
        sha("n0"),
        sha("n1"),
    }
    code, lines = _sync(db_session, url, high_water=2, lease_items=1)
    assert _event(lines, "enqueued")["why"] == "high_water"
    assert db_session.scalar(select(func.count()).select_from(DisplayOriginal)) == 2


def test_a_new_depiction_of_a_seeded_original_puts_it_back_to_pending(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin = sha("a")
    fake.depict_full(origin, seen=T0 - dt.timedelta(hours=1))
    _sync(db_session, url)
    row = db_session.get(DisplayOriginal, origin)
    row.state = "seeded"
    for d in db_session.execute(select(DisplayDepiction)).scalars():
        d.linked_at = T0  # as the seed left it: every depiction linked
    db_session.commit()
    fake.held.clear()
    for d in fake.depictions.values():
        d["state"] = "derived"
    fake.depict_full(origin, seen=T0)
    _code, lines = _sync(db_session, url, report_limit=0)
    assert _event(lines, "reported")["relink"] == 0  # the enqueue sent it back, not the report
    assert db_session.get(DisplayOriginal, origin).state == "pending"
    assert db_session.scalar(select(func.count()).select_from(DisplayDepiction)) == 2


def test_an_undeclared_order_is_refused_by_the_spine(spine: tuple[QueueSpine, str]) -> None:
    _fake, url = spine
    client = _client(url)
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        client._call(  # noqa: SLF001 -- the raw request, as a broken caller would send it
            "LeasePendingImages", media_pb2.LeasePendingImagesRequest(max_items=1, order=7)
        )


def test_the_client_refuses_an_order_or_scope_it_does_not_know() -> None:
    from app.spine import media_client as mc

    client = mc.SpineMediaClient(grpc.insecure_channel("127.0.0.1:1"))
    with pytest.raises(ValueError, match="order"):
        client.lease_queue(order="sideways", scope="all", max_items=1, lease_owner="x")
    with pytest.raises(ValueError, match="scope"):
        client.lease_queue(order="newest", scope="some", max_items=1, lease_owner="x")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _seeded(
    db: Session, fake: QueueSpine, tag: str, *, depictions: int = 1
) -> tuple[str, list[str], dict[str, Any]]:
    """A copy in the media DB (spine_fakes.seed_copy) and its original queued
    as seeded, with ``depictions`` the spine holds."""
    seeded = seed_copy(db, tag)
    origin = seeded["origin"]
    db.add(
        DisplayOriginal(
            origin_sha256=origin,
            storage_key=_key(origin),
            codec="identity",
            first_seen_at=T0,
            state="seeded",
            copy_sha256=seeded["clean"],
            copy_mime="image/jpeg",
            copy_width=100,
            copy_height=200,
            copy_generated_at=AT,
            thumbhash="AAECAwQFBgcICQ==",
            dominant_color="#112233",
        )
    )
    db.flush()  # the queue's foreign key: the original before its depictions
    pids = []
    for n in range(depictions):
        pid = fake.depict_full(origin, seen=T0)
        fake.depictions[pid]["state"] = "captured"
        fake.held.add(pid)  # leased by the enqueue that queued it
        db.add(
            DisplayDepiction(
                product_image_id=pid,
                origin_sha256=origin,
                product_id=str(uuid.UUID(int=n + 1)),
                origin_url="https://img.store.example/a.jpg",
                source_site="amiami",
                first_seen_at=T0,
                linked_at=T0,
            )
        )
        pids.append(pid)
    db.commit()
    return origin, pids, seeded


def test_each_seeded_depiction_is_reported_with_the_copy_and_its_appearance(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a", depictions=2)
    code, lines = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_DONE
    reports = fake.sent("ReportDerivative")
    assert sorted(r.product_image_id for r in reports) == sorted(pids)
    r = reports[0]
    assert r.derivative_sha256 == bytes.fromhex(seeded["clean"])
    assert (r.mime, r.width, r.height, r.pipeline_version) == (
        "image/jpeg",
        100,
        200,
        CLEAN_PIPELINE_VERSION,
    )
    assert r.generated_at == "2026-10-14T09:00:00.000000Z"
    assert r.dominant_color == "#112233" and r.thumbhash
    rows = db_session.execute(select(DisplayDepiction)).scalars().all()
    assert {(d.outcome, d.reported_at is not None) for d in rows} == {("derived", True)}
    assert _event(lines, "reported")["derived"] == 2
    # a rerun sends nothing
    _sync(db_session, url, max_new=0)
    assert len(fake.sent("ReportDerivative")) == 2


def test_a_copy_that_died_since_the_seed_is_not_reported(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a")
    from app.models import Image

    db_session.execute(select(Image).where(Image.sha256 == origin)).scalar_one().deleted_at = AT
    db_session.commit()
    code, lines = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_DONE
    assert fake.sent("ReportDerivative") == []
    assert db_session.get(DisplayDepiction, pids[0]).outcome == "no_copy"


def test_a_copy_other_than_the_recorded_one_sends_the_original_back_to_the_seeder(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a")
    row = db_session.get(DisplayOriginal, origin)
    row.copy_sha256 = "e" * 64
    db_session.commit()
    _sync(db_session, url, max_new=0)
    assert fake.sent("ReportDerivative") == []
    assert db_session.get(DisplayOriginal, origin).state == "pending"
    assert db_session.get(DisplayDepiction, pids[0]).reported_at is None


def test_failed_originals_are_reported_failed_with_their_reason(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin = sha("bad")
    db_session.add(
        DisplayOriginal(
            origin_sha256=origin,
            storage_key=_key(origin),
            codec="identity",
            first_seen_at=T0,
            state="failed",
            reason="raw_hash_mismatch",
            updated_at=T0,
        )
    )
    db_session.flush()
    pid = fake.depict_full(origin, seen=T0)
    db_session.add(
        DisplayDepiction(
            product_image_id=pid,
            origin_sha256=origin,
            product_id=str(uuid.UUID(int=1)),
            origin_url="https://x.example/a.jpg",
            source_site="amiami",
            first_seen_at=T0,
        )
    )
    db_session.commit()
    code, _ = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_DONE
    (report,) = fake.sent("ReportDerivative")
    assert (report.outcome, report.failure_reason) == (
        media_pb2.DERIVE_OUTCOME_FAILED,
        "raw_hash_mismatch",
    )
    assert db_session.get(DisplayDepiction, pid).outcome == "failed"


def test_failed_reports_past_the_cap_are_held_and_the_run_ends_1(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    for n in range(3):
        origin = sha(f"bad{n}")
        db_session.add(
            DisplayOriginal(
                origin_sha256=origin,
                storage_key=_key(origin),
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
                product_id=str(uuid.UUID(int=n + 1)),
                origin_url="https://x.example/a.jpg",
                source_site="amiami",
                first_seen_at=T0,
            )
        )
    db_session.commit()
    code, lines = _sync(db_session, url, max_new=0, max_failed_reports=2)
    assert code == feed.EXIT_REFUSED
    assert fake.sent("ReportDerivative") == []
    assert _event(lines, "failed_held")["originals"] == 3


def test_a_refused_report_is_kept_and_the_run_ends_1(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a")
    fake.fail("ReportDerivative", C.INVALID_ARGUMENT)
    code, lines = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_REFUSED
    d = db_session.get(DisplayDepiction, pids[0])
    assert d.outcome == "refused" and "INVALID_ARGUMENT" in (d.note or "")


def test_a_spine_that_stays_down_stops_the_run_and_reports_nothing_twice(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a")
    fake.fail("ReportDerivative", C.UNAVAILABLE, C.UNAVAILABLE)
    code, lines = _sync(db_session, url, max_new=0, attempts=2)
    assert code == feed.EXIT_STOPPED
    assert db_session.get(DisplayDepiction, pids[0]).reported_at is None
    code, lines = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_DONE
    assert db_session.get(DisplayDepiction, pids[0]).outcome == "derived"


def test_a_mesh_refusal_is_a_configuration_fault_exit_2(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    fake.depict_full(sha("a"), seen=T0)
    fake.fail("LeasePendingImages", C.PERMISSION_DENIED)
    code, lines = _sync(db_session, url)
    assert code == feed.EXIT_CONFIG


def test_reported_rows_are_purged_after_their_days(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, seeded = _seeded(db_session, fake, "a")
    _sync(db_session, url, max_new=0)
    d = db_session.get(DisplayDepiction, pids[0])
    d.reported_at = T0 - dt.timedelta(days=15)
    db_session.commit()
    _sync(db_session, url, max_new=0, purge_days=14)
    assert db_session.get(DisplayDepiction, pids[0]) is None
    assert db_session.get(DisplayOriginal, origin) is None


def test_the_feed_holds_no_bucket_client() -> None:
    from pathlib import Path

    source = Path(feed.__file__).read_text()
    for name in ("boto3", "s3_client", "ObjectStore", "raw_store"):
        assert name not in source, name


def test_lag_of_the_queue_is_in_the_done_line(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    fake.depict_full(sha("a"), seen=T0 - dt.timedelta(hours=2))
    _code, lines = _sync(db_session, url)
    done = _event(lines, "done")
    assert done["pending_originals"] == 1
    assert done["oldest_pending_seen"] == (T0 - dt.timedelta(hours=2)).isoformat()


def test_publicobject_rows_are_read_never_written(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    _seeded(db_session, fake, "a")
    before = db_session.scalar(select(func.count()).select_from(PublicObject))
    _sync(db_session, url, max_new=0)
    assert db_session.scalar(select(func.count()).select_from(PublicObject)) == before
    assert hashlib.sha256(b"x").hexdigest()  # keeps the import honest


#: The reported line of a run that reported every row it selected.
REPORTED_KEYS = ("derived", "failed", "refused", "no_copy", "reseed", "relink", "batch_copy")


def test_reports_stop_at_their_share_of_the_budget_and_the_run_still_enqueues_and_purges(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """The REPORT may spend half the run's budget_seconds: past it no new
    report starts, the line says how far it got and what is left, and ENQUEUE
    and PURGE still run (no Job deadline cuts them off). Each row is committed
    on its own, so those not reached wait, unreported, for the next run."""
    fake, url = spine
    pids = [_seeded(db_session, fake, tag)[1][0] for tag in ("a", "b", "c", "d")]
    _origin, (old,), _ = _seeded(db_session, fake, "old")
    d = db_session.get(DisplayDepiction, old)
    d.reported_at, d.outcome = T0 - dt.timedelta(days=15), "derived"
    db_session.commit()
    new = fake.depict_full(sha("new"), seen=T0)

    def clock() -> float:  # half the budget has passed once two reports are sent
        return 750.0 if len(fake.sent("ReportDerivative")) >= 2 else 0.0

    code, lines = _timed_sync(db_session, url, clock, budget_seconds=1500)

    assert code == feed.EXIT_DONE
    assert _event(lines, "reported") == {
        "event": "reported",
        **dict.fromkeys(REPORTED_KEYS, 0),
        "derived": 2,
        "partial": "budget",
        "remaining": 2,
    }
    sent = {r.product_image_id for r in fake.sent("ReportDerivative")}
    assert len(sent) == 2 and sent < set(pids)
    assert {db_session.get(DisplayDepiction, p).reported_at for p in set(pids) - sent} == {None}
    assert _event(lines, "enqueued")["new_depictions"] == 1  # the enqueue still ran
    assert db_session.get(DisplayDepiction, new) is not None
    assert _event(lines, "done")["purged"] == 1  # and the purge
    assert db_session.get(DisplayDepiction, old) is None
    # the next run reports the rows this one did not reach
    code, lines = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_DONE and _event(lines, "reported")["derived"] == 2
    assert {r.product_image_id for r in fake.sent("ReportDerivative")} == set(pids)


def test_a_run_within_its_budget_reports_every_row_as_before(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """Under the report's share of the budget nothing changes: every row is
    reported, the line carries no partial or remaining, and the run goes on
    to the enqueue and the purge."""
    fake, url = spine
    pids = [_seeded(db_session, fake, tag)[1][0] for tag in ("a", "b", "c")]
    fake.depict_full(sha("new"), seen=T0)

    def clock() -> float:  # 249 s a report: 747 s for the three, under the 750
        return 249.0 * len(fake.sent("ReportDerivative"))

    code, lines = _timed_sync(db_session, url, clock, budget_seconds=1500)

    assert code == feed.EXIT_DONE
    assert [line["event"] for line in lines] == ["reported", "enqueued", "done"]
    assert _event(lines, "reported") == {
        "event": "reported",
        **dict.fromkeys(REPORTED_KEYS, 0),
        "derived": 3,
    }
    assert {r.product_image_id for r in fake.sent("ReportDerivative")} == set(pids)
    assert _event(lines, "enqueued")["new_depictions"] == 1


def test_the_enqueue_gets_what_the_report_left_of_the_runs_budget(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    """budget_seconds is the whole run's: a report phase that took 700 s of
    1500 leaves the enqueue 800, so the run as a whole stays inside the Job's
    deadline (budget_seconds and its margin)."""
    fake, url = spine
    _seeded(db_session, fake, "a")
    for n in range(6):
        fake.depict_full(sha(f"r{n}"), seen=T0 - dt.timedelta(minutes=n))

    def clock() -> float:  # 700 s a report, 400 s a lease round
        return 700.0 * len(fake.sent("ReportDerivative")) + 400.0 * len(
            fake.sent("LeasePendingImages")
        )

    code, lines = _timed_sync(db_session, url, clock, budget_seconds=1500, lease_items=1)

    assert code == feed.EXIT_DONE
    assert "partial" not in _event(lines, "reported")
    enqueued = _event(lines, "enqueued")
    # rounds at 700 and 1100 s; at 1500 the run's budget is spent (on its
    # own 1500 s from 700 the enqueue would have leased until 2300)
    assert (enqueued["why"], enqueued["rounds"], enqueued["new_depictions"]) == ("budget", 2, 2)


# --------------------------------------------------------------------------
# the remaining edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"scope": "some"}, "scope"),
        # only a SHOWABLE lease carries origin_url (media.v1 0.12.0): under ALL
        # the feed would hold every row and seed none
        ({"scope": "all"}, "SHOWABLE"),
        ({"lease_seconds": 0}, "lease_seconds"),
        ({"lease_seconds": 121}, "lease_seconds"),
        ({"max_new": -1}, "max_new"),
        # one round goes to the newest end: a walk under way never moves
        ({"max_rounds": 1}, "max_rounds"),
        ({"max_rounds": 0}, "max_rounds"),
        ({"purge_days": 0}, "purge_days"),
    ],
)
def test_a_bad_feed_config_is_refused(kwargs: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        feed.FeedConfig(**kwargs)


def test_a_lease_the_spine_refuses_is_a_configuration_fault(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    fake.fail("LeasePendingImages", C.INVALID_ARGUMENT)
    code, lines = _sync(db_session, url)
    assert code == feed.EXIT_CONFIG
    assert "refused: INVALID_ARGUMENT" in _event(lines, "stopped")["why"]


def test_a_value_the_client_will_not_send_is_kept_as_refused(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, _ = _seeded(db_session, fake, "a")
    db_session.get(DisplayOriginal, origin).copy_mime = "image/x-unknown"
    db_session.commit()
    code, _ = _sync(db_session, url, max_new=0)
    assert code == feed.EXIT_REFUSED
    assert fake.sent("ReportDerivative") == []
    d = db_session.get(DisplayDepiction, pids[0])
    assert d.outcome == "refused" and (d.note or "").startswith("not sent:")


def test_the_second_depiction_of_an_original_sent_back_is_left_for_the_seeder(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    origin, pids, _ = _seeded(db_session, fake, "a", depictions=2)
    db_session.get(DisplayOriginal, origin).copy_sha256 = "e" * 64
    db_session.commit()
    _code, lines = _sync(db_session, url, max_new=0)
    assert _event(lines, "reported")["reseed"] == 1  # once, not once per depiction
    assert {db_session.get(DisplayDepiction, p).reported_at for p in pids} == {None}


class _Rows(QueueSpine):
    """A spine whose lease answers exactly ``rows`` (then nothing), in the
    total order asked (first_seen_at, then product_image_id)."""

    rows: list[media_pb2.PendingImage] = []

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("LeasePendingImages", request, context)
        out, self.rows = self.rows, []
        out.sort(
            key=lambda i: (i.first_seen_at, i.product_image_id.lower()),
            reverse=request.order == media_pb2.LEASE_ORDER_NEWEST_FIRST,
        )
        return media_pb2.LeasePendingImagesResponse(images=out, lease_expires_at="x")


def _row(origin: str, **fields: Any) -> media_pb2.PendingImage:
    base: dict[str, Any] = {
        "product_image_id": str(uuid.uuid4()),
        "origin_sha256": bytes.fromhex(origin),
        "storage_key": _key(origin),
        "source_site": "amiami",
        "role": "gallery",
        "product_id": str(uuid.uuid4()),
        "codec": "identity",
        "first_seen_at": "2026-10-31T12:00:00.000000Z",
        "origin_url": "https://img.amiami.example/a.jpg",
    }
    base.update(fields)
    return media_pb2.PendingImage(**{k: v for k, v in base.items() if v is not None})


def test_every_unsafe_row_shape_is_skipped(db_session: Session) -> None:
    fake = _Rows()
    fake.rows = [
        _row(sha("a"), product_id="not-a-uuid"),
        _row(sha("b"), codec=""),
        _row(sha("c"), source_site="s" * 51),
        _row(sha("d"), source_class="c" * 33),
        _row(sha("e"), source_class="manufacturer_press"),
        # a SHOWABLE row carries origin_url (A-3); one without it is never
        # seeded into an originless image
        _row(sha("f"), origin_url=None),
    ]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_DONE
    assert _event(lines, "enqueued")["skipped"] == {
        "no_origin_url": 1,
        "bad_identity": 1,
        "no_codec": 1,
        "bad_site": 1,
        "bad_class": 1,
    }
    (queued,) = db_session.execute(select(DisplayDepiction)).scalars()
    assert queued.source_class == "manufacturer_press"


def test_an_unreadable_first_seen_is_a_deployment_fault(db_session: Session) -> None:
    fake = _Rows()
    fake.rows = [_row(sha("a"), first_seen_at="yesterday")]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_STOPPED
    assert "0.12.0" in _event(lines, "stopped")["why"]


def test_an_older_depiction_keeps_the_originals_newest_instant_and_a_repeat_is_counted(
    db_session: Session,
) -> None:
    fake = _Rows()
    first = _row(sha("a"), first_seen_at="2026-10-31T12:00:00.000000Z")
    fake.rows = [first, _row(sha("a"), first_seen_at="2026-10-30T12:00:00.000000Z")]
    with serve(fake) as url:
        _sync(db_session, url)
        fake.rows = [first]
        _code, lines = _sync(db_session, url)
    assert db_session.get(DisplayOriginal, sha("a")).first_seen_at.replace(tzinfo=dt.UTC) == T0
    assert _event(lines, "enqueued")["already"] == 1


def test_enqueue_stops_at_its_rounds_and_its_budget(
    db_session: Session, spine: tuple[QueueSpine, str]
) -> None:
    fake, url = spine
    for n in range(3):
        fake.depict_full(sha(f"r{n}"), seen=T0 - dt.timedelta(minutes=n))
    code, lines = _sync(db_session, url, lease_items=1, max_rounds=2)
    assert _event(lines, "enqueued")["why"] == "rounds"
    out: list[dict[str, Any]] = []
    ticks = iter([0.0, 0.0, 10_000.0])  # the run, its enqueue, its spent budget
    feed.run_sync(
        db_session,
        lambda: _client(url),
        feed.FeedConfig(lease_items=1),
        emit=out.append,
        now=lambda: T0,
        monotonic=lambda: next(ticks),
    )
    assert _event(out, "enqueued")["why"] == "budget"
    code, lines = _sync(db_session, url, lease_items=1, max_new=1)
    assert _event(lines, "enqueued")["why"] == "max_new"


@pytest.mark.parametrize(
    ("error", "code", "why"),
    [
        ("operational", feed.EXIT_STOPPED, "database: OperationalError"),
        ("programming", feed.EXIT_CONFIG, "database: ProgrammingError"),
        # a row another writer committed first: two syncs at once
        ("integrity", feed.EXIT_STOPPED, "database: IntegrityError: another sync"),
    ],
)
def test_a_database_fault_stops_the_sync_with_its_class(
    db_session: Session,
    spine: tuple[QueueSpine, str],
    monkeypatch: pytest.MonkeyPatch,
    error: str,
    code: int,
    why: str,
) -> None:
    from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

    fake, url = spine
    fake.depict_full(sha("a"), seen=T0)
    cls = {
        "operational": OperationalError,
        "programming": ProgrammingError,
        "integrity": IntegrityError,
    }[error]

    def broken(*a: Any, **k: Any) -> Any:
        raise cls("SELECT", {}, Exception("the database said no"))

    monkeypatch.setattr(feed, "report", broken)
    got, lines = _sync(db_session, url)
    assert got == code
    assert _event(lines, "stopped")["why"].startswith(why)
