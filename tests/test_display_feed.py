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
import uuid
from collections.abc import Iterator
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


class QueueSpine(FakeSpine):
    """A FakeSpine that also leases the WHOLE queue (0.12.0): by order, in a
    scope, with the four provenance fields -- or, ``old``, as a server before
    0.12.0 answers (no new fields, the filter-less order ignored)."""

    old: bool = False

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
    ) -> str:
        pid = self.depict(origin)
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
            return super().LeasePendingImages(request, context)
        self._enter("LeasePendingImages", request, context)
        if request.order not in (0, 1, 2) or request.scope not in (0, 1, 2):
            context.abort(C.INVALID_ARGUMENT, "undeclared order or scope")
        limit = min(request.max_items or 50, 200)
        rows = [
            (pid, d)
            for pid, d in self.depictions.items()
            if d["state"] == "captured"
            and pid not in self.held
            and (self.old or request.scope != 2 or (d.get("showable") and d.get("url")))
        ]
        newest = request.order == 2 and not self.old
        rows.sort(key=lambda r: (r[1]["seen"], r[0]), reverse=newest)
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
                image.first_seen_at = d["seen"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                if d["url"] is not None:
                    image.origin_url = d["url"]
            out.append(image)
        self.held.update(i.product_image_id for i in out)
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
    assert list(first.origin_sha256) == [] and first.lease_seconds == 3600
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
    no_url = fake.depict_full(sha("b"), seen=T0, url=None)
    long_url = fake.depict_full(sha("c"), seen=T0, url="https://x.example/" + "a" * 2100)
    bad_key = fake.depict_full(sha("d"), seen=T0, key="elsewhere/" + sha("d"))
    wrong_key = fake.depict_full(sha("e"), seen=T0, key=_key(sha("f")))
    code, lines = _sync(db_session, url, scope="all")
    assert code == feed.EXIT_DONE
    queued = set(db_session.execute(select(DisplayDepiction.product_image_id)).scalars())
    assert queued == {good}
    skipped = _event(lines, "enqueued")["skipped"]
    assert skipped == {"no_origin_url": 1, "url_too_long": 1, "bad_storage_key": 2}
    assert {no_url, long_url, bad_key, wrong_key}.isdisjoint(queued)


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
    db_session.commit()
    fake.held.clear()
    for d in fake.depictions.values():
        d["state"] = "derived"
    fake.depict_full(origin, seen=T0)
    _sync(db_session, url, report_limit=0)
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


# --------------------------------------------------------------------------
# the remaining edges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"scope": "some"}, "scope"),
        ({"lease_seconds": 0}, "lease_seconds"),
        ({"max_new": -1}, "max_new"),
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
    """A spine whose lease answers exactly ``rows`` (then nothing)."""

    rows: list[media_pb2.PendingImage] = []

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("LeasePendingImages", request, context)
        out, self.rows = self.rows, []
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
    ]
    with serve(fake) as url:
        code, lines = _sync(db_session, url)
    assert code == feed.EXIT_DONE
    assert _event(lines, "enqueued")["skipped"] == {
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
    code, lines = _sync(db_session, url, lease_items=1, max_rounds=1)
    assert _event(lines, "enqueued")["why"] == "rounds"
    out: list[dict[str, Any]] = []
    ticks = iter([0.0, 10_000.0])
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
    ("error", "code"), [("operational", feed.EXIT_STOPPED), ("programming", feed.EXIT_CONFIG)]
)
def test_a_database_fault_stops_the_sync_with_its_class(
    db_session: Session,
    spine: tuple[QueueSpine, str],
    monkeypatch: pytest.MonkeyPatch,
    error: str,
    code: int,
) -> None:
    from sqlalchemy.exc import OperationalError, ProgrammingError

    fake, url = spine
    fake.depict_full(sha("a"), seen=T0)

    def broken(*a: Any, **k: Any) -> Any:
        cls = OperationalError if error == "operational" else ProgrammingError
        raise cls("SELECT", {}, Exception("the database said no"))

    monkeypatch.setattr(feed, "report", broken)
    got, lines = _sync(db_session, url)
    assert got == code
    assert _event(lines, "stopped")["why"].startswith("database:")
