"""SHOW1: the display-copy lane's FEED -- the half that holds the spine
identity and no bucket key (``python -m app.cli.display_copy sync``, the
mm-display-sync CronJob, service account media-report: the one identity the
spine's media listener admits).

Q1 (b), the split. The pod that decodes untrusted bytes (the seeder,
app.workers.display_copy) holds the bucket keys and NO spine identity; this
pod holds the spine identity and NO bucket key, and talks to the seeder only
through the media DB queue (``display_originals``, ``display_depictions``)
-- the same trust shape as the WSL mask batch and the mm-report replay. A
run is REPORT, then ENQUEUE, then PURGE:

REPORT. Every unreported depiction of a ``seeded`` original whose gallery
link the seed wrote: ReportDerivative (DERIVED) with the copy the replay
would report, re-read first (app.spine.copies.reported_copy): no live copy
now (a takedown since the seed) -> nothing sent, ``no_copy``; another copy
-> the original goes back to the seeder, nothing sent. Every unreported
depiction of a ``failed`` original (the image's own fault, a closed list):
a FAILED report with its reason -- unless more than ``max_failed_reports``
originals wait for one, when none is sent and the run ends 1: a FAILED
report is permanent on the spine in practice, so a flood of them is looked
at by a person first. ``held`` and ``stuck`` originals are never reported.
A report needs no live lease (fc-aggregation derivative-writer.ts: "a
report is never refused for a stale lease") and is idempotent.

ENQUEUE. While fewer than ``high_water`` originals are pending:
LeasePendingImages{order, scope} (media.v1 0.12.0; newest first and
SHOWABLE by default, so new captures land within the hour whatever the
backlog), each row queued with its product, origin URL, site, class, codec
and first-seen instant. A row the seeder could not seed safely is skipped
and counted, never queued: no origin URL (the seed would write an
originless image, which takes a published mask of its picture offline, F3),
a URL past the gallery link's 2048 characters, a storage key outside the
``sha256-v1`` scheme or naming other bytes, a site past 50 characters. Its
hold lapses; the lag check sees it. AN ANSWER FROM A SERVER BEFORE 0.12.0
(no product_id or first_seen_at) or out of the order asked is a DEPLOYMENT
FAULT: nothing is queued, the run stops (exit 3), the holds lapse.

PURGE. Depictions reported more than ``purge_days`` ago, and the seeded or
failed originals left with none.

Exit status: 0 done; 1 done, but a report was refused or FAILED reports were
held; 2 a configuration fault (the mesh refused the identity, the spine
refused the lease, the database refused the role); 3 stopped, resumable
(the spine or the database stayed down, or a deployment fault).
"""

from __future__ import annotations

import datetime as dt
import os
import re
import socket
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import CursorResult, and_, delete, exists, func, or_, select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from ..models import DisplayDepiction, DisplayOriginal, PublicObject
from ..models import Image as ImageModel
from ..workers.mask_seed import CLEAN_PIPELINE_VERSION
from . import messages as msg
from .copies import measured_appearance, reported_copy
from .field_rules import FieldRefused
from .media_client import LEASE_ORDERS, LEASE_SCOPES, SpineCallError, SpineMediaClient, Verdict
from .media_v1 import media_pb2

EXIT_DONE = 0
EXIT_REFUSED = 1
EXIT_CONFIG = 2
EXIT_STOPPED = 3

#: The ``sha256-v1`` raw-store key scheme (fc-aggregation backfill-captures.ts
#: KEY_RE), any prefix of path segments allowed before it.
KEY_RE = re.compile(r"^(?:[A-Za-z0-9._-]+/)*sha256/([0-9a-f]{2})/([0-9a-f]{64})(\.[A-Za-z0-9.]+)?$")
#: figure_galleries.source_url and .source.
URL_MAX = 2048
SITE_MAX = 50
CLASS_MAX = 32


@dataclass(frozen=True)
class FeedConfig:
    order: str = "newest"
    scope: str = "showable"
    lease_items: int = 200
    lease_seconds: int = 3600
    max_rounds: int = 200
    max_new: int = 20_000
    high_water: int = 20_000
    report_limit: int = 20_000
    max_failed_reports: int = 50
    purge_days: int = 14
    budget_seconds: float = 1500.0
    owner: str = ""

    def __post_init__(self) -> None:
        if self.order not in LEASE_ORDERS:
            raise ValueError(f"order must be one of {', '.join(LEASE_ORDERS)}")
        if self.scope not in LEASE_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(LEASE_SCOPES)}")
        if not 1 <= self.lease_items <= 200:
            raise ValueError("lease_items must be 1-200")
        if not 1 <= self.lease_seconds <= 3600:
            raise ValueError("lease_seconds must be 1-3600")
        for name in ("max_rounds", "max_new", "high_water", "report_limit", "max_failed_reports"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.purge_days < 1 or not self.budget_seconds > 0:
            raise ValueError("purge_days >= 1 and budget_seconds > 0")


class _Stop(Exception):
    def __init__(self, code: int, why: str) -> None:
        super().__init__(why)
        self.code = code
        self.why = why


def _utc(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def _stop_for(exc: SpineCallError) -> _Stop:
    if exc.verdict is Verdict.FATAL:
        return _Stop(EXIT_CONFIG, f"{exc.rpc} {exc.code.name}: the mesh or the deployment refused")
    if exc.verdict is Verdict.RETRY:
        return _Stop(EXIT_STOPPED, f"{exc.rpc} {exc.code.name} after {exc.attempts} attempts")
    return _Stop(EXIT_CONFIG, f"{exc.rpc} refused: {exc.code.name}: {exc.details}"[:256])


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _copy_record(o: DisplayOriginal) -> msg.CopyRecord:
    assert o.copy_sha256 and o.copy_mime and o.copy_generated_at  # a seeded original's
    return msg.CopyRecord(
        sha256=o.copy_sha256,
        mime=o.copy_mime,
        width=o.copy_width or 0,
        height=o.copy_height or 0,
        generated_at=_utc(o.copy_generated_at),
        thumbhash=o.thumbhash,
        dominant_color=o.dominant_color,
        own_grounding=o.own_grounding,
    )


def report(
    db: Session,
    client: SpineMediaClient,
    config: FeedConfig,
    emit: Callable[[dict[str, Any]], None],
    now: dt.datetime,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    waiting_failed = (
        db.scalar(
            select(func.count(func.distinct(DisplayOriginal.origin_sha256)))
            .join(DisplayDepiction, DisplayDepiction.origin_sha256 == DisplayOriginal.origin_sha256)
            .where(DisplayOriginal.state == "failed", DisplayDepiction.reported_at.is_(None))
        )
        or 0
    )
    # The cap counts what this feed made permanent in the last day too: the
    # seeder's breaker stops each RUN at max_failed own faults, so a fault that
    # only looks like the images' would otherwise pass ~max_failed FAILED
    # reports to the spine every run, the cap never reached.
    sent_today = (
        db.scalar(
            select(func.count(func.distinct(DisplayDepiction.origin_sha256))).where(
                DisplayDepiction.outcome == "failed",
                DisplayDepiction.reported_at >= now - dt.timedelta(days=1),
            )
        )
        or 0
    )
    send_failed = not waiting_failed or waiting_failed + sent_today <= config.max_failed_reports
    if not send_failed:
        counts["failed_held"] = waiting_failed
        emit(
            {
                "event": "failed_held",
                "originals": waiting_failed,
                "sent_last_day": sent_today,
                "cap": config.max_failed_reports,
            }
        )
    ready = and_(DisplayOriginal.state == "seeded", DisplayDepiction.linked_at.is_not(None))
    rows = db.execute(
        select(DisplayDepiction, DisplayOriginal)
        .join(DisplayOriginal, DisplayOriginal.origin_sha256 == DisplayDepiction.origin_sha256)
        .where(
            DisplayDepiction.reported_at.is_(None),
            or_(ready, DisplayOriginal.state == "failed") if send_failed else ready,
        )
        .order_by(DisplayDepiction.first_seen_at.desc(), DisplayDepiction.product_image_id)
        .limit(config.report_limit)
    ).all()
    current: dict[str, str | None] = {}
    for depiction, original in rows:
        if original.state == "pending":  # sent back to the seeder earlier in this run
            continue
        pid = depiction.product_image_id
        try:
            if original.state == "seeded":
                origin = original.origin_sha256
                if origin not in current:
                    current[origin] = reported_copy(db, origin)
                copy = current[origin]
                if copy is None:
                    depiction.reported_at, depiction.outcome = now, "no_copy"
                    counts["no_copy"] += 1
                    db.commit()
                    continue
                if copy != original.copy_sha256:
                    original.state, original.updated_at = "pending", now
                    counts["reseed"] += 1
                    db.commit()
                    continue
                built = msg.derivative_report(
                    pid, _copy_record(original), pipeline_version=CLEAN_PIPELINE_VERSION
                )
                outcome = "derived"
            else:
                built = msg.failed_report(
                    pid,
                    pipeline_version=CLEAN_PIPELINE_VERSION,
                    generated_at=_utc(original.updated_at),
                    failure_reason=original.reason or "failed",
                )
                outcome = "failed"
            ack = client.report_derivative(built.message)
        except FieldRefused as exc:
            depiction.reported_at, depiction.outcome = now, "refused"
            depiction.note = f"not sent: {exc}"[:256]
            counts["refused"] += 1
            db.commit()
            continue
        except SpineCallError as exc:
            if exc.verdict is not Verdict.REFUSED:
                db.rollback()
                raise _stop_for(exc) from exc
            depiction.reported_at, depiction.outcome = now, "refused"
            depiction.note = f"{exc.code.name}: {exc.details}"[:256]
            counts["refused"] += 1
            db.commit()
            continue
        depiction.reported_at, depiction.outcome = now, outcome
        warnings = [*built.warnings, *ack.warnings]
        depiction.note = "; ".join(warnings)[:256] or None
        counts[outcome] += 1
        counts["warnings"] += len(warnings)
        db.commit()
    _report_left_to_the_batch(db, client, config, counts, now)
    emit(
        {
            "event": "reported",
            **{
                k: counts[k]
                for k in ("derived", "failed", "refused", "no_copy", "reseed", "batch_copy")
            },
        }
    )
    return counts


def _report_left_to_the_batch(
    db: Session,
    client: SpineMediaClient,
    config: FeedConfig,
    counts: Counter[str],
    now: dt.datetime,
) -> None:
    """Depictions of ``held`` or ``stuck`` originals that another seeder (the
    GPU batch, an operator's force) has since put in the media DB: DERIVED
    with the copy the replay would report (reported_copy, its record as
    replay._load_chunk builds it). The replay reports them too, once the
    feed's short hold on them lapses; both reports are idempotent, and this
    one also settles the lane's own row (purged after ``purge_days``), so a
    row the lane leaves to the batch never waits on the replay's schedule.
    Only originals the media DB holds an image of are looked at."""
    rows = db.execute(
        select(DisplayDepiction, DisplayOriginal)
        .join(DisplayOriginal, DisplayOriginal.origin_sha256 == DisplayDepiction.origin_sha256)
        .where(
            DisplayDepiction.reported_at.is_(None),
            DisplayOriginal.state.in_(("held", "stuck")),
            # a column the role may read: EXISTS (SELECT * ...) needs them all
            select(ImageModel.id)
            .where(ImageModel.sha256 == DisplayOriginal.origin_sha256)
            .exists(),
        )
        .order_by(DisplayDepiction.first_seen_at.desc(), DisplayDepiction.product_image_id)
        .limit(config.report_limit)
    ).all()
    records: dict[str, msg.CopyRecord | None] = {}
    for depiction, original in rows:
        origin = original.origin_sha256
        if origin not in records:
            copy = reported_copy(db, origin)
            obj = db.get(PublicObject, copy) if copy is not None else None
            look = measured_appearance(db, copy) if copy is not None else None
            records[origin] = (
                msg.CopyRecord(
                    sha256=obj.sha256,
                    mime=obj.mime,
                    width=obj.width,
                    height=obj.height,
                    generated_at=_utc(obj.created_at),
                    thumbhash=look.plain_thumbhash if look else None,
                    dominant_color=look.plain_dominant_color if look else None,
                    own_grounding=look.own_alpha_grounding if look else None,
                )
                if obj is not None
                else None
            )
        record = records[origin]
        if record is None:
            continue  # nothing in media yet: the row stays the lane's, unreported
        try:
            built = msg.derivative_report(
                depiction.product_image_id, record, pipeline_version=CLEAN_PIPELINE_VERSION
            )
            ack = client.report_derivative(built.message)
        except FieldRefused as exc:
            depiction.reported_at, depiction.outcome = now, "refused"
            depiction.note = f"not sent: {exc}"[:256]
            counts["refused"] += 1
            db.commit()
            continue
        except SpineCallError as exc:
            if exc.verdict is not Verdict.REFUSED:
                db.rollback()
                raise _stop_for(exc) from exc
            depiction.reported_at, depiction.outcome = now, "refused"
            depiction.note = f"{exc.code.name}: {exc.details}"[:256]
            counts["refused"] += 1
            db.commit()
            continue
        depiction.reported_at, depiction.outcome = now, "derived"
        depiction.note = "; ".join([*built.warnings, *ack.warnings])[:256] or None
        counts["batch_copy"] += 1
        db.commit()


# --------------------------------------------------------------------------
# enqueue
# --------------------------------------------------------------------------


def _uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


def _instant(token: str) -> dt.datetime | None:
    try:
        return _utc(dt.datetime.fromisoformat(token.replace("Z", "+00:00")))
    except ValueError:
        return None


def _check_answer(images: list[media_pb2.PendingImage], order: str) -> list[dt.datetime]:
    """Every row's first_seen_at, once the answer is a 0.12.0 server's in the
    order asked; a _Stop otherwise."""
    seen: list[dt.datetime] = []
    for image in images:
        instant = _instant(image.first_seen_at) if image.first_seen_at else None
        if not image.product_id or instant is None:
            raise _Stop(
                EXIT_STOPPED,
                "the lease answered without product_id or first_seen_at: a spine before media.v1 "
                "0.12.0, which ignores order and scope; nothing queued",
            )
        seen.append(instant)
    ordered = sorted(seen, reverse=order == "newest")
    if seen != ordered:
        raise _Stop(
            EXIT_STOPPED, f"the lease answered out of the {order}-first order asked; nothing queued"
        )
    return seen


def _skip_reason(image: media_pb2.PendingImage) -> str | None:
    origin = image.origin_sha256.hex()
    if (
        len(image.origin_sha256) != 32
        or not _uuid(image.product_image_id)
        or not _uuid(image.product_id)
    ):
        return "bad_identity"
    if not image.HasField("origin_url") or not image.origin_url:
        return "no_origin_url"
    if len(image.origin_url) > URL_MAX:
        return "url_too_long"
    m = KEY_RE.match(image.storage_key)
    if m is None or m.group(2) != origin or m.group(1) != origin[:2]:
        return "bad_storage_key"
    if not image.codec:
        return "no_codec"
    if not image.source_site or len(image.source_site) > SITE_MAX:
        return "bad_site"
    if image.HasField("source_class") and len(image.source_class) > CLASS_MAX:
        return "bad_class"
    return None


def _queue_one(db: Session, image: media_pb2.PendingImage, seen: dt.datetime) -> bool:
    """Queue one leased row; False when it was queued already."""
    if db.get(DisplayDepiction, image.product_image_id) is not None:
        return False
    origin = image.origin_sha256.hex()
    original = db.get(DisplayOriginal, origin)
    if original is None:
        db.add(
            DisplayOriginal(
                origin_sha256=origin,
                storage_key=image.storage_key,
                codec=image.codec,
                first_seen_at=seen,
                state="pending",
                attempts=0,
            )
        )
        db.flush()
    else:
        if original.state == "seeded":  # its new depiction's gallery link is still to write
            original.state = "pending"
        if seen > _utc(original.first_seen_at):
            original.first_seen_at = seen
    db.add(
        DisplayDepiction(
            product_image_id=image.product_image_id,
            origin_sha256=origin,
            product_id=image.product_id,
            origin_url=image.origin_url,
            source_site=image.source_site,
            source_class=image.source_class if image.HasField("source_class") else None,
            first_seen_at=seen,
        )
    )
    return True


def _pending(db: Session) -> int:
    return db.scalar(select(func.count()).where(DisplayOriginal.state == "pending")) or 0


def enqueue(
    db: Session,
    client: SpineMediaClient,
    config: FeedConfig,
    emit: Callable[[dict[str, Any]], None],
    monotonic: Callable[[], float],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    owner = config.owner or f"mm-display-sync/{os.environ.get('JOB_NAME') or socket.gethostname()}"
    start = monotonic()
    pending = _pending(db)
    why = "empty"
    while True:
        if pending >= config.high_water:
            why = "high_water"
            break
        if counts["new_depictions"] >= config.max_new:
            why = "max_new"
            break
        if counts["rounds"] >= config.max_rounds:
            why = "rounds"
            break
        if monotonic() - start >= config.budget_seconds:
            why = "budget"
            break
        try:
            outcome = client.lease_queue(
                order=config.order,
                scope=config.scope,
                max_items=config.lease_items,
                lease_owner=owner,
                lease_seconds=config.lease_seconds,
            )
        except SpineCallError as exc:
            raise _stop_for(exc) from exc
        counts["rounds"] += 1
        images = list(outcome.response.images)
        if not images:
            break
        seen = _check_answer(images, config.order)
        for image, instant in zip(images, seen, strict=True):
            reason = _skip_reason(image)
            if reason is not None:
                skipped[reason] += 1
            elif _queue_one(db, image, instant):
                counts["new_depictions"] += 1
            else:
                counts["already"] += 1
        db.commit()
        pending = _pending(db)
    if why in ("rounds", "budget") and counts["new_depictions"] == 0 and counts["already"]:
        # every leased row was one the lane holds already (held, stuck, still
        # pending or unreportable): the newest-first window is full of them and
        # nothing older is ever reached. A person must look (exit 1).
        counts["stalled"] = 1
        emit({"event": "stalled", "already": counts["already"], "rounds": counts["rounds"]})
    emit(
        {
            "event": "enqueued",
            "why": why,
            "rounds": counts["rounds"],
            "new_depictions": counts["new_depictions"],
            "already": counts["already"],
            "skipped": dict(skipped),
            "pending_originals": pending,
        }
    )
    counts.update({f"skipped_{k}": v for k, v in skipped.items()})
    return counts


# --------------------------------------------------------------------------
# purge and the run
# --------------------------------------------------------------------------


def purge(db: Session, config: FeedConfig, now: dt.datetime) -> int:
    cutoff = now - dt.timedelta(days=config.purge_days)
    # "fetch": the database decides which rows go (never a Python comparison
    # of stored instants), and the session forgets them
    fetch = {"synchronize_session": "fetch"}
    gone: CursorResult[Any] = db.execute(  # type: ignore[assignment] # a DELETE's result
        delete(DisplayDepiction).where(DisplayDepiction.reported_at < cutoff),
        execution_options=fetch,
    )
    db.execute(
        delete(DisplayOriginal).where(
            DisplayOriginal.state.in_(("seeded", "failed")),
            ~exists().where(DisplayDepiction.origin_sha256 == DisplayOriginal.origin_sha256),
        ),
        execution_options=fetch,
    )
    db.commit()
    return int(gone.rowcount or 0)


def _lag(db: Session) -> dict[str, Any]:
    oldest = db.scalar(
        select(func.min(DisplayOriginal.first_seen_at)).where(DisplayOriginal.state == "pending")
    )
    return {
        "pending_originals": _pending(db),
        "oldest_pending_seen": _utc(oldest).isoformat() if oldest is not None else None,
        "unreported_depictions": db.scalar(
            select(func.count()).where(DisplayDepiction.reported_at.is_(None))
        ),
    }


def run_sync(
    db: Session,
    connect: Callable[[], SpineMediaClient],
    config: FeedConfig,
    *,
    emit: Callable[[dict[str, Any]], None],
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """REPORT, then ENQUEUE, then PURGE; the exit status."""
    client = connect()
    try:
        try:
            reported = report(db, client, config, emit, now())
            queued = enqueue(db, client, config, emit, monotonic)
        except (OperationalError, ProgrammingError) as exc:
            # the database down (resumable) or refusing the role (a grant, the schema): never a traceback
            db.rollback()
            raise _Stop(
                EXIT_STOPPED if isinstance(exc, OperationalError) else EXIT_CONFIG,
                f"database: {type(exc).__name__}",
            ) from exc
    except _Stop as stop:
        emit({"event": "stopped", "why": stop.why})
        return stop.code
    purged = purge(db, config, now())
    emit({"event": "done", "purged": purged, **_lag(db)})
    if reported["refused"] or reported["failed_held"] or queued["stalled"]:
        return EXIT_REFUSED
    return EXIT_DONE
