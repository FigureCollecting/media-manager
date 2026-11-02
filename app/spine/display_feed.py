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
a FAILED report with its reason -- unless the originals waiting for one and
those it sent one for in the last day are more than ``max_failed_reports``,
when none is sent and the run ends 1: a FAILED report is permanent on the
spine in practice, so a flood of them is looked at by a person first, across
seeder runs too. A seeded original with a depiction no seed linked (the feed
queued it while the seed settled) goes back to the seeder, ``relink``. A
``stuck`` original (left to the GPU batch) is reported once the media DB
holds an image of its bytes -- the batch's seed of it, or any earlier seed
of the same bytes for another product: DERIVED with the copy the replay
would report, as the replay itself does. A ``held`` one is never reported
here: the media DB may hold its bytes from the start, so a copy there is no
operator's decision; it waits for ``display_copy requeue --state held
--force`` (the mm-report replay is not the lane and does not wait:
app.workers.display_copy, HELD).
A report needs no live lease (fc-aggregation derivative-writer.ts: "a
report is never refused for a stale lease") and is idempotent.

ENQUEUE. While fewer than ``high_water`` originals are pending:
LeasePendingImages{order, scope, after} (media.v1 0.12.0; newest first and
SHOWABLE, the only scope whose rows carry origin_url), each row queued with
its product, origin URL, site, class, codec and first-seen instant. A row the
seeder could not seed safely is skipped and counted, never queued: no origin
URL (the seed would write an originless image, which takes a published mask
of its picture offline, F3), a URL past the gallery link's 2048 characters, a
storage key outside the ``sha256-v1`` scheme or naming other bytes, a site
past 50 characters. Its hold lapses; the lag check sees it.

THE WALK. Rows the lane will never report (held, stuck, skipped) stay pending
on the spine, so a lease that always started at the newest end would hand
them out again on every round and never reach older ones (RT4). The feed
walks the queue with the lease's cursor instead, and keeps its place per
order in ``display_feed_walks``: a run looks at the newest end first (new
captures land within the hour whatever the backlog) and goes on through new
rows, the walk's place moving with it, so a run cut off among them (its
rounds, budget, high water or max_new spent) goes on from there next time,
never past the rest of them. It goes on past an answer of skipped rows
alone, the walk's place left where it is: a skipped row was never queued, so
it is no sign of ground covered, and the next run's pass goes on past it the
same way. It does so in the first half of the run's rounds and of its time
only: skipped rows stay pending, so a pass that went on past them until
either ran out would never reach the walk's place once more of them lie
before it than a run gets through, and every later run would queue new
captures alone, its walk never deeper (round 3 verifier, V3); what lies
behind so many waits for the walk to wrap round. At the first answer with a
row queued already and nothing new, or of skipped rows alone past half the
run, it goes on from where the walk stood when the run began, the place put
back there (or carries the walk on, once it reaches that place). The price of a
run cut off among new rows: past them, the next goes on through rows the
walk had covered already, a round per ``lease_items`` of those still
pending, until it is past the deeper place. Each round's place is committed
with the rows it queued. A walk that reaches the
end of the queue (an empty answer) ends, and the next run starts a new one at
the newest end, which comes back for rows that were held when it passed
them. A run that queued nothing and left the walk where it found it, its
rounds or time spent, is STALLED (exit 1): a person looks.

THE HOLD is ``lease_seconds``, at most 120. The feed reports none of its rows
under it (a report needs no live lease), and a hold ends only by lapsing
(there is no release RPC), so a row it leased goes back to every other caller
-- the mm-report replay's origin-filtered lease among them -- within two
minutes of the feed's run (RT5); the walk's cursor, not its holds, keeps the
feed from leasing its own rows twice.

A DEPLOYMENT FAULT -- an answer from a server before 0.12.0 (no product_id or
first_seen_at), a first_seen_at or product_image_id a cursor could not carry
back verbatim, rows out of the total order asked, or a row not after the
cursor sent (a 0.12.0 draft that ignores it) -- queues nothing from that
answer and stops the run (exit 3); the holds lapse.

PURGE. Depictions reported more than ``purge_days`` ago, and the seeded or
failed originals left with none.

THE BUDGET. ``budget_seconds`` is the whole run's, counted from its start.
The REPORT starts no new report once ``REPORT_SHARE`` of it is spent; the
rows it did not reach (the batch's too) stay unreported for the next run,
and its line says ``partial: budget`` and how many are ``remaining``.
ENQUEUE gets what is left (THE WALK's half of its time is half of that), and
PURGE runs after it. Half: the mm-display-sync Job's activeDeadlineSeconds
is 1680 to a budget of 1500 (fc-infra media-display-sync-cronjob.yaml), and
those 180 s are for starting up, the one call in flight when time runs out
(57.5 s at most with its retries) and the purge, never for a phase. An
enqueue of all its 200 rounds takes ~410 s at the ~2 s a round measured (100
in 206 s, 2026-11-02), so the 750 s left holds them with room for a slower
database; a REPORT at the ~2.2 s a report measured in a drain (450 in 995 s)
still sends ~340 a run. Without the share, a REPORT past the deadline was
killed with the Job, and ENQUEUE and PURGE never ran.

Exit status: 0 done; 1 done, but a report was refused, FAILED reports were
held or the enqueue stalled; 2 a configuration fault (the mesh refused the
identity, the spine refused the lease or its cursor, the database refused the
role); 3 stopped, resumable (the spine or the database stayed down, a
deployment fault, or another sync writing the same rows at the same time).
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import socket
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import CursorResult, and_, delete, exists, func, or_, select, update
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from ..models import DisplayDepiction, DisplayFeedWalk, DisplayOriginal, PublicObject
from ..models import Image as ImageModel
from ..workers.mask_seed import CLEAN_PIPELINE_VERSION
from . import messages as msg
from .copies import measured_appearance, reported_copy
from .field_rules import FieldRefused
from .media_client import LEASE_ORDERS, SpineCallError, SpineMediaClient, Verdict
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
#: PendingImage.first_seen_at as a 0.12.0 server spells it, and the only
#: spelling a LeaseCursor may carry back: UTC, always six fractional digits,
#: ASCII digits only (a str pattern's \d matches every Unicode digit, and
#: strptime reads some of them).
INSTANT_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", re.ASCII)
INSTANT_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
#: The longest hold the feed asks for (see THE HOLD).
MAX_LEASE_SECONDS = 120
#: The part of the run's ``budget_seconds`` the REPORT may spend before it
#: starts no new report; ENQUEUE gets the rest (see THE BUDGET).
REPORT_SHARE = 0.5


@dataclass(frozen=True)
class FeedConfig:
    order: str = "newest"
    scope: str = "showable"
    lease_items: int = 200
    lease_seconds: int = MAX_LEASE_SECONDS
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
        if self.scope != "showable":
            raise ValueError(
                "scope must be showable: only a SHOWABLE lease carries origin_url (media.v1 0.12.0)"
            )
        if not 1 <= self.lease_items <= 200:
            raise ValueError("lease_items must be 1-200")
        if not 1 <= self.lease_seconds <= MAX_LEASE_SECONDS:
            raise ValueError(
                f"lease_seconds must be 1-{MAX_LEASE_SECONDS}: a hold ends only by lapsing"
            )
        if self.max_rounds < 2:
            raise ValueError(
                "max_rounds must be at least 2: a run's first round goes to the newest end, "
                "so with one a walk under way never moves"
            )
        for name in ("max_new", "high_water", "report_limit", "max_failed_reports"):
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
    monotonic: Callable[[], float] = time.monotonic,
    until: float = math.inf,
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
    for n, (depiction, original) in enumerate(rows):
        if monotonic() >= until:  # its share of the run's budget spent: the rest wait
            counts["remaining"] = len(rows) - n
            break
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
    counts["relink"] = _relink(db, now)
    _report_left_to_the_batch(db, client, config, counts, now, monotonic, until)
    line: dict[str, Any] = {
        "event": "reported",
        **{
            k: counts[k]
            for k in ("derived", "failed", "refused", "no_copy", "reseed", "relink", "batch_copy")
        },
    }
    if counts["remaining"]:
        line.update(partial="budget", remaining=counts["remaining"])
    emit(line)
    return counts


def _relink(db: Session, now: dt.datetime) -> int:
    """Seeded originals with a depiction no seed linked, sent back to the
    seeder: the feed queued it between the settle's read of the unlinked
    depictions and its commit of ``seeded`` (it read the original pending,
    so left its state). No claim takes a seeded original and nothing here
    reports an unlinked depiction, so it would wait forever; the seeder
    links it (and only it) and the next REPORT sends it."""
    sent_back: CursorResult[Any] = db.execute(  # type: ignore[assignment] # an UPDATE's result
        update(DisplayOriginal)
        .where(
            DisplayOriginal.state == "seeded",
            select(DisplayDepiction.product_image_id)
            .where(
                DisplayDepiction.origin_sha256 == DisplayOriginal.origin_sha256,
                DisplayDepiction.linked_at.is_(None),
            )
            .exists(),
        )
        .values(state="pending", updated_at=now),
        execution_options={"synchronize_session": "fetch"},
    )
    db.commit()
    return int(sent_back.rowcount or 0)


def _report_left_to_the_batch(
    db: Session,
    client: SpineMediaClient,
    config: FeedConfig,
    counts: Counter[str],
    now: dt.datetime,
    monotonic: Callable[[], float],
    until: float,
) -> None:
    """Depictions of ``stuck`` originals (left to the GPU batch) whose bytes
    the media DB now holds an image of -- the batch's seed of them, or any
    earlier seed of the same bytes for another product: DERIVED with the
    copy the replay would report (reported_copy, its record as
    replay._load_chunk builds it). The replay reports them too, once the
    feed's short hold on them lapses; both reports are idempotent, and this
    one also settles the lane's own row (purged after ``purge_days``), so a
    row the lane leaves to the batch never waits on the replay's schedule.
    Never ``held`` ones: the media DB may hold their bytes from the start
    (another product's capture), so a copy there is no operator's decision;
    they wait for ``display_copy requeue --state held --force``."""
    rows = db.execute(
        select(DisplayDepiction, DisplayOriginal)
        .join(DisplayOriginal, DisplayOriginal.origin_sha256 == DisplayDepiction.origin_sha256)
        .where(
            DisplayDepiction.reported_at.is_(None),
            DisplayOriginal.state == "stuck",
            # a column the role may read: EXISTS (SELECT * ...) needs them all
            select(ImageModel.id)
            .where(ImageModel.sha256 == DisplayOriginal.origin_sha256)
            .exists(),
        )
        .order_by(DisplayDepiction.first_seen_at.desc(), DisplayDepiction.product_image_id)
        .limit(config.report_limit)
    ).all()
    records: dict[str, msg.CopyRecord | None] = {}
    for n, (depiction, original) in enumerate(rows):
        if monotonic() >= until:  # the REPORT's share spent, here or before the batch
            counts["remaining"] += len(rows) - n
            return
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
            continue  # no live copy shows it (a takedown): the row stays the lane's, unreported
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
    """A uuid in its 36-character 8-4-4-4-12 form, either case."""
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


def _instant(token: str) -> dt.datetime | None:
    """A first_seen_at token as 0.12.0 spells it, as an instant; None for any
    other spelling of it (an offset, three digits, none) or an instant that
    does not exist (an hour 24, a second 60, a 30 February)."""
    if not INSTANT_RE.fullmatch(token):
        return None
    try:
        return dt.datetime.strptime(token, INSTANT_FORMAT).replace(tzinfo=dt.UTC)
    except ValueError:
        return None


@dataclass(frozen=True)
class Place:
    """A place in the lease's total order (media.v1 LeaseCursor): a row's
    first_seen_at and product_image_id, VERBATIM as the spine sent them."""

    first_seen_at: str
    product_image_id: str

    @classmethod
    def of(cls, image: media_pb2.PendingImage) -> Place:
        return cls(image.first_seen_at, image.product_image_id)

    def comes_after(self, other: Place, order: str) -> bool:
        """Whether this place is strictly after ``other`` in ``order``. The
        tokens sort as text in time order; a uuid by value is its lowercase
        spelling as text."""
        mine = (self.first_seen_at, self.product_image_id.lower())
        theirs = (other.first_seen_at, other.product_image_id.lower())
        return mine < theirs if order == "newest" else mine > theirs


def _check_answer(
    images: list[media_pb2.PendingImage], order: str, after: Place | None
) -> list[dt.datetime]:
    """Every row's first_seen_at, once the answer is a 0.12.0 server's: every
    row a place a cursor can carry back, in the total order asked, the first
    after the cursor sent. A _Stop otherwise."""
    seen: list[dt.datetime] = []
    previous = after
    for image in images:
        if not image.product_id or not image.first_seen_at:
            raise _Stop(
                EXIT_STOPPED,
                "the lease answered without product_id or first_seen_at: a spine before media.v1 "
                "0.12.0, which ignores order, scope and the cursor; nothing queued",
            )
        instant = _instant(image.first_seen_at)
        if instant is None or not _uuid(image.product_image_id):
            raise _Stop(
                EXIT_STOPPED,
                "the lease answered a first_seen_at or product_image_id a cursor cannot carry "
                "back verbatim (media.v1 0.12.0 spells them exactly); nothing queued",
            )
        place = Place.of(image)
        if previous is not None and not place.comes_after(previous, order):
            what = (
                "a row not after the cursor it was sent (a server that ignores the cursor)"
                if previous is after
                else f"rows out of the {order}-first order asked"
            )
            raise _Stop(EXIT_STOPPED, f"the lease answered {what}; nothing queued")
        previous = place
        seen.append(instant)
    return seen


def _skip_reason(image: media_pb2.PendingImage) -> str | None:
    origin = image.origin_sha256.hex()
    if len(image.origin_sha256) != 32 or not _uuid(image.product_id):
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


def _walk_place(walk: DisplayFeedWalk | None) -> Place | None:
    return Place(walk.after_first_seen_at, walk.after_product_image_id) if walk else None


def _move_walk(
    db: Session, walk: DisplayFeedWalk | None, order: str, place: Place, now: dt.datetime
) -> DisplayFeedWalk:
    if walk is None:
        walk = DisplayFeedWalk(lease_order=order, started_at=now)
        db.add(walk)
    walk.after_first_seen_at = place.first_seen_at
    walk.after_product_image_id = place.product_image_id
    walk.updated_at = now
    return walk


def enqueue(
    db: Session,
    client: SpineMediaClient,
    config: FeedConfig,
    emit: Callable[[dict[str, Any]], None],
    monotonic: Callable[[], float],
    now: dt.datetime,
    until: float,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    owner = config.owner or f"mm-display-sync/{os.environ.get('JOB_NAME') or socket.gethostname()}"
    start = monotonic()
    budget = until - start  # what the REPORT left of the run's budget_seconds
    walk = db.get(DisplayFeedWalk, config.order)
    found = _walk_place(walk)  # where the last run left the walk
    # A new walk starts at the order's end. A walk under way is gone on from
    # once this run's pass from the end meets rows it has queued already.
    walking = walk is None
    after: Place | None = None  # where the next lease goes on after (None: the end)
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
        if monotonic() - start >= budget:
            why = "budget"
            break
        try:
            outcome = client.lease_queue(
                order=config.order,
                scope=config.scope,
                max_items=config.lease_items,
                lease_owner=owner,
                lease_seconds=config.lease_seconds,
                after=(after.first_seen_at, after.product_image_id) if after else None,
            )
        except SpineCallError as exc:
            raise _stop_for(exc) from exc
        counts["rounds"] += 1
        images = list(outcome.response.images)
        if not images:
            # nothing pending past this place, nor past the walk's (never
            # before this one): the walk has reached the end of the queue
            if walk is not None:
                db.delete(walk)
                walk = None
                db.commit()
            counts["walk_done"] = 1
            break
        seen = _check_answer(images, config.order, after)
        new = already = 0
        for image, instant in zip(images, seen, strict=True):
            reason = _skip_reason(image)
            if reason is not None:
                skipped[reason] += 1
            elif _queue_one(db, image, instant):
                new += 1
            else:
                already += 1
        counts["new_depictions"] += new
        counts["already"] += already
        last = Place.of(images[-1])
        if not walking and found is not None and not found.comes_after(last, config.order):
            walking = True  # this pass has reached the walk's place: it carries the walk on
        if walking or new:
            # the walk, or new rows at this end: on through them, the walk's
            # place with them, so a run cut off among new rows (its rounds,
            # budget, high water or max_new spent) goes on from here, never
            # past the rest of them to a deeper place
            walk = _move_walk(db, walk, config.order, last, now)
            after = last
        elif (
            not already
            and counts["rounds"] * 2 <= config.max_rounds
            and (monotonic() - start) * 2 <= budget
        ):
            # rows all skipped: never queued, so no sign of ground covered, nor
            # a place to keep (the next run's pass goes on past them the same
            # way): on past them, the walk's place left where it is. In the
            # first half of the run's rounds and time only: past it, back to
            # the walk, so more skipped rows before it than a run gets through
            # never starve it
            after = last
        else:
            # ground already covered: on from the walk's place, and the walk
            # back there now (all before it is covered), so a run that stops
            # past here keeps the deep place. Skipped rows past half the run
            # go the same way: what lies behind them waits for the walk to wrap
            assert found is not None  # a run with no walk is walking from its first round
            walk = _move_walk(db, walk, config.order, found, now)
            walking, after = True, found
        db.commit()
        pending = _pending(db)
    place = _walk_place(walk)
    if (
        why in ("rounds", "budget")
        and counts["rounds"]
        and not counts["new_depictions"]
        and place == found
    ):
        # it queued nothing and its walk is where it found it: no later run
        # with the same budget gets further (exit 1, a person looks)
        counts["stalled"] = 1
        emit(
            {
                "event": "stalled",
                "rounds": counts["rounds"],
                "already": counts["already"],
                "walk_after": place.first_seen_at if place else None,
            }
        )
    emit(
        {
            "event": "enqueued",
            "why": why,
            "rounds": counts["rounds"],
            "new_depictions": counts["new_depictions"],
            "already": counts["already"],
            "skipped": dict(skipped),
            "pending_originals": pending,
            "walk_after": place.first_seen_at if place else None,
            "walk_started": _utc(walk.started_at).isoformat() if walk else None,
            "walk_done": bool(counts["walk_done"]),
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
    start = monotonic()
    client = connect()
    try:
        try:
            # the REPORT's share of the run's budget, then the ENQUEUE what is left
            reports_until = start + config.budget_seconds * REPORT_SHARE
            reported = report(db, client, config, emit, now(), monotonic, reports_until)
            queued = enqueue(
                db, client, config, emit, monotonic, now(), start + config.budget_seconds
            )
        except (OperationalError, ProgrammingError) as exc:
            # the database down (resumable) or refusing the role (a grant, the schema): never a traceback
            db.rollback()
            raise _Stop(
                EXIT_STOPPED if isinstance(exc, OperationalError) else EXIT_CONFIG,
                f"database: {type(exc).__name__}",
            ) from exc
        except IntegrityError as exc:
            # a row another writer committed first (the walk, an original, a
            # depiction): two syncs at once, the CronJob's Forbid bypassed by a
            # Job made by hand. This round is rolled back; the next run goes on.
            db.rollback()
            said = str(exc.orig).split("\n", 1)[0][:120]  # the constraint, never the row
            raise _Stop(
                EXIT_STOPPED,
                f"database: IntegrityError: another sync wrote the row first (two at once?): {said}",
            ) from exc
    except _Stop as stop:
        emit({"event": "stopped", "why": stop.why})
        return stop.code
    purged = purge(db, config, now())
    emit({"event": "done", "purged": purged, **_lag(db)})
    if reported["refused"] or reported["failed_held"] or queued["stalled"]:
        return EXIT_REFUSED
    return EXIT_DONE
