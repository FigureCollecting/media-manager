"""SHOW1: the display-copy lane's SEEDER -- a clean copy for every captured
original the spine leased to the lane (app.spine.display_feed queued it in
``display_originals``), in the cluster, on CPU, with no mask model.

    python -m app.cli.display_copy seed   (the mm-display-seed CronJob)

WHAT IT HOLDS AND WHAT IT DOES NOT. The media bucket's read-write key and
the raw bucket's READ-ONLY key, and the media DB. NO spine identity: it
never calls SpineMedia (the reporter does, app.spine.display_feed, under the
one identity the spine's media listener admits), so the pod that decodes
untrusted bytes cannot register anything on the spine by itself. It never
contacts a store: an original's bytes come from the raw bucket only.

PER ORIGINAL, newest first (``claim_next``):
  1. CLAIM: ``attempts`` + 1 and ``claimed_until`` committed BEFORE anything
     is read or decoded, so a pod killed in the decode leaves its trace. An
     original whose claims never settled ``max_attempts`` times is ``stuck``
     (``crashed_repeatedly``) and is never decoded again. A claim a dead pod
     left behind is taken next with EVERY decode unit, alone: the pod's one
     poison image cost each original it held an attempt, so only a crash in
     that isolated run is the original's own.
  2. READ the raw object (read only), decode its codec, check its sha, read
     its header. MEMORY: an original holds one decode unit per
     ``unit_pixels`` begun through its seed AND its appearance (measured:
     about 53 MB a megapixel); one past every unit together is ``stuck``
     (``too_large_for_lane:<pixels>``), left to the GPU batch, never FAILED.
  3. SEED exactly as the mask batch does (app.workers.mask_seed
     seed_with_retry: the original private, its clean copy public, one
     gallery link per depiction, the retry after a concurrent first seed),
     once per product its depictions belong to.
  4. RECORD the copy the replay reports (app.spine.copies.reported_copy) and
     its appearance: the mask row's plain values when a mask row measured
     them, else measured now by the mask lane's own code
     (app.workers.mask.plain_appearance). The reporter sends it.

FAILED IS ONLY EVER THE IMAGE'S OWN FAULT, a closed list: ``raw_hash_mismatch``
(the object is not what the spine recorded), ``refused:<limit>`` (a limit
of app.workers.image_limits) and ``undecodable`` (the bytes are not an image
the decoder can read). A FAILED report is permanent on the spine in
practice, so nothing else ever becomes one:
  * a bad credential or configuration (the bucket's AccessDenied,
    InvalidAccessKeyId, SignatureDoesNotMatch, RequestTimeTooSkewed, ...;
    the database's ProgrammingError: a grant the role lacks, a schema not
    migrated) stops the run at once, exit 2, its attempt given back;
  * an outage (the network, a 5xx, the database) gives the attempt back;
    ``max_outages`` in a row stop the run, exit 3;
  * anything else (a missing raw object, a corrupt stored object, a bug) is
    an ERROR: the original stays pending with its attempt counted, and is
    ``stuck`` (``gave_up:<why>``) once ``max_attempts`` runs have tried it;
    ``max_errors`` in one run stop it, exit 1;
  * a codec the lane does not know is ``stuck`` at once
    (``unsupported_codec:<codec>``).
THE BREAKER: more than ``max_failed`` own faults that are also more than
``max_failed_fraction`` of the originals settled stop the run, exit 1, so a
fault that only looks like the images' (a bucket full of wrong objects) is
looked at by a person before the reporter makes it permanent (the reporter
holds FAILED reports past its own cap, which counts the last day's reports
too).

HELD, NOT SEEDED: depictions whose seed would turn the pooled mask verdict
of a picture whose clean copy serves a published mask into a refusal
(``not_granted``, ``unknown_origin``, ``user_photo``, ``store_revoked``, or
``no_origin`` when the product already links the URL to another image).
Mask eligibility is POOLED over every capture of a picture
(app.publication), so seeding it would take that mask offline at the origin
while the spine still advertises it. Such an original is ``held``
(``would_withdraw_mask:<class>``), never reported by the lane, until an
operator decides (``display_copy requeue --state held --force``). The
mm-report replay is not the lane: it reports every pending depiction of an
origin the media DB holds a copy of, so a held capture of bytes the media DB
holds already (another product's capture of them) is reported DERIVED by
the replay's next run, with no gallery link of its own.

Exit status: 0 done (the queue empty or the budget spent); 1 the breaker
(own faults or errors past their limits); 2 a configuration fault; 3 stopped
by outages, resumable.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import os
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session

from ..models import (
    DisplayDepiction,
    DisplayOriginal,
    FigureGallery,
    ImageMask,
    PublicObject,
    PublicObjectImage,
)
from ..models import Image as ImageModel
from ..object_store import ObjectStore
from ..publication import gallery_origins, judged_origins, mask_refusal, store_revocations
from ..spine.copies import measured_appearance, reported_copy
from ..telemetry.db import DbTally
from .grounding import AlphaGrounding
from .image_limits import ImageLimitError, check_image
from .mask import plain_appearance
from .mask_policy import MASKABLE, ImageOrigin, StoreGrants, classify_origin, mask_eligibility
from .mask_seed import _copies_showing, picture_keys, seed_with_retry
from .raw_source import (
    KNOWN_CODECS,
    bucket_error_code,
    decode_stored,
    is_configuration_error,
    is_outage,
)

EXIT_DONE = 0
EXIT_BREAKER = 1
EXIT_CONFIG = 2
EXIT_STOPPED = 3

#: The states an operator may put back to pending (``requeue``).
REQUEUEABLE = ("failed", "held", "stuck")

Kind = Literal["seeded", "failed", "held", "stuck", "error", "outage", "interrupted", "config"]


@dataclass(frozen=True)
class SeedConfig:
    #: Worker threads, each with its own database session (0: one at a time on the caller's).
    threads: int = 6
    #: Seconds after which no new original is claimed (the CronJob's deadline is past it).
    budget_seconds: float = 2700.0
    #: Decode units in flight at once, an original taking one per ``unit_pixels`` begun
    #: (the memory bound: one decode, its copy's encode and its appearance).
    decode_units: int = 6
    unit_pixels: int = 4_000_000
    claim_seconds: int = 900
    max_attempts: int = 3
    max_errors: int = 10
    max_failed: int = 25
    max_failed_fraction: float = 0.05
    max_outages: int = 5
    owner: str = ""
    #: The mask policy's store grants (None: the configured MASK_STORE_GRANTS).
    grants: StoreGrants | None = None

    def __post_init__(self) -> None:
        if self.threads < 0 or self.decode_units < 1 or self.max_attempts < 1:
            raise ValueError("threads >= 0, decode_units >= 1 and max_attempts >= 1")
        if not self.budget_seconds > 0:
            raise ValueError("budget_seconds must be above 0")


@dataclass(frozen=True)
class SeedResult:
    kind: Kind
    reason: str | None = None
    #: Seeded: the copy and its appearance, as ``display_originals`` records them.
    look: Mapping[str, Any] | None = None
    #: Seeded: the depictions whose gallery links this seed wrote.
    linked: tuple[str, ...] = ()


def grounding_dict(grounding: AlphaGrounding | None) -> dict[str, float] | None:
    """A grounding as the mask rows store it (``image_masks.own_alpha_grounding``)."""
    return dataclasses.asdict(grounding) if grounding is not None else None


# --------------------------------------------------------------------------
# the claim
# --------------------------------------------------------------------------


def claim_next(
    db: Session, *, owner: str, claim_seconds: int, now: dt.datetime
) -> DisplayOriginal | None:
    """The newest pending original no one holds, claimed: its attempt
    counted and committed before it is read (FOR UPDATE SKIP LOCKED, so two
    seeders never claim one original)."""
    claimed = claim_with_history(db, owner=owner, claim_seconds=claim_seconds, now=now)
    return claimed[0] if claimed is not None else None


def claim_with_history(
    db: Session, *, owner: str, claim_seconds: int, now: dt.datetime
) -> tuple[DisplayOriginal, bool] | None:
    """``claim_next``, and whether the claim before it was ABANDONED (still
    named a holder: that pod died holding it)."""
    row = db.execute(
        select(DisplayOriginal)
        .where(
            DisplayOriginal.state == "pending",
            or_(DisplayOriginal.claimed_until.is_(None), DisplayOriginal.claimed_until <= now),
        )
        .order_by(DisplayOriginal.first_seen_at.desc(), DisplayOriginal.origin_sha256)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()
    if row is None:
        db.rollback()
        return None
    abandoned = row.claimed_by is not None
    row.attempts += 1
    row.claimed_until = now + dt.timedelta(seconds=claim_seconds)
    row.claimed_by = owner[:128]
    row.updated_at = now
    db.commit()
    return row, abandoned


# --------------------------------------------------------------------------
# one original
# --------------------------------------------------------------------------


class _Units:
    """Decode units: an original takes ``n`` at once, never more than all.
    A request for EVERY unit (an isolated claim) goes first: while one waits
    no smaller request is granted, so it waits out only the seeds already in
    flight -- never past its own claim, beside threads that keep the units
    busy forever."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._free = total
        self._whole_waiting = 0
        self._cond = threading.Condition()

    def acquire(self, n: int) -> int:
        n = max(1, min(n, self.total))
        with self._cond:
            if n == self.total:
                self._whole_waiting += 1
                try:
                    while self._free < n:
                        self._cond.wait()
                finally:
                    self._whole_waiting -= 1
            else:
                while self._free < n or self._whole_waiting:
                    self._cond.wait()
            self._free -= n
        return n

    def release(self, n: int) -> None:
        with self._cond:
            self._free += n
            self._cond.notify_all()


class _Recording:
    """The media bucket as one original's seed sees it: what it PUT is kept,
    so its fresh clean copy is measured without being read back."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self.written: dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        return self.written[key] if key in self.written else self._store.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store.put(key, data, content_type)
        self.written[key] = data


def _origins(depictions: Sequence[DisplayDepiction]) -> list[ImageOrigin]:
    return [
        ImageOrigin(url=d.origin_url, source=d.source_site, source_class=d.source_class)
        for d in depictions
    ]


def _seeded_origins(
    db: Session, origin_sha256: str, depictions: Sequence[DisplayDepiction]
) -> list[ImageOrigin]:
    """The live origins the image will have once ``depictions`` are seeded:
    its live gallery links now, plus each depiction's URL unless the product
    already links that URL (uq_figure_source_url: the seed then writes NO
    link for this image) -- and, with none at all, the originless stand-in
    app.publication.judged_origins pools it as (``no_origin``)."""
    image_id = db.execute(
        select(ImageModel.id).where(ImageModel.sha256 == origin_sha256)
    ).scalar_one_or_none()
    origins = gallery_origins(db, image_id) if image_id is not None else []
    for d in depictions:
        taken = db.execute(
            select(FigureGallery.id).where(
                FigureGallery.figure_id == d.product_id, FigureGallery.source_url == d.origin_url
            )
        ).first()
        if taken is None:
            origins.append(
                ImageOrigin(url=d.origin_url, source=d.source_site, source_class=d.source_class)
            )
    return origins or [ImageOrigin(url=None)]


def _picture_copies(db: Session, copies: set[str], origin_sha256: str) -> set[str]:
    """Every clean copy the seed's _join_picture will put the image behind:
    ``copies`` grown through the recorded copy-image links to their fixed
    point (a capture behind two copies joins them)."""
    images = set(
        db.execute(select(ImageModel.id).where(ImageModel.sha256 == origin_sha256)).scalars()
    )
    while True:
        more_images = set(
            db.execute(
                select(PublicObjectImage.image_id).where(PublicObjectImage.sha256.in_(copies))
            ).scalars()
        )
        more_copies = set(
            db.execute(
                select(PublicObjectImage.sha256).where(PublicObjectImage.image_id.in_(images))
            ).scalars()
        )
        if more_images <= images and more_copies <= copies:
            return copies
        images, copies = images | more_images, copies | more_copies


def _withdraws_a_mask(
    db: Session,
    data: bytes,
    origin_sha256: str,
    depictions: Sequence[DisplayDepiction],
    grants: StoreGrants | None,
) -> str | None:
    """Why seeding ``depictions`` would take a published mask of this picture
    offline (the pooled verdict's refusal: not_granted, unknown_origin,
    user_photo, store_revoked or no_origin), or None. JUDGED, NOT GUESSED:
    every copy of the picture that serves a mask now is judged again over
    its current origins plus the ones this seed would add, with the
    persisted revocations, exactly as app.publication will judge it."""
    revoked = store_revocations(db)
    added = _seeded_origins(db, origin_sha256, depictions)
    if all(classify_origin(o, grants, revoked) in MASKABLE for o in added):
        return None  # every origin it adds is maskable: no verdict can turn (and no decode)
    db.commit()  # MEDIADB (M-1): its reads so far, none open across the decode and encode
    keys = picture_keys(data)
    shown = {c.sha256 for c in _copies_showing(db, keys.digest, origin_sha256)}
    shown |= {c.sha256 for c in _copies_showing(db, keys.own_digest, keys.clean_sha256)}
    shown = _picture_copies(db, shown, origin_sha256)
    for copy in sorted(shown):
        published = db.execute(
            select(ImageMask.id).where(ImageMask.source_sha256 == copy, ImageMask.status == "ok")
        ).first()
        if published is None or mask_refusal(db, copy, grants, revoked) is not None:
            continue
        # The copy's own origins pass (no refusal) and ``added`` holds one that
        # is not maskable (above), so the pooled verdict refuses: its reason.
        return mask_eligibility([*(judged_origins(db, copy) or []), *added], grants, revoked).reason
    return None


def _decodable(data: bytes) -> tuple[SeedResult | None, int]:
    """The image's header read (the limits first): its pixel count, or the
    image's own fault."""
    try:
        check_image(data)
        with PILImage.open(io.BytesIO(data)) as header:
            return None, header.width * header.height
    except ImageLimitError as exc:
        return SeedResult("failed", f"refused:{exc.code}"), 0
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return SeedResult("failed", "undecodable"), 0


#: What Pillow's decoders raise (a bare OSError) for broken image data.
_PIL_DECODE_FAULTS = (
    "image file is truncated",
    "broken data stream",
    "decoder error",
    "Truncated File Read",
    "unrecognized data stream contents",
    "buffer overrun",
)


def _raised(where: str, exc: Exception) -> SeedResult:
    if isinstance(exc, ImageLimitError):
        return SeedResult("failed", f"refused:{exc.code}")
    if isinstance(exc, IntegrityError):
        return SeedResult("interrupted", f"{where}:IntegrityError")
    if isinstance(exc, ProgrammingError):
        # a grant the role lacks, a schema not migrated: every item alike, the deployment's
        return SeedResult("config", f"{where}:ProgrammingError")
    if is_configuration_error(exc):
        return SeedResult("config", f"{where}:{bucket_error_code(exc)}")
    if is_outage(exc):
        return SeedResult("outage", f"{where}:{type(exc).__name__}")
    if isinstance(exc, UnidentifiedImageError | PILImage.DecompressionBombError):
        return SeedResult("failed", "undecodable")
    if type(exc) is OSError and any(m in str(exc) for m in _PIL_DECODE_FAULTS):
        # past the header, the decoder found the bytes broken (a truncated
        # capture): the image's own fault, as an unreadable header is
        return SeedResult("failed", "undecodable")
    return SeedResult("error", f"{where}:{type(exc).__name__}")


def seed_one(
    db: Session,
    raw: ObjectStore,
    media: ObjectStore,
    original: DisplayOriginal,
    *,
    units: _Units,
    config: SeedConfig,
    alone: bool = False,
) -> SeedResult:
    """Read, check and seed one claimed original; what came of it (written
    by ``_settle``, never here). ``alone``: with every decode unit, so no
    other original decodes beside it (a claim a dead pod left behind)."""
    if original.codec not in KNOWN_CODECS:
        return SeedResult("stuck", f"unsupported_codec:{original.codec}"[:256])
    depictions = list(
        db.execute(
            select(DisplayDepiction)
            .where(DisplayDepiction.origin_sha256 == original.origin_sha256)
            .order_by(DisplayDepiction.product_id, DisplayDepiction.product_image_id)
        ).scalars()
    )
    # MEDIADB (M-1): no transaction open across the bucket's GET, the decode
    # and the wait for decode units (the session keeps what it loaded)
    db.commit()
    if not depictions:
        return SeedResult("stuck", "no_depictions")
    try:
        stored = raw.get(original.storage_key)
    except Exception as exc:  # noqa: BLE001 -- classified, never swallowed
        if bucket_error_code(exc) == "NoSuchKey" or isinstance(exc, FileNotFoundError):
            return SeedResult("error", "missing_raw")
        return _raised("fetch", exc)
    try:
        data = decode_stored(stored, original.codec)
    except ImageLimitError as exc:
        return SeedResult("failed", f"refused:{exc.code}")
    except Exception as exc:  # noqa: BLE001 -- a corrupt stored object is the store's, not the image's
        return SeedResult("error", f"decode_stored:{type(exc).__name__}")
    if hashlib.sha256(data).hexdigest() != original.origin_sha256:
        return SeedResult("failed", "raw_hash_mismatch")
    refused, pixels = _decodable(data)
    if refused is not None:
        return refused
    if pixels > units.total * config.unit_pixels:
        # past what every decode unit together covers: the pod's memory limit
        # is at risk, so the lane leaves it to the GPU batch (never a FAILED)
        return SeedResult("stuck", f"too_large_for_lane:{pixels}")
    taken = units.acquire(units.total if alone else -(-pixels // config.unit_pixels))
    try:
        return _seed_and_measure(db, media, original, depictions, data, config)
    finally:
        units.release(taken)


def _seed_and_measure(
    db: Session,
    media: ObjectStore,
    original: DisplayOriginal,
    depictions: list[DisplayDepiction],
    data: bytes,
    config: SeedConfig,
) -> SeedResult:
    origin = original.origin_sha256
    unlinked = [d for d in depictions if d.linked_at is None] or depictions
    try:
        if not original.force_seed:
            withdrawing = _withdraws_a_mask(db, data, origin, unlinked, config.grants)
            if withdrawing is not None:
                return SeedResult("held", f"would_withdraw_mask:{withdrawing}")
            db.commit()  # the check only read: none open across the seed's encode and PUTs
        recording = _Recording(media)
        by_product: dict[str, list[DisplayDepiction]] = {}
        for d in unlinked:
            by_product.setdefault(d.product_id, []).append(d)
        for product, group in by_product.items():
            seed_with_retry(db, recording, data, origins=_origins(group), figure_id=product)
        copy = reported_copy(db, origin)
        look: dict[str, Any] = {"copy_sha256": copy}
        if copy is not None:
            obj = db.get(PublicObject, copy)
            assert obj is not None  # reported_copy names a registered copy
            look.update(
                copy_mime=obj.mime,
                copy_width=obj.width,
                copy_height=obj.height,
                copy_generated_at=obj.created_at,
            )
            row = measured_appearance(db, copy)
            if row is not None:
                look.update(
                    thumbhash=row.plain_thumbhash,
                    dominant_color=row.plain_dominant_color,
                    own_grounding=row.own_alpha_grounding,
                )
            else:
                db.commit()  # read only since the seed: none open across a GET and the measure
                appearance, grounding = plain_appearance(recording.get(obj.storage_key))
                look.update(
                    thumbhash=appearance.thumbhash,
                    dominant_color=appearance.dominant_color,
                    own_grounding=grounding_dict(grounding),
                )
    except Exception as exc:  # noqa: BLE001 -- classified, never swallowed
        db.rollback()
        return _raised("seed", exc)
    return SeedResult("seeded", look=look, linked=tuple(d.product_image_id for d in unlinked))


def _settle(
    db: Session,
    original: DisplayOriginal,
    result: SeedResult,
    *,
    now: dt.datetime,
    max_attempts: int,
    hold_until: dt.datetime,
) -> SeedResult:
    """Write what came of one claim; the result as it was recorded (an
    error past its attempts becomes ``stuck``). An original that stays
    pending keeps its claim until ``hold_until`` (the end of this run's
    budget), so the run never tries it twice and the next run does."""
    original.claimed_until = None
    original.claimed_by = None
    original.updated_at = now
    if result.kind in ("outage", "interrupted", "config", "error"):
        original.claimed_until = hold_until
    if result.kind == "seeded":
        original.state, original.reason, original.attempts = "seeded", None, 0
        original.force_seed = False
        for name, value in (result.look or {}).items():
            setattr(original, name, value)
        late = False
        for d in db.execute(
            select(DisplayDepiction).where(
                DisplayDepiction.origin_sha256 == original.origin_sha256,
                DisplayDepiction.linked_at.is_(None),
            )
        ).scalars():
            if d.product_image_id in result.linked:
                d.linked_at = now
            else:  # queued by the feed while this seed ran: its link is still to write
                late = True
        if late:
            original.state = "pending"
    elif result.kind in ("failed", "held", "stuck"):
        original.state, original.reason = result.kind, result.reason
    elif result.kind in ("outage", "interrupted", "config"):
        original.attempts = max(0, original.attempts - 1)
    else:  # error
        original.reason = result.reason
        if original.attempts >= max_attempts:
            original.state, original.reason = "stuck", f"gave_up:{result.reason}"[:256]
            result = SeedResult("stuck", original.reason)
    db.commit()
    return result


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


class _Run:
    def __init__(
        self,
        config: SeedConfig,
        emit: Callable[[dict[str, Any]], None],
        now: Callable[[], dt.datetime],
        monotonic: Callable[[], float],
    ) -> None:
        self.config = config
        self.emit = emit
        self.now = now
        self.monotonic = monotonic
        self.start = monotonic()
        self.counts: Counter[str] = Counter()
        self.lock = threading.Lock()
        self.stop: tuple[int, str] | None = None
        self.outages_in_a_row = 0
        self.units = _Units(config.decode_units)
        self.owner = config.owner or f"display-seed/{socket.gethostname()}/{os.getpid()}"
        self.hold_until = now() + dt.timedelta(seconds=config.budget_seconds + config.claim_seconds)
        #: MEDIADB (U8): each original's SQL on its item line, the run's on its last
        self.tally: DbTally | None = None

    def _tally(self, db: Session) -> DbTally:
        with self.lock:  # one tally for the run: every worker's session is on one engine
            if self.tally is None:
                self.tally = DbTally(db.get_bind().engine)
            return self.tally

    def sql(self) -> dict[str, float | int]:
        return (
            self.tally.totals()
            if self.tally is not None
            else {"db_seconds": 0.0, "db_statements": 0, "db_rows": 0}
        )

    def over_budget(self) -> bool:
        return self.monotonic() - self.start >= self.config.budget_seconds

    def stopping(self) -> bool:
        with self.lock:
            return self.stop is not None

    def record(self, result: SeedResult, sql: Mapping[str, float | int] | None = None) -> None:
        c = self.config
        with self.lock:
            self.counts[result.kind] += 1
            if result.kind == "outage":
                self.outages_in_a_row += 1
            elif result.kind != "interrupted":
                self.outages_in_a_row = 0
            if self.stop is not None:
                return
            settled = sum(self.counts[k] for k in ("seeded", "failed", "held", "stuck"))
            if result.kind == "config":
                self.stop = (EXIT_CONFIG, "configuration")
            elif self.outages_in_a_row >= c.max_outages:
                self.stop = (EXIT_STOPPED, "outages")
            elif self.counts["error"] >= c.max_errors:
                self.stop = (EXIT_BREAKER, "errors")
            elif (
                self.counts["failed"] > c.max_failed
                and self.counts["failed"] > c.max_failed_fraction * settled
            ):
                self.stop = (EXIT_BREAKER, "failed_rate")
        if result.kind in ("failed", "held", "stuck", "error", "config"):
            self.emit(
                {"event": "item", "kind": result.kind, "reason": result.reason, **(sql or {})}
            )

    def worker(self, sessions: Callable[[], Session], raw: ObjectStore, media: ObjectStore) -> None:
        db = sessions()
        tally = self._tally(db)
        try:
            while not self.stopping():
                if self.over_budget():
                    with self.lock:
                        self.counts["budget"] = 1
                    return
                now = self.now()
                claimed = claim_with_history(
                    db, owner=self.owner, claim_seconds=self.config.claim_seconds, now=now
                )
                if claimed is None:
                    return
                original, abandoned = claimed
                if original.attempts > self.config.max_attempts:
                    result = SeedResult("stuck", "crashed_repeatedly")
                else:
                    if abandoned:
                        self.emit(
                            {"event": "item", "kind": "isolated", "attempts": original.attempts}
                        )
                    try:
                        result = seed_one(
                            db,
                            raw,
                            media,
                            original,
                            units=self.units,
                            config=self.config,
                            alone=abandoned,
                        )
                    except Exception as exc:  # noqa: BLE001 -- the run records it, never dies of it
                        db.rollback()
                        result = _raised("lane", exc)
                    db.refresh(original)
                result = _settle(
                    db,
                    original,
                    result,
                    now=self.now(),
                    max_attempts=self.config.max_attempts,
                    hold_until=self.hold_until,
                )
                self.record(result, tally.take())  # its claim, seed and settle
        except Exception as exc:  # noqa: BLE001 -- the claim or the settle: the run stops, never ends "done"
            with contextlib.suppress(Exception):  # the connection may be gone; the stop says so
                db.rollback()
            if isinstance(exc, OperationalError | InterfaceError):
                why = (EXIT_STOPPED, f"database:{type(exc).__name__}")
            elif isinstance(exc, ProgrammingError):
                why = (EXIT_CONFIG, "database:ProgrammingError")
            else:
                why = (EXIT_BREAKER, f"lane:{type(exc).__name__}")
            with self.lock:
                self.stop = self.stop or why
        finally:
            if self.config.threads:  # threads=0 runs on the caller's session, which it closes
                db.close()


def run_seed(
    sessions: Callable[[], Session],
    raw: ObjectStore,
    media: ObjectStore,
    config: SeedConfig,
    *,
    emit: Callable[[dict[str, Any]], None],
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Seed queued originals until the queue is empty, the budget is spent or
    a limit stops the run; the exit status."""
    run = _Run(config, emit, now, monotonic)
    emit({"event": "start", "threads": config.threads, "budget_seconds": config.budget_seconds})
    if config.threads == 0:
        run.worker(sessions, raw, media)
    else:
        threads = [
            threading.Thread(target=run.worker, args=(sessions, raw, media), daemon=True)
            for _ in range(config.threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    summary: dict[str, Any] = {
        k: run.counts[k]
        for k in ("seeded", "failed", "held", "stuck", "error", "outage", "interrupted")
    }
    summary.update(run.sql())
    if run.tally is not None:
        run.tally.close()
    if run.stop is not None:
        code, why = run.stop
        emit({"event": "stopped", "why": why, **summary})
        return code
    emit({"event": "done", "why": "budget" if run.counts["budget"] else "empty", **summary})
    return EXIT_DONE


# --------------------------------------------------------------------------
# operator commands
# --------------------------------------------------------------------------


def requeue(db: Session, *, state: str, reason_prefix: str | None, force: bool, apply: bool) -> int:
    """Put originals in ``state`` (failed, held or stuck) whose reason starts
    with ``reason_prefix`` back to pending, their attempts reset; ``force``
    seeds a held one although that would withdraw a mask. A dry run
    (``apply`` False) only counts them. A FAILED original the reporter has
    already reported is also failed on the spine: the spine's own requeue
    (fc-aggregation ``media:requeue-failed``) moves those back."""
    if state not in REQUEUEABLE:
        raise ValueError(f"requeue takes one of {', '.join(REQUEUEABLE)}, not {state!r}")
    query = select(DisplayOriginal).where(DisplayOriginal.state == state)
    if reason_prefix:
        query = query.where(DisplayOriginal.reason.startswith(reason_prefix, autoescape=True))
    rows = list(db.execute(query).scalars())
    if apply:
        for row in rows:
            row.state, row.reason, row.attempts = "pending", None, 0
            row.claimed_until = row.claimed_by = None
            row.force_seed = force
        db.commit()
    else:
        db.rollback()
    return len(rows)


def status(db: Session) -> dict[str, Any]:
    """The queue's counts by state, its unreported depictions, and the
    newest held, stuck and failed reasons."""
    by_state = dict(
        db.execute(
            select(DisplayOriginal.state, func.count()).group_by(DisplayOriginal.state)
        ).all()
    )
    unreported = db.scalar(select(func.count()).where(DisplayDepiction.reported_at.is_(None)))
    reasons = {
        reason or "": n
        for reason, n in db.execute(
            select(DisplayOriginal.reason, func.count())
            .where(DisplayOriginal.state.in_(REQUEUEABLE))
            .group_by(DisplayOriginal.reason)
        ).all()
    }
    return {"originals": by_state, "unreported_depictions": unreported, "reasons": reasons}
