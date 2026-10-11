"""MASK-1 batch job: masks for every clean image that has none yet.

DRY RUN BY DEFAULT. A dry run computes everything -- eligibility, the mask,
its grounding -- and reports it, but writes nothing: no object, no row, no
batch record. ``dry_run=False`` is the only way anything is stored, and it
refuses a backend that is not a mask model (``publishes_masks``; the stub,
which is also the configured default) before writing anything: a stub mask
would otherwise be published and, being a result, settle its image for good.

IDEMPOTENT. A clean image is selected only while it has no image_masks row
at the current MASK_PIPELINE_VERSION, so a re-run is a no-op and a version
bump selects everything again (the old masks stay addressable). Mask bytes
are content-addressed, so two images with the same mask share one object.

RESUMABLE. Each clean image is settled in its own transaction -- its object
written first, then its row, its public registration and the batch's cursor
and counts committed together -- so a killed worker loses at most the image
in flight. ``resume_batch_id`` continues a batch that never finished from
its cursor (candidates are taken in sha order, a keyset rather than an
offset, so nothing is skipped or seen twice).

FAILURES are per image: recorded as ``failed`` with the exception's CLASS
only (never a message, which can carry a bucket or key), and re-driven with
``retry_failed=True``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import FigureGallery, ImageMask, ImageVersion, MaskBatch, PublicObject
from ..models import Image as ImageModel
from ..object_store import ObjectStore, derived_key
from .grounding import AlphaGrounding
from .mask import MASK_MIME, MASK_PIPELINE_VERSION, MaskOutcome, build_mask_artifact
from .mask_policy import ImageOrigin, mask_eligibility
from .mask_seed import register_public_object
from .matting import MattingBackend, get_matting_backend

logger = logging.getLogger(__name__)

COUNT_KEYS = ("ok", "held", "empty", "ineligible", "failed")

DEFAULT_PAGE_SIZE = 50


@dataclass(frozen=True)
class ItemReport:
    source_sha256: str
    status: str
    reason: str | None
    origin_class: str | None
    mask_sha256: str | None = None
    grounding: AlphaGrounding | None = None


def _zero_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


@dataclass
class BatchReport:
    batch_id: int | None
    dry_run: bool
    items: list[ItemReport] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=_zero_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "dry_run": self.dry_run,
            "pipeline_version": MASK_PIPELINE_VERSION,
            "counts": dict(self.counts),
            "items": [dataclasses.asdict(i) for i in self.items],
        }


class _Refused(Exception):
    """A per-image refusal whose reason is safe to record verbatim."""


def _candidates(
    db: Session, *, after: str | None, page_size: int, retry_failed: bool
) -> list[PublicObject]:
    settled = select(ImageMask.source_sha256).where(
        ImageMask.pipeline_version == MASK_PIPELINE_VERSION
    )
    if retry_failed:
        settled = settled.where(ImageMask.status != "failed")
    query = (
        select(PublicObject)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .join(ImageModel, ImageModel.id == ImageVersion.image_id)
        .where(
            PublicObject.kind == "sanitized",
            ImageVersion.deleted_at.is_(None),
            ImageModel.deleted_at.is_(None),
            PublicObject.sha256.not_in(settled),
        )
        .order_by(PublicObject.sha256)
        .limit(page_size)
    )
    if after is not None:
        query = query.where(PublicObject.sha256 > after)
    return list(db.execute(query).scalars())


def _origins(db: Session, image_id: int) -> list[ImageOrigin]:
    rows = db.execute(
        select(FigureGallery)
        .where(FigureGallery.image_id == image_id, FigureGallery.deleted_at.is_(None))
        .order_by(FigureGallery.id)
    ).scalars()
    return [ImageOrigin(url=g.source_url, source=g.source) for g in rows]


def _row_fields(outcome: MaskOutcome) -> dict[str, Any]:
    g = outcome.grounding
    fields: dict[str, Any] = {
        "backend": outcome.backend,
        "width": outcome.width,
        "height": outcome.height,
        "mask_sha256": outcome.mask_sha256,
        "plain_thumbhash": outcome.unmasked.thumbhash,
        "plain_dominant_color": outcome.unmasked.dominant_color,
        "masked_thumbhash": outcome.masked.thumbhash if outcome.masked else None,
        "masked_dominant_color": outcome.masked.dominant_color if outcome.masked else None,
        "own_alpha_grounding": (
            dataclasses.asdict(outcome.own_alpha_grounding) if outcome.own_alpha_grounding else None
        ),
    }
    for name in (f.name for f in dataclasses.fields(AlphaGrounding)):
        fields[name] = getattr(g, name) if g else None
    return fields


_RESULT_COLUMNS = (
    "backend",
    "width",
    "height",
    "mask_sha256",
    "mask_storage_key",
    *(f.name for f in dataclasses.fields(AlphaGrounding)),
    "masked_thumbhash",
    "masked_dominant_color",
    "plain_thumbhash",
    "plain_dominant_color",
    "own_alpha_grounding",
)


def _write_row(
    db: Session,
    *,
    source_sha256: str,
    batch: MaskBatch,
    status: str,
    reason: str | None,
    origin_class: str | None,
    **fields: Any,
) -> None:
    now = dt.datetime.now(dt.UTC)
    row = db.execute(
        select(ImageMask).where(
            ImageMask.source_sha256 == source_sha256,
            ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
        )
    ).scalar_one_or_none()
    if row is None:
        row = ImageMask(source_sha256=source_sha256, pipeline_version=MASK_PIPELINE_VERSION)
        row.created_at = now
        db.add(row)
    row.status = status
    row.reason = reason
    row.origin_class = origin_class
    row.batch_id = batch.id
    row.updated_at = now
    # Every result column is written on every settle, so a re-driven row
    # never keeps a value from the attempt it replaces.
    for name in _RESULT_COLUMNS:
        setattr(row, name, fields.get(name))
    batch.cursor_sha256 = source_sha256
    batch.counts = {**batch.counts, status: batch.counts.get(status, 0) + 1}
    db.commit()


def _settle(
    db: Session,
    store: ObjectStore,
    obj: PublicObject,
    backend: MattingBackend,
    batch: MaskBatch | None,
) -> ItemReport:
    """Settle one clean image. ``batch`` is None on a dry run: nothing is
    written then, the report is the whole result."""
    sha = obj.sha256
    version = db.get(ImageVersion, obj.image_version_id)
    assert version is not None  # the candidate query joined it
    eligibility = mask_eligibility(_origins(db, version.image_id))
    if not eligibility.eligible:
        if batch is not None:
            _write_row(
                db,
                source_sha256=sha,
                batch=batch,
                status="ineligible",
                reason=eligibility.reason,
                origin_class=eligibility.origin_class,
            )
        return ItemReport(sha, "ineligible", eligibility.reason, eligibility.origin_class)

    try:
        data = store.get(obj.storage_key)
        if hashlib.sha256(data).hexdigest() != sha:
            raise _Refused("source_hash_mismatch")
        outcome = build_mask_artifact(data, backend=backend)
        fields = _row_fields(outcome)
        if batch is not None and outcome.mask_png is not None and outcome.mask_sha256 is not None:
            if outcome.status == "ok":
                key = derived_key(outcome.mask_sha256, MASK_MIME)
                store.put(key, outcome.mask_png, MASK_MIME)
                register_public_object(
                    db,
                    sha256=outcome.mask_sha256,
                    kind="mask",
                    mime=MASK_MIME,
                    size=len(outcome.mask_png),
                    width=outcome.width,
                    height=outcome.height,
                    storage_key=key,
                )
            else:
                key = f"held-masks/{outcome.mask_sha256}.png"
                store.put(key, outcome.mask_png, MASK_MIME)
            fields["mask_storage_key"] = key
    except Exception as exc:
        db.rollback()
        reason = str(exc) if isinstance(exc, _Refused) else type(exc).__name__
        logger.warning("mask_failed", extra={"source_sha256": sha, "reason": reason})
        if batch is not None:
            _write_row(
                db,
                source_sha256=sha,
                batch=batch,
                status="failed",
                reason=reason,
                origin_class=eligibility.origin_class,
            )
        return ItemReport(sha, "failed", reason, eligibility.origin_class)

    if batch is not None:
        _write_row(
            db,
            source_sha256=sha,
            batch=batch,
            status=outcome.status,
            reason=outcome.reason,
            origin_class=eligibility.origin_class,
            **fields,
        )
    return ItemReport(
        sha,
        outcome.status,
        outcome.reason,
        eligibility.origin_class,
        outcome.mask_sha256,
        outcome.grounding,
    )


def _open_batch(
    db: Session, *, resume_batch_id: int | None, backend_name: str, limit: int | None
) -> MaskBatch:
    if resume_batch_id is not None:
        batch = db.get(MaskBatch, resume_batch_id)
        if (
            batch is None
            or batch.status != "running"
            or batch.pipeline_version != MASK_PIPELINE_VERSION
        ):
            raise ValueError(f"mask batch {resume_batch_id} is not resumable")
        return batch
    batch = MaskBatch(
        pipeline_version=MASK_PIPELINE_VERSION,
        backend=backend_name,
        dry_run=False,
        status="running",
        item_limit=limit,
        counts=_zero_counts(),
    )
    db.add(batch)
    db.commit()
    return batch


def run_mask_batch(
    db: Session,
    store: ObjectStore,
    *,
    backend: MattingBackend | None = None,
    dry_run: bool = True,
    limit: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    resume_batch_id: int | None = None,
    retry_failed: bool = False,
) -> BatchReport:
    """Settle up to ``limit`` clean images that have no mask result yet."""
    if resume_batch_id is not None and dry_run:
        raise ValueError("a dry run cannot resume a batch: it records nothing to resume")
    active = backend or get_matting_backend()
    if not dry_run and not active.publishes_masks:
        raise ValueError(f"backend {active.name!r} is not a mask model: dry run only")
    batch: MaskBatch | None = None
    if not dry_run:
        batch = _open_batch(
            db, resume_batch_id=resume_batch_id, backend_name=active.name, limit=limit
        )
        if resume_batch_id is not None and batch.item_limit is not None:
            limit = max(0, batch.item_limit - sum(batch.counts.values()))

    report = BatchReport(batch_id=batch.id if batch else None, dry_run=dry_run)
    cursor = batch.cursor_sha256 if batch else None
    remaining = limit
    while remaining is None or remaining > 0:
        size = page_size if remaining is None else min(page_size, remaining)
        page = _candidates(db, after=cursor, page_size=size, retry_failed=retry_failed)
        if not page:
            break
        for obj in page:
            item = _settle(db, store, obj, active, batch)
            report.items.append(item)
            report.counts[item.status] += 1
            cursor = obj.sha256
        if remaining is not None:
            remaining -= len(page)

    if batch is not None:
        batch.status = "done"
        batch.finished_at = dt.datetime.now(dt.UTC)
        db.commit()
    return report
