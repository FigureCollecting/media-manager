"""GAL3: re-judge the official gallery photos settled before their class was carried.

The gallery-capture lane records every photo it takes from MFC's official
pictures list as ``manufacturer_press`` (the spine's
``product_image.source_class``). The mask batch now carries that class onto
the gallery link it seeds (app.workers.mask_seed), and the mask policy lets
such a photo have a mask (app.workers.mask_policy, ``official_gallery``). A
photo the batch settled BEFORE that -- its link without a class, judged a
collector's photo, its result row ``ineligible`` -- is skipped by every later
batch (app.workers.mask_job.original_settled), so it needs this one step.

GIVEN the spine's gallery-list captures (the work list ``mask_batch sql
official-gallery-worklist`` prints), in two phases:

1. PROVENANCE. Each capture of source ``mfc`` and class ``manufacturer_press``
   names its original, its product and its URL; the gallery link with that
   product, URL and source on that original's image gains the class when it
   has none. A link with another class keeps it (``provenance_conflict``),
   a soft-deleted link is never touched (``link_deleted``). Every other
   capture is ignored (``not_gallery``): nothing but the gallery list is
   considered.
2. JUDGMENT. Each clean copy behind those images whose result at the current
   mask pipeline version is ``ineligible`` is judged again, in full
   (app.publication.mask_refusal: every image behind it, every origin, the
   store revocations, soft deletes). Allowed now, its ``ineligible`` row is
   deleted, so the next mask run (``mask_batch run`` over the same work list,
   or ``masks run --apply``) mints its mask; refused, it is left as it is and
   named with the reason. A copy with another result -- ``ok``, ``held``,
   ``empty``, ``failed`` or a ``withdrawn`` mask -- is never touched here.

DRY RUN BY DEFAULT: the same judgment, with the classes this run would record
held in memory (app.publication's ``link_classes``); it writes nothing, so it
holds no row lock a mask batch or a seed could wait on. Applied, each capture
and each copy is committed on its own. A capture or a copy that raises is
named (its exception's class only) and rolled back alone, in either mode.
Idempotent: a rerun records nothing new and re-judges what is still
ineligible.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import FigureGallery, ImageMask, PublicObject, PublicObjectImage
from ..models import Image as ImageModel
from ..publication import mask_refusal
from .mask import MASK_PIPELINE_VERSION
from .mask_policy import (
    MFC_SOURCE,
    OFFICIAL_GALLERY_CLASS,
    configured_official_gallery,
    source_key,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GalleryCapture:
    """One depiction the spine recorded: its original's sha, its product (the
    gallery link's figure_id), the URL it was captured from, its source and
    the source class the capture lane gave it."""

    original_sha256: str
    figure_id: str
    url: str
    source: str
    source_class: str | None


def _listed(capture: GalleryCapture) -> bool:
    return (
        capture.source_class == OFFICIAL_GALLERY_CLASS and source_key(capture.source) == MFC_SOURCE
    )


def _copies_behind(db: Session, image_id: int) -> list[str]:
    """The clean copies the image is behind (app.workers.mask_seed links
    every capture of a picture behind every copy of it)."""
    rows = db.execute(
        select(PublicObjectImage.sha256)
        .join(PublicObject, PublicObject.sha256 == PublicObjectImage.sha256)
        .where(PublicObjectImage.image_id == image_id, PublicObject.kind == "sanitized")
        .order_by(PublicObjectImage.sha256)
    ).scalars()
    return list(rows)


def _record(
    db: Session, capture: GalleryCapture, pending: set[int], dry_run: bool
) -> tuple[str, list[str]]:
    """Phase 1 for one capture: its outcome, and the copies to judge."""
    image_id = db.execute(
        select(ImageModel.id).where(ImageModel.sha256 == capture.original_sha256)
    ).scalar_one_or_none()
    if image_id is None:
        return "not_seeded", []
    link = db.execute(
        select(FigureGallery).where(
            FigureGallery.figure_id == capture.figure_id,
            FigureGallery.source_url == capture.url,
            FigureGallery.source == capture.source,
            FigureGallery.image_id == image_id,
        )
    ).scalar_one_or_none()
    if link is None:
        return "no_link", []
    if link.deleted_at is not None:
        return "link_deleted", []
    link_id = link.id
    current = OFFICIAL_GALLERY_CLASS if link_id in pending else link.source_class
    if current not in (None, OFFICIAL_GALLERY_CLASS):
        return "provenance_conflict", []
    copies = _copies_behind(db, image_id)
    if current is not None:
        return "provenance_present", copies
    if not dry_run:
        link.source_class = OFFICIAL_GALLERY_CLASS
        db.commit()
    pending.add(link_id)
    return "provenance_recorded", copies


def rejudge_official_gallery(
    db: Session, captures: Iterable[GalleryCapture], *, dry_run: bool = True
) -> dict[str, Any]:
    """Record the gallery-list class on the links that lack it, then clear the
    ``ineligible`` result of every copy behind them that may now carry a mask.
    Dry run unless ``dry_run=False``."""
    counts: Counter[str] = Counter()
    failed: list[dict[str, str]] = []
    pending: set[int] = set()
    touched: dict[str, None] = {}
    for capture in captures:
        counts["captures"] += 1
        if not _listed(capture):
            counts["not_gallery"] += 1
            continue
        try:
            outcome, copies = _record(db, capture, pending, dry_run)
        except Exception as exc:
            db.rollback()
            reason = type(exc).__name__
            logger.warning(
                "gallery_rejudge_failed",
                extra={"original_sha256": capture.original_sha256, "reason": reason},
            )
            failed.append({"original_sha256": capture.original_sha256, "reason": reason})
            continue
        counts[outcome] += 1
        touched.update(dict.fromkeys(copies))

    classes = dict.fromkeys(pending, OFFICIAL_GALLERY_CLASS)
    requeued: list[str] = []
    refused: list[dict[str, str]] = []
    for sha in touched:
        try:
            row = db.execute(
                select(ImageMask).where(
                    ImageMask.source_sha256 == sha,
                    ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
                )
            ).scalar_one_or_none()
            if row is None:
                counts["unsettled"] += 1
                continue
            if row.status != "ineligible":
                counts["settled"] += 1
                continue
            refusal = mask_refusal(db, sha, link_classes=classes)
            if refusal is not None:
                refused.append({"clean_sha256": sha, "reason": refusal})
                continue
            if not dry_run:
                db.delete(row)
                db.commit()
            requeued.append(sha)
        except Exception as exc:
            db.rollback()
            reason = type(exc).__name__
            logger.warning("gallery_rejudge_failed", extra={"source_sha256": sha, "reason": reason})
            failed.append({"clean_sha256": sha, "reason": reason})
    if dry_run:
        db.rollback()  # ends its read transaction; it wrote nothing
    keys = (
        "captures", "not_gallery", "not_seeded", "no_link", "link_deleted",
        "provenance_conflict", "provenance_present", "provenance_recorded",
        "unsettled", "settled",
    )  # fmt: skip
    return {
        "dry_run": dry_run,
        "official_gallery": configured_official_gallery(),
        "counts": {k: counts[k] for k in keys},
        "requeued": requeued,
        "still_refused": refused,
        "failed": failed,
    }
