"""Whether a registered public object may be served NOW (MASK-1 revocation).

``public_objects`` registers what is ever addressable at ``/d/<sha256>``;
this module decides, on every request, whether it still is:

- a clean derivative (``sanitized``/``matted``) is served while its image
  version and its image are not soft-deleted. A row with no version is never
  served.
- a mask is served while some ``ok`` image_masks row names it whose clean
  image is live (as above) and still mask-eligible from its CURRENT origins
  (app.workers.mask_policy). A mask shared by several images stays while any
  of them qualifies.

So soft-deleting an image or a version revokes its derivative and its masks;
a mask also goes offline the moment its image gains an ineligible origin (a
collector's photo) or loses every origin, and stays offline until
``reevaluate_masks`` restores its row (app.workers.mask_job). What a phone
already cached under the immutable address cannot be recalled.
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session

from .models import FigureGallery, ImageMask, ImageVersion, PublicObject
from .models import Image as ImageModel
from .workers.mask_policy import ImageOrigin, StoreGrants, mask_eligibility


def gallery_origins(db: Session, image_id: int) -> list[ImageOrigin]:
    """The image's current origins: its gallery links that are not deleted."""
    rows = db.execute(
        select(FigureGallery)
        .where(FigureGallery.image_id == image_id, FigureGallery.deleted_at.is_(None))
        .order_by(FigureGallery.id)
    ).scalars()
    return [ImageOrigin(url=g.source_url, source=g.source) for g in rows]


def _live_image_id(db: Session, version: ColumnElement[bool]) -> int | None:
    """The image behind the version ``version`` selects, while neither the
    version nor the image is soft-deleted; None otherwise (a NULL version
    selects nothing)."""
    return db.execute(
        select(ImageVersion.image_id)
        .join(ImageModel, ImageModel.id == ImageVersion.image_id)
        .where(version, ImageVersion.deleted_at.is_(None), ImageModel.deleted_at.is_(None))
    ).scalar_one_or_none()


def mask_refusal(
    db: Session, source_sha256: str, store_grants: StoreGrants | None = None
) -> str | None:
    """None while the clean image ``source_sha256`` may carry a published
    mask; otherwise why not: ``deleted`` or the eligibility refusal."""
    clean_version = (
        select(PublicObject.image_version_id)
        .where(PublicObject.sha256 == source_sha256)
        .scalar_subquery()
    )
    image_id = _live_image_id(db, ImageVersion.id == clean_version)
    if image_id is None:
        return "deleted"
    verdict = mask_eligibility(gallery_origins(db, image_id), store_grants)
    return None if verdict.eligible else verdict.reason


def servable(db: Session, obj: PublicObject, store_grants: StoreGrants | None = None) -> bool:
    if obj.kind == "mask":
        sources = db.execute(
            select(ImageMask.source_sha256).where(
                ImageMask.mask_sha256 == obj.sha256, ImageMask.status == "ok"
            )
        ).scalars()
        return any(mask_refusal(db, s, store_grants) is None for s in sources)
    return _live_image_id(db, ImageVersion.id == obj.image_version_id) is not None
