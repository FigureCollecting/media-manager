"""Whether a registered public object may be served NOW (MASK-1 revocation).

``public_objects`` registers what is ever addressable at ``/d/<sha256>``;
this module decides, on every request, whether it still is. A clean copy is
judged over EVERY image behind it: the image its version belongs to and each
pixel-identical capture linked to it (``public_object_images``):

- a clean copy (``sanitized``) is served while its version and every image
  behind it are not soft-deleted. A row with no version is never served.
- a mask is served while some ``ok`` image_masks row names it whose clean
  copy is live (as above) and still mask-eligible from the CURRENT origins
  of every image behind it (app.workers.mask_policy). A mask shared by
  several clean copies (different pictures, the same mask) stays while any
  of them qualifies.
- a ``matted`` cut-out is our derivative work on the picture, like a mask,
  so it is judged like a mask's clean copy: live AND mask-eligible. Nothing
  registers one yet; the rule is here so that whatever does cannot skip the
  per-source permission split.

ALL, NOT ANY, for the images behind one clean copy: the address names the
pixels, so a soft delete (a takedown) under any capture of a picture
reaches the copy every capture shares, and the same picture captured as an
item image and as a collector's photo is a collector's photo somewhere --
whichever was seeded first. The cost is deliberate: soft-deleting a
pixel-identical duplicate takes the shared copy offline for the capture
that remains.

So soft-deleting an image or a version revokes its derivative and its masks;
a mask also goes offline the moment any image behind it gains an ineligible
origin (a collector's photo) or every origin is gone, and stays offline
until ``reevaluate_masks`` restores its row (app.workers.mask_job).

What a phone already cached under the immutable address cannot be recalled.
Nor, yet, can an edge copy: a 200 is ``public, max-age=31536000, immutable``
and the fc-infra ADDR-1 edge caches it (ignoring a request's no-cache), so a
withdrawal or a soft delete stops at this origin. Once the images tunnel or
cache rule lands, both must also purge the edge (a purge by URL) -- a
follow-up, not done here.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import FigureGallery, ImageMask, ImageVersion, PublicObject, PublicObjectImage
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


def images_behind(db: Session, sha256: str) -> list[int] | None:
    """The images behind the public object ``sha256`` -- its version's image
    and every image linked to it -- while its version and ALL of them are
    live; None otherwise (an unknown sha, a row with no version, a version
    or any one image soft-deleted)."""
    owner = db.execute(
        select(ImageVersion.image_id)
        .join(PublicObject, PublicObject.image_version_id == ImageVersion.id)
        .where(PublicObject.sha256 == sha256, ImageVersion.deleted_at.is_(None))
    ).scalar_one_or_none()
    if owner is None:
        return None
    linked = db.execute(
        select(PublicObjectImage.image_id).where(PublicObjectImage.sha256 == sha256)
    ).scalars()
    ids = sorted({owner, *linked})
    live = db.execute(
        select(ImageModel.id).where(ImageModel.id.in_(ids), ImageModel.deleted_at.is_(None))
    ).scalars()
    return ids if len(list(live)) == len(ids) else None


def judged_origins(db: Session, sha256: str) -> list[ImageOrigin] | None:
    """The current origins of every image behind ``sha256``, in image order;
    None while it is not live (images_behind)."""
    ids = images_behind(db, sha256)
    if ids is None:
        return None
    return [origin for image_id in ids for origin in gallery_origins(db, image_id)]


def mask_refusal(
    db: Session, source_sha256: str, store_grants: StoreGrants | None = None
) -> str | None:
    """None while the clean copy ``source_sha256`` may carry a published
    mask; otherwise why not: ``deleted`` or the eligibility refusal."""
    origins = judged_origins(db, source_sha256)
    if origins is None:
        return "deleted"
    verdict = mask_eligibility(origins, store_grants)
    return None if verdict.eligible else verdict.reason


def servable(db: Session, obj: PublicObject, store_grants: StoreGrants | None = None) -> bool:
    if obj.kind == "mask":
        sources = db.execute(
            select(ImageMask.source_sha256).where(
                ImageMask.mask_sha256 == obj.sha256, ImageMask.status == "ok"
            )
        ).scalars()
        return any(mask_refusal(db, s, store_grants) is None for s in sources)
    if obj.kind == "sanitized":
        return images_behind(db, obj.sha256) is not None
    # 'matted' -- and any kind not named above -- is derivative work: live
    # and mask-eligible, like a mask's clean copy.
    return mask_refusal(db, obj.sha256, store_grants) is None
