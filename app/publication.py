"""Whether a registered public object may be served NOW (MASK-1 revocation).

``public_objects`` registers what is ever addressable at ``/d/<sha256>``;
this module decides, on every request, whether it still is. A clean copy is
judged over EVERY image behind it: the image its version belongs to and each
other capture of the same picture (``public_object_images``), linked by the
pixels the copy was minted from, by the pixels the copy itself decodes to,
or by the copy's very bytes, in either order and whichever route captured
it -- the seed or the gallery ingest (app.workers.mask_seed.link_capture):

- a clean copy (``sanitized``) is served while its version and every image
  behind it are not soft-deleted. A row with no version is never served.
- a mask is served while some ``ok`` image_masks row names it whose clean
  copy is live (as above) and still mask-eligible from the CURRENT origins
  of every image behind it (app.workers.mask_policy); an image behind it
  with no live origin refuses it (``no_origin``), as it would on its own.
  A mask shared by several clean copies (different pictures, the same mask)
  stays while any of them qualifies.
- a ``matted`` cut-out is our derivative work on the picture, like a mask,
  so it is judged over the clean copy it was cut from (its version's
  ``derived_from_version``): served while its own version is live and that
  copy may carry a mask -- live AND mask-eligible over every image behind
  it. One not cut from a registered clean copy is never served. Nothing
  registers one yet; the rule is here so that whatever does cannot skip the
  per-source permission split.

ALL, NOT ANY, for the images behind one clean copy: the address names the
picture, so a soft delete (a takedown) under any capture of a picture
reaches every clean copy of it -- including a capture whose bytes ARE the
copy (our clean copy captured again, a lossy JPEG one included, before or
after it was minted) -- and the same picture captured as an item image and
as a collector's photo is a collector's photo somewhere, in whichever order
the two were captured. The cost is deliberate: soft-deleting a duplicate
capture takes the shared copy offline for the capture that remains.

So soft-deleting an image or a version revokes its derivative and its masks;
a mask also goes offline the moment any image behind it gains an ineligible
origin (a collector's photo) or every origin is gone, or a source of one of
its origins is revoked (MASK-G: ``store_revocations``, read on every
request), and stays offline until ``reevaluate_masks`` restores its row
(app.workers.mask_job). A revocation never touches a clean copy: display is
not decided here.

What a phone already cached under the immutable address cannot be recalled.
Nor, yet, can an edge copy: a 200 is ``public, max-age=31536000, immutable``
and the fc-infra ADDR-1 edge caches it (ignoring a request's no-cache), so a
withdrawal or a soft delete stops at this origin. Once the images tunnel or
cache rule lands, both must also purge the edge (a purge by URL) -- a
follow-up, not done here.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    FigureGallery,
    ImageMask,
    ImageVersion,
    MaskStoreRevocation,
    PublicObject,
    PublicObjectImage,
)
from .models import Image as ImageModel
from .workers.mask_policy import (
    ImageOrigin,
    Revoked,
    StoreGrants,
    configured_revocations,
    mask_eligibility,
    source_key,
)


def store_revocations(db: Session, configured: Iterable[str] | None = None) -> Revoked:
    """Every revoked source (MASK-G): MASK_STORE_REVOKED -- or ``configured``,
    the origin's own settings -- plus every revocation ``masks revoke-store``
    persisted. Read on every judgment, so a revocation takes effect at the
    origin the moment it is committed."""
    names = configured_revocations() if configured is None else frozenset(configured)
    persisted = db.execute(select(MaskStoreRevocation.source)).scalars()
    return frozenset(source_key(s) for s in (*names, *persisted))


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
    None while it is not live (images_behind).

    POOLED, BUT FAILING CLOSED: an image behind it with no live origin at
    all (no gallery row, or every row soft-deleted) stands in as an origin
    with no URL, so it refuses the mask as ``no_origin`` -- exactly as it
    would on its own."""
    ids = images_behind(db, sha256)
    if ids is None:
        return None
    judged: list[ImageOrigin] = []
    for image_id in ids:
        judged.extend(gallery_origins(db, image_id) or [ImageOrigin(url=None)])
    return judged


def mask_refusal(
    db: Session,
    source_sha256: str,
    store_grants: StoreGrants | None = None,
    revoked: Revoked | None = None,
) -> str | None:
    """None while the clean copy ``source_sha256`` may carry a published
    mask; otherwise why not: ``deleted`` or the eligibility refusal.
    ``revoked`` defaults to ``store_revocations(db)``."""
    origins = judged_origins(db, source_sha256)
    if origins is None:
        return "deleted"
    if revoked is None:
        revoked = store_revocations(db)
    verdict = mask_eligibility(origins, store_grants, revoked)
    return None if verdict.eligible else verdict.reason


def cut_from(db: Session, sha256: str) -> str | None:
    """The clean copy the derivative ``sha256`` was cut from: the registered
    clean copy of the version its own version derives from (same image).
    None while its version is soft-deleted or derives from no clean copy."""
    own = db.execute(
        select(ImageVersion.image_id, ImageVersion.derived_from_version)
        .join(PublicObject, PublicObject.image_version_id == ImageVersion.id)
        .where(PublicObject.sha256 == sha256, ImageVersion.deleted_at.is_(None))
    ).one_or_none()
    if own is None:
        return None
    return db.execute(
        select(PublicObject.sha256)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .where(
            ImageVersion.image_id == own.image_id,
            ImageVersion.version_no == own.derived_from_version,
            PublicObject.kind == "sanitized",
        )
    ).scalar_one_or_none()


def servable(
    db: Session,
    obj: PublicObject,
    store_grants: StoreGrants | None = None,
    revoked: Revoked | None = None,
) -> bool:
    """Whether ``obj`` may be served now. ``revoked`` defaults to
    ``store_revocations(db)``, read per judgment."""
    if obj.kind == "sanitized":
        return images_behind(db, obj.sha256) is not None
    if obj.kind == "mask":
        sources = db.execute(
            select(ImageMask.source_sha256).where(
                ImageMask.mask_sha256 == obj.sha256, ImageMask.status == "ok"
            )
        ).scalars()
        return any(mask_refusal(db, s, store_grants, revoked) is None for s in sources)
    # 'matted' -- and any kind not named above -- is derivative work cut from
    # a clean copy: served while its own version is live and that copy may
    # carry a mask, judged over every image behind it.
    source = cut_from(db, obj.sha256)
    return source is not None and mask_refusal(db, source, store_grants, revoked) is None
