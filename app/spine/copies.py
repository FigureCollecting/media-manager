"""SHOW1: the clean copy a depiction of one original is reported with, read
per original from the media DB -- the replay's rule (app.spine.replay
load_plan), without the replay's whole-DB index.

THE RULE, load_plan's: the lowest-addressed LIVE clean copy minted from the
original's image, else the lowest live copy the image is linked behind;
live as app.publication.images_behind judges it (its version and every
image behind it not soft-deleted). None when the original is not in the
media DB, or no live copy shows it (a soft delete, a takedown): then
nothing is reported for it.

WHY THE RULE AND NOT THE SEED'S ANSWER. ``seed_clean_image`` returns the
copy minted from THESE pixels if there is one, else another copy of the
picture, which need not be the one the replay picks; and the spine keeps
the FIRST copy registered for an original (fc-aggregation
derivative-writer.ts: ON CONFLICT DO NOTHING). Reporting another copy first
would leave the replay's later reports, and the masks it reports over the
copy it picked, warned and dropped.
"""

from __future__ import annotations

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ..models import Image as ImageModel
from ..models import ImageMask, ImageVersion, PublicObject, PublicObjectImage
from ..publication import images_behind
from .replay import _appearance

__all__ = ["measured_appearance", "reported_copy"]


def _first_live(db: Session, candidates: list[str]) -> str | None:
    for copy in sorted(candidates):
        if images_behind(db, copy) is not None:
            return copy
    return None


def reported_copy(db: Session, origin_sha256: str) -> str | None:
    """The clean copy the replay reports ``origin_sha256``'s depictions with."""
    image_id = db.execute(
        select(ImageModel.id).where(ImageModel.sha256 == origin_sha256)
    ).scalar_one_or_none()
    if image_id is None:
        return None
    owned = db.execute(
        select(PublicObject.sha256)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .where(PublicObject.kind == "sanitized", ImageVersion.image_id == image_id)
    ).scalars()
    minted = _first_live(db, list(owned))
    if minted is not None:
        return minted
    behind = db.execute(
        select(PublicObjectImage.sha256)
        .join(
            PublicObject,
            and_(PublicObject.sha256 == PublicObjectImage.sha256, PublicObject.kind == "sanitized"),
        )
        .where(PublicObjectImage.image_id == image_id)
    ).scalars()
    return _first_live(db, list(behind))


def measured_appearance(db: Session, copy_sha256: str) -> ImageMask | None:
    """The mask row whose PLAIN appearance the replay sends for ``copy_sha256``
    (the current mask version's when it has one, else the latest measured),
    None when no mask row measured one."""
    rows = list(
        db.execute(
            select(ImageMask).where(ImageMask.source_sha256 == copy_sha256).order_by(ImageMask.id)
        ).scalars()
    )
    return _appearance(rows)
