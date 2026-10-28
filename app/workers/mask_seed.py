"""Seeding clean images into the derivative lane, and the one gate every
public object passes: ``register_public_object``.

``make_clean_copy`` is the MINIMAL clean copy the seeded sample and local
runs need: EXIF orientation baked in, a fresh pixel copy (no EXIF, XMP, ICC
or text chunks survive), re-encoded at UNCHANGED resolution -- JPEG quality
95 with 4:4:4 chroma for an opaque image, lossless PNG when the image has
transparency. The full sanitize contract (byte and pixel caps before decode,
verify-then-reopen, magic-byte checks, the pinned size multiple) is the
image track's I8 unit and supersedes this when it lands.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..hashing import compute_phash, detect_mime
from ..models import FigureGallery, ImageVersion, PublicObject, PublicObjectImage
from ..models import Image as ImageModel
from ..object_store import ObjectStore, derived_key
from .image_limits import check_image
from .mask_policy import ImageOrigin

PUBLIC_KINDS = ("sanitized", "matted", "mask")

#: The pipeline that makes ``make_clean_copy``'s clean copies, as the
#: media.v1 replay reports it (DerivativeReport.pipeline_version). Bump it
#: when the encoding changes: the spine keeps a new version's derivative
#: beside the old one (app.spine.replay). The full sanitize contract (I8)
#: will report under its own version.
CLEAN_PIPELINE_VERSION = "mm-seed-clean-v1"


def _picture(original: bytes) -> Image.Image:
    """The pixels ``original`` shows -- EXIF orientation baked in, RGBA when
    some pixel IS transparent, else RGB -- as a fresh image (no EXIF, XMP,
    ICC or text chunks): exactly what its clean copy encodes. An alpha
    channel whose every pixel is opaque, or a colour key no pixel has, is no
    transparency: such a file shows the same picture as its RGB twin. The
    colour under a FULLY transparent pixel shows nothing (lossless WebP and
    PNG optimisers rewrite it) and is zeroed: two files differing only
    there show one picture, and the clean copy carries no hidden pixels."""
    check_image(original)  # MM-CAPS-1: the header-read limits, before any pixel is decoded
    with Image.open(io.BytesIO(original)) as src:
        src.load()
        oriented = ImageOps.exif_transpose(src) or src
        may_have_alpha = oriented.mode in ("RGBA", "LA", "PA") or "transparency" in oriented.info
        pixels = oriented.convert("RGBA" if may_have_alpha else "RGB")
        if pixels.mode == "RGBA" and pixels.getchannel("A").getextrema()[0] == 255:
            pixels = pixels.convert("RGB")
        arr = np.array(pixels)
        if pixels.mode == "RGBA":
            arr[arr[..., 3] == 0] = 0
        return Image.fromarray(arr)


def _digest(picture: Image.Image) -> str:
    h = hashlib.sha256(f"{picture.mode} {picture.width}x{picture.height}\n".encode())
    h.update(picture.tobytes())
    return h.hexdigest()


def pixel_digest(original: bytes) -> str:
    """The PIXEL DIGEST of ``original``: the sha-256 of the mode, the size and
    the decoded bytes after EXIF orientation (``_picture``: the colour under
    a fully transparent pixel zeroed). Two captures with the same
    pixels share it whatever their bytes (metadata, a lossless re-save, our
    own lossless PNG copy captured again). Our JPEG copy decodes to other
    pixels than its original, so it has a digest of its own, recorded on the
    copy (``own_pixel_sha256``). ``link_capture`` links captures by
    both."""
    return _digest(_picture(original))


def _encode(picture: Image.Image, original: bytes) -> tuple[bytes, str]:
    # The clean copy must never be byte-identical to its original (the
    # original's address must never become public): if the first encoding
    # reproduces the original exactly -- an original that was itself written
    # by the same encoder -- the other setting is used, which never writes
    # the same bytes: for a PNG another zlib level class, whose header
    # differs (levels 8 and 9 often write the same bytes), for a JPEG other
    # quantisation tables.
    has_alpha = picture.mode == "RGBA"

    def save(setting: int) -> bytes:
        out = io.BytesIO()
        if has_alpha:
            picture.save(out, format="PNG", compress_level=setting)
        else:
            picture.save(out, format="JPEG", quality=setting, subsampling=0)
        return out.getvalue()

    first, other = (9, 1) if has_alpha else (95, 94)
    data = save(first)
    if data == original:
        data = save(other)
    return data, "image/png" if has_alpha else "image/jpeg"


def make_clean_copy(original: bytes) -> tuple[bytes, str, int, int]:
    """``(bytes, mime, width, height)`` of a clean copy of ``original``."""
    picture = _picture(original)
    data, mime = _encode(picture, original)
    return data, mime, picture.width, picture.height


def register_public_object(
    db: Session,
    *,
    sha256: str,
    kind: str,
    mime: str,
    size: int,
    width: int,
    height: int,
    storage_key: str,
    image_version_id: int | None = None,
    pixel_sha256: str | None = None,
    own_pixel_sha256: str | None = None,
) -> PublicObject:
    """Make ``sha256`` addressable at ``/d/<sha256>``. Idempotent on the sha.

    Refuses an original's bytes: the public origin must never be able to
    name an original. The one exception is our own clean copy captured
    BEFORE it was minted: a clean copy (``sanitized``) of one image whose
    bytes ANOTHER image's original already has. That original is linked
    behind the copy, as is the copy's own image (``public_object_images``),
    and both are judged with it (app.publication). A clean copy carries both
    pixel digests, the keys later captures of the picture are linked by; no
    other kind carries either.
    """
    if kind not in PUBLIC_KINDS:
        raise ValueError(f"not a public object kind: {kind!r}")
    clean = kind == "sanitized"
    if (pixel_sha256 is not None) != clean or (own_pixel_sha256 is not None) != clean:
        raise ValueError("a clean copy, and only a clean copy, carries a pixel digest")
    owner = None
    if clean and image_version_id is not None:
        owner = db.execute(
            select(ImageVersion.image_id).where(ImageVersion.id == image_version_id)
        ).scalar_one_or_none()
    captured = db.execute(
        select(ImageModel.id).where(ImageModel.sha256 == sha256)
    ).scalar_one_or_none()
    if captured is not None and (owner is None or captured == owner):
        raise ValueError("refusing to publish an original's bytes")
    obj = db.get(PublicObject, sha256)
    if obj is None:
        obj = PublicObject(
            sha256=sha256,
            kind=kind,
            mime=mime,
            bytes=size,
            width=width,
            height=height,
            storage_key=storage_key,
            image_version_id=image_version_id,
            pixel_sha256=pixel_sha256,
            own_pixel_sha256=own_pixel_sha256,
        )
        db.add(obj)
        db.flush()
    _link(db, {sha256}, {i for i in (owner, captured) if i is not None})
    return obj


def _link(db: Session, copies: set[str], images: set[int]) -> None:
    """Every image in ``images`` behind every clean copy in ``copies``."""
    for sha in sorted(copies):
        for image_id in sorted(images):
            if db.get(PublicObjectImage, (sha, image_id)) is None:
                db.add(PublicObjectImage(sha256=sha, image_id=image_id))
    db.flush()


def _copies_showing(db: Session, digest: str, sha256: str) -> list[PublicObject]:
    """The registered clean copies of the picture with pixel digest
    ``digest``: minted from those pixels (``pixel_sha256``) or showing them
    (``own_pixel_sha256``) -- plus the clean copy whose bytes are
    ``sha256``, whatever a decoder makes of them. The copy minted from
    ``digest`` comes first."""
    keys = (
        PublicObject.pixel_sha256 == digest,
        PublicObject.own_pixel_sha256 == digest,
        PublicObject.sha256 == sha256,
    )
    found = db.execute(
        select(PublicObject).where(PublicObject.kind == "sanitized", or_(*keys))
    ).scalars()
    return sorted(found, key=lambda c: (c.pixel_sha256 != digest, c.sha256))


def _join_picture(db: Session, copies: set[str], images: set[int]) -> None:
    """Link every capture of the picture behind every clean copy of it: grow
    ``copies`` and ``images`` through the links already recorded (a copy's
    images, an image's copies) until nothing is added, then link them all.
    So a copy is judged over every capture of its picture whichever order
    they were captured in -- a capture linked to two copies joins them."""
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
            break
        images, copies = images | more_images, copies | more_copies
    _link(db, copies, images)


def _captures_showing(db: Session, *digests: str) -> set[int]:
    """The images whose recorded pixel digest is one of ``digests``."""
    return set(
        db.execute(select(ImageModel.id).where(ImageModel.pixel_sha256.in_(digests))).scalars()
    )


def link_capture(db: Session, image: ImageModel, digest: str) -> list[PublicObject]:
    """ONE PICTURE, EVERY CAPTURE, for one capture of a picture whatever
    route captured it -- ``seed_clean_image`` or the gallery ingest
    (app.workers.tasks.ingest_gallery_images), each calling this for every
    image it records. ``digest`` is the capture's pixel digest
    (``pixel_digest``).

    Records the digest on the image -- the key a clean copy minted LATER
    finds it by (``seed_clean_image``, photo first) -- and links the image
    behind every registered clean copy of its picture: minted from those
    pixels (``pixel_sha256``), decoding to them (``own_pixel_sha256``), or
    whose very bytes it has. Then every capture of the picture is behind
    every copy of it (``_join_picture``). Returns those copies, the one
    minted from these pixels first; none while the picture has no copy.

    A capture whose bytes ARE one of our masks or cut-outs (registered
    before it: the registration refuses an original's bytes) is linked
    behind that object too, which is served only while it is live
    (app.publication)."""
    image.pixel_sha256 = digest
    if db.get(PublicObject, image.sha256) is not None:
        _link(db, {image.sha256}, {image.id})
    copies = _copies_showing(db, digest, image.sha256)
    _join_picture(db, {c.sha256 for c in copies}, {image.id})
    return copies


class PictureKeys(NamedTuple):
    """What ``seed_clean_image`` finds an original's picture by, from its
    bytes alone: its pixel digest, the sha of the clean copy it would mint,
    and that copy's own pixel digest."""

    digest: str
    clean_sha256: str
    own_digest: str


def picture_keys(original: bytes) -> PictureKeys:
    """``original``'s ``PictureKeys``, computed exactly as the seed computes
    them (MASK-PERF-2: the pipelined batch claims them before the seed)."""
    picture = _picture(original)
    clean, _mime = _encode(picture, original)
    return PictureKeys(_digest(picture), hashlib.sha256(clean).hexdigest(), pixel_digest(clean))


def seed_footprint(db: Session, *, shas: set[str], digests: set[str]) -> set[str]:
    """The keys of every row seeding and settling an image may read or write
    (MASK-PERF-2): its own byte shas (``shas``: the original's, the clean
    copy's) and pixel digests (``digests``), and the sha of every image and
    public object they lead to -- an image with one of those shas or pixel
    digests, an object with one of those shas or either digest -- grown
    through every capture-to-copy link to its fixed point, as
    ``link_capture`` and ``_join_picture`` grow a picture. Two images whose
    footprints do not meet cannot see each other's rows; read now, it may
    miss what an image still in flight is about to link (the pipeline joins
    those by their own footprints)."""
    images = set(
        db.execute(
            select(ImageModel.sha256).where(
                or_(ImageModel.sha256.in_(shas), ImageModel.pixel_sha256.in_(digests))
            )
        ).scalars()
    )
    objects = set(
        db.execute(
            select(PublicObject.sha256).where(
                or_(
                    PublicObject.sha256.in_(shas),
                    PublicObject.pixel_sha256.in_(digests),
                    PublicObject.own_pixel_sha256.in_(digests),
                )
            )
        ).scalars()
    )
    while images or objects:
        links = db.execute(
            select(PublicObjectImage.sha256, ImageModel.sha256)
            .join(ImageModel, ImageModel.id == PublicObjectImage.image_id)
            .where(or_(PublicObjectImage.sha256.in_(objects), ImageModel.sha256.in_(images)))
        ).all()
        more_objects = {obj for obj, _image in links} - objects
        more_images = {image for _obj, image in links} - images
        if not more_objects and not more_images:
            break
        objects |= more_objects
        images |= more_images
    return shas | digests | images | objects


def _original_key(sha256: str, mime: str) -> str:
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(mime, "bin")
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}.{ext}"


def seed_clean_image(
    db: Session,
    store: ObjectStore,
    original: bytes,
    *,
    origins: Sequence[ImageOrigin],
    figure_id: str,
) -> PublicObject:
    """Register ``original`` (stored, never public), its clean copy (public,
    kind ``sanitized``), the image as one behind that copy, and one gallery
    link per origin. Idempotent; returns the clean copy minted from these
    pixels if there is one, else another clean copy of the picture.

    ONE PICTURE, EVERY CAPTURE. An original is a capture of a registered
    clean copy's picture when it has the pixels the copy was minted from
    (``pixel_sha256``), the pixels the copy itself decodes to
    (``own_pixel_sha256``: our JPEG copy captured again, re-saved or with
    metadata added), or the copy's very bytes (whatever a decoder makes of
    them). Such an original gets no copy of its own. When none matches, a
    clean copy is minted, and any original whose bytes it turns out to be,
    or whose pixels it shows, is a capture of it too (photo first) -- one
    the gallery ingest recorded included, by the pixel digest it recorded
    (``link_capture``). Every capture of a picture is then linked behind
    every clean copy of it (``public_object_images``), in whichever order
    they arrive, so each copy and its mask are judged over all of them: a
    soft delete or an ineligible origin under any capture -- the one whose
    bytes ARE the copy included -- reaches every copy (app.publication)."""
    check_image(original)  # MM-CAPS-1: refused before anything is hashed, stored or decoded
    orig_sha = hashlib.sha256(original).hexdigest()
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == orig_sha)).scalar_one_or_none()
    if image is None:
        mime = detect_mime(original)
        with Image.open(io.BytesIO(original)) as probe:
            width, height = probe.size
        key = _original_key(orig_sha, mime)
        store.put(key, original, mime)
        image = ImageModel(
            sha256=orig_sha,
            phash=compute_phash(original),
            mime=mime,
            width=width,
            height=height,
            bytes=len(original),
            storage_key=key,
        )
        db.add(image)
        db.flush()
        db.add(
            ImageVersion(
                image_id=image.id,
                version_no=1,
                transform_spec={},
                mime=mime,
                width=width,
                height=height,
                bytes=len(original),
                storage_key=key,
                visibility="private",
                age_rating=0,
            )
        )
        db.flush()

    picture = _picture(original)
    digest = _digest(picture)
    copies = link_capture(db, image, digest)
    if copies:
        obj = copies[0]
    else:
        clean, clean_mime = _encode(picture, original)
        clean_sha = hashlib.sha256(clean).hexdigest()
        own = pixel_digest(clean)
        cw, ch = picture.width, picture.height
        key = derived_key(clean_sha, clean_mime)
        store.put(key, clean, clean_mime)
        version = ImageVersion(
            image_id=image.id,
            version_no=2,
            derived_from_version=1,
            transform_spec={"derive_kind": "sanitized"},
            mime=clean_mime,
            width=cw,
            height=ch,
            bytes=len(clean),
            storage_key=key,
            visibility="public",
            age_rating=0,
        )
        db.add(version)
        db.flush()
        obj = register_public_object(
            db,
            sha256=clean_sha,
            kind="sanitized",
            mime=clean_mime,
            size=len(clean),
            width=cw,
            height=ch,
            storage_key=key,
            image_version_id=version.id,
            pixel_sha256=digest,
            own_pixel_sha256=own,
        )
        # Photo first: every capture recorded earlier with these pixels or
        # the copy's own, whatever route recorded it, joins the new copy.
        _join_picture(db, {obj.sha256}, {image.id, *_captures_showing(db, digest, own)})

    for origin in origins:
        url = origin.url or ""
        linked = db.execute(
            select(FigureGallery).where(
                FigureGallery.figure_id == figure_id, FigureGallery.source_url == url
            )
        ).scalar_one_or_none()
        if linked is None:
            db.add(
                FigureGallery(
                    figure_id=figure_id,
                    source_url=url,
                    image_id=image.id,
                    position=0,
                    source=origin.source or "unknown",
                    source_class=origin.source_class,
                )
            )
        elif linked.source_class is None and origin.source_class is not None:
            # GAL3: a class beats none and is never replaced (the spine's merge rule).
            linked.source_class = origin.source_class
    db.commit()
    return obj
