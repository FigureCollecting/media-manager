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

from PIL import Image, ImageOps
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..hashing import compute_phash, detect_mime
from ..models import FigureGallery, ImageVersion, PublicObject, PublicObjectImage
from ..models import Image as ImageModel
from ..object_store import ObjectStore, derived_key
from .mask_policy import ImageOrigin

PUBLIC_KINDS = ("sanitized", "matted", "mask")


def _picture(original: bytes) -> Image.Image:
    """The pixels ``original`` shows -- EXIF orientation baked in, RGBA when
    it has any transparency, else RGB -- as a fresh image (no EXIF, XMP, ICC
    or text chunks): exactly what its clean copy encodes."""
    with Image.open(io.BytesIO(original)) as src:
        src.load()
        oriented = ImageOps.exif_transpose(src) or src
        has_alpha = oriented.mode in ("RGBA", "LA", "PA") or "transparency" in oriented.info
        mode = "RGBA" if has_alpha else "RGB"
        pixels = oriented.convert(mode)
        return Image.frombytes(mode, pixels.size, pixels.tobytes())


def _digest(picture: Image.Image) -> str:
    h = hashlib.sha256(f"{picture.mode} {picture.width}x{picture.height}\n".encode())
    h.update(picture.tobytes())
    return h.hexdigest()


def pixel_digest(original: bytes) -> str:
    """The PIXEL DIGEST of ``original``: the sha-256 of the mode, the size and
    the decoded bytes after EXIF orientation. Two captures with the same
    pixels share it whatever their bytes (metadata, a lossless re-save, our
    own lossless PNG copy captured again). Our JPEG copy decodes to other
    pixels than its original, so it has a digest of its own, recorded on the
    copy (``own_pixel_sha256``). ``seed_clean_image`` links captures by
    both."""
    return _digest(_picture(original))


def _encode(picture: Image.Image, original: bytes) -> tuple[bytes, str]:
    # The clean copy must never be byte-identical to its original (the
    # original's address must never become public): if the first encoding
    # happens to reproduce the original exactly -- an original that was
    # itself written by the same encoder -- the next setting is used.
    has_alpha = picture.mode == "RGBA"
    mime = "image/png" if has_alpha else "image/jpeg"
    data = b""
    for setting in (9, 8) if has_alpha else (95, 94):
        out = io.BytesIO()
        if has_alpha:
            picture.save(out, format="PNG", compress_level=setting)
        else:
            picture.save(out, format="JPEG", quality=setting, subsampling=0)
        data = out.getvalue()
        if data != original:
            break
    return data, mime


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


def _copies_showing(db: Session, digest: str, sha256: str | None = None) -> list[PublicObject]:
    """The registered clean copies of the picture with pixel digest
    ``digest``: minted from those pixels (``pixel_sha256``) or showing them
    (``own_pixel_sha256``) -- plus the clean copy whose bytes are
    ``sha256``, whatever a decoder makes of them. The copy minted from
    ``digest`` comes first."""
    keys = [PublicObject.pixel_sha256 == digest, PublicObject.own_pixel_sha256 == digest]
    if sha256 is not None:
        keys.append(PublicObject.sha256 == sha256)
    found = db.execute(
        select(PublicObject).where(PublicObject.kind == "sanitized", or_(*keys))
    ).scalars()
    return sorted(found, key=lambda c: (c.pixel_sha256 != digest, c.sha256))


def _join_picture(db: Session, copies: set[str], images: set[int]) -> None:
    """Link every capture of the picture behind every clean copy of it: grow
    ``copies`` and ``images`` through the links already recorded (a copy's
    images, an image's copies) until nothing is added, then link them all.
    So a copy is judged over every capture of its picture whichever order
    they were seeded in -- a capture linked to two copies joins them."""
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
    or whose pixels it shows, is a capture of it too (photo first). Every
    capture of a picture is then linked behind every clean copy of it
    (``public_object_images``), in whichever order they arrive, so each copy
    and its mask are judged over all of them: a soft delete or an ineligible
    origin under any capture -- the one whose bytes ARE the copy included --
    reaches every copy (app.publication)."""
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
    copies = _copies_showing(db, digest, orig_sha)
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
        copies = [obj, *_copies_showing(db, own)]
    _join_picture(db, {c.sha256 for c in copies}, {image.id})

    for origin in origins:
        url = origin.url or ""
        linked = db.execute(
            select(FigureGallery.id).where(
                FigureGallery.figure_id == figure_id, FigureGallery.source_url == url
            )
        ).first()
        if linked is None:
            db.add(
                FigureGallery(
                    figure_id=figure_id,
                    source_url=url,
                    image_id=image.id,
                    position=0,
                    source=origin.source or "unknown",
                )
            )
    db.commit()
    return obj
