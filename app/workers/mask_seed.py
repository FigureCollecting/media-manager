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
from sqlalchemy import select
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
    the decoded bytes after EXIF orientation. Two captures of one picture
    share it whatever their bytes (metadata, compression, or one of them
    being our own clean copy captured again); it is the key that links them
    behind one clean copy (``seed_clean_image``)."""
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
) -> PublicObject:
    """Make ``sha256`` addressable at ``/d/<sha256>``. Idempotent on the sha.

    Refuses an original's sha outright: the public origin must never be able
    to name an original, so an original can never be registered here. A
    clean copy (``sanitized``) needs its pixel digest, the key later captures
    of the same picture are linked by; no other kind carries one.
    """
    if kind not in PUBLIC_KINDS:
        raise ValueError(f"not a public object kind: {kind!r}")
    if (kind == "sanitized") != (pixel_sha256 is not None):
        raise ValueError("a clean copy, and only a clean copy, carries a pixel digest")
    if db.execute(select(ImageModel.id).where(ImageModel.sha256 == sha256)).first():
        raise ValueError("refusing to publish an original's bytes")
    existing = db.get(PublicObject, sha256)
    if existing is not None:
        return existing
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
    )
    db.add(obj)
    db.flush()
    return obj


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
    link per origin. Idempotent.

    The link keys on the PIXELS (``pixel_digest``), never on the minted
    clean sha: an original whose pixels a registered clean copy already has
    -- whatever its bytes, in whichever order the captures arrive -- is one
    more image behind THAT copy (``public_object_images``), never a copy of
    its own. That includes an original whose bytes ARE the copy (our clean
    copy captured again): it is judged with the copy, so its soft delete
    takes the copy offline (app.publication)."""
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
    obj = db.execute(
        select(PublicObject).where(PublicObject.pixel_sha256 == digest)
    ).scalar_one_or_none()
    if obj is None:
        clean, clean_mime = _encode(picture, original)
        clean_sha = hashlib.sha256(clean).hexdigest()
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
        )

    if db.get(PublicObjectImage, (obj.sha256, image.id)) is None:
        db.add(PublicObjectImage(sha256=obj.sha256, image_id=image.id))

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
