"""IMGREND (IR-MM-1): display-sized renditions ("tiers") of a clean copy and
of its mask -- minted, stored, registered and recorded in the ledger.

The app draws a case tile about 95 x 119 device px from a full-size clean copy
and a full-size mask (hundreds of KB to 30 MB). A tier is the same picture at
a display size: the contract's word for a rendition (media.v1 ReportTiers,
fc-api-contract ImageTier). The full derivative and the full mask stay
exactly as they are, at their own addresses.

THE LADDER. 128, 256, 512, 1,024 and 2,048 px on the LONG edge. A parent
whose long edge is L gets every rung r < L, plus one at its own size when L
is at most 2,048 (the same pixels as WebP): at most five, and NEVER one
larger than the parent -- a picture below a rung has no copy at that rung.
The backfill goes small rungs first: ``max_edge=512`` mints 128, 256 and 512
(and a smaller parent's own size) for everything, the default the rest.

THE RULES (each pinned by tests/test_tiers.py):

1. ONE SIZE FUNCTION. ``rung_size``: the long side is exactly the edge, the
   other side the parent's aspect rounded half up, at least one pixel. A
   mask's rungs are sized from its DERIVATIVE's width and height, so the photo
   rung and the mask rung of one edge have identical pixel dimensions. A mask
   stored at another size than its derivative is refused
   (``mask_size_mismatch``), never resized to fit.
2. UNIFORM SCALE OF THE WHOLE FRAME ONLY. Nothing here crops, pads, turns,
   mirrors, skews or re-projects a picture, and no EXIF orientation is ever
   applied. The parent is our own clean copy, which carries none; a parent
   whose Orientation tag is anything but 1 is REFUSED
   (``parent_has_orientation``), not interpreted.
3. A MASK'S RUNGS ARE THE MASK, SCALED. Never re-predicted. Photos are
   resampled with Lanczos on premultiplied alpha; masks with a box filter (an
   area average cannot ring or overshoot). ``reducing_gap`` is pinned to
   None on both: Pillow would otherwise shrink in two steps, and since 13 it
   does so for RGBA and LA as well.
4. THE SAME PARENT ALWAYS MINTS THE SAME BYTES. Photo rungs are lossy WebP
   (quality 82, method 4; 88 for the 2,048 rung and the parent's own size),
   their alpha lossless; mask rungs are lossless WebP RGBA, white, the mask
   as alpha -- read by CSS ``mask-image`` and by a canvas exactly as the PNG
   ``LA`` mask is. A change of encoder or settings is a new
   ``TIER_PIPELINE_VERSION`` (tests/test_tiers.py GOLDEN fails on a drift);
   the first tier registered for a (parent, version, edge) is kept.
5. GEOMETRY NEVER COMES FROM A RUNG. A client sizes its box from the full
   image's width and height; a rung is at most half a rung pixel off per axis.

GIANT PARENTS. A parent over ``TierLimits.giant_pixels`` (64 megapixels) is
not skipped -- the slowest pictures are exactly these. A JPEG one is scaled
in its decoder (DCT scaling to 1/2, 1/4 or 1/8: the deepest that still
leaves ``draft_min_edge``, twice the top rung, on the long edge); a PNG one,
and every mask, is decoded whole. Every decode first takes its working
bytes from a ``DecodedBytes`` gate, which bounds what is decoded at once
across threads; a parent the gate could never hold is refused
(``parent_too_large``), from its header alone. Pillow's own pixel guard
(image_limits.MAX_IMAGE_PIXELS) is not consulted and not changed: the header
is read by the format's own reader, and the gate is the bound.

REGISTRY AND LEDGER. A tier is addressed by the sha-256 of its bytes and
stored once at ``tiers/<aa>/<sha>.webp``. ``register_public_tier`` is the one
gate into ``public_tiers``: an address an original or a public object
already has is refused. ``mint_parent`` does one parent end to end: bytes
into the store BEFORE any row names them, then the rows, then the ledger
(``tier_parents``): ``done`` up to an edge, ``refused`` (the parent's own
fault, never read again, listed by ``refused_parents``) or ``failed`` (tried
again). A store or database error is neither: it propagates, and the parent
is simply not settled.

Serving is app.publication.tier_servable: a tier has no status of its own.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFile, JpegImagePlugin, PngImagePlugin
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..models import Image as ImageModel
from ..models import ImageMask, PublicObject, PublicTier, TierParent
from ..object_store import ObjectStore

#: Bump when the resampling, the encoder or its settings change: a new
#: version mints new tiers beside the old ones (an address cached on a phone
#: never changes meaning).
TIER_PIPELINE_VERSION = "mm-tier-v1"

TIER_MIME = "image/webp"

#: The rungs, on the long edge.
LADDER = (128, 256, 512, 1024, 2048)

#: The contract's cap on a tier's long edge (media.v1 0.13.0).
MAX_TIER_EDGE = 4096

#: The public objects that have tiers: a clean copy and a mask.
PARENT_KINDS = ("sanitized", "mask")

#: Working bytes decoded at once, by default: 768 MiB carries the largest live
#: parent (14,174 x 7,525) as a mask (509 MiB) and as a half-scale JPEG (102
#: MiB), inside a 2 Gi pod.
DEFAULT_DECODED_BYTES = 768 * 1024 * 1024

#: Bytes a pixel as Pillow holds a decoded image: RGB, RGBA and LA are all
#: stored four bytes wide (only L is one).
_PIXEL_BYTES = 4

_QUALITY = 82
#: The 2,048 rung and a rung at the parent's own size.
_TOP_QUALITY = 88
_METHOD = 4
#: Lossless effort for mask rungs: the smallest files at method 4, for about
#: a tenth more encoding time than 75 (measured on a 2,048 px mask).
_MASK_QUALITY = 100

_ORIENTATION = 0x0112
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_READERS: tuple[tuple[bytes, type[ImageFile.ImageFile]], ...] = (
    (b"\xff\xd8\xff", JpegImagePlugin.JpegImageFile),
    (b"\x89PNG\r\n\x1a\n", PngImagePlugin.PngImageFile),
)


class TierRefusal(ValueError):
    """A parent that gets no tiers, for a reason of its own. ``code`` is
    stable (the ledger records it): parent_format, parent_undecodable,
    parent_mode, parent_has_orientation, parent_size_mismatch,
    parent_too_large, not_a_mask, mask_size_mismatch, mask_without_source,
    tier_address_taken."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TierStoreError(RuntimeError):
    """The store answered a parent's key with other bytes than the parent's:
    the store's fault, never the parent's."""


@dataclass(frozen=True)
class TierLimits:
    """Where the giant-parent path starts and how far a JPEG is scaled in its
    decoder. Fixed with the pipeline version (a rung's bytes depend on them):
    not read from the environment."""

    #: A parent of more pixels than this is a giant.
    giant_pixels: int = 64_000_000
    #: A giant JPEG's decoded long edge is at least this: twice the top rung.
    draft_min_edge: int = 2 * LADDER[-1]

    def __post_init__(self) -> None:
        if self.giant_pixels < 1:
            raise ValueError("giant_pixels must be positive")
        if self.draft_min_edge < LADDER[-1]:
            raise ValueError("draft_min_edge must carry the top rung: every rung is a downscale")


class DecodedBytes:
    """A gate on the working bytes decoded at once, across threads: ``hold``
    waits until its bytes fit, and refuses what could never fit."""

    def __init__(self, capacity: int = DEFAULT_DECODED_BYTES) -> None:
        if capacity < 1:
            raise ValueError("the capacity must be positive")
        self.capacity = capacity
        self.in_use = 0
        #: The most ever held at once.
        self.peak = 0
        self._room = threading.Condition()

    @contextmanager
    def hold(self, size: int) -> Iterator[None]:
        if size < 1:
            raise ValueError("a hold must be positive")
        if size > self.capacity:
            raise TierRefusal(
                "parent_too_large", f"decoding needs {size} bytes, the gate holds {self.capacity}"
            )
        with self._room:
            while self.in_use + size > self.capacity:
                self._room.wait()
            self.in_use += size
            self.peak = max(self.peak, self.in_use)
        try:
            yield
        finally:
            with self._room:
                self.in_use -= size
                self._room.notify_all()


@dataclass(frozen=True)
class Tier:
    """One minted rung: its bytes and what the contract says of them."""

    long_edge_px: int
    width: int
    height: int
    data: bytes
    sha256: str
    mime: str = TIER_MIME


# --------------------------------------------------------------------------
# sizes
# --------------------------------------------------------------------------


def rung_size(width: int, height: int, edge: int) -> tuple[int, int]:
    """The pixel size of the rung ``edge`` of a ``width`` x ``height``
    parent: the long side exactly ``edge``, the other the parent's aspect
    rounded half up and at least 1. Refuses an edge past the parent's."""
    if width < 1 or height < 1 or edge < 1:
        raise ValueError("a rung and its parent have positive sizes")
    long_edge = max(width, height)
    if edge > long_edge:
        raise ValueError(f"a {edge} px rung would upscale a {width} x {height} parent")

    def scaled(side: int) -> int:
        return max(1, (2 * side * edge + long_edge) // (2 * long_edge))

    return (edge, scaled(height)) if width >= height else (scaled(width), edge)


def ladder_for(width: int, height: int, *, max_edge: int | None = None) -> tuple[int, ...]:
    """The long edges of a parent's rungs, ascending: every ladder rung below
    its own long edge, plus its own size when that is at most the top rung.
    ``max_edge`` keeps the rungs up to it (a phase of the backfill)."""
    long_edge = max(width, height)
    edges = [rung for rung in LADDER if rung < long_edge]
    if long_edge <= LADDER[-1]:
        edges.append(long_edge)
    return tuple(edge for edge in edges if max_edge is None or edge <= max_edge)


def draft_scale(width: int, height: int, limits: TierLimits) -> int:
    """The scale (1, 2, 4 or 8) a JPEG parent is decoded at: 1 unless it is a
    giant, else the deepest that leaves ``draft_min_edge`` on the long edge."""
    if width * height <= limits.giant_pixels:
        return 1
    for scale in (8, 4, 2):
        if -(-max(width, height) // scale) >= limits.draft_min_edge:
            return scale
    return 1


# --------------------------------------------------------------------------
# decoding a parent
# --------------------------------------------------------------------------


@contextmanager
def _reading() -> Iterator[None]:
    """Pillow reading the parent's own bytes: whatever it raises on them is
    the parent's fault (the bytes are exactly the registered ones)."""
    try:
        yield
    except (TierRefusal, MemoryError):
        raise
    except Exception as exc:
        raise TierRefusal(
            "parent_undecodable", f"the parent does not decode: {type(exc).__name__}"
        ) from exc


def _header(parent: bytes) -> ImageFile.ImageFile:
    """The parent opened by its format's own reader -- header only, no pixel
    decoded -- or a refusal. Not ``Image.open``: that applies the process's
    pixel guard, which would turn a giant away before its bounded path."""
    for magic, reader in _READERS:
        if parent.startswith(magic):
            with _reading():
                return reader(io.BytesIO(parent))
    raise TierRefusal("parent_format", "a parent is a JPEG or a PNG")


def _decode(image: ImageFile.ImageFile) -> None:
    """Decode the pixels, then refuse an orientation tag (EXIF, or XMP's
    ``tiff:Orientation``; a PNG's may follow its pixels) other than upright."""
    with _reading():
        image.load()
        orientation = image.getexif().get(_ORIENTATION)
    if orientation is not None and orientation != 1:
        raise TierRefusal(
            "parent_has_orientation", f"the parent is tagged Orientation {orientation!r}"
        )


def _webp(image: Image.Image, **options: Any) -> bytes:
    out = io.BytesIO()
    image.save(out, format="WEBP", **options)
    return out.getvalue()


def _tier(edge: int, size: tuple[int, int], data: bytes) -> Tier:
    return Tier(edge, size[0], size[1], data, hashlib.sha256(data).hexdigest())


# --------------------------------------------------------------------------
# minting
# --------------------------------------------------------------------------


def mint_photo_tiers(
    parent: bytes,
    *,
    size: tuple[int, int] | None = None,
    max_edge: int | None = None,
    limits: TierLimits | None = None,
    gate: DecodedBytes | None = None,
) -> list[Tier]:
    """The rungs of the clean copy ``parent`` (its bytes), ascending.
    ``size`` is what its row says it is: another size is refused."""
    limits = limits or TierLimits()
    gate = gate or DecodedBytes()
    image = _header(parent)
    width, height = image.size
    if size is not None and size != (width, height):
        raise TierRefusal(
            "parent_size_mismatch", f"the parent is {width} x {height}, its row says {size}"
        )
    if image.mode not in ("RGB", "RGBA") or "transparency" in image.info:
        raise TierRefusal("parent_mode", f"a clean copy is RGB or RGBA, not {image.mode}")
    scale = draft_scale(width, height, limits) if image.format == "JPEG" else 1
    if scale > 1:
        image.draft(image.mode, (max(1, width // scale), max(1, height // scale)))
    transparent = image.mode == "RGBA"
    # a transparent photo is held twice: as stored, and premultiplied for the resampling
    with gate.hold(image.width * image.height * _PIXEL_BYTES * (2 if transparent else 1)):
        _decode(image)
        premultiplied = image.convert("RGBa") if transparent else None
        minted = []
        for edge in ladder_for(width, height, max_edge=max_edge):
            rung = rung_size(width, height, edge)
            if rung == image.size:
                scaled: Image.Image = image
            elif premultiplied is not None:
                scaled = premultiplied.resize(
                    rung, Image.Resampling.LANCZOS, reducing_gap=None
                ).convert("RGBA")
            else:
                scaled = image.resize(rung, Image.Resampling.LANCZOS, reducing_gap=None)
            top = edge == LADDER[-1] or edge == max(width, height)
            data = _webp(
                scaled, quality=_TOP_QUALITY if top else _QUALITY, method=_METHOD, alpha_quality=100
            )
            minted.append(_tier(edge, rung, data))
        return minted


def mint_mask_tiers(
    mask: bytes,
    *,
    width: int,
    height: int,
    max_edge: int | None = None,
    gate: DecodedBytes | None = None,
) -> list[Tier]:
    """The rungs of the mask ``mask`` (a PNG ``LA`` whose alpha is the mask),
    ascending, sized from its derivative's ``width`` and ``height``."""
    gate = gate or DecodedBytes()
    image = _header(mask)
    if image.format != "PNG" or image.mode != "LA":
        raise TierRefusal("not_a_mask", f"a mask is a PNG LA, not {image.format} {image.mode}")
    if image.size != (width, height):
        raise TierRefusal(
            "mask_size_mismatch",
            f"the mask is {image.width} x {image.height}, its derivative {width} x {height}",
        )
    # LA as stored, plus its alpha plane (one byte a pixel)
    with gate.hold(width * height * (_PIXEL_BYTES + 1)):
        _decode(image)
        alpha = image.getchannel("A")
        minted = []
        for edge in ladder_for(width, height, max_edge=max_edge):
            rung = rung_size(width, height, edge)
            scaled = (
                alpha
                if rung == alpha.size
                else alpha.resize(rung, Image.Resampling.BOX, reducing_gap=None)
            )
            white = Image.new("L", rung, 255)
            data = _webp(
                Image.merge("RGBA", (white, white, white, scaled)),
                lossless=True,
                quality=_MASK_QUALITY,
                method=_METHOD,
                exact=True,
            )
            minted.append(_tier(edge, rung, data))
        return minted


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def tier_key(sha256: str) -> str:
    """``tiers/<aa>/<sha>.webp``: a prefix of the tiers' own, so accounting
    and a later cleanup are prefix operations."""
    if not _SHA256_HEX.fullmatch(sha256):
        raise ValueError("a tier key needs a lowercase hex sha-256")
    return f"tiers/{sha256[:2]}/{sha256}.webp"


def tier_address_taken(db: Session, sha256: str) -> str | None:
    """Whose bytes ``sha256`` already names, if anyone's but a tier's: an
    original's (soft-deleted or not) or a public object's."""
    if db.execute(select(ImageModel.id).where(ImageModel.sha256 == sha256)).first() is not None:
        return "an original"
    if db.get(PublicObject, sha256) is not None:
        return "a public object"
    return None


def register_public_tier(
    db: Session,
    *,
    sha256: str,
    parent_sha256: str,
    long_edge_px: int,
    size: int,
    width: int,
    height: int,
    pipeline_version: str = TIER_PIPELINE_VERSION,
    mime: str = TIER_MIME,
) -> PublicTier:
    """Make the tier ``sha256`` addressable at ``/d/<sha256>`` as the
    ``long_edge_px`` rung of ``parent_sha256``. Idempotent on (parent,
    version, edge): the first tier registered is kept and returned, whatever
    bytes a repeat brings.

    Refuses an address an original or a public object already has (the
    public origin must never name an original, and a tier is never a clean
    copy or a mask), a parent that is not a registered clean copy or mask,
    and anything outside the contract's shape: WebP only, a long edge of 1
    to 4,096 px that IS the longer side."""
    if not _SHA256_HEX.fullmatch(sha256):
        raise ValueError("a tier is addressed by a lowercase hex sha-256")
    if mime != TIER_MIME:
        raise ValueError(f"a tier is {TIER_MIME}, not {mime!r}")
    if not 1 <= len(pipeline_version) <= 32:
        raise ValueError("a pipeline version is 1 to 32 characters")
    if size < 1:
        raise ValueError("a tier has bytes")
    if (
        not 1 <= long_edge_px <= MAX_TIER_EDGE
        or min(width, height) < 1
        or max(width, height) != long_edge_px
    ):
        raise ValueError(
            f"a tier's long edge is its longer side, 1 to {MAX_TIER_EDGE} px:"
            f" {long_edge_px} for {width} x {height}"
        )
    parent = db.get(PublicObject, parent_sha256)
    if parent is None or parent.kind not in PARENT_KINDS:
        raise ValueError("a tier's parent is a registered clean copy or mask")
    row = db.get(PublicTier, (parent_sha256, pipeline_version, long_edge_px))
    if row is not None:
        return row
    taken = tier_address_taken(db, sha256)
    if taken is not None:
        raise ValueError(f"refusing to publish {taken}'s bytes as a tier")
    row = PublicTier(
        parent_sha256=parent_sha256,
        pipeline_version=pipeline_version,
        long_edge_px=long_edge_px,
        sha256=sha256,
        parent_kind=parent.kind,
        mime=mime,
        bytes=size,
        width=width,
        height=height,
        storage_key=tier_key(sha256),
    )
    db.add(row)
    db.flush()
    return row


# --------------------------------------------------------------------------
# one parent, end to end
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentOutcome:
    """What ``mint_parent`` settled: the ledger's state and reason, the
    parent's registered tiers up to the edge asked for (ascending; none
    unless ``done``) and how many of them this call stored."""

    state: str
    reason: str | None
    tiers: tuple[PublicTier, ...]
    minted: int


def _registered(db: Session, parent_sha256: str, max_edge: int) -> list[PublicTier]:
    return list(
        db.execute(
            select(PublicTier)
            .where(
                PublicTier.parent_sha256 == parent_sha256,
                PublicTier.pipeline_version == TIER_PIPELINE_VERSION,
                PublicTier.long_edge_px <= max_edge,
            )
            .order_by(PublicTier.long_edge_px)
        ).scalars()
    )


def _derivative_size(db: Session, mask: PublicObject) -> tuple[int, int]:
    """The one size of every clean copy the mask ``mask`` is the mask of,
    which its own row must agree with."""
    sizes = {
        (row.width, row.height)
        for row in db.execute(
            select(PublicObject.width, PublicObject.height)
            .join(ImageMask, ImageMask.source_sha256 == PublicObject.sha256)
            .where(ImageMask.mask_sha256 == mask.sha256)
        )
    }
    if not sizes:
        raise TierRefusal("mask_without_source", "no mask row names this mask")
    if sizes != {(mask.width, mask.height)}:
        raise TierRefusal(
            "mask_size_mismatch",
            f"the mask's row is {mask.width} x {mask.height}, its derivatives {sorted(sizes)}",
        )
    return mask.width, mask.height


def _settle(
    db: Session,
    ledger: TierParent | None,
    parent_sha256: str,
    state: str,
    reason: str | None,
    max_edge: int,
) -> None:
    if ledger is None:
        ledger = TierParent(
            parent_sha256=parent_sha256,
            pipeline_version=TIER_PIPELINE_VERSION,
            max_edge_px=0,
            attempts=0,
        )
        db.add(ledger)
    ledger.state = state
    ledger.reason = reason
    ledger.max_edge_px = max(ledger.max_edge_px, max_edge)
    ledger.attempts += 1
    ledger.updated_at = dt.datetime.now(dt.UTC)
    db.flush()


def mint_parent(
    db: Session,
    store: ObjectStore,
    parent_sha256: str,
    *,
    max_edge: int = LADDER[-1],
    limits: TierLimits | None = None,
    gate: DecodedBytes | None = None,
) -> ParentOutcome:
    """Mint, store, register and record every rung up to ``max_edge`` (a
    ladder rung: 512 for the small-rung phase) of the clean copy or mask
    ``parent_sha256``. Flushes, never commits: the caller owns the
    transaction.

    A parent the ledger has settled -- ``done`` up to the edge, or
    ``refused`` -- is answered from the ledger: the store is not read.
    Otherwise the parent's bytes are read and checked against its address,
    its missing rungs are minted, PUT (before any row names them) and
    registered, and the ledger says ``done``. A ``TierRefusal`` settles it
    ``refused``; any other failure of the minting itself ``failed``, to be
    tried again. A store or database error propagates with nothing settled."""
    if max_edge not in LADDER:
        raise ValueError(f"max_edge must be a rung of the ladder {LADDER}")
    parent = db.get(PublicObject, parent_sha256)
    if parent is None or parent.kind not in PARENT_KINDS:
        raise LookupError(f"no clean copy or mask is registered as {parent_sha256}")
    ledger = db.get(TierParent, (parent_sha256, TIER_PIPELINE_VERSION))
    if ledger is not None and ledger.state == "refused":
        return ParentOutcome("refused", ledger.reason, (), 0)
    if ledger is not None and ledger.state == "done" and ledger.max_edge_px >= max_edge:
        return ParentOutcome("done", None, tuple(_registered(db, parent_sha256, max_edge)), 0)

    data = store.get(parent.storage_key)
    if hashlib.sha256(data).hexdigest() != parent_sha256:
        raise TierStoreError(f"the store's bytes at {parent.storage_key} are not {parent_sha256}")
    try:
        if parent.kind == "mask":
            width, height = _derivative_size(db, parent)
            minted = mint_mask_tiers(data, width=width, height=height, max_edge=max_edge, gate=gate)
        else:
            minted = mint_photo_tiers(
                data,
                size=(parent.width, parent.height),
                max_edge=max_edge,
                limits=limits,
                gate=gate,
            )
        have = {row.long_edge_px for row in _registered(db, parent_sha256, max_edge)}
        new = [tier for tier in minted if tier.long_edge_px not in have]
        for tier in new:
            taken = tier_address_taken(db, tier.sha256)
            if taken is not None:
                raise TierRefusal(
                    "tier_address_taken",
                    f"the {tier.long_edge_px} px rung's bytes are {taken}'s",
                )
    except TierRefusal as refusal:
        _settle(db, ledger, parent_sha256, "refused", refusal.code, 0)
        return ParentOutcome("refused", refusal.code, (), 0)
    except SQLAlchemyError:
        raise  # the database's fault, not the parent's: nothing is settled
    except Exception as exc:
        _settle(db, ledger, parent_sha256, "failed", type(exc).__name__, 0)
        return ParentOutcome("failed", type(exc).__name__, (), 0)

    for tier in new:
        store.put(tier_key(tier.sha256), tier.data, TIER_MIME)
    for tier in new:
        register_public_tier(
            db,
            sha256=tier.sha256,
            parent_sha256=parent_sha256,
            long_edge_px=tier.long_edge_px,
            size=len(tier.data),
            width=tier.width,
            height=tier.height,
        )
    _settle(db, ledger, parent_sha256, "done", None, max_edge)
    return ParentOutcome("done", None, tuple(_registered(db, parent_sha256, max_edge)), len(new))


def refused_parents(
    db: Session,
    *,
    after: str = "",
    limit: int = 100,
    pipeline_version: str = TIER_PIPELINE_VERSION,
) -> list[TierParent]:
    """The parents this version refused, by sha, a page after ``after``: what
    the lane could not handle is listed, never silently skipped."""
    return list(
        db.execute(
            select(TierParent)
            .where(
                TierParent.pipeline_version == pipeline_version,
                TierParent.state == "refused",
                TierParent.parent_sha256 > after,
            )
            .order_by(TierParent.parent_sha256)
            .limit(limit)
        ).scalars()
    )
