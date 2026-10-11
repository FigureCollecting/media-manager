"""MASK-1: the separate alpha mask and the grounding numbers, per clean image.

Ross 2026-09-26: background removal is a NON-DESTRUCTIVE, display-time
overlay. The clean copy (the sanitized derivative) is stored and served
byte-for-byte as it is; this module computes a SEPARATE mask -- our own work
product -- that the client composites over it (CSS ``mask-image`` in alpha
mode, or canvas ``destination-in``), so the case shows through around the
figure. A destructive cut-out stays only as Ross's fallback
(``app.workers.tasks._create_matted_derivative``).

THE MASK FILE. fc-api-contract 0.4.0 ``ImageMask``: same pixel dimensions as
its derivative, ``image/png``, "its alpha channel is the mask". It is one
channel of information carried as PNG ``LA``: luminance is constant white and
alpha is the mask, so a client compositing in alpha mode reads the mask and
one compositing in luminance mode reads ``1.0 x alpha`` -- the same mask. A
grayscale ``L`` PNG would carry no alpha and mask nothing.

THE NUMBERS. Grounding is measured on the mask's alpha (opaque = alpha above
10 of 255, the contract's rule); the appearance (ThumbHash and dominant
color) is computed twice, with the mask applied and without it, because the
coordinator must describe the image "as THIS caller is shown it" and a
``no_mask`` display restriction withholds the mask (catalog.proto DISPLAY
RESTRICTIONS). The lane never decides who sees what; it only stores both.

THE QUALITY GATE. A mask that removed (almost) nothing, or that kept the
backdrop's floor edge to edge -- the July black-dress-on-black-marble failure
(fc-design-assets/madoka-matte-basekept.png) -- is HELD: computed and kept
for review, never published. A mask with nothing opaque is EMPTY and nothing
is stored. Only an ``ok`` mask is ever addressable.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

from .derivatives import compute_dominant_color, compute_thumbhash
from .grounding import ALPHA_THRESHOLD, AlphaGrounding, measure_alpha_grounding
from .matting import MattingBackend, get_matting_backend

#: Bump when the mask algorithm or its encoding changes: a new version mints
#: new masks beside the old ones (an address cached on a phone never changes
#: meaning), and the batch job re-selects every clean image for it.
MASK_PIPELINE_VERSION = "mask-v1"

MASK_MIME = "image/png"

#: A mask whose opaque share of the frame is at least this removed (almost)
#: nothing: held for review rather than published.
FULL_FRAME_COVERAGE = 0.97

#: A mask that reaches the image's last row with a contact band at least
#: this wide kept the floor edge to edge (the black-dress failure): held.
BOTTOM_BLEED_WIDTH_FRAC = 0.98

MaskStatus = Literal["ok", "held", "empty"]


@dataclass(frozen=True)
class Appearance:
    """ThumbHash (base64) and dominant color (``#rrggbb``, lowercase, as the
    contract spells it) of an image as some caller is shown it."""

    thumbhash: str
    dominant_color: str


@dataclass(frozen=True)
class MaskOutcome:
    """One clean image's answer. ``mask_png``/``mask_sha256``/``grounding``/
    ``masked`` are present for ``ok`` and ``held`` and absent for ``empty``."""

    status: MaskStatus
    reason: str | None
    backend: str
    pipeline_version: str
    width: int
    height: int
    mask_png: bytes | None
    mask_sha256: str | None
    grounding: AlphaGrounding | None
    masked: Appearance | None
    unmasked: Appearance
    own_alpha_grounding: AlphaGrounding | None


def _require_mask_shape(alpha: np.ndarray) -> None:
    if alpha.ndim != 2 or alpha.dtype != np.uint8 or alpha.size == 0:
        raise ValueError(
            f"a mask is a non-empty (height, width) uint8 array, got {alpha.dtype} {alpha.shape}"
        )


def encode_mask_png(alpha: np.ndarray) -> bytes:
    """The mask as a PNG whose alpha channel is the mask (``LA``, luminance
    constant white). Deterministic: the same alpha always encodes to the same
    bytes, so the same mask always has the same content address."""
    _require_mask_shape(alpha)
    la = np.dstack([np.full_like(alpha, 255), alpha])
    out = io.BytesIO()
    Image.fromarray(la, mode="LA").save(out, format="PNG", compress_level=9)
    return out.getvalue()


def decode_mask_png(data: bytes) -> np.ndarray:
    """The ``(height, width)`` alpha of a mask written by encode_mask_png."""
    img = Image.open(io.BytesIO(data))
    if img.format != "PNG" or img.mode != "LA":
        raise ValueError(f"not a mask: {img.format} {img.mode}")
    return np.asarray(img)[..., 1].copy()


def composite_with_mask(clean: Image.Image, alpha: np.ndarray) -> Image.Image:
    """What the client draws: the clean copy's pixels, its alpha multiplied
    by the mask (CSS ``mask-image`` alpha mode / canvas ``destination-in``).
    An opaque clean copy simply takes the mask as its alpha."""
    rgba = clean.convert("RGBA")
    if (rgba.height, rgba.width) != alpha.shape:
        raise ValueError(
            f"mask {alpha.shape[::-1]} does not match its image {(rgba.width, rgba.height)}"
        )
    arr = np.asarray(rgba).copy()
    own = arr[..., 3].astype(np.uint32)
    arr[..., 3] = ((own * alpha.astype(np.uint32) + 127) // 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def assess_mask_quality(
    alpha: np.ndarray, grounding: AlphaGrounding | None
) -> tuple[MaskStatus, str | None]:
    """``("ok", None)``, ``("held", reason)`` or ``("empty", reason)``."""
    if grounding is None:
        return "empty", "no_opaque_pixels"
    coverage = float(np.mean(alpha > ALPHA_THRESHOLD))
    if coverage >= FULL_FRAME_COVERAGE:
        return "held", "full_frame"
    if (
        grounding.bottom_margin_frac == 0
        and grounding.contact_band_width_frac >= BOTTOM_BLEED_WIDTH_FRAC
    ):
        return "held", "bottom_edge_bleed"
    return "ok", None


def _appearance(rgba: Image.Image) -> Appearance:
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    data = buf.getvalue()
    return Appearance(
        thumbhash=compute_thumbhash(data),
        dominant_color=compute_dominant_color(data).lower(),
    )


def build_mask_artifact(clean: bytes, *, backend: MattingBackend | None = None) -> MaskOutcome:
    """Compute the mask, its grounding and both appearances for one clean
    image. Never alters ``clean``; never produces a cut-out.

    The image is read with its EXIF orientation applied, because that is how
    a browser draws the clean copy (``image-orientation: from-image``), and
    the mask must line up with what is drawn.
    """
    active = backend or get_matting_backend()
    with Image.open(io.BytesIO(clean)) as src:
        oriented = ImageOps.exif_transpose(src) or src
        rgba = oriented.convert("RGBA")
    width, height = rgba.size

    own = np.asarray(rgba)[..., 3]
    own_alpha_grounding = measure_alpha_grounding(own) if bool((own < 255).any()) else None
    unmasked = _appearance(rgba)

    alpha = np.asarray(active.predict_alpha(rgba), dtype=np.uint8)
    if alpha.shape != (height, width):
        raise ValueError(
            f"backend {active.name!r} answered {alpha.shape} for a {height}x{width} image"
        )
    grounding = measure_alpha_grounding(alpha)
    status, reason = assess_mask_quality(alpha, grounding)

    if status == "empty":
        return MaskOutcome(
            status=status,
            reason=reason,
            backend=active.name,
            pipeline_version=MASK_PIPELINE_VERSION,
            width=width,
            height=height,
            mask_png=None,
            mask_sha256=None,
            grounding=None,
            masked=None,
            unmasked=unmasked,
            own_alpha_grounding=own_alpha_grounding,
        )

    mask_png = encode_mask_png(alpha)
    return MaskOutcome(
        status=status,
        reason=reason,
        backend=active.name,
        pipeline_version=MASK_PIPELINE_VERSION,
        width=width,
        height=height,
        mask_png=mask_png,
        mask_sha256=hashlib.sha256(mask_png).hexdigest(),
        grounding=grounding,
        masked=_appearance(composite_with_mask(rgba, alpha)),
        unmasked=unmasked,
        own_alpha_grounding=own_alpha_grounding,
    )
