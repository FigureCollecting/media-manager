"""Grounding scalars for matted figure images.

Python port of fc-mobile's src/components/display/alphaMargin.ts
(computeBottomMarginFrac / computeContactBand). The JS side deliberately
takes a DOM-decoupled ``AlphaBuffer`` ({width, height, data}) rather than a
real ``ImageData`` so it can be unit tested without a canvas; the Python
port mirrors that shape with a raw ``(height, width, 4)`` uint8 RGBA numpy
array (the PIL/sharp-equivalent of that buffer) so the same pure algorithm
runs here against a matte's real alpha channel.

CRITICAL: keep ALPHA_THRESHOLD and CONTACT_BAND_FRAC numerically identical
to alphaMargin.ts -- these two services must agree on the same figure's
grounding scalars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Ignore near-transparent anti-aliasing dust -- only real content counts.
# Mirrors alphaMargin.ts ALPHA_THRESHOLD.
ALPHA_THRESHOLD = 10

# Bottom 8% of the image's own height. Mirrors alphaMargin.ts CONTACT_BAND_FRAC.
CONTACT_BAND_FRAC = 0.08


def _round_half_up(x: float) -> int:
    """Match JS Math.round semantics (halves round toward +Infinity) rather
    than Python's banker's-rounding builtin round(), for the one place
    (band height) where the two could otherwise disagree at a .5 boundary."""
    return math.floor(x + 0.5)


def compute_bottom_margin_frac(rgba: np.ndarray) -> float:
    """Fraction of the image's own height that's transparent padding BELOW
    the figure's actual visible content (last opaque row from the bottom).

    ``rgba`` is a ``(height, width, 4)`` uint8 array. Returns 0 for a
    degenerate (zero-size) or fully transparent buffer rather than a bogus
    100% margin -- same contract as the JS original.
    """
    height, width = rgba.shape[0], rgba.shape[1]
    if width <= 0 or height <= 0:
        return 0.0

    alpha = rgba[..., 3]
    opaque_rows = np.nonzero(np.any(alpha > ALPHA_THRESHOLD, axis=1))[0]
    if opaque_rows.size == 0:
        return 0.0

    last_opaque_y = int(opaque_rows.max())
    return float((height - 1 - last_opaque_y) / height)


def compute_contact_band(rgba: np.ndarray) -> tuple[float, float] | None:
    """Horizontal ground-contact footprint: the fractional center/width of
    the opaque pixels in the figure's own bottom CONTACT BAND (not just the
    single last opaque row -- too thin/noisy, e.g. a boot tip or stray
    hair strand could skew it).

    Returns ``(center_x_frac, width_frac)`` or ``None`` when there's
    nothing to measure (degenerate or fully transparent buffer).
    """
    height, width = rgba.shape[0], rgba.shape[1]
    if width <= 0 or height <= 0:
        return None

    alpha = rgba[..., 3]
    opaque_rows = np.nonzero(np.any(alpha > ALPHA_THRESHOLD, axis=1))[0]
    if opaque_rows.size == 0:
        return None

    contact_row = int(opaque_rows.max())
    band_height = max(1, _round_half_up(height * CONTACT_BAND_FRAC))
    band_top = max(0, contact_row - band_height + 1)

    band = alpha[band_top : contact_row + 1, :]
    cols = np.nonzero(np.any(band > ALPHA_THRESHOLD, axis=0))[0]
    if cols.size == 0:
        return None  # shouldn't happen -- contact_row was found opaque

    min_x = int(cols.min())
    max_x = int(cols.max())
    center_x_frac = (min_x + max_x) / 2 / width
    width_frac = (max_x - min_x) / width
    return center_x_frac, width_frac


@dataclass(frozen=True)
class AlphaGrounding:
    """Everything the case view needs to stand a figure on a shelf, measured
    on one alpha channel (a MASK-1 mask, or a clean copy's own alpha).

    ``bottom_margin_frac`` and the contact band are fc-api-contract 0.4.0
    ProductImage fields. ``top_margin_frac`` and the opaque box are what the
    case view measures client-side today (fc-mobile computeTopMarginFrac /
    computeSideMarginFracs); they have no contract field yet (CONTRACT-1
    follow-up). The box is the HALF-OPEN opaque extent as fractions of the
    image: rows ``[box_top, box_bottom)`` of the height and columns
    ``[box_left, box_right)`` of the width, so ``1 - box_bottom`` is the clear
    bottom margin and ``1 - box_right`` the clear right margin.
    """

    bottom_margin_frac: float
    top_margin_frac: float
    box_top_frac: float
    box_bottom_frac: float
    box_left_frac: float
    box_right_frac: float
    contact_band_center_x_frac: float
    contact_band_width_frac: float


def measure_alpha_grounding(alpha: np.ndarray) -> AlphaGrounding | None:
    """Grounding on a bare ``(height, width)`` alpha channel, where opaque
    means alpha above ALPHA_THRESHOLD (the contract's "above 10 of 255").

    Same arithmetic as compute_bottom_margin_frac / compute_contact_band, so
    a mask and a cut-out with the same alpha report the same numbers. Returns
    ``None`` -- unmeasured, never a bogus 0 -- when nothing is opaque or the
    buffer is empty.
    """
    if alpha.ndim != 2:
        raise ValueError(f"expected a (height, width) alpha channel, got shape {alpha.shape}")
    height, width = alpha.shape
    opaque = alpha > ALPHA_THRESHOLD
    rows = np.nonzero(np.any(opaque, axis=1))[0]
    if rows.size == 0:
        return None
    cols = np.nonzero(np.any(opaque, axis=0))[0]
    first_row, last_row = int(rows.min()), int(rows.max())
    first_col, last_col = int(cols.min()), int(cols.max())

    band_height = max(1, _round_half_up(height * CONTACT_BAND_FRAC))
    band_top = max(0, last_row - band_height + 1)
    band_cols = np.nonzero(np.any(opaque[band_top : last_row + 1, :], axis=0))[0]
    band_min, band_max = int(band_cols.min()), int(band_cols.max())

    return AlphaGrounding(
        bottom_margin_frac=float((height - 1 - last_row) / height),
        top_margin_frac=first_row / height,
        box_top_frac=first_row / height,
        box_bottom_frac=(last_row + 1) / height,
        box_left_frac=first_col / width,
        box_right_frac=(last_col + 1) / width,
        contact_band_center_x_frac=(band_min + band_max) / 2 / width,
        contact_band_width_frac=(band_max - band_min) / width,
    )
