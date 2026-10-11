"""The July base-keep pass, ported (fc-design-assets/pipeline-scripts/basekept.py,
2026-07, variant V1 -- the raw union that reproduces the July cut-outs).

WHY. Plain BiRefNet tends to drop a figure's BASE (a stand, a rock, a floor
tile) -- measured 2026-10-11 on the seven July reference figures, its answer
overlaps the July cut-outs at IoU 0.72-0.89 on five of them. The base is what
stands on the shelf, so the grounding numbers (bottom margin, contact band)
are wrong without it. IS-Net sees bases well but also sees backdrop haze.

HOW. BiRefNet everywhere; an IS-Net answer added ONLY inside a feathered band
over the bottom of the figure (from BAND_FRAC of its height down), and only
for components connected to the figure. Of the donors, the one that sees more
confident base pixels BiRefNet missed is used. Finally any blob not connected
to the figure is dropped -- from BiRefNet's own answer too.

THE BLACK-DRESS CASE. A pass like this can keep a dark studio floor along
with a dark base; the mask lane's quality gate (app.workers.mask:
bottom_edge_bleed, full_frame) holds such a mask instead of publishing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from .matting import MattingBackend

#: The base band starts at this fraction of the figure's height.
BAND_FRAC = 0.78
#: Vertical feather above the band, as a fraction of the figure's height.
FEATHER_FRAC = 0.06
#: Donor alpha at which a pixel joins a candidate base component.
COMP_T = 35
#: Donor alpha counted as a confident base pixel.
CORE_T = 190
#: BiRefNet alpha above which a pixel is the figure.
FIGURE_T = 127
#: Alpha below which a pixel is not part of any kept blob (final cleanup).
BLOB_T = 25

_EIGHT = np.ones((3, 3), dtype=bool)


def _blur(weights: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray((weights * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def base_kept_alpha(
    birefnet: np.ndarray, donors: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, str | None]:
    """``(alpha, donor)``: BiRefNet's ``(height, width)`` answer with the
    figure's base restored from the best donor. ``donor`` is None (and the
    answer unchanged) when BiRefNet found no figure at all."""
    b = birefnet.astype(np.float32)
    height, width = b.shape
    fig = b > FIGURE_T
    rows = np.nonzero(fig.any(axis=1))[0]
    if rows.size == 0:
        return np.clip(b, 0, 255).astype(np.uint8), None
    ytop, ybot = int(rows[0]), int(rows[-1])
    figh = max(1, ybot - ytop)
    band_top = int(ytop + BAND_FRAC * figh)
    feather = max(4, int(FEATHER_FRAC * figh))

    def _base_seen(i: np.ndarray) -> int:
        missed = (i[band_top:] > 10) & (b[band_top:] < 30)
        return int(((i[band_top:] >= 120) & missed).sum())

    scored = {name: _base_seen(d.astype(np.float32)) for name, d in donors.items()}
    donor = max(scored, key=lambda k: scored[k])
    i = donors[donor].astype(np.float32)

    band0 = max(0, band_top - feather)
    i_band = i.copy()
    i_band[:band0, :] = 0

    comp = i_band >= COMP_T
    labels, n = ndimage.label(comp, structure=_EIGHT)
    fig_dilated = ndimage.binary_dilation(fig, iterations=4)
    added = np.zeros((height, width), dtype=bool)
    if n:
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(comp, labels, idx)
        cores = ndimage.sum_labels(i_band >= CORE_T, labels, idx)
        touching = ndimage.sum_labels(fig_dilated, labels, idx)
        keep = (touching > 0) & ((sizes >= 150) | (cores >= 20))
        added = np.isin(labels, idx[keep])

    vertical = np.zeros(height, dtype=np.float32)
    vertical[band_top:] = 1.0
    if band_top > band0:
        vertical[band0:band_top] = np.linspace(0, 1, band_top - band0, endpoint=False)
    horizontal = np.zeros(width, dtype=np.float32)
    if added.any():
        cols = np.nonzero(added.any(axis=0))[0]
        horizontal[max(0, int(cols[0]) - 12) : min(width, int(cols[-1]) + 13)] = 1.0
    weights = _blur(vertical[:, None] * horizontal[None, :], 5)
    keep_mask = _blur(ndimage.binary_dilation(added, iterations=6).astype(np.float32), 4)

    alpha = np.maximum(b, i * weights * keep_mask)

    blobs, nb = ndimage.label(alpha > BLOB_T, structure=_EIGHT)
    if nb > 1:
        bidx = np.arange(1, nb + 1)
        on_figure = ndimage.sum_labels(fig_dilated, blobs, bidx)
        alpha = np.where(np.isin(blobs, bidx[on_figure > 0]), alpha, 0)
    return np.clip(alpha, 0, 255).astype(np.uint8), donor


class BaseKeptMattingBackend(MattingBackend):
    """BiRefNet (``primary``) plus the July base-keep pass over IS-Net donors."""

    name: ClassVar[str] = "birefnet_basekept"
    publishes_masks: ClassVar[bool] = True

    def __init__(self, *, primary: MattingBackend, donors: Sequence[MattingBackend]) -> None:
        if not donors:
            raise ValueError("the base-keep pass needs at least one donor backend")
        self.primary = primary
        self.donors = list(donors)

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        answers = {f"{n}:{d.name}": d.predict_alpha(rgba) for n, d in enumerate(self.donors)}
        alpha, _donor = base_kept_alpha(self.primary.predict_alpha(rgba), answers)
        return alpha

    def matte(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba)
        combined = np.minimum(self.predict_alpha(rgba), arr[..., 3])
        return Image.fromarray(np.dstack([arr[..., :3], combined]).astype(np.uint8), mode="RGBA")
