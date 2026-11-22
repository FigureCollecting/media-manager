"""IMGREND (IR-MM-1): minting display-sized renditions ("tiers") of a clean
copy and of its mask -- app.workers.tiers.

What a rung may be, each rule against a reference this file computes itself:
never larger than its parent; long side exactly the rung's edge; a photo rung
and the mask rung of the same edge the same pixel size; the whole frame scaled
uniformly and NOTHING ELSE (an asymmetric fixture keeps its corner markers, an
Orientation-tagged parent is refused rather than interpreted); the pair
composited is what the full pair composited and scaled down shows (SSIM); the
same parent always encodes to the same bytes (golden shas: an encoder drift
fails here, and is a new pipeline version).
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import struct
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest
from app.workers import tiers
from app.workers.mask import encode_mask_png
from app.workers.tiers import (
    LADDER,
    TIER_MIME,
    TIER_PIPELINE_VERSION,
    DecodedBytes,
    Tier,
    TierLimits,
    TierRefusal,
    ladder_for,
    mint_mask_tiers,
    mint_photo_tiers,
    rung_size,
)
from PIL import Image, ImageFile
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import gaussian_filter

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "mask_sample"
ENTRIES = json.loads((SAMPLE / "manifest.json").read_text())["entries"]

#: The ladder as decided (IR-1 a, IR-3 b): stated here, not read from the code.
RUNGS = (128, 256, 512, 1024, 2048)

RED, GREEN, BLUE, YELLOW = (220, 30, 30), (30, 200, 40), (30, 40, 220), (230, 220, 30)


def _figure(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A LOPSIDED figure: a body left of centre and an arm reaching right, so
    a mirrored, turned or shifted rung is another picture."""
    yy, xx = np.mgrid[0:height, 0:width]
    body = ((xx - width * 0.45) / (width * 0.25)) ** 2 + (
        (yy - height * 0.55) / (height * 0.38)
    ) ** 2
    arm = (np.abs(yy - height * 0.3) < height * 0.05) & (xx > width * 0.45) & (xx < width * 0.9)
    return body, arm, xx, yy


def _scene(width: int, height: int, seed: int = 7) -> np.ndarray:
    """A photo-like RGB frame: a graded backdrop, a figure striped light and
    dark (detail a wrong pairing cannot hide in), mild sensor noise."""
    body, arm, xx, yy = _figure(width, height)
    arr = np.stack(
        [
            120 + 100 * xx / width,
            110 + 90 * yy / height,
            140 + 60 * (xx + yy) / (width + height),
        ],
        axis=-1,
    )
    stripes = 0.5 + 0.5 * np.sin((xx + yy * 0.6) * 2 * np.pi / (max(width, height) / 18))
    figure = np.stack([60 + 150 * stripes, 40 + 120 * stripes, 50 + 130 * stripes], axis=-1)
    inside = (body < 1) | arm
    arr[inside] = figure[inside]
    noise = np.random.default_rng(seed).integers(-5, 6, (height, width, 1))
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def _hue_striped(width: int, height: int) -> np.ndarray:
    """``_scene`` with its stripes in HUE alone (red against blue at one
    brightness): fine detail a 4:2:0 encoder cannot carry."""
    body, arm, xx, yy = _figure(width, height)
    arr = _scene(width, height)
    stripes = 0.5 + 0.5 * np.sin((xx + yy * 0.6) * 2 * np.pi / (max(width, height) / 18))
    figure = np.stack([60 + 150 * stripes, 40 + 60 * yy / height, 90 + 80 * (1 - stripes)], axis=-1)
    inside = (body < 1) | arm
    arr[inside] = np.clip(figure[inside], 0, 255).astype(np.uint8)
    return arr


def _alpha(width: int, height: int) -> np.ndarray:
    """The figure's mask: a soft edge round the body, a hard one on the arm."""
    body, arm, _xx, _yy = _figure(width, height)
    alpha = np.clip((1.05 - body) * 400, 0, 255)
    alpha[arm] = 255
    return alpha.astype(np.uint8)


def _as_a_store_serves_it(arr: np.ndarray) -> np.ndarray:
    """The pixels after a store's own JPEG (quality 90, 4:2:0): what a real
    original is before its clean copy is made."""
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="JPEG", quality=90)
    return np.asarray(Image.open(io.BytesIO(out.getvalue())).convert("RGB"))


def _corners(width: int, height: int) -> np.ndarray:
    """An ASYMMETRIC frame: a different colour in each corner on grey."""
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    bh, bw = height // 4, width // 4
    arr[:bh, :bw] = RED  # top left
    arr[:bh, -bw:] = GREEN  # top right
    arr[-bh:, :bw] = BLUE  # bottom left
    arr[-bh:, -bw:] = YELLOW  # bottom right
    return arr


def _png(arr: np.ndarray, **save: object) -> bytes:
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG", **save)
    return out.getvalue()


def _jpeg(arr: np.ndarray, **save: object) -> bytes:
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="JPEG", quality=95, subsampling=0, **save)
    return out.getvalue()


def _open(tier: Tier) -> Image.Image:
    img = Image.open(io.BytesIO(tier.data))
    img.load()
    return img


def _by_edge(minted: list[Tier]) -> dict[int, Tier]:
    return {t.long_edge_px: t for t in minted}


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity (Wang et al. 2004: an 11-tap Gaussian
    window, sigma 1.5, K1 0.01, K2 0.03), averaged over the channels."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=3.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(score.mean())


def _over_grey(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """What the case draws: the photo through its mask, over a mid grey."""
    a = alpha.astype(np.float64)[..., None] / 255
    return np.clip(np.rint(rgb[..., :3] * a + 128 * (1 - a)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# the size function and the ladder
# --------------------------------------------------------------------------


class TestRungSize:
    @pytest.mark.parametrize(
        ("width", "height", "edge", "expected"),
        [
            (4000, 3000, 256, (256, 192)),
            (3000, 4000, 256, (192, 256)),
            (850, 850, 512, (512, 512)),
            (600, 400, 600, (600, 400)),  # the native rung is the parent's own size
            (1000, 667, 256, (256, 171)),  # 170.752 rounds up
            (1024, 5, 512, (512, 3)),  # 2.5 rounds half UP, never to even
            (1024, 3, 512, (512, 2)),  # 1.5
            (10000, 3, 256, (256, 1)),  # never a zero side
            (3, 10000, 256, (1, 256)),
            (15254, 22160, 2048, (1410, 2048)),
        ],
    )
    def test_the_long_side_is_exactly_the_edge(
        self, width: int, height: int, edge: int, expected: tuple[int, int]
    ) -> None:
        assert rung_size(width, height, edge) == expected
        assert max(rung_size(width, height, edge)) == edge

    @pytest.mark.parametrize(("width", "height", "edge"), [(600, 400, 601), (10, 10, 11)])
    def test_an_edge_past_the_parents_is_refused_never_upscaled(
        self, width: int, height: int, edge: int
    ) -> None:
        with pytest.raises(ValueError, match="upscale"):
            rung_size(width, height, edge)

    @pytest.mark.parametrize("bad", [(0, 10, 1), (10, 0, 1), (10, 10, 0), (-5, 10, 1)])
    def test_a_size_that_is_not_positive_is_refused(self, bad: tuple[int, int, int]) -> None:
        with pytest.raises(ValueError):
            rung_size(*bad)

    def test_the_aspect_is_the_parents_within_a_pixel(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(500):
            width, height = (int(n) for n in rng.integers(1, 20000, 2))
            for edge in ladder_for(width, height):
                scale = edge / max(width, height)
                for got, side in zip(rung_size(width, height, edge), (width, height), strict=True):
                    # half a pixel of rounding; a side that would round to 0 is 1
                    assert abs(got - side * scale) <= 0.5 or (got == 1 and side * scale < 0.5)


class TestLadder:
    def test_the_ladder_is_128_256_512_1024_2048(self) -> None:
        assert LADDER == RUNGS == (128, 256, 512, 1024, 2048)
        assert TIER_PIPELINE_VERSION == "mm-tier-v1"
        assert TIER_MIME == "image/webp"

    @pytest.mark.parametrize(
        ("long_edge", "expected"),
        [
            (100, (100,)),  # below every rung: its own size only, no copy at 128
            (128, (128,)),  # r < L is strict: the 128 here is the native rung
            (129, (128, 129)),
            (200, (128, 200)),
            (256, (128, 256)),
            (257, (128, 256, 257)),
            (850, (128, 256, 512, 850)),  # the median parent
            (1024, (128, 256, 512, 1024)),
            (1500, (128, 256, 512, 1024, 1500)),
            (2048, (128, 256, 512, 1024, 2048)),
            (2049, (128, 256, 512, 1024, 2048)),  # past 2,048: no native rung
            (15254, (128, 256, 512, 1024, 2048)),
            (1, (1,)),
        ],
    )
    def test_rungs_below_the_parent_plus_its_own_size_up_to_2048(
        self, long_edge: int, expected: tuple[int, ...]
    ) -> None:
        assert ladder_for(long_edge, max(1, long_edge // 2)) == expected
        assert ladder_for(max(1, long_edge // 2), long_edge) == expected

    def test_a_parent_has_at_most_five_rungs_none_past_itself(self) -> None:
        """Never upscaled: a parent below a rung has no copy at that rung. Five
        at most, inside the contract's eight."""
        for long_edge in range(1, 5000, 7):
            edges = ladder_for(long_edge, 1)
            assert len(edges) <= 5 and list(edges) == sorted(set(edges))
            assert all(e <= long_edge and e <= 2048 for e in edges)
            assert [e for e in RUNGS if e < long_edge] == [e for e in edges if e != long_edge]

    @pytest.mark.parametrize(
        ("long_edge", "max_edge", "expected"),
        [
            (850, 512, (128, 256, 512)),  # the small-rung phase: 128, 256 and 512
            (400, 512, (128, 256, 400)),  # a native rung within the phase is in it
            (200, 512, (128, 200)),
            (100, 512, (100,)),
            (3000, 512, (128, 256, 512)),
            (3000, 2048, (128, 256, 512, 1024, 2048)),
            (3000, 128, (128,)),
            (3000, 100, ()),
        ],
    )
    def test_a_phase_takes_the_rungs_up_to_its_edge(
        self, long_edge: int, max_edge: int, expected: tuple[int, ...]
    ) -> None:
        assert ladder_for(long_edge, 10, max_edge=max_edge) == expected


# --------------------------------------------------------------------------
# photo rungs
# --------------------------------------------------------------------------


class TestPhotoRungs:
    @pytest.mark.parametrize(
        ("width", "height"), [(1200, 800), (800, 1200), (523, 550), (2100, 1400), (200, 120)]
    )
    def test_never_upscaled_long_side_exact_whole_ladder(self, width: int, height: int) -> None:
        minted = mint_photo_tiers(_jpeg(_scene(width, height)))
        assert tuple(t.long_edge_px for t in minted) == ladder_for(width, height)
        for tier in minted:
            img = _open(tier)
            assert img.format == "WEBP"
            assert (
                img.size == (tier.width, tier.height) == rung_size(width, height, tier.long_edge_px)
            )
            assert max(img.size) == tier.long_edge_px <= max(width, height)
            assert img.width <= width and img.height <= height
            assert tier.sha256 == hashlib.sha256(tier.data).hexdigest()
            assert tier.mime == "image/webp"

    def test_an_opaque_photo_is_lossy_webp_with_no_alpha_and_no_metadata(self) -> None:
        for tier in mint_photo_tiers(_jpeg(_scene(900, 600))):
            img = _open(tier)
            assert img.mode == "RGB"
            assert tier.data[12:16] == b"VP8 "  # the simple lossy container: no VP8X, no chunks
            assert not {"exif", "icc_profile", "xmp"} & set(img.info)

    def test_a_transparent_photo_keeps_its_alpha(self) -> None:
        rgba = np.dstack([_scene(600, 900), _alpha(600, 900)])
        rgba[rgba[..., 3] == 0] = 0  # as the clean copy stores it
        minted = mint_photo_tiers(_png(rgba))
        assert [t.long_edge_px for t in minted] == [128, 256, 512, 900]
        for tier in minted:
            img = _open(tier)
            assert img.mode == "RGBA"
            alpha = np.asarray(img)[..., 3]
            assert alpha.min() == 0 and alpha.max() == 255
            assert alpha[0, 0] == 0 and alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255
        # the native rung's alpha is the parent's alpha, bit for bit (alpha_quality 100)
        assert np.array_equal(np.asarray(_open(minted[-1]))[..., 3], rgba[..., 3])

    def test_a_native_rung_is_the_parents_own_pixels_reencoded(self) -> None:
        """Not resampled: the parent's pixel grid, lossy. (How close a pair
        must be is TestThePairComposited's; this frame's noise is the hard
        part for any lossy encoder, 0.955 measured.)"""
        scene = _as_a_store_serves_it(_scene(700, 500))
        native = _by_edge(mint_photo_tiers(_png(scene)))[700]
        assert _open(native).size == (700, 500)
        got = np.asarray(_open(native))
        assert _ssim(got, scene) >= 0.93
        # a rung that had been through any resampling would sit a pixel off somewhere
        assert _ssim(got, scene) > _ssim(np.roll(got, 1, axis=1), scene) + 0.05

    def test_the_same_parent_always_mints_the_same_bytes(self) -> None:
        parent = _jpeg(_scene(1100, 700))
        first, again = mint_photo_tiers(parent), mint_photo_tiers(parent)
        assert [t.sha256 for t in first] == [t.sha256 for t in again]

    def test_a_phase_mints_only_its_rungs_and_the_same_bytes(self) -> None:
        parent = _jpeg(_scene(1100, 700))
        small = mint_photo_tiers(parent, max_edge=512)
        assert [t.long_edge_px for t in small] == [128, 256, 512]
        assert [t.sha256 for t in small] == [t.sha256 for t in mint_photo_tiers(parent)[:3]]

    def test_a_picture_below_a_rung_gets_no_copy_at_that_rung(self) -> None:
        """100 px: its own size only. 200 px: 128 and its own size, never 256."""
        tiny = mint_photo_tiers(_png(_scene(100, 70)))
        assert [(t.long_edge_px, t.width, t.height) for t in tiny] == [(100, 100, 70)]
        small = mint_photo_tiers(_png(_scene(150, 200)))
        assert [(t.long_edge_px, t.width, t.height) for t in small] == [
            (128, 96, 128),
            (200, 150, 200),
        ]
        masks = mint_mask_tiers(encode_mask_png(_alpha(150, 200)), width=150, height=200)
        assert [(t.width, t.height) for t in masks] == [(96, 128), (150, 200)]

    def test_the_parents_metadata_never_reaches_a_rung(self) -> None:
        info = PngInfo()
        info.add_text("Comment", "secret")
        parent = _png(_scene(400, 300), pnginfo=info, icc_profile=b"\x00" * 200, dpi=(300, 300))
        for tier in mint_photo_tiers(parent):
            assert b"secret" not in tier.data and b"ICCP" not in tier.data
            assert not {"exif", "icc_profile", "xmp", "Comment"} & set(_open(tier).info)

    @pytest.mark.parametrize(
        ("width", "height", "top_edges"),
        [(2100, 1400, {2048}), (1100, 733, {1100}), (3000, 600, {2048})],
    )
    def test_the_top_rung_is_q88_and_every_other_rung_q82(
        self,
        monkeypatch: pytest.MonkeyPatch,
        width: int,
        height: int,
        top_edges: set[int],
    ) -> None:
        """The 2,048 rung cut from a LARGER parent is the top rung, as is a
        parent's own size below it; a drift here changes every such rung under
        mm-tier-v1 (rule 4: that is a new pipeline version)."""
        qualities: dict[int, int] = {}
        Image.init()  # registers every plugin's save handler
        real = Image.SAVE["WEBP"]

        def spy(im: Image.Image, fp: object, filename: object) -> None:
            qualities[max(im.size)] = int(im.encoderinfo["quality"])
            real(im, fp, filename)  # type: ignore[arg-type]

        # the encoder Pillow itself calls, so a wrapper that drops or rewrites
        # the asked quality is seen too
        monkeypatch.setitem(Image.SAVE, "WEBP", spy)
        mint_photo_tiers(_jpeg(_scene(width, height)))
        assert set(qualities) == set(ladder_for(width, height))
        assert {edge for edge, q in qualities.items() if q == 88} == top_edges
        assert {q for edge, q in qualities.items() if edge not in top_edges} == {82}


class TestNoRotation:
    """Uniform scale of the whole frame only: every corner stays where it was."""

    @staticmethod
    def _corner_colours(arr: np.ndarray) -> list[tuple[int, ...]]:
        h, w = arr.shape[:2]
        dy, dx = max(1, h // 10), max(1, w // 10)
        patches = (arr[:dy, :dx], arr[:dy, -dx:], arr[-dy:, :dx], arr[-dy:, -dx:])
        return [tuple(int(v) for v in p.reshape(-1, p.shape[-1]).mean(axis=0)[:3]) for p in patches]

    @staticmethod
    def _close(got: tuple[int, ...], want: tuple[int, ...]) -> bool:
        return all(abs(g - w) <= 30 for g, w in zip(got, want, strict=True))

    @pytest.mark.parametrize("encode", [_jpeg, _png])
    @pytest.mark.parametrize(("width", "height"), [(1300, 700), (700, 1300), (640, 640)])
    def test_an_asymmetric_photo_keeps_its_corner_markers(
        self, encode: object, width: int, height: int
    ) -> None:
        parent = encode(_corners(width, height))  # type: ignore[operator]
        minted = mint_photo_tiers(parent)
        assert len(minted) >= 3
        for tier in minted:
            img = _open(tier)
            assert (img.width > img.height) == (width > height)
            assert (img.width == img.height) == (width == height)
            got = self._corner_colours(np.asarray(img))
            for corner, want in zip(got, (RED, GREEN, BLUE, YELLOW), strict=True):
                assert self._close(corner, want), (tier.long_edge_px, got)

    def test_an_asymmetric_mask_keeps_its_corner(self) -> None:
        alpha = np.zeros((700, 1300), dtype=np.uint8)
        alpha[:150, :300] = 255  # opaque in the top left only
        for tier in mint_mask_tiers(encode_mask_png(alpha), width=1300, height=700):
            a = np.asarray(_open(tier))[..., 3]
            h, w = a.shape
            assert w > h
            assert a[: h // 10, : w // 10].min() == 255
            assert a[: h // 10, -w // 10 :].max() == 0
            assert a[-h // 10 :, : w // 10].max() == 0
            assert a[-h // 10 :, -w // 10 :].max() == 0

    @staticmethod
    def _oriented(kind: str, orientation: int) -> bytes:
        exif = Image.Exif()
        exif[0x0112] = orientation
        arr = _corners(300, 200)
        if kind == "jpeg":
            return _jpeg(arr, exif=exif)
        if kind == "png":
            return _png(arr, exif=exif)
        info = PngInfo()
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF'
            ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description'
            f' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="{orientation}"/>'
            "</rdf:RDF></x:xmpmeta>"
        )
        info.add_itxt("XML:com.adobe.xmp", xmp)
        return _png(arr, pnginfo=info)

    @pytest.mark.parametrize("kind", ["jpeg", "png", "png-xmp"])
    @pytest.mark.parametrize("orientation", [2, 3, 4, 5, 6, 7, 8])
    def test_an_orientation_tagged_parent_is_refused_never_rotated(
        self, kind: str, orientation: int
    ) -> None:
        with pytest.raises(TierRefusal) as refused:
            mint_photo_tiers(self._oriented(kind, orientation))
        assert refused.value.code == "parent_has_orientation"

    @pytest.mark.parametrize("orientation", [0, 9, 65535])
    def test_an_orientation_no_reader_knows_is_refused_too(self, orientation: int) -> None:
        with pytest.raises(TierRefusal) as refused:
            mint_photo_tiers(self._oriented("jpeg", orientation))
        assert refused.value.code == "parent_has_orientation"

    @pytest.mark.parametrize("kind", ["jpeg", "png", "png-xmp"])
    def test_orientation_1_is_upright_and_minted_as_stored(self, kind: str) -> None:
        minted = mint_photo_tiers(self._oriented(kind, 1))
        plain = mint_photo_tiers(
            _jpeg(_corners(300, 200)) if kind == "jpeg" else _png(_corners(300, 200))
        )
        assert [t.sha256 for t in minted] == [t.sha256 for t in plain]
        assert [_open(t).size for t in minted] == [(128, 85), (256, 171), (300, 200)]

    def test_an_orientation_tagged_mask_is_refused(self) -> None:
        exif = Image.Exif()
        exif[0x0112] = 6
        la = np.dstack([np.full((200, 300), 255, np.uint8), _alpha(300, 200)])
        with pytest.raises(TierRefusal) as refused:
            mint_mask_tiers(_png(la, exif=exif), width=300, height=200)
        assert refused.value.code == "parent_has_orientation"

    def test_the_module_calls_no_transposing_or_warping_api(self) -> None:
        tree = ast.parse(Path(tiers.__file__).read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        banned = {"transpose", "rotate", "transform", "exif_transpose", "ImageOps", "crop"}
        banned |= {"Transpose", "Transform", "perspective", "affine", "expand", "pad", "fit"}
        assert not names & banned


# --------------------------------------------------------------------------
# mask rungs, and the pair
# --------------------------------------------------------------------------


class TestMaskRungs:
    @pytest.mark.parametrize(("width", "height"), [(1200, 800), (801, 1203), (523, 550), (90, 60)])
    def test_a_mask_rung_is_the_size_of_the_photo_rung(self, width: int, height: int) -> None:
        photo = _by_edge(mint_photo_tiers(_jpeg(_scene(width, height))))
        mask = _by_edge(
            mint_mask_tiers(encode_mask_png(_alpha(width, height)), width=width, height=height)
        )
        assert sorted(photo) == sorted(mask) == list(ladder_for(width, height))
        for edge, tier in mask.items():
            assert (tier.width, tier.height) == (photo[edge].width, photo[edge].height)
            assert _open(tier).size == _open(photo[edge]).size

    def test_a_mask_rung_is_lossless_webp_white_with_the_mask_as_alpha(self) -> None:
        alpha = _alpha(1024, 512)
        minted = _by_edge(mint_mask_tiers(encode_mask_png(alpha), width=1024, height=512))
        for tier in minted.values():
            img = _open(tier)
            assert img.format == "WEBP" and img.mode == "RGBA"
            assert tier.data[12:16] == b"VP8L"  # lossless
            assert np.asarray(img)[..., :3].min() == 255  # white, under transparent pixels too
        # 1024 -> 256 is a 4 x 4 area average of the full mask (within one level: the
        # resampler rounds once per axis)
        blocks = alpha.reshape(128, 4, 256, 4).astype(np.float64).mean(axis=(1, 3))
        got = np.asarray(_open(minted[256]))[..., 3].astype(np.float64)
        assert np.abs(got - blocks).max() <= 1.0
        assert np.abs(got - blocks).mean() <= 0.1
        # the native rung is the stored mask, bit for bit
        assert np.array_equal(np.asarray(_open(minted[1024]))[..., 3], alpha)

    def test_a_hard_edged_mask_never_rings_or_overshoots(self) -> None:
        alpha = np.zeros((1000, 1000), dtype=np.uint8)
        alpha[250:750, 250:750] = 255
        for tier in mint_mask_tiers(encode_mask_png(alpha), width=1000, height=1000):
            a = np.asarray(_open(tier))[..., 3]
            n = a.shape[0]
            lo, hi = n // 4, 3 * n // 4
            assert a[: lo - 1].max() == 0 and a[hi + 1 :].max() == 0
            assert a[:, : lo - 1].max() == 0 and a[:, hi + 1 :].max() == 0
            assert a[lo + 1 : hi - 1, lo + 1 : hi - 1].min() == 255

    def test_a_mask_of_another_size_is_refused_never_resized(self) -> None:
        mask = encode_mask_png(_alpha(600, 400))
        for width, height in ((400, 600), (601, 400), (300, 200)):
            with pytest.raises(TierRefusal) as refused:
                mint_mask_tiers(mask, width=width, height=height)
            assert refused.value.code == "mask_size_mismatch"

    @pytest.mark.parametrize(
        "not_a_mask",
        [
            _png(_scene(60, 40)),  # RGB
            _png(_alpha(60, 40)),  # L: no alpha channel, masks nothing
            _png(np.dstack([_scene(60, 40), _alpha(60, 40)])),  # RGBA
            _jpeg(_scene(60, 40)),
        ],
    )
    def test_only_a_png_la_mask_is_a_mask(self, not_a_mask: bytes) -> None:
        with pytest.raises(TierRefusal) as refused:
            mint_mask_tiers(not_a_mask, width=60, height=40)
        assert refused.value.code == "not_a_mask"

    def test_the_same_mask_always_mints_the_same_bytes(self) -> None:
        mask = encode_mask_png(_alpha(900, 700))
        first = mint_mask_tiers(mask, width=900, height=700)
        again = mint_mask_tiers(mask, width=900, height=700)
        assert [t.sha256 for t in first] == [t.sha256 for t in again]
        assert [
            t.long_edge_px for t in mint_mask_tiers(mask, width=900, height=700, max_edge=512)
        ] == [128, 256, 512]


class TestThePairComposited:
    """A photo rung through the mask rung of the same edge shows what the full
    photo through the full mask shows, scaled down: SSIM at least 0.98
    against the full composite downscaled (Lanczos)."""

    @staticmethod
    def _reference(rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        full = Image.fromarray(_over_grey(rgb, alpha))
        return np.asarray(full.resize(size, Image.Resampling.LANCZOS))

    @pytest.mark.parametrize(("width", "height"), [(2400, 1600), (1500, 2300), (1200, 800)])
    def test_every_rung_pair_matches_and_no_wrong_pairing_does(
        self, width: int, height: int
    ) -> None:
        """A store's photo of a striped, lopsided figure. The right pair clears
        the bar on every rung; the same rungs paired wrongly -- the mask
        mirrored, the photo mirrored, the photo turned half round -- do not."""
        rgb, alpha = _as_a_store_serves_it(_scene(width, height)), _alpha(width, height)
        photos = _by_edge(mint_photo_tiers(_png(rgb)))
        masks = _by_edge(mint_mask_tiers(encode_mask_png(alpha), width=width, height=height))
        assert sorted(photos) == sorted(masks) == list(ladder_for(width, height))
        assert len(photos) == 5
        for edge, tier in photos.items():
            photo, mask = np.asarray(_open(tier)), np.asarray(_open(masks[edge]))[..., 3]
            reference = self._reference(rgb, alpha, (tier.width, tier.height))
            assert _ssim(_over_grey(photo, mask), reference) >= 0.98, edge
            assert _ssim(_over_grey(photo, mask[:, ::-1]), reference) < 0.97, edge
            assert _ssim(_over_grey(photo[:, ::-1], mask), reference) < 0.95, edge
            assert _ssim(_over_grey(photo[::-1, ::-1], mask), reference) < 0.97, edge

    def test_fine_detail_in_hue_alone_is_where_the_small_rungs_fall_short(self) -> None:
        """A KNOWN LIMIT, pinned at what was measured: stripes of red against
        blue at one brightness, about 7 px apart on the 128 px rung. Lossy
        WebP stores colour at half resolution (4:2:0), so no quality setting
        carries them (0.937 at quality 100); the pair scores 0.924 at 128 px
        and clears 0.98 from 256 px up. Detail in brightness, as above, is
        unaffected."""
        rgb, alpha = _as_a_store_serves_it(_hue_striped(1200, 800)), _alpha(1200, 800)
        photos = _by_edge(mint_photo_tiers(_png(rgb)))
        masks = _by_edge(mint_mask_tiers(encode_mask_png(alpha), width=1200, height=800))
        scores = {}
        for edge, tier in photos.items():
            pair = _over_grey(np.asarray(_open(tier)), np.asarray(_open(masks[edge]))[..., 3])
            scores[edge] = _ssim(pair, self._reference(rgb, alpha, (tier.width, tier.height)))
        assert 0.92 <= scores[128] < 0.98, scores
        assert all(score >= 0.98 for edge, score in scores.items() if edge != 128), scores

    #: The sandbox sample's floors below 512 px, and what the encoder may
    #: cost there: see the test below.
    SAMPLE_FLOOR = {128: 0.92, 256: 0.96}
    ENCODER_COST = {128: 0.035}

    @pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["name"])
    def test_the_sample_pairs_match_the_downscaled_full_composite(
        self, entry: dict[str, object]
    ) -> None:
        """The committed sandbox sample: flat geometric figures with hard
        edges on a studio backdrop -- the worst case for a photo and a mask
        scaled separately. 0.98 holds from 512 px up. AT 256 AND 128 PX IT
        DOES NOT, and the floors here are the measured ones: the UNENCODED
        rung pair itself scores 0.9815 to 0.997 at 256 px and 0.949 to 0.995
        at 128 (a photo rung's edge pixels blend the backdrop in before the
        mask is applied; the browser scaling the full pair does the same
        today), and lossy WebP's 4:2:0 chroma takes the saturated flat
        colours to 0.965 and 0.924 at the lowest (dark-angel); no quality
        setting clears 0.98 for it (0.9696 at 256 px, quality 95). What the
        encoder costs is bounded here too."""
        original = (SAMPLE / "originals" / f"{entry['name']}.png").read_bytes()
        rgb = np.asarray(Image.open(io.BytesIO(original)).convert("RGB"))
        ref = Image.open(SAMPLE / "reference" / f"{entry['reference']}.png").convert("RGBA")
        alpha = np.asarray(ref)[..., 3].copy()
        height, width = alpha.shape
        clean = _jpeg(rgb)  # its clean copy: JPEG quality 95, 4:4:4
        photos = _by_edge(mint_photo_tiers(clean))
        masks = _by_edge(mint_mask_tiers(encode_mask_png(alpha), width=width, height=height))
        assert len(photos) == 4 and sorted(photos) == sorted(masks)
        shown = np.asarray(Image.open(io.BytesIO(clean)).convert("RGB"))
        for edge, tier in photos.items():
            mask = np.asarray(_open(masks[edge]))[..., 3]
            reference = self._reference(shown, alpha, (tier.width, tier.height))
            score = _ssim(_over_grey(np.asarray(_open(tier)), mask), reference)
            assert score >= self.SAMPLE_FLOOR.get(edge, 0.98), (entry["name"], edge, score)
            unencoded = Image.fromarray(shown).resize(
                (tier.width, tier.height), Image.Resampling.LANCZOS, reducing_gap=None
            )
            lossless = _ssim(_over_grey(np.asarray(unencoded), mask), reference)
            cost = self.ENCODER_COST.get(edge, 0.02)
            assert lossless - score <= cost, (entry["name"], edge, lossless, score)


# --------------------------------------------------------------------------
# golden bytes
# --------------------------------------------------------------------------

#: sha-256 of every rung of three fixed parents, minted by mm-tier-v1 on the
#: locked Pillow 13.0.0 wheel (libwebp 1.6.0). A change here is an encoder
#: drift: ship it as a NEW pipeline version, never under mm-tier-v1.
GOLDEN: dict[str, dict[int, str]] = {
    "photo-rgb": {
        128: "4bcd6d5dc98ac7ed6e8db86d2e76ce428bea4112e953da39bd875a1bf7b3e88d",
        256: "d81a2dd1ac81a5d34786bf2b584a27e012f453597b7fd1863cc28fe5de7297f2",
        512: "42c264c6bc1a00682b3b143965cd2b39ea4d3a78ba346c4d57d2b7d68dc7f2ca",
        1024: "c067d85b378ca499124e05056271f8640b8fb9bf4c3a0b1e4354292d7fc297a0",
        1100: "62585a63f72955cd1040918150d723ed63e9bfa9837e42f9f9f3523ea3467b53",
    },
    "photo-rgba": {
        128: "c2a51e1d436eaca04e4fe6da41b4506dd2f2ddf1406d347f32e166f1552cfca6",
        256: "fd5f75915e2e8e701343a65facaf308da09e46ce1d84f906578535a71b1a404e",
        512: "9c5fcdf735b8ff04169f7cbe2817a0e3e6e48fde199048d6a16bfacbe704d9d8",
        600: "47740e240fd5996950c58704055dc71a685157fa46f7c33b6a9d0fec96f0e800",
    },
    "mask": {
        128: "a10edccf96e0f8330fd54755159b24371ab873ae9c63843ccc79cba57d42606e",
        256: "0feb3a44a8e953b7d42a35e83fbf61a2262dff530a8b89c03f208b541f7e9d49",
        512: "86926d0b86eb9de15d75d1f75d2d7b714b80fc1fb83ada536c324de6dac132a1",
        1024: "6e8f18913e4acd0c3b7e87d3ef127694420f0b4210fc741729c571922e452adf",
        1100: "fc818494fb82feab7d8527ba19755fdd9c5440a36db063891542c5301480e93e",
    },
}


def _golden_rgb(width: int, height: int, seed: int) -> np.ndarray:
    """The golden photo, in INTEGER arithmetic only: the same array on every
    machine (no float rounding stands between a CPU and the recorded shas)."""
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.stack(
        [
            100 + 120 * xx // width,
            90 + 110 * yy // height,
            60 + 150 * (xx + yy) // (width + height),
        ],
        axis=-1,
    )
    inside = (
        (xx > width // 4) & (xx < 2 * width // 3) & (yy > height // 5) & (yy < 9 * height // 10)
    )
    stripe = ((xx + 2 * yy) // 9) % 2
    figure = np.stack([70 + 140 * stripe, 50 + 0 * xx, 200 - 110 * stripe], axis=-1)
    arr[inside] = figure[inside]
    noise = np.random.default_rng(seed).integers(-4, 5, (height, width, 1))
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def _golden_alpha(width: int, height: int) -> np.ndarray:
    """The golden mask, integer arithmetic only: the figure's box, a ramp at
    its left edge, clear outside."""
    yy, xx = np.mgrid[0:height, 0:width]
    inside = (
        (xx > width // 4) & (xx < 2 * width // 3) & (yy > height // 5) & (yy < 9 * height // 10)
    )
    ramp = np.clip((xx - width // 4) * 6, 0, 255)
    # a soft, curved inner edge too, so the encoder's effort setting shows in the bytes
    dent = (xx - width // 2) ** 2 + (yy - height // 2) ** 2
    soft = np.clip(dent * 255 // (height * height // 16), 0, 255)
    return np.where(inside, np.minimum(ramp, soft), 0).astype(np.uint8)


class TestGoldenBytes:
    @staticmethod
    def _minted() -> dict[str, dict[int, str]]:
        rgb = _golden_rgb(1100, 733, seed=11)
        rgba = np.dstack([_golden_rgb(400, 600, seed=12), _golden_alpha(400, 600)])
        rgba[rgba[..., 3] == 0] = 0
        mask = encode_mask_png(_golden_alpha(1100, 733))
        return {
            "photo-rgb": {t.long_edge_px: t.sha256 for t in mint_photo_tiers(_png(rgb))},
            "photo-rgba": {t.long_edge_px: t.sha256 for t in mint_photo_tiers(_png(rgba))},
            "mask": {
                t.long_edge_px: t.sha256 for t in mint_mask_tiers(mask, width=1100, height=733)
            },
        }

    def test_the_fixture_parents_are_the_recorded_ones(self) -> None:
        """The goldens pin the encoder only while their inputs are fixed: if
        THIS fails, the fixture moved, not the encoder."""
        assert hashlib.sha256(_golden_rgb(1100, 733, seed=11).tobytes()).hexdigest() == SCENE_SHA
        assert hashlib.sha256(_golden_alpha(1100, 733).tobytes()).hexdigest() == ALPHA_SHA

    def test_every_rung_is_the_recorded_bytes(self) -> None:
        assert self._minted() == GOLDEN


SCENE_SHA = "935075634e9700c547b157e9e46efa6fc0316b15f78047521e41e6c6820bf7b3"
ALPHA_SHA = "a55546d651e0a0662be4de0aa42ff0f66974fe2086514bf565786c45e2efd6b7"


# --------------------------------------------------------------------------
# what is refused, and the giant-parent path
# --------------------------------------------------------------------------


class TestRefusals:
    def _code(self, parent: bytes, **kw: object) -> str:
        with pytest.raises(TierRefusal) as refused:
            mint_photo_tiers(parent, **kw)  # type: ignore[arg-type]
        return refused.value.code

    def test_a_parent_that_is_neither_jpeg_nor_png_is_refused(self) -> None:
        webp, gif = io.BytesIO(), io.BytesIO()
        Image.fromarray(_scene(60, 40)).save(webp, format="WEBP")
        Image.fromarray(_scene(60, 40)).convert("P").save(gif, format="GIF")
        for parent in (webp.getvalue(), gif.getvalue(), b"", b"not an image at all"):
            assert self._code(parent) == "parent_format"

    def test_a_truncated_or_corrupt_parent_is_refused_as_undecodable(self) -> None:
        jpeg, png = _jpeg(_scene(300, 200)), _png(_scene(300, 200))
        assert self._code(jpeg[: len(jpeg) // 2]) == "parent_undecodable"
        assert self._code(png[: len(png) // 2]) == "parent_undecodable"
        assert self._code(png[:20]) == "parent_undecodable"
        assert self._code(jpeg[:3] + b"\x00" * 40) == "parent_undecodable"

    def test_running_out_of_memory_is_not_the_parents_fault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A MemoryError while decoding is the worker's, not a refusal: the
        parent must stay retryable, never be written off as undecodable."""

        def starved(self: Image.Image) -> None:
            raise MemoryError

        monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.load", starved)
        with pytest.raises(MemoryError):
            mint_photo_tiers(_png(_scene(300, 200)))

    def test_a_truncated_mask_is_refused_as_undecodable(self) -> None:
        mask = encode_mask_png(_alpha(300, 200))
        with pytest.raises(TierRefusal) as refused:
            mint_mask_tiers(mask[: len(mask) // 2], width=300, height=200)
        assert refused.value.code == "parent_undecodable"

    def test_a_photo_that_is_not_rgb_or_rgba_is_refused(self) -> None:
        grey = io.BytesIO()
        Image.fromarray(_alpha(60, 40)).save(grey, format="JPEG")
        cmyk = io.BytesIO()
        Image.fromarray(_scene(60, 40)).convert("CMYK").save(cmyk, format="JPEG")
        palette = io.BytesIO()
        Image.fromarray(_scene(60, 40)).convert("P").save(palette, format="PNG")
        for parent in (grey.getvalue(), cmyk.getvalue(), palette.getvalue()):
            assert self._code(parent) == "parent_mode"

    def test_a_refusal_is_a_value_error_with_a_stable_code(self) -> None:
        err = TierRefusal("parent_format", "why")
        assert isinstance(err, ValueError) and err.code == "parent_format" and str(err) == "why"


class TestGiantParents:
    """A parent over the pixel bound takes a bounded path, never a silent
    skip: a JPEG is scaled in its decoder (DCT, 1/2 to 1/8), anything else is
    decoded whole when the decoded-bytes gate has the room."""

    #: the small-scale stand-in: "giant" from 500k pixels, the draft keeps the
    #: top rung (a 4,200 px strip is a giant here, and cheap to make)
    SMALL = TierLimits(giant_pixels=500_000, draft_min_edge=2048)

    def test_the_defaults_are_64_megapixels_and_twice_the_top_rung(self) -> None:
        assert TierLimits() == TierLimits(giant_pixels=64_000_000, draft_min_edge=4096)
        assert DecodedBytes().capacity == 768 * 1024 * 1024
        # the largest live parent (14,174 x 7,525) fits, as a mask and as a half-scale JPEG
        assert DecodedBytes().capacity > 14174 * 7525 * 5

    def test_a_draft_that_could_not_carry_the_top_rung_is_no_limit(self) -> None:
        """Every rung must be a downscale of what was decoded: a draft bound
        under the top rung would mint upscales (or a partial ladder)."""
        with pytest.raises(ValueError, match="draft_min_edge"):
            TierLimits(draft_min_edge=2047)
        with pytest.raises(ValueError, match="giant_pixels"):
            TierLimits(giant_pixels=0)

    def test_a_giant_jpeg_is_decoded_scaled_down(self) -> None:
        parent = _jpeg(_scene(4200, 300))
        gate = DecodedBytes(64 * 1024 * 1024)
        minted = mint_photo_tiers(parent, limits=self.SMALL, gate=gate)
        assert gate.peak == 2100 * 150 * 4  # decoded at 1/2: never the full 4200 x 300
        assert gate.in_use == 0
        # every rung, each sized from the parent's TRUE size, not the draft's
        assert [(t.width, t.height) for t in minted] == [
            rung_size(4200, 300, edge) for edge in RUNGS
        ]
        for tier in minted:
            assert _open(tier).size == (tier.width, tier.height)

    def test_the_draft_is_the_deepest_scale_that_keeps_its_bound(self) -> None:
        deep = TierLimits(giant_pixels=500_000, draft_min_edge=2048)
        assert tiers.draft_scale(16400, 100, deep) == 8  # 2050 px left
        assert tiers.draft_scale(16300, 100, deep) == 4  # 1/8 would leave 2038
        assert tiers.draft_scale(8100, 100, deep) == 2
        assert tiers.draft_scale(4096, 200, deep) == 2  # 2048 exactly: kept
        assert tiers.draft_scale(4094, 200, deep) == 1  # over the bound, no scale keeps 2,048
        assert tiers.draft_scale(700, 700, deep) == 1  # not a giant

    def test_the_real_bound_keeps_every_rung_of_a_giant(self) -> None:
        """At the real limits a giant's draft is at least 4,096 px on its long
        edge (twice the top rung), so every rung is a downscale of it."""
        for width, height in ((14174, 7525), (9000, 9000), (15254, 22160), (65000, 1000)):
            scale = tiers.draft_scale(width, height, TierLimits())
            assert scale in (2, 4, 8)
            assert max(-(-width // scale), -(-height // scale)) >= 4096
        assert tiers.draft_scale(14174, 7525, TierLimits()) == 2  # the largest live parent
        assert tiers.draft_scale(8000, 8000, TierLimits()) == 1  # at the bound: not a giant
        assert tiers.draft_scale(8100, 7950, TierLimits()) == 1  # 1/2 would leave 4,050

    def test_a_drafted_giant_shows_what_the_full_decode_shows(self) -> None:
        scene = _corners(4200, 600)
        scene[150:450, 1000:3200] = _scene(2200, 300)
        parent = _jpeg(scene)
        full = _by_edge(mint_photo_tiers(parent))
        gate = DecodedBytes(64 * 1024 * 1024)
        drafted = _by_edge(mint_photo_tiers(parent, limits=self.SMALL, gate=gate))
        assert gate.peak == 2100 * 300 * 4
        assert sorted(full) == sorted(drafted) == list(RUNGS)
        for edge in full:
            a, b = np.asarray(_open(full[edge])), np.asarray(_open(drafted[edge]))
            assert a.shape == b.shape and _ssim(a, b) >= 0.98, edge
        got = TestNoRotation._corner_colours(np.asarray(_open(drafted[2048])))
        for corner, want in zip(got, (RED, GREEN, BLUE, YELLOW), strict=True):
            assert TestNoRotation._close(corner, want)

    def test_a_giant_png_is_decoded_whole_when_the_gate_has_room(self) -> None:
        gate = DecodedBytes(64 * 1024 * 1024)
        minted = mint_photo_tiers(_png(_scene(4200, 300)), limits=self.SMALL, gate=gate)
        assert gate.peak == 4200 * 300 * 4 and gate.in_use == 0
        assert [t.long_edge_px for t in minted] == list(RUNGS)

    def test_a_giant_mask_is_decoded_whole_when_the_gate_has_room(self) -> None:
        gate = DecodedBytes(64 * 1024 * 1024)
        mask = encode_mask_png(_alpha(4200, 300))
        minted = mint_mask_tiers(mask, width=4200, height=300, gate=gate)
        assert gate.peak == 4200 * 300 * 5 and gate.in_use == 0
        assert [(t.width, t.height) for t in minted] == [
            rung_size(4200, 300, edge) for edge in RUNGS
        ]

    def test_a_parent_the_gate_can_never_hold_is_refused_as_too_large(self) -> None:
        for parent in (_png(_scene(1600, 1200)), _jpeg(_scene(1600, 1200))):
            with pytest.raises(TierRefusal) as refused:
                mint_photo_tiers(parent, gate=DecodedBytes(1_000_000))
            assert refused.value.code == "parent_too_large"
        with pytest.raises(TierRefusal) as refused:
            mint_mask_tiers(
                encode_mask_png(_alpha(1600, 1200)),
                width=1600,
                height=1200,
                gate=DecodedBytes(1_000_000),
            )
        assert refused.value.code == "parent_too_large"

    def test_a_refusal_on_the_gate_decodes_no_pixel_of_a_photo_or_a_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate is taken BEFORE the decode: a parent too big for it is
        refused from its header, never decoded first (an OOM in a small pod).
        The spy is on Pillow's real pixel load, so ANY decode ahead of the gate
        (the wrapper, a convert, a getchannel) is seen, not only tiers._decode."""
        decoded: list[tuple[int, int]] = []
        real = ImageFile.ImageFile.load

        def spy(self: ImageFile.ImageFile, *args: object) -> object:
            decoded.append(self.size)
            return real(self, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(ImageFile.ImageFile, "load", spy)
        rgba = np.dstack([_scene(1600, 1200), _alpha(1600, 1200)])
        for parent in (_png(_scene(1600, 1200)), _jpeg(_scene(1600, 1200)), _png(rgba)):
            with pytest.raises(TierRefusal) as refused:
                mint_photo_tiers(parent, gate=DecodedBytes(1_000_000))
            assert refused.value.code == "parent_too_large"
        with pytest.raises(TierRefusal) as refused:
            mint_mask_tiers(
                encode_mask_png(_alpha(1600, 1200)),
                width=1600,
                height=1200,
                gate=DecodedBytes(1_000_000),
            )
        assert refused.value.code == "parent_too_large"
        assert decoded == []
        # the spy sees a decode when the gate has room, so an empty list means none was made
        mint_mask_tiers(encode_mask_png(_alpha(60, 40)), width=60, height=40)
        assert decoded and set(decoded) == {(60, 40)}

    def test_the_gate_counts_the_working_copy_of_each_kind(self) -> None:
        """As Pillow holds them: 4 bytes a pixel for RGB, RGBA and LA alike
        (tests/test_tier_memory.py measures the real thing). A transparent
        photo is held twice (as stored, and premultiplied); a mask is its LA
        plus its one-byte alpha plane."""
        rgba = np.dstack([_scene(300, 200), _alpha(300, 200)])
        for parent, per_pixel in ((_jpeg(_scene(300, 200)), 4), (_png(rgba), 8)):
            gate = DecodedBytes(10_000_000)
            mint_photo_tiers(parent, gate=gate)
            assert (gate.peak, gate.in_use) == (300 * 200 * per_pixel, 0)
        gate = DecodedBytes(10_000_000)
        mint_mask_tiers(encode_mask_png(_alpha(300, 200)), width=300, height=200, gate=gate)
        assert (gate.peak, gate.in_use) == (300 * 200 * 5, 0)

    def test_the_gate_is_released_when_minting_fails(self) -> None:
        gate = DecodedBytes(10_000_000)
        png = _png(_scene(300, 200))
        with pytest.raises(TierRefusal):
            mint_photo_tiers(png[: len(png) // 2], gate=gate)
        assert gate.in_use == 0

    def test_a_header_that_lies_about_its_size_is_bounded_by_the_gate(self) -> None:
        """A PNG whose header claims 60000 x 60000 is refused from the header
        alone: nothing is decoded to find out."""
        png = bytearray(_png(_scene(60, 40)))
        ihdr = struct.pack(">II", 60000, 60000) + bytes(png[24:29])
        png[16:29] = ihdr
        png[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
        with pytest.raises(TierRefusal) as refused:
            mint_photo_tiers(bytes(png))
        assert refused.value.code == "parent_too_large"


class TestDecodedBytes:
    def test_it_counts_what_is_held_and_the_peak(self) -> None:
        gate = DecodedBytes(100)
        with gate.hold(60):
            assert gate.in_use == 60
            with gate.hold(40):
                assert gate.in_use == 100
        assert (gate.in_use, gate.peak, gate.capacity) == (0, 100, 100)

    def test_more_than_the_whole_capacity_is_refused_not_awaited(self) -> None:
        gate = DecodedBytes(100)
        with pytest.raises(TierRefusal) as refused, gate.hold(101):
            pass  # pragma: no cover
        assert refused.value.code == "parent_too_large" and gate.in_use == 0

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_capacity_or_a_hold_that_is_not_positive_is_an_error(self, bad: int) -> None:
        with pytest.raises(ValueError):
            DecodedBytes(bad)
        with pytest.raises(ValueError), DecodedBytes(10).hold(bad):
            pass  # pragma: no cover

    def test_a_second_decode_waits_for_room_and_never_exceeds_the_capacity(self) -> None:
        gate = DecodedBytes(100)
        order: list[str] = []

        def second() -> None:
            with gate.hold(70):
                order.append("second")

        with gate.hold(60):
            worker = threading.Thread(target=second, daemon=True)
            worker.start()
            worker.join(0.3)
            assert worker.is_alive() and order == []  # 60 + 70 would pass 100: it waits
            order.append("first")
        worker.join(5)
        assert not worker.is_alive() and order == ["first", "second"]
        assert (gate.in_use, gate.peak) == (0, 70)

    def test_the_hold_is_released_when_its_body_raises(self) -> None:
        gate = DecodedBytes(100)
        with pytest.raises(RuntimeError), gate.hold(50):
            raise RuntimeError("decode failed")
        assert gate.in_use == 0
