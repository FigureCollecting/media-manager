"""Tests for the MASK-1 mask artifact (app.workers.mask) and the alpha-native
grounding it reports (app.workers.grounding.measure_alpha_grounding).

The mask is OUR work product: a separate single-channel alpha mask the client
composites over the clean derivative at display time (Ross 2026-09-26). The
clean derivative itself is never cut out. The grounding numbers follow
fc-api-contract 0.4.0 ProductImage (bottom_margin_frac, contact_band: opaque
means alpha above 10 of 255) plus the CONTRACT-1 follow-up fields the case
view measures client-side today (top margin, opaque box).

No real model runs here: the stub backend and fake BiRefNet sessions only.
"""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import pytest
from app.workers.grounding import (
    ALPHA_THRESHOLD,
    AlphaGrounding,
    compute_bottom_margin_frac,
    compute_contact_band,
    measure_alpha_grounding,
)
from app.workers.mask import (
    BOTTOM_BLEED_WIDTH_FRAC,
    FULL_FRAME_COVERAGE,
    MASK_MIME,
    MASK_PIPELINE_VERSION,
    MaskOutcome,
    assess_mask_quality,
    build_mask_artifact,
    composite_with_mask,
    decode_mask_png,
    encode_mask_png,
)
from app.workers.matting import (
    BiRefNetMattingBackend,
    MattingBackend,
    StubMattingBackend,
)
from PIL import Image

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _golden_alpha() -> np.ndarray:
    """200 x 100 (W x H): a body over rows 10..79, cols 50..149, standing on
    a narrower base (rows 72..79, cols 80..119). Every grounding number is
    exact by construction (see TestMeasureAlphaGrounding.test_golden_values)."""
    alpha = np.zeros((100, 200), dtype=np.uint8)
    alpha[10:72, 50:150] = 255
    alpha[72:80, 80:120] = 255
    return alpha


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _studio_photo(fmt: str = "PNG") -> bytes:
    """A near-white backdrop with a dark figure on a narrower base -- the
    stub backend separates it exactly (uniform backdrop)."""
    arr = np.full((100, 80, 3), (250, 250, 248), dtype=np.uint8)
    arr[20:85, 25:55] = (40, 30, 90)  # body
    arr[85:90, 30:50] = (40, 30, 90)  # base
    return _encode(Image.fromarray(arr, mode="RGB"), fmt)


class _ConstantAlphaBackend(MattingBackend):
    """predict_alpha answers one fixed array, whatever the image."""

    name = "constant"

    def __init__(self, alpha: np.ndarray) -> None:
        self._alpha = alpha

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        assert (image.height, image.width) == self._alpha.shape
        return self._alpha.copy()

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError("the mask lane must never ask for a cut-out")


class _FakeOrtSession:
    """Duck-types onnxruntime.InferenceSession: a logit map that is strongly
    foreground on the left half and background on the right half."""

    class _Input:
        name = "input_image"

    def get_inputs(self) -> list[_FakeOrtSession._Input]:
        return [self._Input()]

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        (arr,) = feed.values()
        h, w = arr.shape[2], arr.shape[3]
        logits = np.full((1, 1, h, w), -12.0, dtype=np.float32)
        logits[..., : w // 2] = 12.0
        return [logits]


# ---------------------------------------------------------------------------
# grounding on a bare alpha mask
# ---------------------------------------------------------------------------


class TestMeasureAlphaGrounding:
    def test_golden_values(self) -> None:
        g = measure_alpha_grounding(_golden_alpha())
        assert g == AlphaGrounding(
            bottom_margin_frac=pytest.approx(20 / 100),
            top_margin_frac=pytest.approx(10 / 100),
            box_top_frac=pytest.approx(10 / 100),
            box_bottom_frac=pytest.approx(80 / 100),
            box_left_frac=pytest.approx(50 / 200),
            box_right_frac=pytest.approx(150 / 200),
            contact_band_center_x_frac=pytest.approx((80 + 119) / 2 / 200),
            contact_band_width_frac=pytest.approx((119 - 80) / 200),
        )

    def test_box_edges_convert_exactly_to_the_case_view_margins(self) -> None:
        """fc-mobile computeTopMarginFrac / computeSideMarginFracs /
        computeBottomMarginFrac read the clear margins; the box is the
        half-open opaque extent, so 1 - box edge == the far margin."""
        g = measure_alpha_grounding(_golden_alpha())
        assert g is not None
        assert g.box_top_frac == g.top_margin_frac
        assert 1 - g.box_bottom_frac == pytest.approx(g.bottom_margin_frac)
        assert g.box_left_frac == pytest.approx(50 / 200)  # computeSideMarginFracs left
        assert 1 - g.box_right_frac == pytest.approx((200 - 1 - 149) / 200)  # its right

    def test_single_pixel(self) -> None:
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[5, 7] = 200
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.top_margin_frac == pytest.approx(0.5)
        assert g.bottom_margin_frac == pytest.approx(0.4)
        assert (g.box_top_frac, g.box_bottom_frac) == (pytest.approx(0.5), pytest.approx(0.6))
        assert (g.box_left_frac, g.box_right_frac) == (pytest.approx(0.35), pytest.approx(0.4))
        assert g.contact_band_center_x_frac == pytest.approx(7 / 20)
        assert g.contact_band_width_frac == 0

    def test_full_frame(self) -> None:
        g = measure_alpha_grounding(np.full((40, 30), 255, dtype=np.uint8))
        assert g is not None
        assert (g.top_margin_frac, g.bottom_margin_frac) == (0, 0)
        assert (g.box_top_frac, g.box_bottom_frac, g.box_left_frac, g.box_right_frac) == (
            0,
            1,
            0,
            1,
        )
        assert g.contact_band_width_frac == pytest.approx(29 / 30)

    def test_alpha_at_threshold_is_not_opaque(self) -> None:
        alpha = np.full((8, 8), ALPHA_THRESHOLD, dtype=np.uint8)
        assert measure_alpha_grounding(alpha) is None

    def test_alpha_one_above_threshold_is_opaque(self) -> None:
        alpha = np.zeros((8, 8), dtype=np.uint8)
        alpha[3, 3] = ALPHA_THRESHOLD + 1
        assert measure_alpha_grounding(alpha) is not None

    def test_threshold_is_the_contract_value(self) -> None:
        assert ALPHA_THRESHOLD == 10

    def test_empty_and_degenerate_buffers_are_unmeasured(self) -> None:
        assert measure_alpha_grounding(np.zeros((5, 5), dtype=np.uint8)) is None
        assert measure_alpha_grounding(np.zeros((0, 5), dtype=np.uint8)) is None

    def test_parity_with_the_july_rgba_port(self) -> None:
        alpha = _golden_alpha()
        rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.bottom_margin_frac == compute_bottom_margin_frac(rgba)
        assert (g.contact_band_center_x_frac, g.contact_band_width_frac) == compute_contact_band(
            rgba
        )

    def test_rejects_a_non_2d_buffer(self) -> None:
        with pytest.raises(ValueError):
            measure_alpha_grounding(np.zeros((4, 4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# the mask PNG: one channel of information, carried as alpha
# ---------------------------------------------------------------------------


class TestMaskPng:
    def test_alpha_carries_the_mask_and_luminance_is_constant_white(self) -> None:
        alpha = _golden_alpha()
        img = Image.open(io.BytesIO(encode_mask_png(alpha)))
        assert img.format == "PNG"
        assert img.mode == "LA"
        assert img.size == (200, 100)
        arr = np.asarray(img)
        assert bool((arr[..., 0] == 255).all())
        assert np.array_equal(arr[..., 1], alpha)

    def test_mime_is_png(self) -> None:
        assert MASK_MIME == "image/png"

    def test_encoding_is_deterministic(self) -> None:
        alpha = _golden_alpha()
        assert encode_mask_png(alpha) == encode_mask_png(alpha.copy())

    def test_round_trip(self) -> None:
        alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
        assert np.array_equal(decode_mask_png(encode_mask_png(alpha)), alpha)

    def test_decode_rejects_a_picture_that_is_not_a_mask(self) -> None:
        with pytest.raises(ValueError):
            decode_mask_png(_encode(Image.new("RGB", (4, 4)), "PNG"))

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((4, 4, 1), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.uint8),
        ],
    )
    def test_encode_rejects_anything_but_a_2d_uint8_mask(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            encode_mask_png(bad)


# ---------------------------------------------------------------------------
# the client composite: CSS mask-image (alpha mode) / canvas destination-in
# ---------------------------------------------------------------------------


class TestCompositeWithMask:
    def test_opaque_clean_copy_takes_the_mask_as_its_alpha(self) -> None:
        clean = Image.new("RGB", (200, 100), (9, 99, 199))
        out = composite_with_mask(clean, _golden_alpha())
        arr = np.asarray(out)
        assert out.mode == "RGBA"
        assert bool((arr[..., :3] == (9, 99, 199)).all())
        assert np.array_equal(arr[..., 3], _golden_alpha())

    def test_multiplies_into_a_clean_copy_that_has_its_own_alpha(self) -> None:
        clean = Image.new("RGBA", (2, 1), (1, 2, 3, 128))
        alpha = np.array([[255, 128]], dtype=np.uint8)
        arr = np.asarray(composite_with_mask(clean, alpha))
        assert arr[0, 0, 3] == 128
        assert arr[0, 1, 3] == 64  # round(128 * 128 / 255)

    def test_rejects_a_mask_of_other_dimensions(self) -> None:
        with pytest.raises(ValueError):
            composite_with_mask(Image.new("RGB", (5, 5)), np.zeros((5, 6), dtype=np.uint8))


# ---------------------------------------------------------------------------
# the backends: one alpha, whether asked for a mask or a cut-out
# ---------------------------------------------------------------------------


class TestPredictAlpha:
    def test_stub_alpha_is_the_alpha_of_its_cutout(self) -> None:
        img = Image.open(io.BytesIO(_studio_photo())).convert("RGBA")
        backend = StubMattingBackend()
        alpha = backend.predict_alpha(img)
        assert alpha.shape == (100, 80)
        assert alpha.dtype == np.uint8
        assert np.array_equal(alpha, np.asarray(backend.matte(img))[..., 3])

    def test_birefnet_alpha_is_the_alpha_of_its_cutout(self) -> None:
        img = Image.new("RGBA", (64, 32), (10, 20, 30, 255))
        backend = BiRefNetMattingBackend(session=_FakeOrtSession())
        alpha = backend.predict_alpha(img)
        assert alpha.shape == (32, 64)
        assert alpha[16, 5] == 255
        assert alpha[16, 60] == 0
        assert np.array_equal(alpha, np.asarray(backend.matte(img))[..., 3])

    def test_predict_alpha_ignores_the_images_own_alpha(self) -> None:
        """The mask is the model's answer alone; the composite multiplies in
        the clean copy's own alpha at display time."""
        img = Image.new("RGBA", (64, 32), (10, 20, 30, 0))
        backend = BiRefNetMattingBackend(session=_FakeOrtSession())
        assert backend.predict_alpha(img)[16, 5] == 255

    def test_a_backend_that_only_mattes_still_yields_an_alpha(self) -> None:
        class _HalfCut(MattingBackend):
            def matte(self, image: Image.Image) -> Image.Image:
                arr = np.asarray(image.convert("RGBA")).copy()
                arr[:, arr.shape[1] // 2 :, 3] = 0
                return Image.fromarray(arr, mode="RGBA")

        alpha = _HalfCut().predict_alpha(Image.new("RGB", (10, 4), (1, 1, 1)))
        assert alpha.tolist() == [[255] * 5 + [0] * 5] * 4

    def test_backends_name_themselves_for_provenance(self) -> None:
        assert StubMattingBackend.name == "stub"
        assert BiRefNetMattingBackend.name == "birefnet"


# ---------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------


class TestAssessMaskQuality:
    def test_nothing_opaque_is_empty(self) -> None:
        alpha = np.zeros((10, 10), dtype=np.uint8)
        assert assess_mask_quality(alpha, measure_alpha_grounding(alpha)) == (
            "empty",
            "no_opaque_pixels",
        )

    def test_a_figure_with_clear_surroundings_is_ok(self) -> None:
        alpha = _golden_alpha()
        assert assess_mask_quality(alpha, measure_alpha_grounding(alpha)) == ("ok", None)

    def test_coverage_at_the_full_frame_line_is_held(self) -> None:
        alpha = np.zeros((10, 100), dtype=np.uint8)
        alpha[:, : int(FULL_FRAME_COVERAGE * 100)] = 255
        alpha[0, 0] = 255
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("held", "full_frame")

    def test_coverage_just_under_the_full_frame_line_is_not_held_for_it(self) -> None:
        alpha = np.zeros((10, 100), dtype=np.uint8)
        alpha[:, : int(FULL_FRAME_COVERAGE * 100) - 1] = 255
        alpha[-1, :] = 0  # lift it off the bottom edge so no bleed either
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("ok", None)

    def test_a_base_band_across_the_whole_bottom_edge_is_held(self) -> None:
        """The black-dress failure shape: the backdrop's floor is kept, so the
        contact band runs edge to edge on the image's last row."""
        alpha = _golden_alpha()
        alpha[90:100, :] = 255
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.contact_band_width_frac >= BOTTOM_BLEED_WIDTH_FRAC
        assert assess_mask_quality(alpha, g) == ("held", "bottom_edge_bleed")

    def test_a_wide_base_that_stops_short_of_the_last_row_is_ok(self) -> None:
        alpha = _golden_alpha()
        alpha[85:95, :] = 255  # edge to edge, but five clear rows below it
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("ok", None)

    def test_a_base_on_the_last_row_short_of_the_bleed_width_is_ok(self) -> None:
        alpha = _golden_alpha()
        alpha[90:100, 2:198] = 255  # 196/200 = 0.975 wide band
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.contact_band_width_frac < BOTTOM_BLEED_WIDTH_FRAC
        assert assess_mask_quality(alpha, g) == ("ok", None)


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------


class TestBuildMaskArtifact:
    def test_produces_a_separate_mask_with_contract_grounding(self) -> None:
        clean = _studio_photo("PNG")
        out = build_mask_artifact(clean, backend=StubMattingBackend())
        assert isinstance(out, MaskOutcome)
        assert out.status == "ok"
        assert out.reason is None
        assert out.backend == "stub"
        assert out.pipeline_version == MASK_PIPELINE_VERSION
        assert out.mask_png is not None
        assert out.mask_sha256 == hashlib.sha256(out.mask_png).hexdigest()
        assert (out.width, out.height) == (80, 100)
        mask = decode_mask_png(out.mask_png)
        assert mask.shape == (100, 80)
        g = out.grounding
        assert g is not None
        assert g.top_margin_frac == pytest.approx(20 / 100)
        assert g.bottom_margin_frac == pytest.approx((100 - 1 - 89) / 100)
        assert (g.box_left_frac, g.box_right_frac) == (pytest.approx(25 / 80), pytest.approx(55 / 80))
        assert g.contact_band_center_x_frac == pytest.approx((30 + 49) / 2 / 80)
        assert g.contact_band_width_frac == pytest.approx((49 - 30) / 80)

    def test_the_clean_copy_is_never_altered(self) -> None:
        clean = _studio_photo("JPEG")
        before = bytes(clean)
        build_mask_artifact(clean, backend=StubMattingBackend())
        assert clean == before

    def test_masked_and_unmasked_appearance_differ(self) -> None:
        out = build_mask_artifact(_studio_photo("PNG"), backend=StubMattingBackend())
        assert out.masked is not None
        assert out.masked.dominant_color == "#281e5a"  # the figure, lowercase
        assert out.unmasked.dominant_color == "#fafaf8"  # the backdrop dominates
        # ThumbHash header bit 23 = has-alpha: set only with the mask applied.
        masked_hdr = base64.b64decode(out.masked.thumbhash)
        plain_hdr = base64.b64decode(out.unmasked.thumbhash)
        assert masked_hdr[2] & 0x80
        assert not plain_hdr[2] & 0x80

    def test_an_opaque_clean_copy_has_no_own_alpha_grounding(self) -> None:
        out = build_mask_artifact(_studio_photo("JPEG"), backend=StubMattingBackend())
        assert out.own_alpha_grounding is None

    def test_a_clean_copy_with_its_own_alpha_is_measured_on_it(self) -> None:
        rgba = np.zeros((20, 10, 4), dtype=np.uint8)
        rgba[5:15, 2:8] = (200, 10, 10, 255)
        clean = _encode(Image.fromarray(rgba, mode="RGBA"), "PNG")
        out = build_mask_artifact(clean, backend=_ConstantAlphaBackend(np.full((20, 10), 255, np.uint8)))
        assert out.own_alpha_grounding is not None
        assert out.own_alpha_grounding.top_margin_frac == pytest.approx(5 / 20)

    def test_an_empty_answer_stores_no_mask(self) -> None:
        out = build_mask_artifact(
            _studio_photo("PNG"), backend=_ConstantAlphaBackend(np.zeros((100, 80), np.uint8))
        )
        assert (out.status, out.reason) == ("empty", "no_opaque_pixels")
        assert out.mask_png is None
        assert out.mask_sha256 is None
        assert out.grounding is None
        assert out.masked is None

    def test_a_full_frame_answer_is_held_but_kept_for_review(self) -> None:
        out = build_mask_artifact(
            _studio_photo("PNG"),
            backend=_ConstantAlphaBackend(np.full((100, 80), 255, np.uint8)),
        )
        assert (out.status, out.reason) == ("held", "full_frame")
        assert out.mask_png is not None
        assert out.grounding is not None

    def test_same_bytes_same_mask_address(self) -> None:
        clean = _studio_photo("PNG")
        a = build_mask_artifact(clean, backend=StubMattingBackend())
        b = build_mask_artifact(clean, backend=StubMattingBackend())
        assert a.mask_sha256 == b.mask_sha256

    def test_reads_exif_orientation_like_the_clean_copy_is_shown(self) -> None:
        """A browser draws the clean copy with its EXIF orientation applied,
        so the mask must be computed on (and sized like) the oriented image."""
        arr = np.full((10, 30, 3), 250, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 CW on display -> 10 wide x 30 tall
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        out = build_mask_artifact(
            buf.getvalue(), backend=_ConstantAlphaBackend(np.full((30, 10), 255, np.uint8))
        )
        assert (out.width, out.height) == (10, 30)
