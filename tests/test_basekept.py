"""The July base-keep pass (fc-design-assets/pipeline-scripts/basekept.py, July
2026), ported: BiRefNet everywhere, plus an IS-Net answer restricted to a
feathered band at the bottom of the figure, kept only where it is connected
to the figure. Plain BiRefNet drops figure BASES (measured 2026-10-11 on the
July references: IoU 0.72-0.89 on five of seven); the base is what stands on
the shelf, so the grounding numbers depend on it.

No model runs here: fake backends answer fixed alphas, and IS-Net's
pre/post-processing is exercised against a fake ONNX session.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.workers.basekept import (
    BAND_FRAC,
    BaseKeptMattingBackend,
    base_kept_alpha,
)
from app.workers.matting import IsNetMattingBackend, MattingBackend, get_matting_backend
from PIL import Image

H, W = 200, 100


def _figure_without_base() -> np.ndarray:
    """BiRefNet's answer: the figure's body (rows 20..159), no base."""
    b = np.zeros((H, W), dtype=np.float32)
    b[20:160, 40:60] = 255
    return b


def _donor_with_base() -> np.ndarray:
    """An IS-Net answer that also sees the base: a wide disc under the feet
    (rows 160..179), touching the body, plus a far-away speck of backdrop."""
    i = _figure_without_base().copy()
    i[160:180, 15:85] = 230
    i[5:12, 2:9] = 200  # backdrop speck, top-left, not connected to the figure
    return i


class _Fixed(MattingBackend):
    def __init__(self, alpha: np.ndarray, name: str = "fixed") -> None:
        self.alpha = alpha
        self.name = name  # type: ignore[misc]

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        return self.alpha.astype(np.uint8)

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError


class TestBaseKeptAlpha:
    def test_the_base_is_restored_under_the_figure(self) -> None:
        out, donor = base_kept_alpha(_figure_without_base(), {"isnet": _donor_with_base()})
        assert donor == "isnet"
        assert out[170, 50] > 200  # inside the base, under the feet
        assert out[170, 20] > 100  # the base's wide part
        assert out[20:160, 40:60].min() == 255  # the body is untouched

    def test_nothing_is_added_above_the_band(self) -> None:
        b = _figure_without_base()
        out, _ = base_kept_alpha(b, {"isnet": _donor_with_base()})
        band_top = int(20 + BAND_FRAC * (159 - 20))
        assert np.array_equal(out[: band_top - 40], np.clip(b[: band_top - 40], 0, 255))

    def test_backdrop_specks_never_join_the_figure(self) -> None:
        out, _ = base_kept_alpha(_figure_without_base(), {"isnet": _donor_with_base()})
        assert out[5:12, 2:9].max() == 0

    def test_a_stray_blob_in_birefnets_own_answer_is_dropped(self) -> None:
        b = _figure_without_base()
        b[185:195, 2:8] = 255  # a blob BiRefNet kept, nowhere near the figure
        out, _ = base_kept_alpha(b, {"isnet": _donor_with_base()})
        assert out[185:195, 2:8].max() == 0
        assert out[100, 50] == 255

    def test_the_donor_that_sees_more_base_is_chosen(self) -> None:
        weak = _figure_without_base().copy()
        weak[160:165, 45:55] = 200
        _, donor = base_kept_alpha(
            _figure_without_base(), {"weak": weak, "strong": _donor_with_base()}
        )
        assert donor == "strong"

    def test_a_base_not_connected_to_the_figure_is_not_kept(self) -> None:
        i = _figure_without_base().copy()
        i[185:195, 10:90] = 230  # a slab well clear of the feet
        out, _ = base_kept_alpha(_figure_without_base(), {"isnet": i})
        assert out[190, 50] == 0

    def test_an_empty_answer_is_returned_as_it_is(self) -> None:
        b = np.zeros((H, W), dtype=np.float32)
        out, donor = base_kept_alpha(b, {"isnet": _donor_with_base()})
        assert donor is None
        assert out.max() == 0

    def test_output_is_uint8(self) -> None:
        out, _ = base_kept_alpha(_figure_without_base(), {"isnet": _donor_with_base()})
        assert out.dtype == np.uint8


class TestBaseKeptBackend:
    def test_predict_alpha_combines_primary_and_donors(self) -> None:
        backend = BaseKeptMattingBackend(
            primary=_Fixed(_figure_without_base(), "birefnet"),
            donors=[_Fixed(_donor_with_base(), "isnet")],
        )
        img = Image.new("RGB", (W, H), (9, 9, 9))
        alpha = backend.predict_alpha(img)
        assert alpha.shape == (H, W)
        assert alpha[170, 50] > 200
        assert backend.name == "birefnet_basekept"

    def test_matte_never_widens_existing_transparency(self) -> None:
        backend = BaseKeptMattingBackend(
            primary=_Fixed(_figure_without_base(), "birefnet"),
            donors=[_Fixed(_donor_with_base(), "isnet")],
        )
        arr = np.zeros((H, W, 4), dtype=np.uint8)
        arr[..., 3] = 255
        arr[100, 50, 3] = 7
        out = np.asarray(backend.matte(Image.fromarray(arr, mode="RGBA")))
        assert out[100, 50, 3] == 7
        assert out[170, 50, 3] > 200

    def test_it_needs_at_least_one_donor(self) -> None:
        with pytest.raises(ValueError):
            BaseKeptMattingBackend(primary=_Fixed(_figure_without_base()), donors=[])


class _FakeIsNetSession:
    """Duck-types onnxruntime: records its input, answers a raw (un-normalised)
    map that is high on the left half."""

    class _Input:
        name = "input"

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def get_inputs(self) -> list[_FakeIsNetSession._Input]:
        return [self._Input()]

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        (x,) = feed.values()
        self.seen = x
        out = np.full((1, 1, 1024, 1024), 3.0, dtype=np.float32)
        out[..., :512] = 9.0
        return [out]


class TestIsNet:
    def test_preprocess_scales_by_the_image_maximum_and_subtracts_the_mean(self) -> None:
        session = _FakeIsNetSession()
        backend = IsNetMattingBackend(session=session)
        backend.predict_alpha(Image.new("RGB", (64, 32), (100, 50, 25)))
        assert session.seen is not None
        assert session.seen.shape == (1, 3, 1024, 1024)
        # red is the image maximum, so it scales to 1.0 before the mean
        assert session.seen[0, 0, 0, 0] == pytest.approx(1.0 - 0.485, abs=1e-5)
        assert session.seen[0, 1, 0, 0] == pytest.approx(0.5 - 0.456, abs=1e-5)

    def test_postprocess_is_min_max_without_a_sigmoid(self) -> None:
        backend = IsNetMattingBackend(session=_FakeIsNetSession())
        alpha = backend.predict_alpha(Image.new("RGB", (64, 32), (1, 2, 3)))
        assert alpha.shape == (32, 64)
        assert alpha[16, 2] == 255
        assert alpha[16, 61] == 0

    def test_it_names_itself(self) -> None:
        assert IsNetMattingBackend.name == "isnet"

    def test_unconfigured_weights_are_refused_by_name(self) -> None:
        backend = IsNetMattingBackend()
        with pytest.raises(RuntimeError, match="IsNetMattingBackend"):
            backend.predict_alpha(Image.new("RGB", (4, 4)))


class TestFactory:
    def test_the_configured_base_keep_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setattr(
            "app.config.get_settings",
            lambda: Settings(
                matting_backend="birefnet_basekept",
                matting_model_path="/m/birefnet.onnx",
                matting_isnet_general_path="/m/isnet-general-use.onnx",
                matting_isnet_anime_path="/m/isnet-anime.onnx",
            ),
        )
        backend = get_matting_backend()
        assert isinstance(backend, BaseKeptMattingBackend)
        assert backend.primary._model_path == "/m/birefnet.onnx"  # type: ignore[attr-defined]
        assert [d._model_path for d in backend.donors] == [  # type: ignore[attr-defined]
            "/m/isnet-general-use.onnx",
            "/m/isnet-anime.onnx",
        ]
