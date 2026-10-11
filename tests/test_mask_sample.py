"""MASK-1 on the committed synthetic sample (tests/fixtures/mask_sample).

Three claims, each against an independent reference:
  * GROUNDING GOLDENS -- for the same alpha, the lane reports exactly what
    fc-mobile's own client code measures (client_golden.json, measured by
    alphaMargin.ts under node);
  * THE COMPOSITE -- the client's mask-over-clean-copy composite is pixel for
    pixel the July destructive cut-out (tasks._apply_transforms with
    matte=True) made from the same backend answer, so the non-destructive
    overlay shows exactly what the July pipeline showed;
  * THE BLACK-DRESS CASE -- the July failure shape (the dark floor kept edge
    to edge) is held and never published, while a correct answer on the same
    dark-on-dark image is.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from app.workers import matting
from app.workers.mask import build_mask_artifact, composite_with_mask, decode_mask_png
from app.workers.mask_policy import ImageOrigin, mask_eligibility
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.tasks import _apply_transforms
from PIL import Image

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "mask_sample"
MANIFEST = json.loads((SAMPLE / "manifest.json").read_text())
ENTRIES = {e["name"]: e for e in MANIFEST["entries"]}
CLIENT = json.loads((SAMPLE / "client_golden.json").read_text())["measured"]
REFERENCES = sorted(CLIENT)
UNIFORM = [n for n, e in ENTRIES.items() if isinstance(e["backdrop"], list)]


def _original(name: str) -> bytes:
    return (SAMPLE / "originals" / f"{name}.png").read_bytes()


def _ref_alpha(reference: str) -> np.ndarray:
    return np.asarray(Image.open(SAMPLE / "reference" / f"{reference}.png").convert("RGBA"))[
        ..., 3
    ].copy()


class _Answer(MattingBackend):
    """A backend whose answer is a fixed alpha (what BiRefNet said in July)."""

    name = "reference"

    def __init__(self, alpha: np.ndarray) -> None:
        self.alpha = alpha

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        assert (image.height, image.width) == self.alpha.shape
        return self.alpha.copy()

    def matte(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGBA")).copy()
        arr[..., 3] = np.minimum(arr[..., 3], self.predict_alpha(image))
        return Image.fromarray(arr, mode="RGBA")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 10, b > 10
    return float((a & b).sum() / (a | b).sum())


class TestTheSample:
    def test_every_entry_has_an_opaque_original_and_a_reference(self) -> None:
        for name, entry in ENTRIES.items():
            img = Image.open(SAMPLE / "originals" / f"{name}.png")
            assert img.mode == "RGB", name
            ref = Image.open(SAMPLE / "reference" / f"{entry['reference']}.png")
            assert ref.mode == "RGBA" and ref.size == img.size, name

    def test_the_committed_originals_are_what_the_generator_makes(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("gen", SAMPLE / "_generate.py")
        assert spec and spec.loader
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        for name, entry in ENTRIES.items():
            ref = np.asarray(
                Image.open(SAMPLE / "reference" / f"{entry['reference']}.png").convert("RGBA")
            )
            if entry.get("variant") == "black_dress":
                ref = gen._black_dress(ref)
            backdrop = (
                gen._marble(ref)
                if entry["backdrop"] == "marble"
                else gen._flat(ref, tuple(entry["backdrop"]))
            )
            committed = np.asarray(Image.open(SAMPLE / "originals" / f"{name}.png"))
            assert np.array_equal(gen._over(ref, backdrop), committed), name

    def test_origins_split_item_images_from_the_collector_photo(self) -> None:
        verdicts = {
            name: mask_eligibility([ImageOrigin(**o) for o in e["origins"]]).eligible
            for name, e in ENTRIES.items()
        }
        assert [n for n, ok in verdicts.items() if not ok] == ["spike-collector-photo"]


@pytest.mark.parametrize("reference", REFERENCES)
class TestAgainstTheReferenceAnswer:
    def test_mask_and_grounding_match_the_client(self, reference: str) -> None:
        alpha = _ref_alpha(reference)
        out = build_mask_artifact(_original(reference), backend=_Answer(alpha))
        assert (out.status, out.reason) == ("ok", None)
        assert out.mask_png is not None
        assert np.array_equal(decode_mask_png(out.mask_png), alpha)
        g, c = out.grounding, CLIENT[reference]
        assert g is not None
        assert (out.width, out.height) == (c["width"], c["height"])
        assert g.bottom_margin_frac == c["bottom"]
        assert g.top_margin_frac == c["top"]
        assert g.box_left_frac == c["left"]
        assert 1 - g.box_right_frac == pytest.approx(c["right"], abs=1e-12)
        assert g.contact_band_center_x_frac == c["contactBand"]["centerXFrac"]
        assert g.contact_band_width_frac == c["contactBand"]["widthFrac"]

    def test_the_client_composite_is_the_july_cutout(
        self, reference: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answer = _Answer(_ref_alpha(reference))
        monkeypatch.setattr(matting, "get_matting_backend", lambda name=None: answer)
        july_png, _mime, _w, _h = _apply_transforms(
            _original(reference), {"matte": True, "format": "PNG"}
        )
        july = np.asarray(Image.open(io.BytesIO(july_png)).convert("RGBA"))

        out = build_mask_artifact(_original(reference), backend=answer)
        assert out.mask_png is not None
        clean = Image.open(io.BytesIO(_original(reference)))
        ours = np.asarray(composite_with_mask(clean, decode_mask_png(out.mask_png)))
        assert np.array_equal(ours, july)


#: White garments on a near-white backdrop: the stub thresholds colour distance
#: from the backdrop, so it cuts them away. It is a dev stand-in, not
#: segmentation; deployments run BiRefNet (see app.workers.matting).
STUB_WHITE_ON_WHITE = {"madoka", "rem"}


@pytest.mark.parametrize("name", [n for n in UNIFORM if n not in STUB_WHITE_ON_WHITE])
def test_the_default_stub_recovers_figures_on_a_uniform_backdrop(name: str) -> None:
    entry = ENTRIES[name]
    out = build_mask_artifact(_original(name), backend=StubMattingBackend())
    assert out.status == "ok"
    assert out.mask_png is not None
    reference = _ref_alpha(entry["reference"])
    assert _iou(decode_mask_png(out.mask_png), reference) >= 0.97
    g, c = out.grounding, CLIENT[entry["reference"]]
    assert g is not None
    assert abs(g.bottom_margin_frac - c["bottom"]) <= 2 / c["height"]
    assert abs(g.top_margin_frac - c["top"]) <= 2 / c["height"]
    assert abs(g.contact_band_center_x_frac - c["contactBand"]["centerXFrac"]) <= 2 / c["width"]


@pytest.mark.parametrize("name", sorted(STUB_WHITE_ON_WHITE))
def test_the_stub_cuts_away_white_on_white_which_is_why_it_is_not_deployed(name: str) -> None:
    out = build_mask_artifact(_original(name), backend=StubMattingBackend())
    assert out.mask_png is not None
    assert _iou(decode_mask_png(out.mask_png), _ref_alpha(ENTRIES[name]["reference"])) < 0.9


class TestTheBlackDressCase:
    NAME = "madoka-black-dress"

    def _july_failure(self) -> np.ndarray:
        """What the July base-keep pass produced on the black dress: the
        figure plus the dark marble floor, kept edge to edge."""
        alpha = _ref_alpha("madoka")
        floor = int(alpha.shape[0] * 0.94)
        alpha[floor:, :] = 255
        return alpha

    def test_the_july_failure_shape_is_held(self) -> None:
        out = build_mask_artifact(_original(self.NAME), backend=_Answer(self._july_failure()))
        assert (out.status, out.reason) == ("held", "bottom_edge_bleed")

    @pytest.mark.parametrize(
        ("clear_rows", "left", "right"),
        [(2, 0.0, 1.0), (0, 0.015, 0.985), (0, 0.025, 0.975), (1, 0.05, 0.95)],
        ids=["last-two-rows-clear", "97-percent-wide", "95-percent-wide", "edge-and-vignette"],
    )
    def test_near_variants_of_the_july_failure_are_held(
        self, clear_rows: int, left: float, right: float
    ) -> None:
        alpha = _ref_alpha("madoka")
        h, w = alpha.shape
        alpha[int(h * 0.94) : h - clear_rows, int(w * left) : int(round(w * right))] = 255
        out = build_mask_artifact(_original(self.NAME), backend=_Answer(alpha))
        assert (out.status, out.reason) == ("held", "bottom_edge_bleed")

    def test_a_mask_that_removed_nothing_is_held(self) -> None:
        full = np.full_like(_ref_alpha("madoka"), 255)
        out = build_mask_artifact(_original(self.NAME), backend=_Answer(full))
        assert (out.status, out.reason) == ("held", "full_frame")

    def test_a_correct_answer_on_the_dark_image_is_published_with_the_clients_numbers(
        self,
    ) -> None:
        alpha = _ref_alpha("madoka")
        out = build_mask_artifact(_original(self.NAME), backend=_Answer(alpha))
        assert (out.status, out.reason) == ("ok", None)
        g, c = out.grounding, CLIENT["madoka"]
        assert g is not None
        assert g.bottom_margin_frac == c["bottom"]
        assert g.contact_band_center_x_frac == c["contactBand"]["centerXFrac"]
        assert out.masked is not None
        assert out.masked.dominant_color != out.unmasked.dominant_color
