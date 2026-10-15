"""MM-REPORT-1: the media.v1 messages built from the media DB's records, each
field through the contract's field rule (app.spine.field_rules)."""

from __future__ import annotations

import base64
import datetime as dt

import pytest
from app.spine import messages as msg
from app.spine.field_rules import FieldRefused
from app.spine.media_v1 import media_pb2

PI = "0b6e8f9c-1d2e-4f30-8a4b-5c6d7e8f9a0b"
CLEAN = "c1" * 32
MASK = "a7" * 32
AT = dt.datetime(2026, 10, 14, 9, 0, tzinfo=dt.UTC)
TH = base64.b64encode(bytes(range(20))).decode()
GROUND = {
    "bottom_margin_frac": 0.0125,
    "top_margin_frac": 0.04,
    "box_top_frac": 0.04,
    "box_bottom_frac": 0.9875,
    "box_left_frac": 0.2,
    "box_right_frac": 0.8,
    "contact_band_center_x_frac": 0.51,
    "contact_band_width_frac": 0.3,
}


def _copy(**kw: object) -> msg.CopyRecord:
    base: dict[str, object] = {
        "sha256": CLEAN,
        "mime": "image/jpeg",
        "width": 1200,
        "height": 1600,
        "generated_at": AT,
        "thumbhash": TH,
        "dominant_color": "#a1b2c3",
        "own_grounding": None,
    }
    base.update(kw)
    return msg.CopyRecord(**base)  # type: ignore[arg-type]


def _mask(**kw: object) -> msg.MaskRecord:
    base: dict[str, object] = {
        "clean_sha256": CLEAN,
        "pipeline_version": "mask-v1",
        "mask_sha256": MASK,
        "mime": "image/png",
        "generated_at": AT,
        "grounding": GROUND,
        "thumbhash": TH,
        "dominant_color": "#0a0b0c",
    }
    base.update(kw)
    return msg.MaskRecord(**base)  # type: ignore[arg-type]


def test_a_derived_report_carries_the_clean_copy_and_its_own_appearance() -> None:
    built = msg.derivative_report(PI, _copy(), pipeline_version="mm-seed-clean-v1")
    m = built.message
    assert m.product_image_id == PI
    assert m.outcome == media_pb2.DERIVE_OUTCOME_DERIVED
    assert m.derivative_sha256 == bytes.fromhex(CLEAN)
    assert (m.mime, m.width, m.height) == ("image/jpeg", 1200, 1600)
    assert (m.derive_kind, m.pipeline_version) == ("sanitized", "mm-seed-clean-v1")
    assert m.generated_at == "2026-10-14T09:00:00.000000Z"
    assert m.thumbhash == bytes(range(20)) and m.dominant_color == "#a1b2c3"
    assert not m.HasField("grounding")  # an opaque copy has no alpha to stand by
    assert not m.HasField("failure_reason")
    assert built.warnings == []


def test_a_transparent_copy_carries_its_own_grounding() -> None:
    m = msg.derivative_report(PI, _copy(own_grounding=GROUND), pipeline_version="v").message
    assert m.grounding.box_bottom_frac == "0.9875"
    assert m.grounding.contact_band_width_frac == "0.3"


def test_an_unmeasured_or_unsendable_appearance_is_sent_absent() -> None:
    built = msg.derivative_report(
        PI,
        _copy(thumbhash=None, dominant_color="red", own_grounding={**GROUND, "box_top_frac": 2.0}),
        pipeline_version="v",
    )
    m = built.message
    assert not m.HasField("thumbhash") and not m.HasField("dominant_color")
    assert not m.HasField("grounding")
    assert built.warnings == [
        "dominant_color off the contract spelling: dropped",
        "own grounding dropped: box_top_frac: not box_top_frac < box_bottom_frac",
    ]


def test_dimensions_outside_the_contract_are_sent_absent() -> None:
    m = msg.derivative_report(PI, _copy(width=0, height=2**31), pipeline_version="v").message
    assert not m.HasField("width") and not m.HasField("height")


@pytest.mark.parametrize(
    ("kw", "field"),
    [
        ({"sha256": "nope"}, "derivative_sha256"),
        ({"mime": "image/gif"}, "mime"),
    ],
)
def test_a_derivative_the_spine_would_refuse_is_not_built(kw: dict[str, str], field: str) -> None:
    with pytest.raises(FieldRefused) as caught:
        msg.derivative_report(PI, _copy(**kw), pipeline_version="v")
    assert caught.value.field == field


def test_a_bad_depiction_id_or_version_is_not_built() -> None:
    with pytest.raises(FieldRefused, match="uuid"):
        msg.derivative_report("x", _copy(), pipeline_version="v")
    with pytest.raises(FieldRefused, match="pipeline_version"):
        msg.derivative_report(PI, _copy(), pipeline_version="")


def test_a_failed_report_cuts_its_reason_to_one_kilobyte() -> None:
    built = msg.failed_report(
        PI, pipeline_version="v", generated_at=AT, failure_reason="decode\x00" + "x" * 2000
    )
    m = built.message
    assert m.outcome == media_pb2.DERIVE_OUTCOME_FAILED
    assert len(m.failure_reason.encode()) == 1024 and "\x00" not in m.failure_reason
    assert not m.HasField("derivative_sha256") and not m.HasField("mime")
    assert built.warnings == ["failure_reason: NUL removed", "failure_reason: cut to 1024 bytes"]


def test_a_mask_report_carries_the_mask_and_its_masked_appearance() -> None:
    built = msg.mask_report(_mask())
    m = built.message
    assert m.derivative_sha256 == bytes.fromhex(CLEAN)
    assert m.mask_sha256 == bytes.fromhex(MASK)
    assert (m.pipeline_version, m.mime) == ("mask-v1", "image/png")
    assert m.generated_at == "2026-10-14T09:00:00.000000Z"
    assert m.grounding.bottom_margin_frac == "0.0125"
    assert m.thumbhash == bytes(range(20)) and m.dominant_color == "#0a0b0c"
    assert built.warnings == []


def test_a_mask_without_a_whole_grounding_is_not_built() -> None:
    with pytest.raises(FieldRefused, match="bottom_margin_frac"):
        msg.mask_report(_mask(grounding={**GROUND, "bottom_margin_frac": None}))
    with pytest.raises(FieldRefused, match="mime"):
        msg.mask_report(_mask(mime="image/jpeg"))


def test_a_withdrawal_states_the_status_reason_and_instant() -> None:
    built = msg.withdrawal(MASK, reason="store_revoked", withdrawn_at=AT)
    m = built.message
    assert (m.mask_sha256, m.reason, m.withdrawn_at, m.restore) == (
        bytes.fromhex(MASK),
        "store_revoked",
        "2026-10-14T09:00:00.000000Z",
        False,
    )
    restore = msg.restore(MASK)
    assert restore.restore is True and restore.mask_sha256 == bytes.fromhex(MASK)
    assert restore.reason == "" and restore.withdrawn_at == ""


def test_a_long_withdrawal_reason_is_cut_and_warned() -> None:
    built = msg.withdrawal(MASK, reason="r" * 80, withdrawn_at=AT)
    assert built.message.reason == "r" * 64
    assert built.warnings == ["reason: cut to 64 characters"]


def test_an_unmeasured_masked_appearance_is_sent_absent() -> None:
    built = msg.mask_report(_mask(thumbhash=None, dominant_color=None))
    assert not built.message.HasField("thumbhash")
    assert not built.message.HasField("dominant_color")
    assert built.warnings == []
