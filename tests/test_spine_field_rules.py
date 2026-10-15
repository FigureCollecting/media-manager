"""MM-REPORT-1: the media.v1 field rule, applied client-side so the spine
never refuses a call for a data-shaped reason (proto/media/v1/media.proto,
THE FIELD RULE, 0.10.0).

An IDENTITY string is never repaired: one that would be refused is not sent
(``FieldRefused``). FREE TEXT is cleaned and cut here exactly as the spine
would, so it never reaches the 64 KB request cap. Appearance and grounding
values the spine would refuse are dropped (sent absent = not measured) where
the contract lets them be absent, and refused where it does not."""

from __future__ import annotations

import base64
import datetime as dt
import math

import pytest
from app.spine import field_rules as fr

# ---------------------------------------------------------------------------
# identity strings
# ---------------------------------------------------------------------------


def test_an_identity_string_passes_unchanged() -> None:
    assert fr.identity("pipeline_version", "mask-v1") == "mask-v1"
    assert fr.identity("pipeline_version", "é" * 128) == "é" * 128  # 256 bytes exactly


@pytest.mark.parametrize(
    ("value", "why"),
    [("", "empty"), ("a\x00b", "NUL"), ("é" * 128 + "x", "256 bytes")],
)
def test_an_identity_string_is_refused_never_repaired(value: str, why: str) -> None:
    with pytest.raises(fr.FieldRefused, match=why) as caught:
        fr.identity("pipeline_version", value)
    assert caught.value.field == "pipeline_version"


def test_a_uuid_is_an_identity_with_a_shape() -> None:
    good = "0b6e8f9c-1d2e-4f30-8a4b-5c6d7e8f9a0b"
    assert fr.uuid_identity("product_image_id", good) == good
    with pytest.raises(fr.FieldRefused, match="uuid"):
        fr.uuid_identity("product_image_id", "not-a-uuid")


# ---------------------------------------------------------------------------
# free text
# ---------------------------------------------------------------------------


def test_free_text_drops_nul_and_is_cut_at_a_character_boundary() -> None:
    text, warnings = fr.clean_text("ab\x00c", max_bytes=10)
    assert text == "abc" and warnings == ["NUL removed"]
    text, warnings = fr.clean_text("é" * 10, max_bytes=5)  # 2 bytes each
    assert text == "éé" and warnings == ["cut to 5 bytes"]
    assert len(text.encode()) <= 5
    text, warnings = fr.clean_text("plain", max_bytes=5)
    assert text == "plain" and warnings == []


def test_a_failure_reason_is_cut_to_one_kilobyte() -> None:
    reason, warnings = fr.failure_reason("x" * 70_000 + "\x00")
    assert len(reason.encode()) == fr.FAILURE_REASON_MAX_BYTES == 1024
    assert warnings == ["NUL removed", "cut to 1024 bytes"]


def test_a_withdrawal_reason_is_cut_to_64_characters() -> None:
    reason, warnings = fr.withdraw_reason("é" * 70)
    assert reason == "é" * 64 and warnings == ["cut to 64 characters"]
    assert fr.withdraw_reason("deleted") == ("deleted", [])
    assert fr.withdraw_reason("r" * 64) == ("r" * 64, [])  # the bound itself is kept


@pytest.mark.parametrize("value", ["", "\x00\x00", None])
def test_a_withdrawal_with_no_reason_left_is_refused(value: str | None) -> None:
    with pytest.raises(fr.FieldRefused, match="reason"):
        fr.withdraw_reason(value)


def test_a_lease_owner_is_a_label_cut_to_256_bytes() -> None:
    owner, warnings = fr.lease_owner("r" * 300)
    assert len(owner.encode()) == 256 and warnings == ["cut to 256 bytes"]


# ---------------------------------------------------------------------------
# content addresses
# ---------------------------------------------------------------------------


def test_a_sha_is_sent_as_its_32_raw_bytes() -> None:
    hexed = "ab" * 32
    assert fr.sha_bytes("mask_sha256", hexed) == bytes.fromhex(hexed)


@pytest.mark.parametrize("value", ["AB" * 32, "ab" * 31, "zz" * 32, "", None])
def test_anything_else_is_refused(value: str | None) -> None:
    with pytest.raises(fr.FieldRefused, match="64 lowercase hex"):
        fr.sha_bytes("mask_sha256", value)


# ---------------------------------------------------------------------------
# appearance
# ---------------------------------------------------------------------------


def test_a_dominant_color_is_sent_lowercase_or_not_at_all() -> None:
    assert fr.dominant_color("#a1b2c3") == ("#a1b2c3", [])
    assert fr.dominant_color("#A1B2C3") == ("#a1b2c3", [])
    assert fr.dominant_color(None) == (None, [])
    for bad in ("#abc", "red", "#a1b2c3\x00", "#a1b2c", "#a1b2c3d"):
        value, warnings = fr.dominant_color(bad)
        assert value is None and warnings == ["dominant_color off the contract spelling: dropped"]


def test_a_thumbhash_is_decoded_and_sent_only_inside_5_to_64_bytes() -> None:
    raw = bytes(range(25))
    assert fr.thumbhash(base64.b64encode(raw).decode()) == (raw, [])
    assert fr.thumbhash(None) == (None, [])
    for bad in (base64.b64encode(b"1234").decode(), base64.b64encode(b"x" * 65).decode(), "%%%"):
        value, warnings = fr.thumbhash(bad)
        assert value is None and warnings == ["thumbhash outside 5-64 bytes or not base64: dropped"]
    assert fr.thumbhash(base64.b64encode(b"x" * 5).decode())[0] == b"x" * 5
    assert fr.thumbhash(base64.b64encode(b"x" * 64).decode())[0] == b"x" * 64


# ---------------------------------------------------------------------------
# grounding: decimal text, all eight or none, in range
# ---------------------------------------------------------------------------


def test_a_fraction_is_decimal_text_on_the_contract_grammar() -> None:
    assert fr.decimal_text(0.0125) == "0.0125"
    assert fr.decimal_text(0.0) == "0.0"
    assert fr.decimal_text(1.0) == "1.0"
    assert fr.decimal_text(1e-05) == "0.00001"  # never an exponent
    assert fr.decimal_text(0.1 + 0.2) == "0.30000000000000004"
    for bad in (-0.25, math.nan, math.inf, -math.inf, None):
        assert fr.decimal_text(bad) is None
    tiny = fr.decimal_text(1e-40)
    assert tiny is None  # over 32 characters: off the grammar


GOOD = {
    "bottom_margin_frac": 0.0125,
    "top_margin_frac": 0.04,
    "box_top_frac": 0.04,
    "box_bottom_frac": 0.9875,
    "box_left_frac": 0.2,
    "box_right_frac": 0.8,
    "contact_band_center_x_frac": 0.51,
    "contact_band_width_frac": 0.3,
}


def test_a_whole_grounding_is_sent_as_eight_decimal_strings() -> None:
    sent = fr.grounding(GOOD)
    assert sent == {
        "bottom_margin_frac": "0.0125",
        "top_margin_frac": "0.04",
        "box_top_frac": "0.04",
        "box_bottom_frac": "0.9875",
        "box_left_frac": "0.2",
        "box_right_frac": "0.8",
        "contact_band_center_x_frac": "0.51",
        "contact_band_width_frac": "0.3",
    }


def test_the_edges_of_every_range_are_kept() -> None:
    edges = {
        **GOOD,
        "bottom_margin_frac": 0.0,
        "box_top_frac": 0.0,
        "box_bottom_frac": 1.0,
        "box_left_frac": 0.0,
        "box_right_frac": 1.0,
    }
    assert fr.grounding(edges) is not None


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"bottom_margin_frac": None}, "bottom_margin_frac"),
        ({"top_margin_frac": 1.0}, "top_margin_frac"),
        ({"contact_band_width_frac": 1.0}, "contact_band_width_frac"),
        ({"contact_band_center_x_frac": -0.1}, "contact_band_center_x_frac"),
        ({"box_top_frac": 0.5, "box_bottom_frac": 0.5}, "box_top_frac < box_bottom_frac"),
        ({"box_left_frac": 0.9, "box_right_frac": 0.8}, "box_left_frac < box_right_frac"),
        ({"box_bottom_frac": 1.0000001}, "box_bottom_frac"),
        ({"box_right_frac": 1.5}, "box_right_frac"),
        ({"box_top_frac": math.nan}, "box_top_frac"),
    ],
)
def test_a_grounding_the_spine_would_refuse_is_refused_whole(
    change: dict[str, float | None], why: str
) -> None:
    with pytest.raises(fr.FieldRefused, match=why):
        fr.grounding({**GOOD, **change})


def test_a_grounding_missing_a_field_is_refused() -> None:
    partial = dict(GOOD)
    del partial["box_left_frac"]
    with pytest.raises(fr.FieldRefused, match="box_left_frac"):
        fr.grounding(partial)


# ---------------------------------------------------------------------------
# instants and vocabularies
# ---------------------------------------------------------------------------


def test_an_instant_is_an_iso_token_in_utc() -> None:
    aware = dt.datetime(2026, 10, 14, 9, 30, 5, 120000, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert fr.instant(aware) == "2026-10-14T07:30:05.120000Z"
    naive = dt.datetime(2026, 10, 14, 9, 30, 5)  # SQLite hands back naive UTC
    assert fr.instant(naive) == "2026-10-14T09:30:05.000000Z"


def test_the_mime_lists_are_the_contracts() -> None:
    assert fr.derivative_mime("image/jpeg") == "image/jpeg"
    assert fr.mask_mime("image/png") == "image/png"
    with pytest.raises(fr.FieldRefused, match="mime"):
        fr.mask_mime("image/jpeg")  # no alpha, so never a mask
    with pytest.raises(fr.FieldRefused, match="mime"):
        fr.derivative_mime("image/gif")


def test_a_whole_number_fraction_keeps_its_decimal_point() -> None:
    assert fr.decimal_text(1e16) == "10000000000000000.0"
    assert fr.decimal_text(1e40) is None  # over 32 characters
