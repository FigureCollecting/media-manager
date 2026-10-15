"""media.v1's FIELD RULE (fc-ingest-contract 0.10.0), applied before a call.

The spine refuses an IDENTITY string (a key, a version, an instant it keys
on) that is empty, carries a NUL or is over 256 bytes of UTF-8, and never
repairs one; it absorbs FREE TEXT (NUL removed, cut at a character boundary)
with a warning; it refuses an appearance or grounding value off the
contract's spelling. This module makes the same judgments on our side, so
that no call is ever refused for a data-shaped reason:

- an identity that would be refused raises ``FieldRefused`` and is not sent;
- free text is cleaned and cut here, exactly as the spine would cut it (a
  ``failure_reason`` to 1 KB, which also keeps a FAILED report far under the
  spine's 64 KB request cap: RESOURCE_EXHAUSTED is not retryable, so an
  oversized reason would never be recorded, DEPLOY.md section 8);
- an optional appearance value the spine would refuse is DROPPED (absent =
  not measured), with a warning;
- a grounding is all eight fractions or none, as decimal text on the
  contract's grammar and inside its ranges, or ``FieldRefused``.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import math
import re
from collections.abc import Mapping
from decimal import Decimal

IDENTITY_MAX_BYTES = 256
LEASE_OWNER_MAX_BYTES = 256
FAILURE_REASON_MAX_BYTES = 1024
WITHDRAW_REASON_MAX_CHARS = 64
MAX_ORIGIN_FILTER = 200
SHA256_BYTES = 32
THUMBHASH_MIN_BYTES = 5
THUMBHASH_MAX_BYTES = 64
DECIMAL_MAX_CHARS = 32
DERIVATIVE_MIMES = ("image/jpeg", "image/png", "image/webp")
MASK_MIMES = ("image/png", "image/webp")
DERIVE_KINDS = ("sanitized", "matted")

#: Grounding's eight fields, in proto order.
GROUNDING_FIELDS = (
    "bottom_margin_frac",
    "top_margin_frac",
    "box_top_frac",
    "box_bottom_frac",
    "box_left_frac",
    "box_right_frac",
    "contact_band_center_x_frac",
    "contact_band_width_frac",
)
#: The four fractions that lie in [0, 1).
_HALF_OPEN = (
    "bottom_margin_frac",
    "top_margin_frac",
    "contact_band_center_x_frac",
    "contact_band_width_frac",
)

_SHA_HEX = re.compile(r"[0-9a-f]{64}")
_COLOR = re.compile(r"#[0-9a-f]{6}")
_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


class FieldRefused(ValueError):
    """A value the spine would refuse, so it is not sent. ``field`` names it."""

    def __init__(self, field: str, why: str) -> None:
        super().__init__(f"{field}: {why}")
        self.field = field


def identity(field: str, value: str) -> str:
    """An identity string, unchanged, or ``FieldRefused``: never repaired."""
    if not value:
        raise FieldRefused(field, "empty")
    if "\x00" in value:
        raise FieldRefused(field, "carries a NUL")
    if len(value.encode("utf-8")) > IDENTITY_MAX_BYTES:
        raise FieldRefused(field, f"over {IDENTITY_MAX_BYTES} bytes of UTF-8")
    return value


def uuid_identity(field: str, value: str) -> str:
    """A depiction id: an identity that must also be a uuid."""
    identity(field, value)
    if not _UUID.fullmatch(value):
        raise FieldRefused(field, "not a uuid")
    return value


def _cut_bytes(text: str, max_bytes: int) -> str:
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def clean_text(value: str, *, max_bytes: int) -> tuple[str, list[str]]:
    """Free text as the spine would store it: NUL removed, cut to
    ``max_bytes`` of UTF-8 at a character boundary; one warning per repair."""
    warnings: list[str] = []
    if "\x00" in value:
        value = value.replace("\x00", "")
        warnings.append("NUL removed")
    if len(value.encode("utf-8")) > max_bytes:
        value = _cut_bytes(value, max_bytes)
        warnings.append(f"cut to {max_bytes} bytes")
    return value, warnings


def failure_reason(value: str) -> tuple[str, list[str]]:
    """DerivativeReport.failure_reason, cut to 1 KB before it is sent."""
    return clean_text(value, max_bytes=FAILURE_REASON_MAX_BYTES)


def lease_owner(value: str) -> tuple[str, list[str]]:
    """LeasePendingImagesRequest.lease_owner: a label, cut, never refused."""
    return clean_text(value, max_bytes=LEASE_OWNER_MAX_BYTES)


def withdraw_reason(value: str | None) -> tuple[str, list[str]]:
    """MaskWithdrawal.reason on a withdrawal: NUL removed and cut to 64
    characters; refused when nothing is left (a withdrawal nobody can
    explain cannot be audited)."""
    text = (value or "").replace("\x00", "")
    warnings = ["NUL removed"] if value and "\x00" in value else []
    if not text:
        raise FieldRefused("reason", "empty once cleaned")
    if len(text) > WITHDRAW_REASON_MAX_CHARS:
        text = text[:WITHDRAW_REASON_MAX_CHARS]
        warnings.append(f"cut to {WITHDRAW_REASON_MAX_CHARS} characters")
    return text, warnings


def sha_bytes(field: str, value: str | None) -> bytes:
    """A content address as its 32 raw bytes, from the media DB's lowercase hex."""
    if not isinstance(value, str) or not _SHA_HEX.fullmatch(value):
        raise FieldRefused(field, "not 64 lowercase hex characters")
    return bytes.fromhex(value)


def dominant_color(value: str | None) -> tuple[str | None, list[str]]:
    """'#rrggbb' lowercase, or absent. Upper case is lowered (the same
    colour); any other spelling is dropped, warned."""
    if value is None:
        return None, []
    lowered = value.lower()
    if _COLOR.fullmatch(lowered):
        return lowered, []
    return None, ["dominant_color off the contract spelling: dropped"]


def thumbhash(value: str | None) -> tuple[bytes | None, list[str]]:
    """The media DB's base64 ThumbHash as raw bytes, or absent when it is not
    base64 or falls outside the contract's 5-64 bytes (dropped, warned)."""
    if value is None:
        return None, []
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raw = b""
    if THUMBHASH_MIN_BYTES <= len(raw) <= THUMBHASH_MAX_BYTES:
        return raw, []
    return None, ["thumbhash outside 5-64 bytes or not base64: dropped"]


def decimal_text(value: float | None) -> str | None:
    """A measured fraction as the contract's decimal text ("0.0125"): the
    float's shortest round-trip digits, never an exponent. None when it has
    no such spelling (absent, negative, NaN, infinite, over 32 characters)."""
    if value is None or not math.isfinite(value) or value < 0:
        return None
    text = format(Decimal(repr(float(value))), "f")
    if "." not in text:
        text += ".0"
    if len(text) > DECIMAL_MAX_CHARS or not _DECIMAL.fullmatch(text):
        return None
    return text


def grounding(values: Mapping[str, float | None]) -> dict[str, str]:
    """All eight fractions as decimal text, checked against the contract's
    ranges exactly as the spine compares them (as decimals), or
    ``FieldRefused`` naming the first that fails."""
    sent: dict[str, str] = {}
    for name in GROUNDING_FIELDS:
        text = decimal_text(values.get(name))
        if text is None:
            raise FieldRefused(name, "absent or not a fraction")
        sent[name] = text
    num = {name: Decimal(text) for name, text in sent.items()}
    for name in _HALF_OPEN:
        if not num[name] < 1:
            raise FieldRefused(name, "not in [0, 1)")
    for low, high in (("box_top_frac", "box_bottom_frac"), ("box_left_frac", "box_right_frac")):
        if not num[low] < num[high]:
            raise FieldRefused(low, f"not {low} < {high}")
        if not num[high] <= 1:
            raise FieldRefused(high, "over 1")
    return sent


def instant(value: dt.datetime) -> str:
    """A raw instant token in UTC; a naive value is taken as UTC (SQLite
    hands back naive datetimes for the media DB's timezone-aware columns)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def derivative_mime(value: str) -> str:
    """A DERIVED report's mime: one of the three a clean copy is encoded as."""
    if value not in DERIVATIVE_MIMES:
        raise FieldRefused("mime", f"{value!r} is not one of {DERIVATIVE_MIMES}")
    return value


def mask_mime(value: str) -> str:
    """A mask's mime: one of the two that carry an alpha channel."""
    if value not in MASK_MIMES:
        raise FieldRefused("mime", f"{value!r} is not one of {MASK_MIMES}")
    return value
