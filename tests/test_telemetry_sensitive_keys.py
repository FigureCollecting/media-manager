"""LG-S1: 'otp' names a secret only at an edge of a word of the key.

The key lists are shared with fc-shared, whose DEFAULT_SENSITIVE_KEY_PATTERN
and DEFAULT_SENSITIVE_CAMEL_KEY_PATTERN (src/utils/sanitize.ts) this module's
SENSITIVE_KEY ports as one pattern:
tests/fixtures/sensitive-keys.json there is a byte-for-byte copy, read by its
tests/utils/sensitive-keys.test.ts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from app.telemetry import redact as r
from app.telemetry.log import Logger

KEYS: dict[str, list[str]] = {
    group: keys
    for group, keys in json.loads(
        (Path(__file__).parent / "fixtures" / "sensitive-keys.json").read_text(encoding="utf-8")
    ).items()
    if group != "about"
}

#: The pattern as it was (fc-shared 1.8.0 to 1.10.0): 'otp' anywhere in a key.
_BEFORE_SOURCE = (
    r"pass(?:word|wd)?|secret|token|api[-_]?key|access[-_]?key|private[-_]?key|client[-_]?secret"
    r"|authorization|credential|cookie|session[-_]?id|bearer|otp|passphrase|\bpin\b|\bssn\b|\bmfa\b"
)
BEFORE = re.compile(_BEFORE_SOURCE, re.IGNORECASE | re.ASCII)
#: The same without its 'otp' alternative: what every other alternative redacts.
BEFORE_WITHOUT_OTP = re.compile(_BEFORE_SOURCE.replace("|otp|", "|"), re.IGNORECASE | re.ASCII)


def _sensitive(key: str) -> bool:
    return r.SENSITIVE_KEY.search(key) is not None


def _line(msg: str = "", **fields: Any) -> dict[str, Any]:
    lines: list[str] = []
    Logger(service="t", version="1", env={}, sink=lines.append).info(msg, **fields)
    return json.loads(lines[0])  # type: ignore[no-any-return]


def test_keeps_footprint_footprint_mm_laptop_and_captopril() -> None:
    fields = {
        "footprint": 0.42,
        "footprintMm": 118,
        "claim_footprint": 0.6,
        "rootPath": "/srv",
        "laptop": "x1",
        "captopril": "25mg",
    }
    assert r.redact_value(fields) == fields
    assert all(not r.is_sensitive(key) for key in fields)


def test_still_redacts_otp_otp_code_otp_code_camel_totp_x_otp_and_mfa_otp() -> None:
    fields = {
        "otp": "1",
        "otp_code": "2",
        "otpCode": "3",
        "totp": "4",
        "x-otp": "5",
        "mfa.otp": "6",
    }
    assert r.redact_value(fields) == dict.fromkeys(fields, "[REDACTED]")
    assert all(r.is_sensitive(key) for key in fields)


def test_prints_a_footprint_field_of_a_log_line_and_masks_its_otp_fields() -> None:
    line = _line(claim_footprint=0.6, footprintMm=118, otpCode="493817", userOtp="493818")
    assert line["claim_footprint"] == 0.6
    assert line["footprint_mm"] == 118
    assert line["otp_code"] == "[REDACTED]"
    assert line["user_otp"] == "[REDACTED]"
    nested = _line(
        timings={"footprint": 0.42, "rootPath": "/srv", "totp": "493819", "userOtpCode": "493820"}
    )
    assert nested["timings"] == (
        '{"footprint":0.42,"rootPath":"/srv","totp":"[REDACTED]","userOtpCode":"[REDACTED]"}'
    )


def test_still_redacts_a_camel_case_otp_word_in_the_middle_of_a_key() -> None:
    fields = {
        "userOtpCode": "1",
        "userOTPCode": "2",
        "userTotpCode": "3",
        "smsOtpValue": "4",
        "hotpCounter": "5",
        "userHotpCounter": "6",
    }
    assert r.redact_value(fields) == dict.fromkeys(fields, "[REDACTED]")
    assert all(r.is_sensitive(key) for key in fields)
    assert _line(userOTPCode="493821")["user_otp_code"] == "[REDACTED]"


def _every_otp_is_mid_word_without_a_hump(key: str) -> bool:
    """The keys a mid-word otp may be released for, read one occurrence at a
    time: a letter on both sides, and no word that the key's case shows
    starting there (Otp, Totp, Hotp; OTP, TOTP, HOTP after a lower-case letter
    or a digit)."""
    for found in re.finditer(r"otp", key, re.I):
        at = found.start()
        before = key[at - 1] if at > 0 else ""
        after = key[at + 3 : at + 4]
        if not (before.isascii() and before.isalpha() and after.isascii() and after.isalpha()):
            return False
        word = found.group(0)
        if word == "Otp" or (word == "otp" and before in "TH"):
            return False
        if word == "OTP":
            lead = (key[at - 2] if at > 1 else "") if before in "TH" else before
            if lead.isascii() and (lead.islower() or lead.isdigit()):
                return False
    return True


@pytest.mark.parametrize("key", KEYS["redact"])
def test_redacts_as_before(key: str) -> None:
    assert BEFORE.search(key)
    assert _sensitive(key)
    assert r.redact_value({key: "hunter2"}) == {key: "[REDACTED]"}


@pytest.mark.parametrize("key", KEYS["keep"])
def test_keeps_as_before(key: str) -> None:
    assert not BEFORE.search(key)
    assert not _sensitive(key)


@pytest.mark.parametrize("key", KEYS["released"])
def test_no_longer_redacts(key: str) -> None:
    assert BEFORE.search(key)
    assert not _sensitive(key)
    assert r.redact_value({key: "visible"}) == {key: "visible"}


@pytest.mark.parametrize("key", KEYS["limit"])
def test_known_limit_no_longer_redacts(key: str) -> None:
    """With a letter on both sides and no change of case, otp reads as the
    middle of a word whether the key is userotpcode or footprint. No pattern
    can tell them apart; a key in camelCase, or with a separator, redacts."""
    assert BEFORE.search(key)
    assert not _sensitive(key)


def test_nothing_but_a_mid_word_otp_is_released() -> None:
    """Over every fixture key with a prefix, a suffix and a case change."""
    prefixes = ["", "x-", "user_", "user.", "user", "User", "v2", "2"]
    suffixes = ["", "_id", "-id", ".value", "Value", "value", "2", "_2"]
    corpus = [
        recase(prefix + key + suffix)
        for keys in KEYS.values()
        for key in keys
        for prefix in prefixes
        for suffix in suffixes
        for recase in (str, str.upper, str.lower)
    ]
    widened = []
    wrongly_released = []
    released = 0
    for key in corpus:
        before, after = BEFORE.search(key) is not None, _sensitive(key)
        if after and not before:
            widened.append(key)
        if before and not after:
            released += 1
            if BEFORE_WITHOUT_OTP.search(key) or not _every_otp_is_mid_word_without_a_hump(key):
                wrongly_released.append(key)
    assert len(corpus) > 20000
    assert released > 1000
    assert widened == []
    assert wrongly_released == []


def test_keeps_every_other_alternative_and_its_flags() -> None:
    before = [alternative for alternative in _BEFORE_SOURCE.split("|") if alternative != "otp"]
    after = r.SENSITIVE_KEY.pattern.split("|")
    assert len(before) == 17
    assert [alternative for alternative in before if alternative not in after] == []
    assert r.SENSITIVE_KEY.flags & (re.IGNORECASE | re.ASCII) == re.IGNORECASE | re.ASCII
    # The camelCase words are the one case-sensitive group, as fc-shared's separate pattern holds them.
    assert after[-4:] == ["(?-i:Otp", "Totp", "Hotp", "[a-z][TH]?OTP)"]
