"""LG-S1: 'otp' names a secret only as a word of the key, and not before a non-secret word.

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


def _words_of(entry: str) -> list[str]:
    """A word list entry's words: expiresAt is expires, then at."""
    return [word.lower() for word in re.split(r"(?=[A-Z])", entry) if word]


#: What may follow an otp word for the key to print: separators, then one
#: non-secret entry, then the end.
NON_SECRET_REST = [
    re.compile(rf"[^a-z0-9]*{'[^a-z0-9]*'.join(_words_of(entry))}", re.IGNORECASE | re.ASCII)
    for entry in KEYS["otpNonSecretSuffixes"]
]
#: The letters after otp when a guard word runs on from its p: footprint, footpath, potpourri.
GUARD_TAILS = tuple(word[1:] for word in KEYS["otpGuardWords"])


def _sensitive(key: str) -> bool:
    return r.SENSITIVE_KEY.search(key) is not None


def _line(msg: str = "", **fields: Any) -> dict[str, Any]:
    lines: list[str] = []
    Logger(service="t", version="1", env={}, sink=lines.append).info(msg, **fields)
    return json.loads(lines[0])  # type: ignore[no-any-return]


def _letter(char: str) -> bool:
    return char.isascii() and char.isalpha()


def _every_otp_prints_by_the_rule(key: str) -> bool:
    """The keys the otp rule lets print, read one otp at a time (any case).

    Each one prints when what follows it is separators and one non-secret
    entry, and nothing else. Otherwise it redacts when a separator or the key's
    edge is on either side (totp and hotp count from their t or h), when its
    case starts a word (Otp, OtP, OTp; OTP after a lower-case letter) or ends
    one (otp before a capital). A lower-case o with a capital inside (rootPath,
    IoTPlatform) is no otp word. What is left (footprint, USEROTPNUMBER,
    MFAOTPCode) redacts unless a guard word runs on from the p.
    """
    for found in re.finditer(r"otp", key, re.I):
        at = found.start()
        word = found.group(0)
        before = key[at - 1] if at > 0 else ""
        after = key[at + 3 : at + 4]
        rest = key[at + 3 :]
        if any(words.fullmatch(rest) for words in NON_SECRET_REST):
            continue
        two_back = key[at - 2] if at > 1 else ""
        starts_word = not _letter(before) or (before in "thTH" and not _letter(two_back))
        if starts_word or not _letter(after):
            return False
        if word[0] == "o" and word != "otp":
            continue
        if word in ("Otp", "OtP", "OTp"):
            return False
        if word == "otp" and after.isupper():
            return False
        if word == "OTP" and before.islower():
            return False
        if not rest.lower().startswith(GUARD_TAILS):
            return False
    return True


def _tail_keys(tail: str) -> list[str]:
    """Each casing of an otp key with a tail word after the otp, totp or hotp: userotpqrcode, USEROTPQRCODE ..."""
    return [
        key
        for prefix in ("user", "mfa", "sms", "pending", "last", "new")
        for core in ("otp", "totp", "hotp")
        for key in (
            prefix + core + tail,
            (prefix + core + tail).upper(),
            prefix + core.capitalize() + tail.capitalize(),
            f"{prefix}_{core}_{tail}",
        )
    ]


def _non_secret_keys(entry: str) -> list[str]:
    """Each casing of an otp key whose word after otp is one non-secret entry: otp_expires_at, userotpexpiresat ..."""
    words = _words_of(entry)
    camel = "".join(word.capitalize() for word in words)
    return [
        f"otp_{'_'.join(words)}",
        f"otp{camel}",
        f"userOtp{camel}",
        f"userotp{''.join(words)}",
        f"USEROTP{''.join(words).upper()}",
        f"USER_TOTP_{'_'.join(words).upper()}",
        f"x-hotp-{'-'.join(words)}",
        f"smsOTP{camel}",
        f"SMSOTP{camel}",
    ]


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


def test_still_redacts_a_key_whose_case_shows_where_the_otp_word_ends() -> None:
    fields = {
        "MFAOTPCode": "1",
        "SMSOTPCode": "2",
        "MFATOTPSeed": "3",
        "userotpCode": "4",
        "smsotpValue": "5",
        "MFAOTPNumber": "6",
        "userotpNumber": "7",
    }
    assert r.redact_value(fields) == dict.fromkeys(fields, "[REDACTED]")
    assert all(r.is_sensitive(key) for key in fields)


def test_redacts_a_single_case_otp_key_whatever_word_follows_unless_it_names_no_secret() -> None:
    fields = {
        "userotpcode": "1",
        "USEROTPCODE": "2",
        "smsotpvalue": "3",
        "MFAOTPCODE": "4",
        "userotpauthurl": "5",
        "userOtPCode": "6",
        "userotps": "7",
        "USEROTPNUMBER": "8",
        "userotpbase32": "9",
        "userotpqrcode": "10",
        "userotpprovisioninguri": "11",
        "userotpbackupcodes": "12",
        "userotpsharedkey": "13",
    }
    assert r.redact_value(fields) == dict.fromkeys(fields, "[REDACTED]")
    assert all(r.is_sensitive(key) for key in fields)
    printed = {"userotpattempts": 3, "USEROTPEXPIRY": "x"}
    assert r.redact_value(printed) == printed


@pytest.mark.parametrize("tail", KEYS["otpSecretTails"])
def test_redacts_otp_totp_and_hotp_followed_by_a_tail_in_four_casings_as_before(tail: str) -> None:
    for key in _tail_keys(tail):
        assert BEFORE.search(key)
        assert _sensitive(key)
        assert r.is_sensitive(key)
        assert r.redact_value({key: "hunter2"}) == {key: "[REDACTED]"}


@pytest.mark.parametrize("entry", KEYS["otpNonSecretSuffixes"])
def test_prints_an_otp_key_followed_by_a_non_secret_word_alone_and_redacts_more(entry: str) -> None:
    for key in _non_secret_keys(entry):
        assert BEFORE.search(key)
        assert not _sensitive(key)
        assert r.redact_value({key: "visible"}) == {key: "visible"}
        for more in (f"{key}s", f"{key}2", f"{key}_code", f"{key}Code"):
            assert _sensitive(more)


@pytest.mark.parametrize("key", KEYS["unicodePrints"])
def test_prints_a_non_ascii_letter_as_a_separator_before_a_non_secret_word(key: str) -> None:
    """re.ASCII: without it the long s (U+017F) would read as s and the Kelvin sign (U+212A) as k."""
    assert not _sensitive(key)
    assert not r.is_sensitive(key)


@pytest.mark.parametrize("key", KEYS["unicodeRedacts"])
def test_redacts_a_key_whose_non_ascii_letter_does_not_fold_to_an_ascii_one(key: str) -> None:
    assert _sensitive(key)
    assert r.is_sensitive(key)


def test_reads_only_the_end_of_the_key_as_its_end_not_a_line_break_before_it() -> None:
    # Python's $ also matches before a final line break; JavaScript's does not. \Z keeps the two the same.
    assert _sensitive("otp_attempts\n")
    assert _sensitive("userotpattempts\n")
    assert _sensitive("userOtpAttempts\n")
    assert not _sensitive("otp_attempts")


def test_masks_a_camel_case_or_acronym_otp_key_in_the_text_of_a_log_line() -> None:
    assert _line("smsOtpValue: 123456")["msg"] == "smsOtpValue: [REDACTED]"
    assert _line("MFAOTPCode: 123457")["msg"] == "MFAOTPCode: [REDACTED]"
    text = _line(
        "header lines\nuserOtpNumber: 654321\nMFAOTPNumber: 654322\nuserotpNumber: 654323\nfootprint: 0.42"
    )["msg"]
    assert text == (
        "header lines userOtpNumber: [REDACTED] MFAOTPNumber: [REDACTED]"
        " userotpNumber: [REDACTED] footprint: 0.42"
    )
    line = _line(
        raw=["userOtpNumber", "654324", "rootPath", "/srv"], form="userOtpNumber=654325&footprint=1"
    )
    assert line["raw"] == '["userOtpNumber","[REDACTED]","rootPath","/srv"]'
    assert line["form"] == "userOtpNumber=[REDACTED]&footprint=1"


def test_masks_a_single_case_otp_key_in_a_log_line_and_prints_a_non_secret_one() -> None:
    text = _line(
        "header lines\nuserotpbase32: 765432\nUSEROTPNUMBER: 765433\nuserotpattempts: 3\notpExpiresAt: 9"
    )["msg"]
    assert text == (
        "header lines userotpbase32: [REDACTED] USEROTPNUMBER: [REDACTED]"
        " userotpattempts: 3 otpExpiresAt: 9"
    )


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


@pytest.mark.parametrize("key", KEYS["otpNonSecret"])
def test_no_longer_redacts_a_key_whose_word_after_otp_names_no_secret(key: str) -> None:
    assert BEFORE.search(key)
    assert not _sensitive(key)
    assert r.redact_value({key: "visible"}) == {key: "visible"}


@pytest.mark.parametrize("key", KEYS["limit"])
def test_known_limit_no_longer_redacts(key: str) -> None:
    """A lower-case o with a capital inside the otp (useroTpnumber) reads as two
    words, as rootPath and IoTPlatform do, so it is no otp word. Any other case
    form of otp, totp or hotp redacts."""
    assert BEFORE.search(key)
    assert not _sensitive(key)
    assert not r.is_sensitive(key)


def _acronym(key: str) -> str:
    """The otp word in capitals and a capital after it (userotpcode as userOTPCode)."""
    return re.sub(r"[th]?otp[a-z]?", lambda word: word.group(0).upper(), key, flags=re.I)


def test_nothing_but_what_the_otp_rule_releases_is_released() -> None:
    """Over every fixture key with a prefix, a suffix and a case change."""
    prefixes = ["", "x-", "user_", "user.", "user", "User", "v2", "2", "MFA"]
    suffixes = ["", "_id", "-id", ".value", "Value", "value", "2", "_2"]
    corpus = [
        recase(prefix + key + suffix)
        for group in ("redact", "keep", "released", "otpNonSecret", "limit")
        for key in KEYS[group]
        for prefix in prefixes
        for suffix in suffixes
        for recase in (str, str.upper, str.lower, _acronym)
    ]
    widened = []
    wrongly_released = []
    wrongly_redacted = []
    released = 0
    for key in corpus:
        before, after = BEFORE.search(key) is not None, _sensitive(key)
        if after and not before:
            widened.append(key)
        should_print = (
            before and not BEFORE_WITHOUT_OTP.search(key) and _every_otp_prints_by_the_rule(key)
        )
        if before and not after:
            released += 1
            if not should_print:
                wrongly_released.append(key)
        if after and should_print:
            wrongly_redacted.append(key)
    # The same two counts are pinned in fc-shared's test: both repos read the same corpus.
    assert len(corpus) == 71712
    assert len(set(corpus)) == 41583
    assert released > 1000
    assert widened == []
    assert wrongly_released == []
    assert wrongly_redacted == []


def _any_case(word: str) -> str:
    return "".join(f"[{letter}{letter.upper()}]" if letter.isalpha() else letter for letter in word)


def _one_of(entries: list[str] | tuple[str, ...], sep: str, recase: Any) -> str:
    return "(?:" + "|".join(sep.join(recase(w) for w in _words_of(e)) for e in entries) + ")"


def test_keeps_every_other_alternative_and_builds_the_otp_words_from_the_fixture_lists() -> None:
    before = [alternative for alternative in _BEFORE_SOURCE.split("|") if alternative != "otp"]
    after = r.SENSITIVE_KEY.pattern.split("|")
    assert len(before) == 17
    assert [alternative for alternative in before if alternative not in after] == []
    assert r.SENSITIVE_KEY.flags & (re.IGNORECASE | re.ASCII) == re.IGNORECASE | re.ASCII
    # fc-shared's two patterns hold the same pieces, and its test pins them too.
    non_secret = _one_of(KEYS["otpNonSecretSuffixes"], "[^a-z0-9]*", str)
    non_secret_any_case = _one_of(KEYS["otpNonSecretSuffixes"], "[^A-Za-z0-9]*", _any_case)
    guard = _one_of(GUARD_TAILS, "", _any_case)
    assert (
        rf"|bearer|(?:(?:^|[^a-z])[ht]?otp|otp(?=$|[^a-z]))(?![^a-z0-9]*{non_secret}\Z)|passphrase|"
        in r.SENSITIVE_KEY.pattern
    )
    camel = (
        "(?-i:(?:Otp|[a-z]OTP|otp(?=[A-Z])|O(?:tP|Tp)"
        f"|(?:[A-Za-z]otp(?=[a-z])|[A-Z]OTP(?=[A-Za-z]))(?!{guard}))"
        rf"(?![^A-Za-z0-9]*{non_secret_any_case}\Z))"
    )
    assert r.SENSITIVE_KEY.pattern.endswith("|" + camel)
