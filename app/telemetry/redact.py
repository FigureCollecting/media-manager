"""What a log line may print: fc-shared 1.8.0's redaction policy, in Python.

Ported from fc-shared ``src/utils/sanitize.ts`` (redactString, redactValue)
and the text rules of ``src/server/log.ts``, so a media-manager line hides
exactly what an fc-shared line hides for the same input:

* a key that names a secret (password, token, api_key, cookie, sig,
  signature, code_verifier ..., or an OAuth code key holding a code) prints
  ``[REDACTED]``, at any depth;
* a secret-shaped value (Bearer, JWT, GitHub, Slack, AWS, Google keys, PEM
  private keys) is masked inside any string;
* a URL loses its userinfo, query and fragment; a value under a URL-named key
  (url, href, path, endpoint ...) is cut at its first ``?`` or ``#``;
* a sensitive header line, header-list entry or form-encoded value is masked;
* a string that is a whole JSON object or array is key-redacted;
* bytes print as ``[binary]``; an object the policy cannot read prints as
  ``[ClassName]`` or ``[unserializable]``, never its attributes;
* a printed value is one line of at most 1000 characters.

Not ported: what only node has (streams, util.inspect hooks, printf), the
FC_LOG_FORMAT=text escape hatch, and the span-attribute exporter wrapper
(nothing is exported yet: the unit that switches export on adds it).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import math
import re
import traceback
import uuid
from collections.abc import Mapping
from decimal import Decimal
from pathlib import PurePath
from typing import Any

PLACEHOLDER = "[REDACTED]"
#: redact_string's cap (sanitize.ts maxStringLength).
MAX_STRING_LENGTH = 8192
#: A printed value's cap (fc-shared utils/logger.ts sanitizeLogValue).
MAX_LOG_VALUE = 1000
#: redact_value's walk depth (sanitize.ts maxDepth).
MAX_DEPTH = 12
#: A printed field's depth, and how deep JSON held in strings is read (log.ts).
MAX_FIELD_DEPTH = 4
#: Entries kept per array or object; past it, MAX_ENTRIES - 1 and a count of the rest.
MAX_ENTRIES = 100
#: Entries read per printed value, all levels and the JSON strings in it together.
MAX_TOTAL_ENTRIES = 1000
TRUNCATED = "[truncated:max-depth]"
TRUNCATED_ENTRIES = "[truncated:max-entries]"
UNSERIALIZABLE = "[unserializable]"

SENSITIVE_KEY = re.compile(
    r"pass(?:word|wd)?|secret|token|api[-_]?key|access[-_]?key|private[-_]?key|client[-_]?secret"
    r"|authorization|credential|cookie|session[-_]?id|bearer|otp|passphrase|\bpin\b|\bssn\b|\bmfa\b",
    re.IGNORECASE | re.ASCII,
)
SECRET_VALUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9._-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z._-]{30,}"),
    re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z]*PRIVATE KEY-----"),
)

# A URL signature (sig, signature, X-Amz-Signature) or PKCE code_verifier as a word of the snake_case key.
_QUERY_SECRET_KEY = re.compile(r"(?:^|_)(?:sig|signature|code_verifier)(?:_|$)")
# In a form, an OAuth code too (code, device_code).
_FORM_SECRET_KEY = re.compile(r"(?:^|_)code(?:_|$)")
_OAUTH_CODE_KEY = re.compile(
    r"(?:auth_|authorization_|device_|user_|oauth_|verification_|reset_|mfa_|confirmation_)?code(?:_[0-9]+)?"
)
_SECRET_WORD_HINT = re.compile(r"sig|code", re.IGNORECASE)
_CODE_WORD = re.compile(r"code", re.IGNORECASE)

_URL_IN_TEXT = re.compile(
    r"((?:[a-z][a-z0-9+.-]{0,31}:)?//)(?:[^\s/?#]*@)?([^\s?#]*)([?#]\S*)?", re.I
)
_REQUEST_TARGET = re.compile(
    r"\b(GET|HEAD|POST|PUT|DELETE|CONNECT|OPTIONS|TRACE|PATCH)( +[^\s?#]*|:[^\s?#:]*)([?#]\S*)"
)
_CLOSERS = "'\"`)]}>.,;:!"
_LEADING_TOKEN = re.compile(r"^([^\s?#]*)[?#]")
_URL_SHAPED = re.compile(
    r"^[a-z][a-z0-9+.-]{0,31}:|[\\/]|^[a-z0-9-]+(?:\.[a-z0-9-]+)+(?::\d+)?\Z", re.I
)
_QUERY_PAIRS = re.compile(r"^[?#][^\s=]*=")
_HEADER_LINE = re.compile(r"(^|[\r\n])([!#$%&'*+.^_`|~0-9A-Za-z-]+)([ \t]*:[ \t]*)[^\r\n]*")
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ESCAPES = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# JavaScript's trim() also drops a byte-order mark.
_EDGE_SPACE = re.compile(r"^[\s﻿]+|[\s﻿]+$")
_URL_KEY_WORDS = frozenset(
    {
        "url", "urls", "uri", "uris", "href", "hrefs", "link", "links", "path", "paths", "target",
        "endpoint", "endpoints", "location", "referer", "referrer", "redirect", "redirects",
    }
)  # fmt: skip


def _trim(text: str) -> str:
    return _EDGE_SPACE.sub("", text)


def to_snake_case(key: str) -> str:
    """``itemId`` -> ``item_id``, ``HTTP-Status`` -> ``http_status``; never empty, never a leading digit."""
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    snake = re.sub(r"([A-Z])(?=[A-Z][a-z])", r"\1_", snake).lower()
    snake = re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
    if snake == "":
        return "field"
    return f"f_{snake}" if snake[0].isdigit() else snake


def redact_string(
    value: str,
    *,
    placeholder: str = PLACEHOLDER,
    max_length: int = MAX_STRING_LENGTH,
) -> str:
    """Secret shapes masked, then cut to ``max_length`` characters."""
    out = value
    for pattern in SECRET_VALUES:
        out = pattern.sub(placeholder, out)
    if len(out) > max_length:
        out = out[:max_length] + "…[truncated]"
    return out


def _iso(value: dt.date) -> str:
    if isinstance(value, dt.datetime) and value.tzinfo is not None:
        utc = value.astimezone(dt.UTC)
        return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond // 1000:03d}Z"
    return value.isoformat()


def _error_parts(exc: BaseException) -> dict[str, Any]:
    out: dict[str, Any] = {"name": type(exc).__name__, "message": str(exc)}
    if exc.__traceback__ is not None:
        out["stack"] = "".join(traceback.format_exception(exc))
    return out


def redact_value(
    value: Any,
    *,
    placeholder: str = PLACEHOLDER,
    max_depth: int = MAX_DEPTH,
) -> Any:
    """A redacted, JSON-safe deep copy (sanitize.ts redactValue): sensitive keys
    masked, secret shapes masked in strings, cycles and depth collapsed."""

    def walk(item: Any, depth: int, seen: set[int]) -> Any:
        if item is None or isinstance(item, bool | int | float):
            return item
        if isinstance(item, str):
            return redact_string(item, placeholder=placeholder)
        if isinstance(item, dt.date):
            return _iso(item)
        if isinstance(item, BaseException):
            parts = _error_parts(item)
            return {k: redact_string(v, placeholder=placeholder) for k, v in parts.items()}
        if isinstance(item, bytes | bytearray | memoryview):
            return "[binary]"
        if callable(item) and not isinstance(item, Mapping):
            return "[function]"
        if not isinstance(item, Mapping | list | tuple | set | frozenset):
            return _scalar_text(item)
        if depth >= max_depth:
            return TRUNCATED
        if id(item) in seen:
            return "[circular]"
        seen.add(id(item))
        try:
            if isinstance(item, Mapping):
                return {
                    str(k): placeholder
                    if SENSITIVE_KEY.search(str(k))
                    else walk(v, depth + 1, seen)
                    for k, v in item.items()
                }
            return [walk(v, depth + 1, seen) for v in item]
        finally:
            seen.discard(id(item))

    return walk(value, 0, set())


def _scalar_text(item: Any) -> str:
    """A value with a natural text form prints as it; any other object as its class name."""
    if isinstance(item, uuid.UUID | Decimal | PurePath):
        return str(item)
    if isinstance(item, enum.Enum):
        return str(item.value)
    return f"[{type(item).__name__ or 'object'}]"


# --- keys ---------------------------------------------------------------------------------------


def is_sensitive(key: str, *, form: bool = False) -> bool:
    if SENSITIVE_KEY.search(key):
        return True
    if not _SECRET_WORD_HINT.search(key):
        return False
    snake = to_snake_case(key)
    return bool(_QUERY_SECRET_KEY.search(snake) or (form and _FORM_SECRET_KEY.search(snake)))


def _is_secret_code(key: str, value: Any) -> bool:
    if not _CODE_WORD.search(key) or not _OAUTH_CODE_KEY.fullmatch(to_snake_case(key)):
        return False
    return isinstance(value, str | list | tuple)


def masks(key: str, value: Any) -> bool:
    """A key whose value is masked, unread: a sensitive key, or an OAuth code key holding a code."""
    return is_sensitive(key) or _is_secret_code(key, value)


def is_url_key(key: str) -> bool:
    return any(word in _URL_KEY_WORDS for word in to_snake_case(key).split("_"))


# --- text ---------------------------------------------------------------------------------------


def _closers(query: str) -> str:
    start = len(query)
    while start > 0 and query[start - 1] in _CLOSERS:
        start -= 1
    return query[start:]


def _strip_query(match: re.Match[str]) -> str:
    query = match.group(3)
    return match.group(1) + match.group(2) + ("" if query is None else _closers(query))


def strip_urls(text: str) -> str:
    """Text without the query, fragment and userinfo of any URL in it. A value
    whose first token reads as a URL or path, or is followed by key=value
    pairs, is cut at its first ``?`` or ``#``."""
    trimmed = _trim(text)
    leading = _LEADING_TOKEN.match(trimmed)
    kept = text
    if leading is not None and (
        _URL_SHAPED.search(leading.group(1)) or _QUERY_PAIRS.match(trimmed[len(leading.group(1)) :])
    ):
        kept = leading.group(1)
    return _URL_IN_TEXT.sub(_strip_query, _REQUEST_TARGET.sub(_strip_query, kept))


def _mask_header_lines(text: str, placeholder: str) -> str:
    def one(m: re.Match[str]) -> str:
        if is_sensitive(m.group(2)):
            return f"{m.group(1)}{m.group(2)}{m.group(3)}{placeholder}"
        return m.group(0)

    return _HEADER_LINE.sub(one, text)


def _form_key(raw: str) -> str:
    """A form key as a server reads it: '+' is a space, each run of escapes decoded as UTF-8."""
    return _ESCAPES.sub(
        lambda m: bytes.fromhex(m.group(0).replace("%", "")).decode("utf-8", "replace"),
        raw.replace("+", " "),
    )


def _mask_form(text: str, placeholder: str) -> str:
    form = _trim(text)
    if re.search(r"\s", form):
        return text
    pairs = []
    for pair in form.split("&"):
        equals = pair.find("=")
        if equals > 0 and is_sensitive(_form_key(pair[:equals]), form=True):
            pairs.append(pair[: equals + 1] + placeholder)
        else:
            pairs.append(pair)
    masked = "&".join(pairs)
    return text if masked == form else masked


def plain_text(text: str, placeholder: str = PLACEHOLDER) -> str:
    """Text that is not a JSON document as it prints; also every object key."""
    return strip_urls(
        _mask_form(
            _mask_header_lines(redact_string(text, placeholder=placeholder), placeholder),
            placeholder,
        )
    )


@dataclasses.dataclass
class _Budget:
    left: int = MAX_TOTAL_ENTRIES


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def _json_document(text: str) -> Any:
    """``text`` (already seen to start with ``{`` or ``[``) as JSON; None when it is not."""
    try:
        return json.loads(_trim(text), parse_constant=_reject_constant)
    except ValueError:
        return None


def dumps(value: Any) -> str:
    """Compact JSON, as JSON.stringify writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def redact_text(
    text: str, nesting: int = 0, in_url: bool = False, budget: _Budget | None = None
) -> str:
    """A string as the logger prints it: a whole JSON object or array is
    printed as an object is and re-emitted compact (four strings deep at
    most); any other text goes through plain_text, a URL value (in_url) first
    cut at its first ``?`` or ``#``."""
    budget = budget if budget is not None else _Budget()
    candidate = _trim(text).startswith(("{", "["))
    if candidate and budget.left < 0:
        return TRUNCATED_ENTRIES
    doc = _json_document(text) if candidate else None
    if doc is None:
        if in_url:
            return _trim(plain_text(re.split(r"[?#]", text, maxsplit=1)[0]))
        return plain_text(text)
    if nesting >= MAX_FIELD_DEPTH:
        return TRUNCATED
    return dumps(print_value(doc, nesting + 1, in_url, budget))


def _bounded_list(items: list[Any]) -> list[Any]:
    if len(items) <= MAX_ENTRIES:
        return items
    return [*items[: MAX_ENTRIES - 1], f"[truncated:{len(items) - MAX_ENTRIES + 1} more]"]


def _bounded_dict(entries: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    if len(entries) <= MAX_ENTRIES:
        return entries
    return [*entries[: MAX_ENTRIES - 1], ("[truncated]", f"{len(entries) - MAX_ENTRIES + 1} more")]


def _header_list(items: list[Any]) -> list[Any]:
    """A raw header list (name, value, name, value ...) with the value after each sensitive name masked."""
    names = len(items) % 2 == 0 and all(
        i % 2 == 1 or _HEADER_NAME.fullmatch(str(item)) is not None for i, item in enumerate(items)
    )
    if not names:
        return items
    return [
        PLACEHOLDER if i % 2 == 1 and is_sensitive(str(items[i - 1])) else item
        for i, item in enumerate(items)
    ]


class _Printer:
    """One printed value: the shape walk of log.ts's JSON.stringify replacer."""

    def __init__(self, nesting: int, root_in_url: bool, budget: _Budget) -> None:
        self.nesting = nesting
        self.root_in_url = root_in_url
        self.budget = budget

    def shape(
        self, key: str | None, value: Any, holder_in_url: bool, ancestors: tuple[int, ...]
    ) -> Any:
        # A masked key never gets here: the object branch below masks it unread.
        url_valued = self.root_in_url or holder_in_url or (key is not None and is_url_key(key))
        if isinstance(value, str):
            return redact_text(value, self.nesting, url_valued, self.budget)
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dt.date):
            return redact_text(_iso(value), self.nesting, url_valued, self.budget)
        if isinstance(value, BaseException):
            return self.shape(None, _error_parts(value), url_valued, ancestors)
        if isinstance(value, bytes | bytearray | memoryview):
            return "[binary]"
        if isinstance(value, Mapping):
            entries: list[tuple[str, Any]] | None = [(str(k), v) for k, v in value.items()]
        elif isinstance(value, list | tuple | set | frozenset):
            entries = None
        elif callable(value):
            return "[function]"
        else:
            return redact_text(_scalar_text(value), self.nesting, url_valued, self.budget)
        if id(value) in ancestors:
            return "[circular]"
        if len(ancestors) + 1 > MAX_FIELD_DEPTH:
            return TRUNCATED
        below = (*ancestors, id(value))
        if entries is None:
            items = _bounded_list(_header_list(list(value)))
            self.budget.left -= len(items)
            if self.budget.left < 0:
                return TRUNCATED_ENTRIES
            return [self.shape(str(i), item, url_valued, below) for i, item in enumerate(items)]
        kept = _bounded_dict(entries)
        self.budget.left -= len(kept)
        if self.budget.left < 0:
            return TRUNCATED_ENTRIES
        out: dict[str, Any] = {}
        for raw_key, item in kept:
            printed = plain_text(raw_key)
            # A key that names a secret before it is printed ('/login?password') is masked unread.
            out[printed] = (
                PLACEHOLDER
                if masks(raw_key, item)
                else self.shape(printed, item, url_valued, below)
            )
        return out


def print_value(
    value: Any, nesting: int = 0, in_url: bool = False, budget: _Budget | None = None
) -> Any:
    """Any value as the logger prints it: a string through redact_text; an
    object shaped (key masking, URL keys, bounds), then key-redacted again; a
    value that cannot be read as ``[unserializable]``."""
    budget = budget if budget is not None else _Budget()
    if isinstance(value, str):
        return redact_text(value, nesting, in_url, budget)
    if value is None or isinstance(value, bool | int | float):
        return value
    try:
        return redact_value(_Printer(nesting, in_url, budget).shape(None, value, False, ()))
    except Exception:
        return UNSERIALIZABLE


def sanitize_log_value(value: Any) -> str:
    """One line, no ANSI escapes, at most MAX_LOG_VALUE characters (utils/logger.ts)."""
    if value is None:
        text = "null"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, BaseException):
        text = str(value) or "Error (no message)"
    else:
        try:
            text = dumps(value)
        except (TypeError, ValueError):
            text = str(value)
    text = _ANSI.sub("", re.sub(r"[\r\n]", " ", text))
    if len(text) > MAX_LOG_VALUE:
        text = text[:MAX_LOG_VALUE] + "...[truncated]"
    return text


def safe_text(value: str) -> str:
    return sanitize_log_value(redact_text(value))


def stack_tail(exc: BaseException, limit: int = MAX_LOG_VALUE) -> str:
    """The traceback as one redacted line, keeping its END: Python prints the raising frame last."""
    text = " ".join(plain_text("".join(traceback.format_exception(exc))).split("\n"))
    text = _ANSI.sub("", text.replace("\r", " ")).strip()
    if len(text) > limit:
        text = "...[truncated]" + text[-limit:]
    return text
