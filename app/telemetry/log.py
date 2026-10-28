"""The estate's log line, written by media-manager (fc-shared 1.8.0 server/log, in Python).

One JSON object per stdout line, never multi-line::

    time level service version event msg call code duration_ms peer
    trace_id span_id job err queue_ms, then the line's own fields

The reserved keys are typed and always in that order. ``trace_id`` and
``span_id`` come only from the ACTIVE span and are absent (never zeroed)
without one. Every other key is lowercase snake_case with a scalar value:
strings pass the redaction policy (app.telemetry.redact) and objects become
one-line JSON. tests/test_telemetry_log.py compares the lines with lines
fc-shared's own logger wrote for the same input.

Python's own ``logging`` joins the shape through LogShapeHandler, which
``configure()`` installs on the root logger: a record becomes one ``app.log``
line (its ``extra`` as fields, ``logger`` its name, an exception as ``err``
plus ``stack``).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import re
import sys
import threading
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

from . import redact as r

LOG_LEVELS = ("trace", "debug", "info", "warn", "error", "fatal")
RESERVED_LOG_KEYS = (
    "time", "level", "service", "version", "event", "msg", "call", "code", "duration_ms",
    "peer", "trace_id", "span_id", "job", "err", "queue_ms",
)  # fmt: skip
#: Keys the logger owns; a caller's value for one of them is dropped.
_OWNED = frozenset({"time", "level", "service", "version", "msg", "trace_id", "span_id"})
#: Reserved keys a caller sets; each is typed on the way out.
_CALLER_RESERVED = frozenset(
    {"event", "call", "code", "duration_ms", "peer", "job", "err", "queue_ms"}
)
#: The shape's code: ok, a run's error, a Connect code name or an HTTP status. Any other string prints the placeholder.
_SHAPE_CODE = re.compile(
    r"ok|error|canceled|unknown|invalid_argument|deadline_exceeded|not_found|already_exists"
    r"|permission_denied|resource_exhausted|failed_precondition|aborted|out_of_range|unimplemented"
    r"|internal|unavailable|data_loss|unauthenticated|[1-5][0-9]{2}"
)
_SEVERITY = {
    "trace": 10,
    "debug": 20,
    "info": 30,
    "warn": 40,
    "error": 50,
    "fatal": 60,
    "silent": math.inf,
}
#: Python's level names, as the shape spells them.
_ALIASES = {"warning": "warn", "critical": "fatal", "notset": "trace"}

Env = Mapping[str, str]
Sink = Callable[[str], None]

_write_lock = threading.Lock()


def _stdout(line: str) -> None:
    # sys.stdout is read per line, so a redirected stdout (pytest's capture) gets it.
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _first_non_blank(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


def _package_version() -> str | None:
    try:
        return metadata.version("media-manager")
    except metadata.PackageNotFoundError:  # the images copy app/ without installing it
        return None


def service_name(fallback: str, env: Env | None = None) -> str:
    """OTEL_SERVICE_NAME when the deployment sets one, else the process's own name."""
    return (
        _first_non_blank((os.environ if env is None else env).get("OTEL_SERVICE_NAME")) or fallback
    )


def resolve_identity(
    env: Env, service: str | None = None, version: str | None = None
) -> tuple[str, str]:
    name = _first_non_blank(service, env.get("OTEL_SERVICE_NAME")) or "unknown_service"
    ver = _first_non_blank(version, env.get("SERVICE_VERSION"), _package_version()) or "unknown"
    return name, ver


def parse_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    name = _ALIASES.get(name, name)
    return name if name in _SEVERITY else None


def round_ms(ms: float) -> float | int:
    """To 0.1 ms, half up as JavaScript's Math.round; a whole number prints without a fraction."""
    rounded = math.floor(ms * 10 + 0.5) / 10
    return int(rounded) if rounded.is_integer() else rounded


def _non_negative_ms(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return round_ms(float(value))


def _reserved_text(value: object) -> str:
    return r.sanitize_log_value(r.print_value(value))


def _err_field(value: object) -> dict[str, str]:
    if isinstance(value, BaseException):
        return {"type": _reserved_text(type(value).__name__), "message": r.safe_text(str(value))}
    return {"type": type(value).__name__, "message": _reserved_text(value)}


def _flat_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if r.masks(key, value):
        return r.PLACEHOLDER
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if callable(value) and not isinstance(value, Mapping | type):
        return "[function]"
    return r.sanitize_log_value(r.print_value(value, 0, r.is_url_key(key)))


#: The run a line belongs to: its job run's trace id, set by tracing.run_job for
#: the run and every item of it, so ``run_id`` finds a whole run and
#: ``trace_id`` one item. Printed after the reserved keys; a caller's own
#: run_id field wins.
RUN_ID_KEY = otel_context.create_key("fc.run_id")


def _active_run_id() -> str | None:
    value = otel_context.get_value(RUN_ID_KEY)
    return value if isinstance(value, str) else None


def _active_ids() -> dict[str, str] | None:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def _now() -> str:
    now = dt.datetime.now(dt.UTC)
    return f"{now:%Y-%m-%dT%H:%M:%S}.{now.microsecond // 1000:03d}Z"


def _build_entry(
    level: str,
    service: str,
    version: str,
    job: str | None,
    fields: Mapping[str, Any],
    message: object,
) -> dict[str, Any]:
    reserved: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for raw_key, value in fields.items():
        key = r.to_snake_case(r.plain_text(str(raw_key)))
        if key in _OWNED:
            continue
        if key in _CALLER_RESERVED:
            reserved[key] = value
        else:
            # The key as given decides too: printing may drop the word that names a secret.
            extras[key] = r.PLACEHOLDER if r.masks(str(raw_key), value) else _flat_value(key, value)
    msg = r.safe_text(message) if isinstance(message, str) else _reserved_text(message)
    event = reserved.get("event")
    entry: dict[str, Any] = {
        "time": _now(),
        "level": level,
        "service": service,
        "version": version,
        "event": _reserved_text(event) if isinstance(event, str) else "app.log",
        "msg": msg,
    }
    if reserved.get("call") is not None:
        call = re.split(r"[?#]", _reserved_text(reserved["call"]), maxsplit=1)[0].strip()
        if call:
            entry["call"] = call
    code = reserved.get("code")
    if isinstance(code, str | int) and not isinstance(code, bool):
        shaped = not isinstance(code, str) or _SHAPE_CODE.fullmatch(code) is not None
        entry["code"] = _reserved_text(str(code)) if shaped else r.PLACEHOLDER
    duration = _non_negative_ms(reserved.get("duration_ms"))
    if duration is not None:
        entry["duration_ms"] = duration
    if reserved.get("peer") is not None:
        entry["peer"] = _reserved_text(reserved["peer"])
    ids = _active_ids()
    if ids is not None:
        entry.update(ids)
    run = _reserved_text(reserved["job"]) if reserved.get("job") is not None else job
    if run is not None:
        entry["job"] = run
    if reserved.get("err") is not None:
        entry["err"] = _err_field(reserved["err"])
    queued = _non_negative_ms(reserved.get("queue_ms"))
    if queued is not None:
        entry["queue_ms"] = queued
    run_id = _active_run_id()
    if run_id is not None and "run_id" not in extras:
        entry["run_id"] = run_id
    entry.update(extras)
    return entry


class Logger:
    """One service's logger. ``log.info("text", key=value, ...)``; the reserved
    keys (event, call, code, duration_ms, peer, job, err, queue_ms) are
    passed the same way."""

    def __init__(
        self,
        *,
        service: str | None = None,
        version: str | None = None,
        level: str | None = None,
        env: Env | None = None,
        sink: Sink | None = None,
        bindings: Mapping[str, Any] | None = None,
    ) -> None:
        environ: Env = os.environ if env is None else env
        self.service, self.version = resolve_identity(environ, service, version)
        self.level = parse_level(level) or parse_level(environ.get("LOG_LEVEL")) or "info"
        self.job = _first_non_blank(environ.get("JOB_NAME"))
        self._sink = sink or _stdout
        self._bindings = dict(bindings or {})

    def child(self, **bindings: Any) -> Logger:
        child = Logger.__new__(Logger)
        child.__dict__.update(self.__dict__)
        child._bindings = {**self._bindings, **bindings}
        return child

    def is_enabled(self, level: str) -> bool:
        return level in LOG_LEVELS and _SEVERITY[level] >= _SEVERITY[self.level]

    def log(
        self, level: str, message: object = "", fields: Mapping[str, Any] | None = None
    ) -> None:
        if not self.is_enabled(level):
            return
        merged = {**self._bindings, **(fields or {})}
        try:
            entry = _build_entry(level, self.service, self.version, self.job, merged, message)
        except Exception:
            entry = _build_entry(level, self.service, self.version, self.job, {}, r.UNSERIALIZABLE)
        self._sink(r.dumps(entry))

    def trace(self, message: object = "", /, **fields: Any) -> None:
        self.log("trace", message, fields)

    def debug(self, message: object = "", /, **fields: Any) -> None:
        self.log("debug", message, fields)

    def info(self, message: object = "", /, **fields: Any) -> None:
        self.log("info", message, fields)

    def warn(self, message: object = "", /, **fields: Any) -> None:
        self.log("warn", message, fields)

    warning = warn

    def error(self, message: object = "", /, **fields: Any) -> None:
        self.log("error", message, fields)

    def fatal(self, message: object = "", /, **fields: Any) -> None:
        self.log("fatal", message, fields)


# --- the process logger -------------------------------------------------------------------------

_process: Logger | None = None


def use_logger(logger: Logger | None) -> Logger | None:
    """Make ``logger`` the process logger; returns the previous one."""
    global _process
    previous, _process = _process, logger
    return previous


def process_logger() -> Logger:
    """The logger configure() set, else one on stdout named by OTEL_SERVICE_NAME."""
    global _process
    if _process is None:
        _process = Logger(service=service_name("media-manager"))
    return _process


# --- Python's logging ---------------------------------------------------------------------------

_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


def _level_of(levelno: int) -> str:
    if levelno >= logging.CRITICAL:
        return "fatal"
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warn"
    if levelno >= logging.INFO:
        return "info"
    return "debug" if levelno >= logging.DEBUG else "trace"


class LogShapeHandler(logging.Handler):
    """A stdlib record as one line of the shape."""

    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.target = logger

    def emit(self, record: logging.LogRecord) -> None:
        fields: dict[str, Any] = {k: v for k, v in record.__dict__.items() if k not in _RECORD_KEYS}
        fields.setdefault("logger", record.name)
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            fields.setdefault("err", exc)
            fields["stack"] = r.stack_tail(exc)
        try:
            message = record.getMessage()
        except Exception:  # a bad format: the format string as written
            message = str(record.msg)
        self.target.log(_level_of(record.levelno), message, fields)


_STDLIB_LEVEL = {
    "trace": 5,
    "debug": 10,
    "info": 20,
    "warn": 30,
    "error": 40,
    "fatal": 50,
    "silent": 60,
}


def configure(service: str, *, level: str | None = None, env: Env | None = None) -> Logger:
    """The process logger for ``service`` on stdout, and every stdlib logger
    routed into it. uvicorn's own loggers propagate here; its access log is
    turned off (the http.in line replaces it)."""
    logger = Logger(service=service_name(service, env), level=level, env=env)
    use_logger(logger)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(LogShapeHandler(logger))
    root.setLevel(_STDLIB_LEVEL[logger.level])
    for name in ("uvicorn", "uvicorn.error"):
        child = logging.getLogger(name)
        child.handlers.clear()
        child.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True
    return logger
