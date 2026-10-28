"""LG-U11: media-manager's log line is the estate's line.

GOLDEN: tests/fixtures/log_shape/fc-shared-1.8.0-golden.jsonl holds the lines
fc-shared 1.8.0's own server logger wrote (tests/fixtures/log_shape/golden.mjs)
for the cases below; the Python logger, given the same input, must write the
same keys in the same order with the same values (``time`` aside). Every line
is also checked against fc-shared's published schema (log-shape.schema.json,
the 1.8.0 file byte for byte).
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from app import telemetry
from app.telemetry import log as tlog
from app.telemetry.log import RESERVED_LOG_KEYS, Logger, LogShapeHandler
from jsonschema import Draft202012Validator
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

FIXTURES = Path(__file__).parent / "fixtures" / "log_shape"
SCHEMA = json.loads((FIXTURES / "log-shape.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
GOLDEN = [
    json.loads(line)
    for line in (FIXTURES / "fc-shared-1.8.0-golden.jsonl").read_text().splitlines()
]
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"
ISO_MS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def valid(line: dict[str, Any]) -> None:
    errors = [e.message for e in VALIDATOR.iter_errors(line)]
    assert errors == [], (line, errors)


@contextmanager
def remote_span(trace_id: str = TRACE_ID, span_id: str = SPAN_ID) -> Iterator[None]:
    ctx = SpanContext(
        int(trace_id, 16), int(span_id, 16), is_remote=True, trace_flags=TraceFlags(1)
    )
    token = otel_context.attach(trace.set_span_in_context(NonRecordingSpan(ctx)))
    try:
        yield
    finally:
        otel_context.detach(token)


def capture(**options: Any) -> tuple[Logger, Callable[[], list[dict[str, Any]]], list[str]]:
    raw: list[str] = []
    options.setdefault("env", {})
    log = Logger(sink=raw.append, **options)
    return log, lambda: [json.loads(line) for line in raw], raw


def _golden_cases() -> list[dict[str, Any]]:
    out: list[str] = []
    api = Logger(service="media-api", version="0.1.0", env={}, sink=out.append)
    job = Logger(
        service="mm-report",
        version="0.1.0",
        env={"JOB_NAME": "mm-report-29312345"},
        sink=out.append,
    )
    with remote_span():
        api.info(
            "",
            event="rpc.in",
            call="media.v1.SpineMedia/ReportDerivative",
            code="ok",
            duration_ms=41.23,
            peer="media-report.media.serviceaccount.identity.linkerd.app.mesh.estate",
            site="orzgk",
            claims=23,
        )
        job.info(
            "pass finished", event="job.end", code="ok", duration_ms=671000, stores=27, enqueued=412
        )
    api.info("", site="orzgk")
    with remote_span():
        api.error("failed hard", err=TypeError("boom"))
        api.warn(
            "see https://x.example/p?token=1 now",
            imageUrl="https://u:p@cdn.example/a.jpg?sig=1#f",
            password="x",
            note="use Bearer abc.def now",
            detail={"a": 1, "token": "t", "nested": {"href": "https://h.example/x?y=1"}},
            itemId=7,
            ratio=0.5,
            ok=True,
            gone=None,
        )
        api.info(
            "",
            event="http.in",
            call="GET /d/{sha256}",
            code="200",
            duration_ms=1.25,
            peer="unmeshed",
        )
        api.warn(
            "",
            event="http.in",
            call="GET <unmatched>",
            code=404,
            duration_ms=0.04,
            peer="invalid",
            aborted=False,
        )
    api.info(
        "multi\nline",
        event="app.log",
        form="user=a&password=b&code=c",
        header="Cookie: a=b",
        json='{"token":"t","n":1}',
        HTTPStatus="x",
        **{"9lives": 1},
    )
    return [json.loads(line) for line in out]


def test_every_golden_line_is_written_key_for_key_as_fc_shared_writes_it() -> None:
    ours = _golden_cases()
    assert len(ours) == len(GOLDEN) == 8
    for mine, theirs in zip(ours, GOLDEN, strict=True):
        assert ISO_MS.match(mine["time"])
        mine = {k: v for k, v in mine.items() if k != "time"}
        theirs = {k: v for k, v in theirs.items() if k != "time"}
        assert list(mine) == list(theirs)
        assert mine == theirs
    for line in [*ours, *GOLDEN]:
        valid(line)


def test_the_schema_fixture_is_fc_shared_1_8_0s_file() -> None:
    import hashlib

    digest = hashlib.sha256((FIXTURES / "log-shape.schema.json").read_bytes()).hexdigest()
    assert digest == "a9243e54eba9c073c41bd5bc43f440acefb2b8c944f4d00278fc4fc62aafed20"
    Draft202012Validator.check_schema(SCHEMA)


def test_exposes_the_reserved_keys_in_contract_order() -> None:
    assert RESERVED_LOG_KEYS == (
        "time", "level", "service", "version", "event", "msg", "call", "code",
        "duration_ms", "peer", "trace_id", "span_id", "job", "err", "queue_ms",
    )  # fmt: skip


def test_trace_ids_are_absent_never_zeroed_without_a_span() -> None:
    log, lines, _ = capture(service="s", version="1")
    log.info("no span here")
    with remote_span("0" * 32, "0" * 16):
        log.info("invalid context")
    for line in lines():
        assert "trace_id" not in line and "span_id" not in line
        valid(line)


def test_types_the_reserved_fields() -> None:
    log, lines, _ = capture(service="s", version="1")
    log.info(
        event="http.in",
        call="POST /ingest/scrape?token=abc#frag",
        code=202,
        duration_ms="slow",
        queue_ms=12.345,
        peer="unmeshed",
    )
    log.info(
        event="app.timing", call="GET /a#frag", duration_ms=-5, queue_ms=-1, peer="forged\npeer"
    )
    log.info(event="job.end", code="ok", durationMs=671000, queueMs=5)
    log.info(event="rpc.in", call="a.v1.S/M", code="oauth-code", duration_ms=1, peer="p")
    log.info(event="rpc.in", call="a.v1.S/M", code=True, duration_ms=float("nan"), peer="p")
    first, second, third, fourth, fifth = lines()
    assert (first["call"], first["code"], first["queue_ms"]) == ("POST /ingest/scrape", "202", 12.3)
    assert "duration_ms" not in first
    assert second["call"] == "GET /a" and second["peer"] == "forged peer"
    assert "duration_ms" not in second and "queue_ms" not in second
    assert (third["duration_ms"], third["queue_ms"]) == (671000, 5) and "durationMs" not in third
    assert fourth["code"] == "[REDACTED]"
    assert "code" not in fifth and "duration_ms" not in fifth
    for line in (second, third):
        valid(line)


def test_owned_keys_stay_owned() -> None:
    log, lines, _ = capture(service="s", version="1")
    with remote_span():
        log.info(
            "real",
            time="x",
            level="fatal",
            service="other",
            version="9",
            msg="fake",
            trace_id="f" * 32,
            span_id="f" * 16,
        )
    line = lines()[0]
    assert (line["level"], line["service"], line["version"], line["msg"]) == (
        "info",
        "s",
        "1",
        "real",
    )
    assert (line["trace_id"], line["span_id"]) == (TRACE_ID, SPAN_ID)


def test_err_is_type_and_message_and_a_non_error_is_typed() -> None:
    log, lines, _ = capture(service="s", version="1")
    log.error("x", err=ValueError("line one\nBearer abcdefghijkl"))
    log.error("y", err="plain words")
    first, second = lines()
    assert first["err"] == {"type": "ValueError", "message": "line one [REDACTED]"}
    assert second["err"] == {"type": "str", "message": "plain words"}
    valid(first)


def test_a_message_that_is_not_a_string_is_printed_as_a_value() -> None:
    log, lines, _ = capture(service="s", version="1")
    log.info(["a", {"password": "p"}])
    assert lines()[0]["msg"] == '["a",{"password":"[REDACTED]"}]'


def test_a_call_it_cannot_render_is_one_line_saying_so() -> None:
    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("cannot")

    log, lines, _ = capture(service="s", version="1")
    log.log("warn", "x", {Unprintable(): 1})  # type: ignore[dict-item]
    line = lines()[0]
    assert (line["msg"], line["level"]) == ("[unserializable]", "warn")
    valid(line)


def test_filters_below_the_level_and_silent_writes_nothing() -> None:
    log, lines, _ = capture(service="s", version="1", level="warn")
    log.trace("t")
    log.debug("d")
    log.info("i")
    log.warn("w")
    log.warning("w2")
    log.error("e")
    log.fatal("f")
    assert [line["level"] for line in lines()] == ["warn", "warn", "error", "fatal"]
    assert log.is_enabled("error") and not log.is_enabled("info") and not log.is_enabled("nope")
    quiet, quiet_lines, _ = capture(service="s", version="1", level="silent")
    quiet.fatal("x")
    assert quiet_lines() == []


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("INFO", "info"),
        ("WARNING", "warn"),
        ("warn", "warn"),
        ("CRITICAL", "fatal"),
        ("DEBUG", "debug"),
        ("constructor", "info"),
        ("", "info"),
    ],
)
def test_reads_log_level_from_env(setting: str, expected: str) -> None:
    log = Logger(service="s", version="1", env={"LOG_LEVEL": setting}, sink=lambda _l: None)
    assert log.level == expected


def test_child_loggers_accumulate_bindings() -> None:
    log, lines, _ = capture(service="s", version="1", bindings={"site": "a"})
    child = log.child(batch=3)
    child.info("x", rpc="ReportMask")
    child.child(site="b").info("y")
    first, second = lines()
    assert (first["site"], first["batch"], first["rpc"]) == ("a", 3, "ReportMask")
    assert second["site"] == "b"


def test_stamps_job_from_job_name() -> None:
    log, lines, _ = capture(service="s", version="1", env={"JOB_NAME": " mm-report-1 "})
    log.info("x")
    log.info("y", job="explicit")
    assert [line["job"] for line in lines()] == ["mm-report-1", "explicit"]


def test_resolves_service_and_version() -> None:
    sink: list[str] = []
    Logger(
        env={"OTEL_SERVICE_NAME": " media-x ", "SERVICE_VERSION": "2.0.1"}, sink=sink.append
    ).info("a")
    Logger(service="given", env={"OTEL_SERVICE_NAME": "media-x"}, sink=sink.append).info("b")
    Logger(env={}, sink=sink.append).info("c")
    first, second, third = (json.loads(line) for line in sink)
    assert (first["service"], first["version"]) == ("media-x", "2.0.1")
    assert second["service"] == "given"
    assert third["service"] == "unknown_service" and third["version"]


def test_an_image_without_the_installed_package_is_version_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import metadata

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(tlog.metadata, "version", missing)
    sink: list[str] = []
    Logger(env={}, sink=sink.append).info("a")
    assert json.loads(sink[0])["version"] == "unknown"


def test_the_printed_key_decides_too() -> None:
    """fc-shared flatValue: 'authCode?x=1' prints as auth_code, and so is masked
    (fc-shared 1.8.0's own logger prints this line for the same input)."""
    sink: list[str] = []
    Logger(env={}, sink=sink.append).info("", **{"authCode?x=1": "hunter81", "login?next=1": "a"})
    line = json.loads(sink[0])
    assert (line["auth_code"], line["login"]) == ("[REDACTED]", "a")


def test_service_name_prefers_the_deployments_name() -> None:
    assert telemetry.service_name("media-api", {"OTEL_SERVICE_NAME": "media-api-canary"}) == (
        "media-api-canary"
    )
    assert telemetry.service_name("media-api", {"OTEL_SERVICE_NAME": "  "}) == "media-api"


def test_writes_one_newline_terminated_line_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    Logger(service="s", version="1", env={}).info("hello", n=1)
    out = capsys.readouterr().out
    assert out.endswith("\n") and out.count("\n") == 1
    assert json.loads(out)["msg"] == "hello"


# --- the stdlib logging bridge -------------------------------------------------------------


@pytest.fixture()
def root_restored() -> Iterator[None]:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    others = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate,
                  logging.getLogger(n).disabled) for n in names}  # fmt: skip
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    for n, (handlers, propagate, disabled) in others.items():
        lg = logging.getLogger(n)
        lg.handlers[:] = handlers
        lg.propagate, lg.disabled = propagate, disabled
    tlog.use_logger(None)


def test_a_stdlib_record_becomes_one_line_with_its_extras() -> None:
    log, lines, _ = capture(service="s", version="1", level="debug")
    stdlib = logging.getLogger("lg-u11.bridge")
    stdlib.propagate = False
    stdlib.setLevel(logging.DEBUG)
    handler = LogShapeHandler(log)
    stdlib.addHandler(handler)
    try:
        stdlib.warning(
            "gallery_url_blocked",
            extra={"url": "https://u:p@static.example/upload/a.jpg?sig=S#f", "figure_id": "f1"},
        )
        stdlib.info("count %d of %s", 3, "x")
        stdlib.debug("bad %d", "not-a-number")
        try:
            raise KeyError("Bearer abcdefghijkl")
        except KeyError:
            stdlib.exception(
                "unhandled_exception", extra={"request_id": "r1", "event": "app.crash"}
            )
        stdlib.critical("down")
    finally:
        stdlib.removeHandler(handler)
    blocked, count, bad, crash, down = lines()
    assert blocked["level"] == "warn" and blocked["msg"] == "gallery_url_blocked"
    assert blocked["url"] == "https://static.example/upload/a.jpg"
    assert (blocked["figure_id"], blocked["logger"], blocked["event"]) == (
        "f1",
        "lg-u11.bridge",
        "app.log",
    )
    assert count["msg"] == "count 3 of x"
    assert bad["msg"] == "bad %d" and bad["level"] == "debug"
    assert crash["event"] == "app.crash" and crash["request_id"] == "r1"
    assert crash["err"] == {"type": "KeyError", "message": "'[REDACTED]'"}
    assert "Traceback" in crash["stack"] and "abcdefghijkl" not in crash["stack"]
    assert down["level"] == "fatal"
    for line in (blocked, count, crash, down):
        valid(line)


def test_a_record_below_debug_is_trace() -> None:
    log, lines, _ = capture(service="s", version="1", level="trace")
    handler = LogShapeHandler(log)
    record = logging.LogRecord("x", 5, __file__, 1, "low", (), None)
    handler.handle(record)
    assert lines()[0]["level"] == "trace"


def test_configure_routes_every_stdlib_logger_into_the_shape(
    root_restored: None, capsys: pytest.CaptureFixture[str]
) -> None:
    logging.getLogger("uvicorn.error").addHandler(logging.StreamHandler(io.StringIO()))
    log = telemetry.configure("media-api", level="WARNING", env={})
    assert tlog.process_logger() is log
    assert log.level == "warn" and logging.getLogger().level == logging.WARNING
    logging.getLogger("app.anything").warning("w")
    logging.getLogger("uvicorn.error").error("e")
    logging.getLogger("uvicorn.access").warning("GET /x?token=1 HTTP/1.1 200")
    logging.getLogger("app.anything").info("dropped below the level")
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [(x["msg"], x["service"]) for x in lines] == [("w", "media-api"), ("e", "media-api")]
    assert logging.getLogger("uvicorn.access").disabled


def test_the_process_logger_defaults_to_stdout(root_restored: None) -> None:
    tlog.use_logger(None)
    first = tlog.process_logger()
    assert tlog.process_logger() is first
    previous = tlog.use_logger(Logger(service="x", env={}, sink=lambda _l: None))
    assert previous is first
