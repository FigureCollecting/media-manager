"""LG-U11: real trace ids with no exporter, W3C traceparent, the propagation
allowlist, and one root trace per CLI run (fc-shared server/tracing, job and
propagation, in Python)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from app.telemetry import tracing as tr
from app.telemetry.log import Logger
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def capture(env: dict[str, str] | None = None) -> tuple[Logger, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    log = Logger(
        service="mm-report", version="1", env=env or {}, sink=lambda s: lines.append(json.loads(s))
    )
    return log, lines


def test_tracing_records_real_ids_and_exports_nothing() -> None:
    provider = tr.start_tracing("media-api", env={})
    assert isinstance(provider, TracerProvider)
    assert tr.start_tracing("other", env={}) is provider  # the first registration wins
    # No span processor: nothing is exported, and nothing is queued for export.
    assert provider._active_span_processor._span_processors == ()
    with trace.get_tracer("t").start_as_current_span("s") as span:
        ctx = span.get_span_context()
        assert ctx.is_valid and ctx.trace_id != 0 and ctx.span_id != 0
        # Sampled, and (W3C Trace Context level 2) a random trace id: flags 03 from this SDK.
        flags = int(ctx.trace_flags)
        assert flags & 0x01
        assert tr.current_traceparent() == f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{flags:02x}"
    assert tr.current_traceparent() is None


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (TRACEPARENT, True),
        (f"  {TRACEPARENT} ", True),
        ("00-00000000000000000000000000000000-00f067aa0ba902b7-01", False),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", False),
        ("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", False),
        ("00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01", False),
        ("garbage", False),
        ("", False),
        (None, False),
    ],
)
def test_parses_a_version_00_traceparent(value: str | None, ok: bool) -> None:
    parsed = tr.parse_traceparent(value)
    assert (parsed is not None) is ok
    if parsed is not None:
        assert (parsed.trace_id, parsed.span_id, parsed.is_remote) == (
            0x4BF92F3577B34DA6A3CE929D0E0E4736,
            0x00F067AA0BA902B7,
            True,
        )


def test_continues_a_stored_traceparent_or_starts_clean() -> None:
    tr.start_tracing("media-worker", env={})
    token = otel_context.attach(tr.context_from_traceparent(TRACEPARENT))
    try:
        with trace.get_tracer("t").start_as_current_span("child") as span:
            assert f"{span.get_span_context().trace_id:032x}" == "4bf92f3577b34da6a3ce929d0e0e4736"
    finally:
        otel_context.detach(token)
    assert (
        not trace.get_current_span(tr.context_from_traceparent("junk")).get_span_context().is_valid
    )


def test_extracts_and_injects_through_header_carriers() -> None:
    ctx = tr.extract_context({"traceparent": TRACEPARENT, "tracestate": "a=1"})
    assert f"{trace.get_current_span(ctx).get_span_context().trace_id:032x}" == (
        "4bf92f3577b34da6a3ce929d0e0e4736"
    )
    carrier: dict[str, str] = {}
    tr.inject_context(carrier, ctx)
    assert carrier["traceparent"] == TRACEPARENT


@pytest.mark.parametrize(
    ("env", "host", "allowed"),
    [
        ({}, "ingest-server.fc.svc.cluster.local", True),
        ({}, "ingest-server.fc.svc", True),
        ({}, "localhost", True),
        ({}, "LOCALHOST.", True),
        ({}, "127.0.0.1", False),
        ({}, "svc.cluster.local", False),
        ({}, "static.myfigurecollection.net", False),
        ({}, "evil.svc.cluster.local.example", False),
        ({"OTEL_PROPAGATION_ALLOWLIST": "10.1.2.3"}, "10.1.2.3", True),
        ({"OTEL_PROPAGATION_ALLOWLIST": "10.1.2.3"}, "localhost", False),
        (
            {"OTEL_PROPAGATION_ALLOWLIST": " http://scraper.fc.svc:3050/ , *.mesh "},
            "scraper.fc.svc",
            True,
        ),
        ({"OTEL_PROPAGATION_ALLOWLIST": "*.mesh"}, "a.mesh", True),
        ({"OTEL_PROPAGATION_ALLOWLIST": "*"}, "anything", False),
        ({"OTEL_PROPAGATION_ALLOWLIST": "  ,  "}, "localhost", True),
    ],
)
def test_the_propagation_allowlist(env: dict[str, str], host: str, allowed: bool) -> None:
    assert tr.propagation_allowed(host, env) is allowed


def test_a_run_is_one_root_trace_bracketed_by_start_and_end() -> None:
    tr.start_tracing("mm-report", env={})
    env = {"JOB_NAME": "mm-report-29312345"}
    log, lines = capture(env)
    seen: list[str | None] = []
    token = otel_context.attach(
        tr.context_from_traceparent(TRACEPARENT)
    )  # an unrelated active span
    try:

        def body() -> int:
            log.info("inside")
            seen.append(tr.current_traceparent())
            return 0

        counts = {"exit": 0, "Leased": 3}
        assert tr.run_job("mm-report", body, logger=log, counts=lambda _c: counts, env=env) == 0
    finally:
        otel_context.detach(token)
    start, inside, end = lines
    assert (start["event"], start["msg"], start["job"]) == (
        "job.start",
        "mm-report started",
        "mm-report-29312345",
    )
    assert (end["event"], end["code"], end["level"], end["exit"], end["leased"]) == (
        "job.end",
        "ok",
        "info",
        0,
        3,
    )
    assert isinstance(end["duration_ms"], int | float) and end["duration_ms"] >= 0
    ids = {line["trace_id"] for line in lines}
    assert len(ids) == 1 and "4bf92f3577b34da6a3ce929d0e0e4736" not in ids  # a ROOT trace
    assert seen[0] is not None and seen[0].split("-")[1] == start["trace_id"]


def test_a_run_that_ends_badly_says_so() -> None:
    tr.start_tracing("mm-report", env={})
    log, lines = capture()
    assert (
        tr.run_job(
            "mm-report",
            lambda: 3,
            logger=log,
            counts=lambda code: {"exit": code},
            ok=lambda c: c == 0,
        )
        == 3
    )
    end = lines[-1]
    assert (end["event"], end["code"], end["level"], end["exit"], end["job"]) == (
        "job.end",
        "error",
        "warn",
        3,
        "mm-report",
    )

    def boom() -> int:
        raise RuntimeError("spine down: Bearer abcdefghijkl")

    with pytest.raises(RuntimeError):
        tr.run_job("mm-report", boom, logger=log)
    end = lines[-1]
    assert (end["event"], end["code"], end["level"]) == ("job.end", "error", "error")
    assert end["err"] == {"type": "RuntimeError", "message": "spine down: [REDACTED]"}
    assert end["msg"] == "mm-report failed"
