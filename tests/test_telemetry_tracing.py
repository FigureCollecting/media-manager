"""LG-U11: real trace ids with no exporter, W3C traceparent, the propagation
allowlist, and the job model (fc-shared server/tracing, job and propagation,
in Python): a run is a root span job.<name> bracketed by job.start and
job.end; each ITEM of it is a NEW root trace (item.enqueue, PRODUCER) with a
span link and fc.run_id back to the run (Ross's choice: one trace per item).
Every line of a run carries run_id, the run's trace id."""

from __future__ import annotations

import json
from typing import Any

import pytest
from app.telemetry import tracing as tr
from app.telemetry.log import Logger
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

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

        def body(run: tr.JobRun) -> int:
            assert run.name == "mm-report"
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
    assert {line["run_id"] for line in lines} == {start["trace_id"]}


def test_a_run_that_ends_badly_says_so() -> None:
    tr.start_tracing("mm-report", env={})
    log, lines = capture()
    assert (
        tr.run_job(
            "mm-report",
            lambda _run: 3,
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

    def boom(_run: tr.JobRun) -> int:
        raise RuntimeError("spine down: Bearer abcdefghijkl")

    with pytest.raises(RuntimeError):
        tr.run_job("mm-report", boom, logger=log)
    end = lines[-1]
    assert (end["event"], end["code"], end["level"]) == ("job.end", "error", "error")
    assert end["err"] == {"type": "RuntimeError", "message": "spine down: [REDACTED]"}
    assert end["msg"] == "mm-report failed"


# --- one trace per ITEM (fc-shared server/job runJob + run.item, Ross's choice) ---------------


def recording() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """A tracer whose finished spans can be read (this test's own provider, not the global one)."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _trace_id() -> str:
    return format(trace.get_current_span().get_span_context().trace_id, "032x")


def test_every_item_is_its_own_root_trace_linked_to_the_run() -> None:
    tracer, exporter = recording()
    log, lines = capture({"JOB_NAME": "mm-report-29312345"})
    item_ids: list[str] = []

    def body(run: tr.JobRun) -> str:
        for site in ("orzgk", "amiami"):

            def one(site: str = site) -> None:
                log.info("enqueued", event="item.enqueue", site=site)
                item_ids.append(_trace_id())

            run.item(one, attributes={"fc.site": site})
        log.info("between items", event="app.pass")
        return run.run_id

    run_id = tr.run_job("mm-report", body, logger=log, tracer=tracer, env={})
    spans = exporter.get_finished_spans()
    (run_span,) = (s for s in spans if s.name == "job.mm-report")
    items = [s for s in spans if s.name == "item.enqueue"]
    assert run_id == format(run_span.context.trace_id, "032x")
    assert len(items) == 2 and len({run_id, *item_ids}) == 3
    for item in items:
        assert item.parent is None
        assert item.kind is SpanKind.PRODUCER
        assert item.status.status_code is StatusCode.OK
        assert item.attributes is not None and item.attributes["fc.run_id"] == run_id
        (link,) = item.links
        assert (link.context.trace_id, link.context.span_id) == (
            run_span.context.trace_id,
            run_span.context.span_id,
        )
    assert [s.attributes["fc.site"] for s in items if s.attributes] == ["orzgk", "amiami"]
    item_lines = [line for line in lines if line["event"] == "item.enqueue"]
    assert [line["trace_id"] for line in item_lines] == item_ids  # trace_id finds one item
    assert {line["run_id"] for line in lines} == {run_id}  # run_id finds the whole run
    assert [line["trace_id"] for line in lines if line["event"] != "item.enqueue"] == [run_id] * 3
    keys = list(item_lines[0])
    assert keys.index("run_id") == keys.index("job") + 1 and keys[-1] == "site"


def test_an_item_with_no_options_is_item_enqueue_tagged_only_with_fc_run_id() -> None:
    tracer, exporter = recording()
    log, _ = capture()
    assert (
        tr.run_job("x", lambda run: run.item(lambda: "queued"), logger=log, tracer=tracer)
        == "queued"
    )
    (item,) = (s for s in exporter.get_finished_spans() if s.name == "item.enqueue")
    assert list(item.attributes or {}) == ["fc.run_id"]


def test_a_failed_item_is_error_rethrown_and_the_run_decides() -> None:
    tracer, exporter = recording()
    log, lines = capture()

    def body(run: tr.JobRun) -> str:
        def broken() -> None:
            raise RuntimeError("item broke")

        try:
            run.item(broken, name="item.fetch")
        except RuntimeError as exc:
            return str(exc)
        return "unreached"

    assert tr.run_job("x", body, logger=log, tracer=tracer) == "item broke"
    (item,) = (s for s in exporter.get_finished_spans() if s.name == "item.fetch")
    assert item.status.status_code is StatusCode.ERROR
    assert lines[-1]["code"] == "ok"


def test_an_item_opened_by_hand_runs_in_its_trace_until_it_is_ended() -> None:
    """fc-shared itemContext: the caller ends the span (a chunk that closes after the next opened)."""
    tracer, exporter = recording()
    log, lines = capture()
    seen: dict[str, str] = {}

    def body(run: tr.JobRun) -> None:
        first = run.open_item(name="item.chunk", attributes={"fc.chunk": 1})
        second = run.open_item(name="item.chunk", attributes={"fc.chunk": 2})
        seen["first"] = first.run(_trace_id)
        seen["second"] = second.run(_trace_id)
        first.run(lambda: log.info("closing", event="app.chunk"))
        first.end()
        first.end()  # idempotent
        second.end(error=True)
        seen["after"] = _trace_id()

    tr.run_job("x", body, logger=log, tracer=tracer)
    chunks = [s for s in exporter.get_finished_spans() if s.name == "item.chunk"]
    assert len(chunks) == 2
    assert [s.status.status_code for s in chunks] == [StatusCode.OK, StatusCode.ERROR]
    assert len({seen["first"], seen["second"], seen["after"]}) == 3
    (chunk_line,) = (line for line in lines if line["event"] == "app.chunk")
    assert chunk_line["trace_id"] == seen["first"] and chunk_line["run_id"] == seen["after"]


def test_the_null_item_runs_in_the_callers_trace() -> None:
    tracer, _ = recording()
    with tracer.start_as_current_span("caller") as span:
        here = format(span.get_span_context().trace_id, "032x")
        assert tr.NULL_ITEM.run(_trace_id) == here
        tr.NULL_ITEM.end(error=True)


def test_without_a_recording_tracer_items_run_and_lines_carry_no_ids() -> None:
    log, lines = capture()
    noop = trace.NoOpTracer()
    assert tr.run_job("x", lambda run: run.item(lambda: 3), logger=log, tracer=noop) == 3
    assert [(line["event"], "trace_id" in line, "run_id" in line) for line in lines] == [
        ("job.start", False, False),
        ("job.end", False, False),
    ]


def test_spans_are_flushed_before_the_run_returns_and_a_failed_flush_is_not_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushed: list[str] = []

    class Provider:
        def __init__(self, fails: bool) -> None:
            self.fails = fails

        def get_tracer(self, *_: Any, **__: Any) -> trace.Tracer:
            return trace.NoOpTracer()

        def force_flush(self) -> bool:
            flushed.append("flush")
            if self.fails:
                raise RuntimeError("collector gone")
            return True

    log, _ = capture()
    for fails in (False, True):
        monkeypatch.setattr(tr.trace, "get_tracer_provider", lambda f=fails: Provider(f))
        assert tr.run_job("x", lambda _run: 7, logger=log) == 7
    assert flushed == ["flush", "flush"]


def test_a_callers_own_run_id_field_is_kept() -> None:
    tracer, _ = recording()
    log, lines = capture()
    tr.run_job("x", lambda run: log.info("", run_id="given"), logger=log, tracer=tracer)
    assert lines[1]["run_id"] == "given"
