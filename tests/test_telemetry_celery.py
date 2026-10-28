"""LG-U11: the Celery worker in the estate's shape.

The worker's logging is ours (Celery's setup_logging signal), each task runs
in a span continuing the trace that enqueued it (the traceparent travels as
a message header, fc-shared job.ts currentTraceparent/withTraceparent), and
each task ends with one item.process line.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from app.telemetry import celery as tcel
from app.telemetry import log as tlog
from app.telemetry import tracing as tr
from app.telemetry.log import Logger
from app.workers import celery_app
from celery import signals
from opentelemetry import trace

from test_telemetry_log import VALIDATOR

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@celery_app.task(name="lg_u11_probe")
def probe(fail: bool = False) -> str | None:
    logging.getLogger("lg-u11.task").warning("inside the task")
    if fail:
        raise RuntimeError("probe failed: Bearer abcdefghijkl")
    return tr.current_traceparent()


@pytest.fixture()
def lines() -> Iterator[list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []
    log = Logger(
        service="media-worker", version="1", env={}, sink=lambda s: captured.append(json.loads(s))
    )
    previous = tlog.use_logger(log)
    handler = tlog.LogShapeHandler(log)
    task_logger = logging.getLogger("lg-u11.task")
    task_logger.addHandler(handler)
    task_logger.propagate = False
    yield captured
    task_logger.removeHandler(handler)
    task_logger.propagate = True
    tlog.use_logger(previous)


def _valid(line: dict[str, Any]) -> None:
    assert [e.message for e in VALIDATOR.iter_errors(line)] == [], line


def test_the_handlers_are_connected() -> None:
    def uids(signal: Any) -> set[Any]:
        return {key[0] for key, _receiver in signal.receivers}

    assert "lg-u11.setup_logging" in uids(signals.setup_logging)
    assert "lg-u11.before_task_publish" in uids(signals.before_task_publish)
    assert {"lg-u11.task_prerun"} <= uids(signals.task_prerun)
    assert {"lg-u11.task_postrun"} <= uids(signals.task_postrun)
    assert {"lg-u11.task_failure"} <= uids(signals.task_failure)


def test_the_enqueuers_trace_rides_the_message() -> None:
    tr.start_tracing("media-api", env={})
    headers: dict[str, Any] = {}
    tcel.before_publish(headers=headers)
    assert "traceparent" not in headers  # no active span: nothing to carry
    with trace.get_tracer("t").start_as_current_span("request") as span:
        tcel.before_publish(headers=headers)
        tcel.before_publish(headers=None)
        kept = {"traceparent": TP}
        tcel.before_publish(headers=kept)
    ctx = span.get_span_context()
    assert headers["traceparent"].split("-")[1:3] == [f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"]
    assert kept == {"traceparent": TP}


def test_a_task_continues_its_enqueuers_trace_and_ends_with_one_line(
    lines: list[dict[str, Any]],
) -> None:
    tr.start_tracing("media-worker", env={})
    seen = probe.apply(headers={"traceparent": TP}).get()
    assert seen is not None and seen.split("-")[1] == "4bf92f3577b34da6a3ce929d0e0e4736"
    inside, done = lines
    assert (
        inside["msg"] == "inside the task"
        and inside["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    )
    assert (done["event"], done["code"], done["task"], done["level"]) == (
        "item.process",
        "ok",
        "lg_u11_probe",
        "info",
    )
    assert done["trace_id"] == inside["trace_id"] and isinstance(done["duration_ms"], int | float)
    _valid(done)


def test_a_failed_task_says_so_and_a_task_without_a_trace_starts_one(
    lines: list[dict[str, Any]],
) -> None:
    tr.start_tracing("media-worker", env={})
    result = probe.apply(kwargs={"fail": True})
    assert result.failed()
    done = lines[-1]
    assert (done["event"], done["code"], done["level"], done["state"]) == (
        "item.process",
        "error",
        "error",
        "FAILURE",
    )
    assert done["err"] == {"type": "RuntimeError", "message": "probe failed: [REDACTED]"}
    assert done["trace_id"] != "4bf92f3577b34da6a3ce929d0e0e4736"
    _valid(done)


def test_postrun_for_a_task_it_never_saw_does_nothing(lines: list[dict[str, Any]]) -> None:
    tcel.task_postrun(task_id="never", task=SimpleNamespace(name="x"), state="SUCCESS")
    tcel.task_failure(task_id="never", exception=RuntimeError("x"))
    assert lines == []


def test_the_worker_logs_in_the_shape(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, tlog.use_logger(None))
    try:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        tcel.setup_logging(loglevel=logging.WARNING)
        logging.getLogger("celery.worker").warning("ready")
        line = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert (line["service"], line["msg"], line["logger"]) == (
            "media-worker",
            "ready",
            "celery.worker",
        )
        tcel.setup_logging(loglevel="debug")
        assert tlog.process_logger().level == "debug"
        tcel.setup_logging()
        assert tlog.process_logger().level == "info"
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        tlog.use_logger(saved[2])


def test_the_trace_crosses_a_real_broker_to_a_real_worker() -> None:
    """Publish -> broker -> worker, Celery's message protocol 2 (memory transport):
    the task runs in the trace that enqueued it."""
    from celery import Celery
    from celery.contrib.testing.worker import start_worker

    tr.start_tracing("media-api", env={})
    app = Celery("lg-u11-roundtrip", broker="memory://", backend="cache+memory://")

    @app.task(name="lg_u11_roundtrip")
    def roundtrip() -> str | None:
        return tr.current_traceparent()

    with start_worker(app, perform_ping_check=False, pool="solo", loglevel=logging.WARNING):
        with trace.get_tracer("t").start_as_current_span("enqueue") as span:
            pending = roundtrip.delay()
        seen = pending.get(timeout=10)
    assert seen is not None
    assert seen.split("-")[1] == f"{span.get_span_context().trace_id:032x}"
    assert seen.split("-")[2] != f"{span.get_span_context().span_id:016x}"  # the task's own span
