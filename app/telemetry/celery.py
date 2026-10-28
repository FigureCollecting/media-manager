"""The Celery worker in the estate's shape.

* setup_logging: Celery leaves the process's logging to us when a receiver is
  connected, so the worker's own lines and every task's are the shape.
* before_task_publish (the API process): the active trace's traceparent rides
  the message as a header, as fc-shared's currentTraceparent stores it with
  a queued item.
* task_prerun / task_postrun / task_failure (the worker): the task runs in a
  CONSUMER span continuing that trace (a new root without one) and ends with
  ONE item.process line: task, task_id, state, code ok|error, duration_ms,
  err on failure.

The broker is our own Valkey, so the header goes whatever the allowlist says;
nothing here touches the HTTP fetches a task makes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from celery import signals
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .log import configure, process_logger
from .tracing import context_from_traceparent, current_traceparent, elapsed_ms, start_tracing

SERVICE = "media-worker"
TRACER_NAME = "media-manager/celery"
_running: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def setup_logging(loglevel: int | str | None = None, **_: Any) -> None:
    level = logging.getLevelName(loglevel) if isinstance(loglevel, int) else loglevel
    configure(SERVICE, level=level)
    start_tracing(SERVICE)


def before_publish(headers: dict[str, Any] | None = None, **_: Any) -> None:
    traceparent = current_traceparent()
    if traceparent is not None and isinstance(headers, dict) and "traceparent" not in headers:
        headers["traceparent"] = traceparent


def _traceparent_of(request: Any) -> str | None:
    value = getattr(request, "traceparent", None)
    if value is None:
        value = (getattr(request, "headers", None) or {}).get("traceparent")
    return value if isinstance(value, str) else None


def task_prerun(task_id: str, task: Any, **_: Any) -> None:
    parent = context_from_traceparent(_traceparent_of(task.request))
    span = trace.get_tracer(TRACER_NAME).start_span(
        f"task.{task.name}",
        context=parent,
        kind=SpanKind.CONSUMER,
        attributes={"messaging.system": "celery", "celery.task_name": task.name},
    )
    token = otel_context.attach(trace.set_span_in_context(span, parent))
    with _lock:
        _running[task_id] = {
            "span": span,
            "token": token,
            "started": time.perf_counter(),
            "err": None,
        }


def task_failure(task_id: str, exception: BaseException | None = None, **_: Any) -> None:
    with _lock:
        if task_id in _running:
            _running[task_id]["err"] = exception


def task_postrun(task_id: str, task: Any, state: str | None = None, **_: Any) -> None:
    with _lock:
        run = _running.pop(task_id, None)
    if run is None:
        return
    failed = run["err"] is not None or state == "FAILURE"
    fields: dict[str, Any] = {
        "event": "item.process",
        "code": "error" if failed else "ok",
        "duration_ms": elapsed_ms(run["started"]),
        "task": task.name,
        "task_id": task_id,
        "state": state,
    }
    if run["err"] is not None:
        fields["err"] = run["err"]
    span = run["span"]
    span.set_status(Status(StatusCode.ERROR if failed else StatusCode.OK))
    try:
        process_logger().log("error" if failed else "info", "", fields)
    finally:
        span.end()
        otel_context.detach(run["token"])


def install() -> None:
    """Connect the handlers (idempotent: each has its own dispatch uid)."""
    signals.setup_logging.connect(setup_logging, weak=False, dispatch_uid="lg-u11.setup_logging")
    signals.before_task_publish.connect(
        before_publish, weak=False, dispatch_uid="lg-u11.before_task_publish"
    )
    signals.task_prerun.connect(task_prerun, weak=False, dispatch_uid="lg-u11.task_prerun")
    signals.task_postrun.connect(task_postrun, weak=False, dispatch_uid="lg-u11.task_postrun")
    signals.task_failure.connect(task_failure, weak=False, dispatch_uid="lg-u11.task_failure")
