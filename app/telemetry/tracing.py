"""Trace ids for every line, with nothing exported (fc-shared server/tracing,
job and propagation, in Python).

start_tracing registers an SDK TracerProvider with NO span processor: spans
record with real, non-zero ids, so every log line under one carries
trace_id/span_id, and no span leaves the process. Export is a later unit's
to switch on (OTEL_EXPORTER_OTLP_*); this file never reads those variables,
and never OTEL_TRACES_EXPORTER=none, which gives all-zero ids.

Propagation: W3C traceparent (+ tracestate) only to cluster-internal hosts,
OTEL_PROPAGATION_ALLOWLIST (default ``*.svc.cluster.local,*.svc,localhost``;
loopback IPs deliberately absent). Nothing here instruments httpx, so the
worker's fetches of store and user-photo URLs never carry a trace header.

run_job (fc-shared server/job runJob): a CLI run is a ROOT span
``job.<name>``, bracketed by a job.start and a job.end line (code,
duration_ms, the run's counts, err on failure). One trace per ITEM (Ross's
choice): ``run.item(fn)`` runs fn as a NEW root trace (``item.enqueue`` by
default, kind PRODUCER) with a span link and ``fc.run_id`` pointing back at
the run, so requests made inside the item carry the item's traceparent;
``run.open_item()`` (fc-shared itemContext) is the same item for work that
does not fit one call (the caller ends it). Every line of the run and of its
items carries ``run_id`` (log.RUN_ID_KEY): ``run_id`` finds the whole run,
``trace_id`` one item. Spans are flushed before run_job returns (a CronJob
exits right after); a failed flush never masks the run's own result.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Link, Span, SpanContext, SpanKind, Status, StatusCode, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import Attributes

from .log import RUN_ID_KEY, Env, Logger, resolve_identity, round_ms

T = TypeVar("T")

TRACER_NAME = "media-manager/job"
ITEM_SPAN = "item.enqueue"
_DIAG = logging.getLogger("opentelemetry.media_manager")
DEFAULT_PROPAGATION_HOSTS = ("*.svc.cluster.local", "*.svc", "localhost")
PROPAGATION_ALLOWLIST_ENV = "OTEL_PROPAGATION_ALLOWLIST"
_PROPAGATOR = TraceContextTextMapPropagator()
_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})")


def start_tracing(
    service: str, *, version: str | None = None, env: Env | None = None
) -> TracerProvider:
    """Register the process's tracer provider, export off. Idempotent: the first registration wins."""
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    name, ver = resolve_identity(os.environ if env is None else env, service, version)
    provider = TracerProvider(resource=Resource({"service.name": name, "service.version": ver}))
    trace.set_tracer_provider(provider)
    return provider


def elapsed_ms(started: float) -> float | int:
    """Milliseconds since a time.perf_counter() reading, to 0.1 ms."""
    return round_ms(max(0.0, (time.perf_counter() - started) * 1000))


def current_traceparent() -> str | None:
    """W3C traceparent of the active span, for storing with a queued item."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{int(ctx.trace_flags):02x}"


def parse_traceparent(value: str | None) -> SpanContext | None:
    """A version-00 traceparent as a remote span context; None if malformed or zero."""
    match = _TRACEPARENT.fullmatch((value or "").strip())
    if match is None:
        return None
    ctx = SpanContext(
        trace_id=int(match.group(1), 16),
        span_id=int(match.group(2), 16),
        is_remote=True,
        trace_flags=TraceFlags(int(match.group(3), 16)),
    )
    return ctx if ctx.is_valid else None


def context_from_traceparent(value: str | None) -> Context:
    """A context whose parent is the stored traceparent; an empty one (a new root) when it is unusable."""
    parent = parse_traceparent(value)
    if parent is None:
        return Context()
    return trace.set_span_in_context(trace.NonRecordingSpan(parent), Context())


def extract_context(carrier: Mapping[str, str]) -> Context:
    """The caller's trace from W3C headers (lower-case keys); an empty context when there is none."""
    return _PROPAGATOR.extract(carrier, context=Context())


def inject_context(carrier: MutableMapping[str, str], ctx: Context | None = None) -> None:
    """traceparent (and tracestate) of ``ctx``'s span into ``carrier``. Callers gate on propagation_allowed."""
    _PROPAGATOR.inject(carrier, context=ctx)


def _entry_host(entry: str) -> str:
    host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", entry.strip(), flags=re.I)
    host = re.sub(r"[/?#].*$", "", host)
    return re.sub(r":\d+$", "", host).strip().lower().rstrip(".")


def propagation_hosts(env: Env | None = None) -> list[str]:
    raw = (os.environ if env is None else env).get(PROPAGATION_ALLOWLIST_ENV, "")
    listed = [entry.strip() for entry in raw.split(",") if entry.strip()]
    return listed or list(DEFAULT_PROPAGATION_HOSTS)


def propagation_allowed(host: str, env: Env | None = None) -> bool:
    """True when ``host`` is on the allowlist: an exact name, or a ``*.suffix`` it is strictly longer than."""
    name = host.strip().lower().rstrip(".")
    for entry in (_entry_host(e) for e in propagation_hosts(env)):
        if entry.startswith("*."):
            suffix = entry[1:]
            if len(entry) > 2 and len(name) > len(suffix) and name.endswith(suffix):
                return True
        elif entry and name == entry:
            return True
    return False


class Item(Protocol):
    """What a run's item offers its caller: ItemScope, or NULL_ITEM outside a run."""

    def run(self, fn: Callable[[], T]) -> T: ...

    def end(self, error: bool = False) -> None: ...


class ItemScope:
    """One item of a run: its context (the item's span, the run's id) and its
    span. ``run(fn)`` runs fn in it, as often as needed and on any thread;
    ``end()`` ends the span (once; ``error`` marks it failed)."""

    def __init__(self, context: Context, span: Span) -> None:
        self.context = context
        self.span = span
        self._ended = False
        self._lock = threading.Lock()

    def run(self, fn: Callable[[], T]) -> T:
        token = otel_context.attach(self.context)
        try:
            return fn()
        finally:
            otel_context.detach(token)

    def end(self, error: bool = False) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        self.span.set_status(Status(StatusCode.ERROR if error else StatusCode.OK))
        self.span.end()


class _NullItem:
    """No item: fn runs in the caller's own context, and there is nothing to end."""

    def run(self, fn: Callable[[], T]) -> T:
        return fn()

    def end(self, error: bool = False) -> None:
        return None


#: For a library caller with no run: items are not separate traces.
NULL_ITEM = _NullItem()


def _with_run_id(ctx: Context, run_id: str | None) -> Context:
    return ctx if run_id is None else otel_context.set_value(RUN_ID_KEY, run_id, ctx)


@dataclass(frozen=True)
class JobRun:
    name: str
    #: The run's trace id: what every item's fc.run_id points at.
    run_id: str
    span: Span
    tracer: trace.Tracer
    _valid: bool = field(default=True, repr=False)

    def open_item(self, name: str | None = None, attributes: Attributes = None) -> ItemScope:
        """A new root trace for one item, linked to this run (fc-shared itemContext). The caller ends it."""
        run_context = self.span.get_span_context()
        span = self.tracer.start_span(
            name or ITEM_SPAN,
            context=Context(),
            kind=SpanKind.PRODUCER,
            links=[Link(run_context)],
            attributes={**(attributes or {}), "fc.run_id": self.run_id},
        )
        ctx = trace.set_span_in_context(span, Context())
        return ItemScope(_with_run_id(ctx, self.run_id if self._valid else None), span)

    def item(
        self, fn: Callable[[], T], *, name: str | None = None, attributes: Attributes = None
    ) -> T:
        """Run fn as its own root trace, linked to this run. Rethrows fn's error (the item failed)."""
        scope = self.open_item(name, attributes)
        failed = True
        try:
            result = scope.run(fn)
            failed = False
            return result
        finally:
            scope.end(error=failed)


def _flush() -> None:
    """Flush whatever provider is registered; one with nothing to flush is fine."""
    try:
        flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if flush is not None:
            flush()
    except Exception:
        _DIAG.warning("run_job: span flush failed", exc_info=True)


def run_job(
    name: str,
    fn: Callable[[JobRun], T],
    *,
    logger: Logger,
    counts: Callable[[T], Mapping[str, Any]] | None = None,
    ok: Callable[[T], bool] | None = None,
    env: Env | None = None,
    tracer: trace.Tracer | None = None,
) -> T:
    """Run ``fn(run)`` as the root span ``job.<name>``: a job.start line, then
    a job.end line with ``counts(result)`` (code ok, or error at warn when
    ``ok(result)`` is false). A failure ends with code error and err, and is
    re-raised unchanged. Spans are flushed before it returns."""
    environ = os.environ if env is None else env
    k8s_job = (environ.get("JOB_NAME") or "").strip() or None
    job = k8s_job or name
    attributes = {"fc.job.name": name, **({"k8s.job.name": k8s_job} if k8s_job else {})}
    tracer = tracer or trace.get_tracer(TRACER_NAME)
    span = tracer.start_span(f"job.{name}", context=Context(), attributes=attributes)
    span_context = span.get_span_context()
    run = JobRun(name, format(span_context.trace_id, "032x"), span, tracer, span_context.is_valid)
    run_context = _with_run_id(
        trace.set_span_in_context(span, Context()), run.run_id if span_context.is_valid else None
    )
    token = otel_context.attach(run_context)
    started = time.perf_counter()
    try:
        logger.info(f"{name} started", event="job.start", job=job)
        try:
            result = fn(run)
        except BaseException as exc:
            span.set_status(Status(StatusCode.ERROR))
            logger.error(
                f"{name} failed", event="job.end", job=job, code="error",
                duration_ms=elapsed_ms(started), err=exc,
            )  # fmt: skip
            raise
        fields = dict(counts(result)) if counts is not None else {}
        good = ok(result) if ok is not None else True
        span.set_status(Status(StatusCode.OK if good else StatusCode.ERROR))
        logger.log(
            "info" if good else "warn",
            f"{name} finished",
            {**fields, "event": "job.end", "job": job, "code": "ok" if good else "error",
             "duration_ms": elapsed_ms(started)},
        )  # fmt: skip
        return result
    finally:
        span.end()
        otel_context.detach(token)
        _flush()
