"""One http.in line per request (fc-shared server/express httpLogMiddleware, as ASGI).

    call         'METHOD /route/template' (FastAPI's matched route), never the
                 path a client sent or its query; '<unmatched>' when no route
                 matched (the origin's grammar refusal included).
    code         the status as a string; '499' when no answer was started.
    duration_ms  from the middleware to the handler's return.
    peer         l5d-client-id, 'unmeshed' or 'invalid'.

Trace: the request runs inside a SERVER span continuing the caller's
traceparent (a new root without one), so the handler's own lines carry its
trace_id. Bodies are never read: no image bytes, no form, no query.

Add it LAST (outermost of the app's middleware) so the line also covers the
rate limiter's and the origin grammar's answers. Starlette's own error
middleware stays outside it, so an unhandled exception is seen here as one:
the line says 500 and carries err.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .log import Logger
from .tracing import elapsed_ms, extract_context

TRACER_NAME = "media-manager/http"
_DNS_NAME = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*"
)


def peer_from_header(value: str | None) -> str:
    """The caller's Linkerd identity: 'unmeshed' when absent, 'invalid' when not a DNS name."""
    name = (value or "").strip().lower()
    if not name:
        return "unmeshed"
    return name if len(name) <= 253 and _DNS_NAME.fullmatch(name) else "invalid"


def _headers(scope: Scope) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers") or ():
        name = raw_name.decode("latin-1").lower()
        value = raw_value.decode("latin-1")
        out[name] = f"{out[name]},{value}" if name in out else value
    return out


def _level(status: int) -> str:
    if status >= 500:
        return "error"
    return "warn" if status >= 400 else "info"


class AccessLogMiddleware:
    def __init__(
        self, app: ASGIApp, *, logger: Logger, ignore_paths: Iterable[str] = ("/healthz",)
    ) -> None:
        self.app = app
        self.logger = logger
        self.ignore = frozenset(ignore_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.ignore:
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", "GET"))
        headers = _headers(scope)
        parent = extract_context(headers)
        span = trace.get_tracer(TRACER_NAME).start_span(
            method, context=parent, kind=SpanKind.SERVER, attributes={"http.request.method": method}
        )
        token = otel_context.attach(trace.set_span_in_context(span, parent))
        started = time.perf_counter()
        status: int | None = None
        error: Exception | None = None

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        except Exception as exc:
            error = exc
            raise
        finally:
            route = getattr(scope.get("route"), "path", None)
            call = f"{method} {route if isinstance(route, str) else '<unmatched>'}"
            code = status if status is not None else (500 if error is not None else 499)
            if isinstance(route, str):
                span.update_name(call)
                span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", code)
            if code >= 500:
                span.set_status(Status(StatusCode.ERROR))
            fields: dict[str, Any] = {
                "event": "http.in",
                "call": call,
                "code": str(code),
                "duration_ms": elapsed_ms(started),
                "peer": peer_from_header(headers.get("l5d-client-id")),
            }
            if error is not None:
                fields["err"] = error
            if status is None and error is None:
                fields["aborted"] = True
            self.logger.log(_level(code), "", fields)
            span.end()
            otel_context.detach(token)
