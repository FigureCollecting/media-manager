"""A CLIENT span and its W3C traceparent on every gRPC call media-manager
makes (fc-shared server/connect rpcClientInterceptor, in Python).

The traceparent names THIS call's span, so the spine's rpc.in line (fc-
aggregation withRequestLog, which extracts it) is a child of it, in the
trace the run started. It goes only to a target on the propagation allowlist
(cluster-internal names by default). Each call also writes an rpc.out line
at fc-shared's level (connect.ts levelFor): info when it succeeds, error when
the server failed (unknown, deadline_exceeded, unimplemented, internal,
unavailable, data_loss), warn otherwise.
"""

from __future__ import annotations

import collections
import time
from typing import Any

import grpc
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .log import process_logger
from .tracing import elapsed_ms, inject_context, propagation_allowed

TRACER_NAME = "media-manager/grpc"
#: Codes that mean the server (or its dependency) failed, not the caller.
_SERVER_FAULTS = frozenset(
    {"unknown", "deadline_exceeded", "unimplemented", "internal", "unavailable", "data_loss"}
)


def code_name(code: grpc.StatusCode | None) -> str:
    """A gRPC status as the shape's Connect code name ('ok', 'invalid_argument' ...)."""
    if code is None:
        return "unknown"
    return "canceled" if code is grpc.StatusCode.CANCELLED else code.name.lower()


class RpcFailed(Exception):
    """The err of a failed call's line: the status details the server sent."""


class _Details(
    collections.namedtuple(
        "_Details",
        ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
    ),
    grpc.ClientCallDetails,
):
    pass


class TraceClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, target: str) -> None:
        self.peer = target
        self.host, _, port = target.rpartition(":")
        self.port = int(port) if port.isdigit() else None
        self.propagates = propagation_allowed(self.host)

    def intercept_unary_unary(self, continuation: Any, details: Any, request: Any) -> Any:
        method = (
            details.method.decode() if isinstance(details.method, bytes) else str(details.method)
        )
        call = method.lstrip("/")
        service, _, rpc = call.rpartition("/")
        attributes: dict[str, Any] = {
            "rpc.system": "grpc",
            "rpc.service": service,
            "rpc.method": rpc,
            "server.address": self.host,
        }
        if self.port is not None:
            attributes["server.port"] = self.port
        span = trace.get_tracer(TRACER_NAME).start_span(
            call, kind=SpanKind.CLIENT, attributes=attributes
        )
        ctx = trace.set_span_in_context(span)
        metadata = list(details.metadata or ())
        if self.propagates:
            carrier: dict[str, str] = {}
            inject_context(carrier, ctx)
            metadata.extend(carrier.items())
        started = time.perf_counter()
        token = otel_context.attach(ctx)
        try:
            outcome = continuation(
                _Details(
                    details.method, details.timeout, metadata, details.credentials,
                    details.wait_for_ready, details.compression,
                ),
                request,
            )  # fmt: skip
            code = code_name(outcome.code())
            fields: dict[str, Any] = {
                "event": "rpc.out",
                "call": call,
                "code": code,
                "duration_ms": elapsed_ms(started),
                "peer": self.peer,
            }
            if code == "ok":
                span.set_status(Status(StatusCode.OK))
                level = "info"
            else:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("rpc.grpc.status_code", code)
                fields["err"] = RpcFailed(outcome.details() or code)
                level = "error" if code in _SERVER_FAULTS else "warn"
            process_logger().log(level, "", fields)
            return outcome
        finally:
            span.end()
            otel_context.detach(token)


def intercept(channel: grpc.Channel, target: str) -> grpc.Channel:
    """``channel`` with a span, a traceparent and an rpc.out line on every unary call."""
    return grpc.intercept_channel(channel, TraceClientInterceptor(target))
