"""The media.v1 SpineMedia client (fc-ingest-contract 0.10.0).

PLAINTEXT TO THE LOCAL MESH. The client speaks gRPC over h2c to
``SPINE_MEDIA_URL`` (``http://ingest-server.fc.svc.cluster.local:50063``);
the pod's Linkerd proxy carries it over mTLS and the spine's inbound proxy
names the caller (``l5d-client-id``). The client never sets that header and
never speaks TLS itself: an ``https://`` target is refused, because TLS on
top of the mesh's mTLS would hide the call from the proxy that authorizes it.

DEADLINES AND RETRIES. Every call carries a deadline. gRPC's own retry is
switched off on the channel; this module retries, so that the policy is the
contract's and nothing else:

- UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL: the spine failed and wrote
  nothing; every RPC is retried, with exponential backoff.
- FAILED_PRECONDITION: retried for a ReportMask only once the derivative's
  own report has landed (the caller says so); otherwise a refusal.
- UNAUTHENTICATED, PERMISSION_DENIED, UNIMPLEMENTED: the mesh or the
  deployment is wrong (no identity, an identity not on the allow list, a
  port without SpineMedia). Fatal: retrying cannot help, the run stops.
- anything else (INVALID_ARGUMENT, RESOURCE_EXHAUSTED, ...): a refusal of
  this request, never retried unchanged.

A RETRIED LEASE IS NOT IDEMPOTENT: an attempt that failed may still have
held its rows, so the next attempt returns a different batch. ``lease``
therefore reports how many attempts failed, and the caller treats a batch
whose lease had a failed attempt as not finished (app.spine.replay).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import grpc

from ..telemetry.grpc_client import intercept
from . import field_rules as fr
from .media_v1 import media_pb2, media_pb2_grpc

RPCS = ("LeasePendingImages", "ReportDerivative", "ReportMask", "WithdrawMask")

RETRYABLE = frozenset(
    {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL}
)
FATAL = frozenset(
    {
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNIMPLEMENTED,
    }
)

DEFAULT_DEADLINE_SECONDS = 10.0
DEFAULT_LEASE_SECONDS = 120


class Verdict(enum.Enum):
    RETRY = "retry"
    REFUSED = "refused"
    FATAL = "fatal"


def classify(code: grpc.StatusCode, *, rpc: str, derivative_landed: bool = False) -> Verdict:
    """What the contract says to do with ``code`` from ``rpc``."""
    if code in FATAL:
        return Verdict.FATAL
    if code in RETRYABLE:
        return Verdict.RETRY
    if code is grpc.StatusCode.FAILED_PRECONDITION and rpc == "ReportMask" and derivative_landed:
        return Verdict.RETRY
    return Verdict.REFUSED


@dataclass(frozen=True)
class RetryPolicy:
    """``max_attempts`` calls in all; between them ``initial_backoff``
    seconds, multiplied each time, never more than ``max_backoff``."""

    max_attempts: int = 5
    initial_backoff: float = 0.5
    max_backoff: float = 8.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def delays(self) -> list[float]:
        out: list[float] = []
        delay = self.initial_backoff
        for _ in range(self.max_attempts - 1):
            out.append(min(delay, self.max_backoff))
            delay *= self.multiplier
        return out


class SpineCallError(Exception):
    """A call that did not succeed: its last status, the verdict on it, and
    how many attempts were made. ``exhausted``: retryable, but out of attempts."""

    def __init__(
        self, rpc: str, code: grpc.StatusCode, details: str, verdict: Verdict, attempts: int
    ) -> None:
        super().__init__(f"{rpc} {code.name} after {attempts} attempt(s): {details}")
        self.rpc = rpc
        self.code = code
        self.details = details
        self.verdict = verdict
        self.attempts = attempts

    @property
    def exhausted(self) -> bool:
        return self.verdict is Verdict.RETRY


@dataclass
class LeaseOutcome:
    response: media_pb2.LeasePendingImagesResponse
    #: Attempts that failed before this answer: each may have held rows.
    failed_attempts: int
    #: What the client cleaned in the request (the lease owner), locally.
    warnings: list[str] = field(default_factory=list)


def target_from_url(url: str) -> str:
    """``host:port`` from ``http://host:port`` (or a bare ``host:port``)."""
    if not url:
        raise ValueError("SPINE_MEDIA_URL is empty")
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme == "https":
        raise ValueError(
            "https is refused: the client sends plaintext to the local Linkerd proxy, "
            "and the mesh carries it over mTLS"
        )
    if parts.scheme != "http":
        raise ValueError(f"unsupported scheme {parts.scheme!r}: use http://host:port")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("SPINE_MEDIA_URL takes no path, query or fragment")
    if not parts.hostname or parts.port is None:
        raise ValueError("SPINE_MEDIA_URL needs a host and a port")
    return f"{parts.hostname}:{parts.port}"


def open_channel(url: str) -> grpc.Channel:
    """A plaintext channel to the mesh, gRPC's built-in retries off. Every call
    carries the active trace's traceparent (app.telemetry.grpc_client), so the
    spine's rpc.in line is in the run's trace."""
    target = target_from_url(url)
    channel = grpc.insecure_channel(target, options=[("grpc.enable_retries", 0)])
    return intercept(channel, target)


class SpineMediaClient:
    def __init__(
        self,
        channel: grpc.Channel,
        *,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._stub = media_pb2_grpc.SpineMediaStub(channel)
        self._deadline = deadline_seconds
        self._policy = policy or RetryPolicy()
        self._sleep = sleep

    @property
    def worst_case_seconds(self) -> float:
        """The longest one call can take: every attempt at its deadline, and
        every backoff between them."""
        return self._policy.max_attempts * self._deadline + sum(self._policy.delays())

    def _call(self, rpc: str, request: Any, *, derivative_landed: bool = False) -> tuple[Any, int]:
        method = getattr(self._stub, rpc)
        delays = self._policy.delays()
        attempt = 0
        while True:
            attempt += 1
            try:
                return method(request, timeout=self._deadline), attempt - 1
            except grpc.RpcError as exc:
                code = exc.code()
                details = exc.details() or ""
                verdict = classify(code, rpc=rpc, derivative_landed=derivative_landed)
                if verdict is not Verdict.RETRY or attempt > len(delays):
                    raise SpineCallError(rpc, code, details, verdict, attempt) from exc
                self._sleep(delays[attempt - 1])

    def lease(
        self,
        origins: Sequence[bytes],
        *,
        max_items: int,
        lease_owner: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> LeaseOutcome:
        """Lease the pending depictions of exactly ``origins`` (raw sha-256s).
        The filter is checked as the spine checks it (1-200 entries of 32
        bytes; an empty filter would lease the whole queue, so it is refused
        here); a repeated entry is sent once."""
        if not origins:
            raise fr.FieldRefused("origin_sha256", "empty: an unfiltered lease is never sent")
        unique = list(dict.fromkeys(origins))
        if len(unique) > fr.MAX_ORIGIN_FILTER:
            raise fr.FieldRefused("origin_sha256", f"over {fr.MAX_ORIGIN_FILTER} entries")
        if any(len(sha) != fr.SHA256_BYTES for sha in unique):
            raise fr.FieldRefused("origin_sha256", "an entry is not exactly 32 bytes")
        owner, cleaned = fr.lease_owner(lease_owner)
        request = media_pb2.LeasePendingImagesRequest(
            max_items=max_items,
            lease_owner=owner,
            lease_seconds=lease_seconds,
            origin_sha256=unique,
        )
        response, failed = self._call("LeasePendingImages", request)
        return LeaseOutcome(response, failed, [f"lease_owner: {w}" for w in cleaned])

    def report_derivative(self, report: media_pb2.DerivativeReport) -> media_pb2.DerivativeAck:
        response, _ = self._call("ReportDerivative", report)
        return response  # type: ignore[no-any-return]

    def report_mask(
        self, report: media_pb2.MaskReport, *, derivative_landed: bool
    ) -> media_pb2.MaskAck:
        response, _ = self._call("ReportMask", report, derivative_landed=derivative_landed)
        return response  # type: ignore[no-any-return]

    def withdraw_mask(self, withdrawal: media_pb2.MaskWithdrawal) -> media_pb2.MaskAck:
        response, _ = self._call("WithdrawMask", withdrawal)
        return response  # type: ignore[no-any-return]
