"""MM-REPORT-1: the SpineMedia gRPC client, over a real in-process gRPC
server whose servicer answers what each test scripts.

The contract's error section decides the retry policy: UNAVAILABLE,
DEADLINE_EXCEEDED and INTERNAL are retried (the spine failed, nothing was
written); FAILED_PRECONDITION is retried for a ReportMask only once the
derivative's own report has landed; nothing else is retried, INVALID_ARGUMENT
never unchanged. UNAUTHENTICATED, PERMISSION_DENIED and UNIMPLEMENTED say the
mesh or the deployment is wrong, so the run stops on them."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any

import grpc
import pytest
from app.spine import media_client as mc
from app.spine.field_rules import FieldRefused
from app.spine.media_v1 import media_pb2, media_pb2_grpc

C = grpc.StatusCode


@dataclass
class Scripted(media_pb2_grpc.SpineMediaServicer):
    """Answers each RPC from its queue: a message, a status code to abort
    with, or ("sleep", seconds, message)."""

    answers: dict[str, deque[Any]] = field(default_factory=dict)
    seen: list[tuple[str, Any, dict[str, str], float | None]] = field(default_factory=list)

    def _answer(self, rpc: str, request: Any, context: grpc.ServicerContext) -> Any:
        self.seen.append(
            (rpc, request, dict(context.invocation_metadata()), context.time_remaining())
        )
        answer = self.answers[rpc].popleft()
        if isinstance(answer, tuple) and answer[0] == "sleep":
            time.sleep(answer[1])
            return answer[2]
        if isinstance(answer, grpc.StatusCode):
            context.abort(answer, f"scripted {answer.name}")
        return answer

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("LeasePendingImages", request, context)

    def ReportDerivative(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("ReportDerivative", request, context)

    def ReportMask(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("ReportMask", request, context)

    def WithdrawMask(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("WithdrawMask", request, context)

    def script(self, rpc: str, *answers: Any) -> None:
        self.answers.setdefault(rpc, deque()).extend(answers)

    def calls(self, rpc: str) -> list[Any]:
        return [s for s in self.seen if s[0] == rpc]


@pytest.fixture()
def spine() -> Iterator[tuple[Scripted, str]]:
    servicer = Scripted()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    media_pb2_grpc.add_SpineMediaServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield servicer, f"http://127.0.0.1:{port}"
    finally:
        server.stop(grace=None)


def _client(url: str, **kw: Any) -> tuple[mc.SpineMediaClient, list[float]]:
    slept: list[float] = []
    client = mc.SpineMediaClient(mc.open_channel(url), sleep=slept.append, **kw)
    return client, slept


SHA = bytes(range(32))
ACK = media_pb2.MaskAck(stored=1, matched=1)
DACK = media_pb2.DerivativeAck(stored=True, image_id="i", product_image_state="derived")


def _mask() -> media_pb2.MaskReport:
    return media_pb2.MaskReport(derivative_sha256=SHA, pipeline_version="mask-v1")


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rpc", mc.RPCS)
@pytest.mark.parametrize("code", [C.UNAVAILABLE, C.DEADLINE_EXCEEDED, C.INTERNAL])
def test_the_spines_own_failures_are_retried_on_every_rpc(rpc: str, code: grpc.StatusCode) -> None:
    assert mc.classify(code, rpc=rpc) is mc.Verdict.RETRY


@pytest.mark.parametrize("rpc", mc.RPCS)
@pytest.mark.parametrize("code", [C.UNAUTHENTICATED, C.PERMISSION_DENIED, C.UNIMPLEMENTED])
def test_a_mesh_or_deployment_fault_stops_the_run(rpc: str, code: grpc.StatusCode) -> None:
    assert mc.classify(code, rpc=rpc, derivative_landed=True) is mc.Verdict.FATAL


@pytest.mark.parametrize("rpc", mc.RPCS)
@pytest.mark.parametrize(
    "code",
    [C.INVALID_ARGUMENT, C.RESOURCE_EXHAUSTED, C.NOT_FOUND, C.ALREADY_EXISTS, C.UNKNOWN],
)
def test_everything_else_is_a_refusal_never_retried(rpc: str, code: grpc.StatusCode) -> None:
    assert mc.classify(code, rpc=rpc, derivative_landed=True) is mc.Verdict.REFUSED


def test_failed_precondition_is_retried_only_for_a_mask_whose_derivative_landed() -> None:
    assert mc.classify(C.FAILED_PRECONDITION, rpc="ReportMask", derivative_landed=True) is (
        mc.Verdict.RETRY
    )
    assert mc.classify(C.FAILED_PRECONDITION, rpc="ReportMask") is mc.Verdict.REFUSED
    for rpc in ("LeasePendingImages", "ReportDerivative", "WithdrawMask"):
        assert mc.classify(C.FAILED_PRECONDITION, rpc=rpc, derivative_landed=True) is (
            mc.Verdict.REFUSED
        )


def test_backoff_doubles_up_to_its_cap() -> None:
    policy = mc.RetryPolicy(max_attempts=6, initial_backoff=0.5, max_backoff=3.0, multiplier=2.0)
    assert policy.delays() == [0.5, 1.0, 2.0, 3.0, 3.0]
    assert mc.RetryPolicy(max_attempts=1).delays() == []
    with pytest.raises(ValueError, match="max_attempts"):
        mc.RetryPolicy(max_attempts=0)


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------


def test_a_transient_failure_is_retried_with_backoff_then_answered(
    spine: tuple[Scripted, str],
) -> None:
    servicer, url = spine
    servicer.script("ReportDerivative", C.UNAVAILABLE, C.INTERNAL, DACK)
    client, slept = _client(url, policy=mc.RetryPolicy(max_attempts=5, initial_backoff=0.5))
    ack = client.report_derivative(media_pb2.DerivativeReport(product_image_id="p"))
    assert ack.stored is True
    assert len(servicer.calls("ReportDerivative")) == 3
    assert slept == [0.5, 1.0]


def test_a_refusal_is_not_retried(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    servicer.script("ReportMask", C.INVALID_ARGUMENT)
    client, slept = _client(url)
    with pytest.raises(mc.SpineCallError) as caught:
        client.report_mask(_mask(), derivative_landed=True)
    err = caught.value
    assert (err.rpc, err.code, err.verdict, err.attempts) == (
        "ReportMask",
        C.INVALID_ARGUMENT,
        mc.Verdict.REFUSED,
        1,
    )
    assert "scripted INVALID_ARGUMENT" in str(err)
    assert len(servicer.calls("ReportMask")) == 1 and slept == []


def test_retries_stop_at_max_attempts(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    servicer.script("WithdrawMask", *[C.UNAVAILABLE] * 3)
    client, slept = _client(url, policy=mc.RetryPolicy(max_attempts=3, initial_backoff=0.25))
    with pytest.raises(mc.SpineCallError) as caught:
        client.withdraw_mask(media_pb2.MaskWithdrawal(mask_sha256=SHA, restore=True))
    assert caught.value.verdict is mc.Verdict.RETRY and caught.value.attempts == 3
    assert caught.value.exhausted is True
    assert len(servicer.calls("WithdrawMask")) == 3 and slept == [0.25, 0.5]


def test_a_fatal_status_is_raised_at_once(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    servicer.script("ReportMask", C.PERMISSION_DENIED)
    client, _ = _client(url)
    with pytest.raises(mc.SpineCallError) as caught:
        client.report_mask(_mask(), derivative_landed=False)
    assert caught.value.verdict is mc.Verdict.FATAL and caught.value.exhausted is False


def test_failed_precondition_on_a_mask_is_retried_once_its_derivative_landed(
    spine: tuple[Scripted, str],
) -> None:
    servicer, url = spine
    servicer.script("ReportMask", C.FAILED_PRECONDITION, ACK, C.FAILED_PRECONDITION)
    client, slept = _client(url)
    assert client.report_mask(_mask(), derivative_landed=True).stored == 1
    assert slept == [0.5]
    with pytest.raises(mc.SpineCallError) as caught:
        client.report_mask(_mask(), derivative_landed=False)
    assert caught.value.code is C.FAILED_PRECONDITION
    assert caught.value.verdict is mc.Verdict.REFUSED
    assert len(servicer.calls("ReportMask")) == 3


def test_every_call_carries_the_deadline_and_a_slow_answer_is_retried(
    spine: tuple[Scripted, str],
) -> None:
    servicer, url = spine
    servicer.script("ReportMask", ("sleep", 1.0, ACK), ACK)
    client, slept = _client(url, deadline_seconds=0.3)
    assert client.report_mask(_mask(), derivative_landed=False).stored == 1
    remaining = [s[3] for s in servicer.calls("ReportMask")]
    assert len(remaining) == 2
    # the deadline (0.3 s), plus the timers' own slack; the default would be 10 s
    assert all(r is not None and 0 < r <= 0.35 for r in remaining), remaining
    assert slept == [0.5]


def test_the_client_never_claims_a_mesh_identity(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    servicer.script("ReportMask", ACK)
    client, _ = _client(url)
    client.report_mask(_mask(), derivative_landed=False)
    metadata = servicer.calls("ReportMask")[0][2]
    assert "l5d-client-id" not in metadata


# ---------------------------------------------------------------------------
# the lease
# ---------------------------------------------------------------------------


def test_a_lease_names_its_originals_and_counts_failed_attempts(
    spine: tuple[Scripted, str],
) -> None:
    servicer, url = spine
    answer = media_pb2.LeasePendingImagesResponse(
        images=[media_pb2.PendingImage(product_image_id="p", origin_sha256=SHA)],
        lease_expires_at="2026-10-15T00:02:00.000000Z",
    )
    servicer.script("LeasePendingImages", C.UNAVAILABLE, answer)
    client, _ = _client(url)
    out = client.lease([SHA, SHA], max_items=200, lease_owner="run\x00 1", lease_seconds=120)
    assert out.failed_attempts == 1
    assert out.response.images[0].product_image_id == "p"
    assert out.warnings == ["lease_owner: NUL removed"]
    sent = servicer.calls("LeasePendingImages")[-1][1]
    assert list(sent.origin_sha256) == [SHA]  # a repeat counted once, here
    assert (sent.max_items, sent.lease_owner, sent.lease_seconds) == (200, "run 1", 120)


def test_a_lease_filter_the_spine_would_refuse_is_not_sent(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    client, _ = _client(url)
    with pytest.raises(FieldRefused, match="200"):
        client.lease([bytes([i % 256]) * 32 for i in range(201)], max_items=1, lease_owner="r")
    with pytest.raises(FieldRefused, match="32 bytes"):
        client.lease([b"short"], max_items=1, lease_owner="r")
    with pytest.raises(FieldRefused, match="empty"):
        client.lease([], max_items=1, lease_owner="r")
    assert servicer.calls("LeasePendingImages") == []


# ---------------------------------------------------------------------------
# the target
# ---------------------------------------------------------------------------


def test_the_target_is_plaintext_to_the_local_mesh_proxy() -> None:
    assert (
        mc.target_from_url("http://ingest-server.fc.svc.cluster.local:50063")
        == "ingest-server.fc.svc.cluster.local:50063"
    )
    assert mc.target_from_url("10.0.0.7:50063") == "10.0.0.7:50063"


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("https://ingest-server.fc.svc:50063", "mesh"),
        ("http://ingest-server.fc.svc", "port"),
        ("http://ingest-server.fc.svc:50063/x", "path"),
        ("", "SPINE_MEDIA_URL"),
        ("ftp://h:1", "scheme"),
    ],
)
def test_any_other_target_is_refused(url: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        mc.target_from_url(url)


def test_a_lease_filter_of_exactly_200_is_sent(spine: tuple[Scripted, str]) -> None:
    servicer, url = spine
    servicer.script("LeasePendingImages", media_pb2.LeasePendingImagesResponse())
    client, _ = _client(url)
    client.lease([i.to_bytes(32, "big") for i in range(200)], max_items=200, lease_owner="r")
    assert len(servicer.calls("LeasePendingImages")[0][1].origin_sha256) == 200


def test_the_channel_leaves_retrying_to_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def capture(target: str, options: Any = None) -> str:
        seen.update(target=target, options=options)
        return "channel"

    monkeypatch.setattr(mc.grpc, "insecure_channel", capture)
    assert mc.open_channel("http://ingest-server.fc.svc.cluster.local:50063") == "channel"
    assert seen == {
        "target": "ingest-server.fc.svc.cluster.local:50063",
        "options": [("grpc.enable_retries", 0)],
    }
