"""LG-U11 / U12: the trace a media-manager run starts is the trace the spine sees.

Every SpineMedia call carries a W3C traceparent for a CLIENT span of the
active trace (fc-shared server/connect rpcClientInterceptor, in Python), to
cluster-internal targets only (OTEL_PROPAGATION_ALLOWLIST). The spine side,
fc-aggregation's withRequestLog (rpcServerInterceptor), reads exactly that
header to continue the trace; FakeSpine reads it the same way here.
tests/test_spine_media_live.py proves the same hop against the real
ingest server when one is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import grpc
import pytest
from app.cli import mask_report as cli
from app.models import Base
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from app.spine.media_v1 import media_pb2
from app.telemetry import log as tlog
from app.telemetry import tracing as tr
from app.telemetry.grpc_client import TraceClientInterceptor, code_name
from app.telemetry.log import Logger
from opentelemetry import trace
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from spine_fakes import FakeSpine, seed_copy, serve
from test_telemetry_log import VALIDATOR


def _valid(line: dict[str, Any]) -> None:
    assert [e.message for e in VALIDATOR.iter_errors(line)] == [], line


@pytest.fixture()
def captured() -> Any:
    lines: list[dict[str, Any]] = []
    previous = tlog.use_logger(
        Logger(
            service="mm-report",
            version="1",
            level="debug",
            env={},
            sink=lambda s: lines.append(json.loads(s)),
        )
    )
    yield lines
    tlog.use_logger(previous)


def _traceparents(fake: FakeSpine) -> list[str | None]:
    return [md.get("traceparent") for _rpc, md in fake.metadata]


def test_each_call_carries_a_client_span_of_the_active_trace(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("OTEL_PROPAGATION_ALLOWLIST", "127.0.0.1")
    tr.start_tracing("mm-report", env={})
    fake = FakeSpine()
    with serve(fake) as url, trace.get_tracer("t").start_as_current_span("run") as run:
        channel = open_channel(url)
        client = SpineMediaClient(channel, deadline_seconds=5, policy=RetryPolicy(max_attempts=1))
        client.lease([b"\x01" * 32], max_items=1, lease_owner="mm-report test")
        channel.close()
    ctx = run.get_span_context()
    (sent,) = _traceparents(fake)
    assert sent is not None
    version, trace_id, parent, flags = sent.split("-")
    assert (version, trace_id) == ("00", f"{ctx.trace_id:032x}")
    assert parent != f"{ctx.span_id:016x}"  # the CLIENT span of this call, as fc-shared injects it
    (out,) = (x for x in captured if x["event"] == "rpc.out")
    assert (out["level"], out["call"], out["code"], out["peer"]) == (
        "debug",
        "media.v1.SpineMedia/LeasePendingImages",
        "ok",
        url.removeprefix("http://"),
    )
    assert out["trace_id"] == trace_id and out["span_id"] == parent
    _valid(out)


def test_no_trace_header_leaves_for_a_host_off_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("OTEL_PROPAGATION_ALLOWLIST", raising=False)
    tr.start_tracing("mm-report", env={})
    fake = FakeSpine()
    with serve(fake) as url, trace.get_tracer("t").start_as_current_span("run"):
        channel = open_channel(url)  # 127.0.0.1: loopback IPs are not on the default list
        SpineMediaClient(channel, deadline_seconds=5).lease(
            [b"\x02" * 32], max_items=1, lease_owner="x"
        )
        channel.close()
    assert _traceparents(fake) == [None]
    assert [x["code"] for x in captured if x["event"] == "rpc.out"] == ["ok"]


def test_a_failed_call_is_an_rpc_out_line_at_its_level(
    monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("OTEL_PROPAGATION_ALLOWLIST", "127.0.0.1")
    tr.start_tracing("mm-report", env={})
    fake = FakeSpine()
    fake.fail("ReportDerivative", grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INVALID_ARGUMENT)
    report = media_pb2.DerivativeReport()
    with serve(fake) as url:
        channel = open_channel(url)
        client = SpineMediaClient(channel, deadline_seconds=5, policy=RetryPolicy(max_attempts=1))
        for _ in range(2):
            with pytest.raises(Exception):  # noqa: B017 - SpineCallError; the line is what is under test
                client.report_derivative(report)
        channel.close()
    unavailable, invalid = (x for x in captured if x["event"] == "rpc.out")
    assert (unavailable["level"], unavailable["code"]) == ("error", "unavailable")
    assert unavailable["err"] == {"type": "RpcFailed", "message": "fake UNAVAILABLE"}
    assert (invalid["level"], invalid["code"]) == ("warn", "invalid_argument")
    for line in (unavailable, invalid):
        _valid(line)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (grpc.StatusCode.OK, "ok"),
        (grpc.StatusCode.CANCELLED, "canceled"),
        (grpc.StatusCode.DEADLINE_EXCEEDED, "deadline_exceeded"),
        (grpc.StatusCode.UNAUTHENTICATED, "unauthenticated"),
        (None, "unknown"),
    ],
)
def test_grpc_codes_print_as_connect_code_names(code: grpc.StatusCode | None, name: str) -> None:
    assert code_name(code) == name


def test_the_interceptor_names_its_peer_without_a_scheme() -> None:
    interceptor = TraceClientInterceptor("ingest-server.fc.svc.cluster.local:50063")
    assert (interceptor.host, interceptor.peer, interceptor.propagates) == (
        "ingest-server.fc.svc.cluster.local",
        "ingest-server.fc.svc.cluster.local:50063",
        True,
    )


# --- U12: the run's trace id is the one the spine side sees ---------------------------------


def _sandbox(tmp_path: Path, *tags: str) -> tuple[str, dict[str, dict[str, object]]]:
    engine = create_engine(f"sqlite:///{tmp_path / 'masks.db'}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        seeded = {tag: seed_copy(db, tag) for tag in tags}
    engine.dispose()
    return str(tmp_path), seeded


def test_u12_the_mm_report_runs_trace_reaches_every_spine_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox, seeded = _sandbox(tmp_path, "a", "b")
    fake = FakeSpine()
    for tag in ("a", "b"):
        fake.depict(str(seeded[tag]["origin"]))
    monkeypatch.setenv("OTEL_PROPAGATION_ALLOWLIST", "127.0.0.1")
    monkeypatch.setenv("JOB_NAME", "mm-report-29312345")
    with serve(fake) as url:
        monkeypatch.setenv("SPINE_MEDIA_URL", url)
        assert cli.main(["--sandbox", sandbox, "--batch-size", "10"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    start, end = lines[0], lines[-1]
    assert (start["event"], start["service"], start["job"]) == (
        "job.start",
        "mm-report",
        "mm-report-29312345",
    )
    run_trace = start["trace_id"]
    sent = _traceparents(fake)
    assert len(sent) >= 3 and {"LeasePendingImages", "ReportDerivative"} <= {
        rpc for rpc, _ in fake.metadata
    }
    assert all(tp is not None and tp.split("-")[1] == run_trace for tp in sent), sent
    assert {line["trace_id"] for line in lines} == {run_trace}
    assert (end["event"], end["code"], end["exit"]) == ("job.end", "ok", 0)
    for line in lines:
        _valid(line)
