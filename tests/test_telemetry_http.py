"""LG-U11: one http.in line per request on both HTTP apps (fc-shared
server/express httpLogMiddleware, as ASGI middleware).

call is 'METHOD /route/template' (never the path a client sent, never its
query); code the status as a string; peer the caller's Linkerd identity; the
request runs inside a SERVER span continuing the caller's traceparent.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Generator
from typing import Any

import pytest
from app.telemetry import tracing as tr
from app.telemetry.http import AccessLogMiddleware, peer_from_header
from app.telemetry.log import Logger
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from starlette.types import Receive, Scope, Send

from test_telemetry_log import SCHEMA

VALIDATOR = Draft202012Validator(SCHEMA)
CALLER_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN = "00f067aa0ba902b7"
TRACEPARENT = f"00-{CALLER_TRACE}-{CALLER_SPAN}-01"
WORKER = "media-report.media.serviceaccount.identity.linkerd.app.mesh.estate"


def valid(line: dict[str, Any]) -> None:
    assert [e.message for e in VALIDATOR.iter_errors(line)] == [], line


@pytest.fixture()
def lines() -> list[dict[str, Any]]:
    return []


@pytest.fixture()
def demo(lines: list[dict[str, Any]]) -> Generator[TestClient, None, None]:
    tr.start_tracing("media-api", env={})
    log = Logger(
        service="media-api", version="1", env={}, sink=lambda s: lines.append(json.loads(s))
    )
    app = FastAPI()
    inner = logging.getLogger("lg-u11.http")

    @app.get("/items/{item_id}")
    def item(item_id: int) -> dict[str, Any]:
        log.info("inside", item_id=item_id)
        inner.debug("stdlib inside")
        return {"id": item_id}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("db at postgresql://app:SECRET@pg:5432/media?sslmode=require")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(AccessLogMiddleware, logger=log)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _http(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in lines if line["event"] == "http.in"]


def test_one_line_per_request_with_the_route_template_and_no_query(
    demo: TestClient, lines: list[dict[str, Any]]
) -> None:
    assert demo.get("/items/42?token=SECRET&sig=SECRET#frag").status_code == 200
    inside, access = lines
    assert (access["call"], access["code"], access["level"], access["peer"]) == (
        "GET /items/{item_id}", "200", "info", "unmeshed",
    )  # fmt: skip
    assert isinstance(access["duration_ms"], int | float) and access["duration_ms"] >= 0
    assert "SECRET" not in json.dumps(lines) and "42?" not in json.dumps(access)
    # The handler's own lines are in the request's trace.
    assert inside["trace_id"] == access["trace_id"] and inside["span_id"] == access["span_id"]
    valid(access)


def test_a_new_trace_without_a_traceparent_and_the_callers_with_one(
    demo: TestClient, lines: list[dict[str, Any]]
) -> None:
    demo.get("/items/1")
    demo.get("/items/2", headers={"traceparent": TRACEPARENT, "l5d-client-id": WORKER})
    demo.get("/items/3", headers={"traceparent": "00-" + "0" * 32 + "-" + CALLER_SPAN + "-01"})
    fresh, continued, rejected = _http(lines)
    assert fresh["trace_id"] != CALLER_TRACE and len(fresh["trace_id"]) == 32
    assert continued["trace_id"] == CALLER_TRACE and continued["span_id"] != CALLER_SPAN
    assert continued["peer"] == WORKER
    assert rejected["trace_id"] not in (CALLER_TRACE, "0" * 32)


def test_unmatched_errors_and_probes(demo: TestClient, lines: list[dict[str, Any]]) -> None:
    assert demo.get("/nothere?token=SECRET").status_code == 404
    assert demo.get("/boom").status_code == 500
    assert demo.get("/healthz").status_code == 200
    missing, failed = _http(lines)
    assert (missing["call"], missing["code"], missing["level"]) == (
        "GET <unmatched>",
        "404",
        "warn",
    )
    assert (failed["call"], failed["code"], failed["level"]) == ("GET /boom", "500", "error")
    assert failed["err"] == {"type": "RuntimeError", "message": "db at postgresql://pg:5432/media"}
    assert "SECRET" not in json.dumps(lines)
    for line in (missing, failed):
        valid(line)


@pytest.mark.asyncio
async def test_a_request_that_never_answers_is_499(lines: list[dict[str, Any]]) -> None:
    tr.start_tracing("media-api", env={})
    log = Logger(
        service="media-api", version="1", env={}, sink=lambda s: lines.append(json.loads(s))
    )

    async def silent(scope: Scope, receive: Receive, send: Send) -> None:
        return None

    async def noop_send(message: Any) -> None:
        return None

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    middleware = AccessLogMiddleware(silent, logger=log)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/x",
        "headers": [(b"l5d-client-id", b"bad name!")],
    }
    await middleware(scope, receive, noop_send)
    await middleware({"type": "lifespan"}, receive, noop_send)  # not a request: no line
    (line,) = lines
    assert (line["code"], line["aborted"], line["peer"], line["call"]) == (
        "499",
        True,
        "invalid",
        "POST <unmatched>",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "unmeshed"),
        ("", "unmeshed"),
        ("  ", "unmeshed"),
        (WORKER, WORKER),
        (
            "Media-Report.Media.ServiceAccount.identity.linkerd.app",
            "media-report.media.serviceaccount.identity.linkerd.app",
        ),
        ("bad name", "invalid"),
        ("a" * 254, "invalid"),
        ("-leading.dash", "invalid"),
    ],
)
def test_peer_is_the_mesh_identity(value: str | None, expected: str) -> None:
    assert peer_from_header(value) == expected


# --- the two apps ---------------------------------------------------------------------------


def _stdout_http(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    out = []
    for raw in capsys.readouterr().out.splitlines():
        if raw.startswith("{"):
            line = json.loads(raw)
            if line.get("event") == "http.in":
                out.append(line)
    return out


def test_the_api_writes_an_access_line_per_request(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert client.get("/images/7?access_token=SECRET").status_code == 401
    assert client.get("/healthz").status_code in (200, 503)
    (line,) = _stdout_http(capsys)
    assert (line["service"], line["call"], line["code"]) == (
        "media-api",
        "GET /images/{image_id}",
        "401",
    )
    assert "SECRET" not in json.dumps(line)
    valid(line)


def test_the_origin_writes_an_access_line_and_never_the_bytes(
    capsys: pytest.CaptureFixture[str], tmp_path: Any, db_session: Any
) -> None:
    from app.config import Settings, get_settings
    from app.db import get_db
    from app.derivative_main import app as origin_app
    from app.derivative_main import get_object_store
    from app.object_store import LocalObjectStore

    store = LocalObjectStore(tmp_path / "objects")
    origin_app.dependency_overrides[get_db] = lambda: db_session
    origin_app.dependency_overrides[get_object_store] = lambda: store
    origin_app.dependency_overrides[get_settings] = lambda: Settings(environment="test")
    try:
        with TestClient(origin_app, raise_server_exceptions=False) as client:
            capsys.readouterr()
            from app.workers.mask_policy import ImageOrigin
            from app.workers.mask_seed import seed_clean_image
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, "PNG")
            clean = seed_clean_image(
                db_session, store, buf.getvalue(),
                origins=[ImageOrigin(url="https://shop.example/i/1.jpg?sig=SECRET", source="mfc")],
                figure_id="mfc:1",
            )  # fmt: skip
            served = client.get(f"/d/{clean.sha256}")
            assert served.status_code == 200 and served.content
            sha = "ab" * 32
            assert client.get(f"/d/{sha}").status_code == 404
            assert client.head(f"/d/{sha}").status_code == 404
            assert client.get(f"/d/{sha}?sig=SECRET").status_code == 404
            assert client.get("/images/1").status_code == 404
            lines = _stdout_http(capsys)
    finally:
        origin_app.dependency_overrides.clear()
    assert [(x["service"], x["call"], x["code"]) for x in lines] == [
        ("media-origin", "GET /d/{sha256}", "200"),
        ("media-origin", "GET /d/{sha256}", "404"),
        ("media-origin", "HEAD /d/{sha256}", "404"),
        ("media-origin", "GET <unmatched>", "404"),
        ("media-origin", "GET <unmatched>", "404"),
    ]
    printed = json.dumps(lines)
    assert sha not in printed and clean.sha256 not in printed and "SECRET" not in printed
    assert all(len(json.dumps(x)) < 600 for x in lines)  # a line, never a body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "service"), [("app.main", "media-api"), ("app.derivative_main", "media-origin")]
)
async def test_each_app_starts_and_stops_in_the_shape(
    module: str, service: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    served = importlib.import_module(module)
    capsys.readouterr()
    async with served.lifespan(served.app):
        logging.getLogger("app.routes").warning("inside")
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [(x["event"], x["service"]) for x in lines] == [
        ("app.start", service),
        ("app.log", service),
        ("app.stop", service),
    ]
    for line in lines:
        valid(line)
