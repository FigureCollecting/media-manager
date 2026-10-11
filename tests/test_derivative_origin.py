"""The public derivative origin: GET/HEAD /d/<64 lowercase hex> and nothing else.

fc-infra nodes/fc-app-01/edge/README.md "The public image address (ADDR-1)"
is the contract: MEDIA_PUBLIC_BASE_URL = https://images.figurecollecting.com/d;
the body is exactly the bytes whose SHA-256 is the path, with the stored
Content-Type; Cache-Control public, max-age=31536000, immutable; the request's
Cache-Control/Pragma are ignored; CORS allow-origin for the app origins with
Vary: Origin; nosniff, CORP cross-origin, CSP default-src 'none'; sandbox.
It is its own ASGI app (image track I11): app.main's routes are never mounted.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from app.config import Settings, get_settings
from app.db import get_db
from app.derivative_main import app as origin_app
from app.derivative_main import get_object_store
from app.models import Image as ImageModel
from app.models import ImageMask, PublicObject
from app.object_store import LocalObjectStore
from app.workers.mask_job import run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image
from app.workers.matting import MattingBackend, StubMattingBackend
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

APP = "https://figurecollecting.com"
CANARY = "https://fc-api-canary.mindsignals1.com"
ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"


def _photo(n: int) -> bytes:
    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (31 + 10 * n, 30, 90)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask model (the real stub never publishes)."""

    name = "stub-model"
    publishes_masks = True


class _Full(MattingBackend):
    name = "full"
    publishes_masks = True

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        return np.full((image.height, image.width), 255, dtype=np.uint8)


@pytest.fixture()
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


@pytest.fixture()
def seeded(db_session: Session, store: LocalObjectStore) -> dict[str, str]:
    clean = seed_clean_image(
        db_session,
        store,
        _photo(1),
        origins=[ImageOrigin(url=ITEM.format(n=1), source="mfc")],
        figure_id="mfc:1",
    )
    run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
    mask = db_session.execute(select(ImageMask)).scalar_one()
    held_src = seed_clean_image(
        db_session,
        store,
        _photo(2),
        origins=[ImageOrigin(url=ITEM.format(n=2), source="mfc")],
        figure_id="mfc:2",
    )
    run_mask_batch(db_session, store, backend=_Full(), dry_run=False)
    held = db_session.execute(
        select(ImageMask).where(ImageMask.source_sha256 == held_src.sha256)
    ).scalar_one()
    original = db_session.execute(select(ImageModel).order_by(ImageModel.id)).scalars().first()
    assert mask.mask_sha256 and held.mask_sha256 and original
    return {
        "clean": clean.sha256,
        "mask": mask.mask_sha256,
        "held_mask": held.mask_sha256,
        "original": original.sha256,
    }


@pytest.fixture()
def client(db_session: Session, store: LocalObjectStore) -> Generator[TestClient, None, None]:
    def _db() -> Generator[Session, None, None]:
        yield db_session

    origin_app.dependency_overrides[get_db] = _db
    origin_app.dependency_overrides[get_object_store] = lambda: store
    origin_app.dependency_overrides[get_settings] = lambda: Settings(environment="test")
    yield TestClient(origin_app, raise_server_exceptions=False)
    origin_app.dependency_overrides.clear()


def _hardening(resp: object) -> None:
    headers = resp.headers  # type: ignore[attr-defined]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["cross-origin-resource-policy"] == "cross-origin"
    assert headers["content-security-policy"] == "default-src 'none'; sandbox"


class TestRoutes:
    def test_the_app_mounts_only_healthz_and_the_derivative_path(self) -> None:
        paths = {getattr(r, "path", None) for r in origin_app.routes}
        assert paths == {"/healthz", "/d/{sha256}"}

    def test_healthz(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_healthz_degraded_when_the_database_is_down(self, client: TestClient) -> None:
        class _DownSession:
            def execute(self, *_a: object) -> None:
                raise ConnectionError("db down")

        def _db() -> Generator[object, None, None]:
            yield _DownSession()

        origin_app.dependency_overrides[get_db] = _db
        resp = client.get("/healthz")
        assert resp.status_code == 503
        assert resp.json() == {"status": "degraded"}


class TestServe:
    @pytest.mark.parametrize(("which", "mime"), [("clean", "image/jpeg"), ("mask", "image/png")])
    def test_serves_exactly_the_addressed_bytes(
        self, client: TestClient, seeded: dict[str, str], which: str, mime: str
    ) -> None:
        sha = seeded[which]
        resp = client.get(f"/d/{sha}")
        assert resp.status_code == 200
        assert hashlib.sha256(resp.content).hexdigest() == sha
        assert resp.headers["content-type"] == mime
        assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert resp.headers["etag"] == f'"{sha}"'
        assert resp.headers["vary"] == "Origin"
        _hardening(resp)

    def test_the_clean_copy_of_a_transparent_png_is_served_as_png(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        buf = io.BytesIO()
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[4:12, 4:12] = (9, 99, 199, 255)
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", compress_level=1)
        obj = seed_clean_image(db_session, store, buf.getvalue(), origins=[], figure_id="x")
        resp = client.get(f"/d/{obj.sha256}")
        assert resp.headers["content-type"] == "image/png"

    def test_head_sends_the_headers_and_no_body(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        sha = seeded["mask"]
        obj = db_session.get(PublicObject, sha)
        assert obj is not None
        resp = client.head(f"/d/{sha}", headers={"Origin": APP})
        assert resp.status_code == 200
        assert resp.content == b""
        assert resp.headers["content-length"] == str(obj.bytes)
        assert resp.headers["access-control-allow-origin"] == APP
        assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"

    def test_the_requests_cache_control_and_pragma_change_nothing(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        sha = seeded["mask"]
        plain = client.get(f"/d/{sha}", headers={"Origin": APP})
        busted = client.get(
            f"/d/{sha}",
            headers={"Origin": APP, "Cache-Control": "no-cache", "Pragma": "no-cache"},
        )
        conditional = client.get(f"/d/{sha}", headers={"Origin": APP, "If-None-Match": f'"{sha}"'})
        for resp in (busted, conditional):
            assert resp.status_code == 200
            assert resp.content == plain.content
            assert resp.headers["cache-control"] == plain.headers["cache-control"]


class TestCors:
    @pytest.mark.parametrize("origin", [APP, CANARY])
    def test_an_app_origin_is_allowed(
        self, client: TestClient, seeded: dict[str, str], origin: str
    ) -> None:
        resp = client.get(f"/d/{seeded['mask']}", headers={"Origin": origin})
        assert resp.headers["access-control-allow-origin"] == origin
        assert resp.headers["vary"] == "Origin"
        assert "access-control-allow-credentials" not in resp.headers

    @pytest.mark.parametrize(
        "origin", ["https://evil.example", "https://figurecollecting.com.evil.example", "null"]
    )
    def test_any_other_origin_gets_the_bytes_but_no_allow_origin(
        self, client: TestClient, seeded: dict[str, str], origin: str
    ) -> None:
        resp = client.get(f"/d/{seeded['mask']}", headers={"Origin": origin})
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers
        assert resp.headers["vary"] == "Origin"

    def test_no_origin_header_no_allow_origin(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        resp = client.get(f"/d/{seeded['mask']}")
        assert "access-control-allow-origin" not in resp.headers

    def test_the_allowed_origins_come_from_settings(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        origin_app.dependency_overrides[get_settings] = lambda: Settings(
            environment="test", media_cors_origins=["https://other.example"]
        )
        resp = client.get(f"/d/{seeded['mask']}", headers={"Origin": APP})
        assert "access-control-allow-origin" not in resp.headers
        resp = client.get(f"/d/{seeded['mask']}", headers={"Origin": "https://other.example"})
        assert resp.headers["access-control-allow-origin"] == "https://other.example"


class TestNotFound:
    def _assert_404(self, resp: object) -> None:
        assert resp.status_code == 404  # type: ignore[attr-defined]
        assert resp.headers["cache-control"] == "no-store"  # type: ignore[attr-defined]
        assert resp.content == b""  # type: ignore[attr-defined]
        _hardening(resp)

    def test_an_originals_address_is_not_served(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        self._assert_404(client.get(f"/d/{seeded['original']}"))

    def test_a_held_mask_is_not_served(self, client: TestClient, seeded: dict[str, str]) -> None:
        self._assert_404(client.get(f"/d/{seeded['held_mask']}"))

    def test_an_unknown_address(self, client: TestClient, seeded: dict[str, str]) -> None:
        self._assert_404(client.get("/d/" + "0" * 64))

    @pytest.mark.parametrize(
        "path",
        [
            "/d/{sha_upper}",
            "/d/{sha63}",
            "/d/{sha}0",
            "/d/{sha}/",
            "/d/{sha}/x",
            "/d/{sha}?raw=1",
            "/e/{sha}",
            "/d//{sha}",
            "/d/" + "g" * 64,
            "/d/",
            "/d",
            "/",
            "/serve/1@1",
            "/public/1@1",
            "/images/1",
            "/albums/1",
            "/external/x",
            "/docs",
            "/openapi.json",
        ],
    )
    def test_anything_but_the_exact_grammar(
        self, client: TestClient, seeded: dict[str, str], path: str
    ) -> None:
        sha = seeded["mask"]
        url = path.format(sha=sha, sha_upper=sha.upper(), sha63=sha[:63])
        self._assert_404(client.get(url))

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    def test_any_other_method(
        self, client: TestClient, seeded: dict[str, str], method: str
    ) -> None:
        resp = client.request(method, f"/d/{seeded['mask']}", headers={"Origin": APP})
        self._assert_404(resp)
        assert "access-control-allow-origin" not in resp.headers

    @pytest.mark.parametrize("method", ["POST", "HEAD", "OPTIONS"])
    def test_healthz_is_get_only(self, client: TestClient, method: str) -> None:
        self._assert_404(client.request(method, "/healthz"))

    def test_a_malformed_address_never_reaches_the_database_or_the_store(
        self, client: TestClient
    ) -> None:
        def _no_db() -> Generator[Session, None, None]:
            raise AssertionError("database touched")
            yield  # pragma: no cover

        class _NoStore:
            def get(self, key: str) -> bytes:
                raise AssertionError("store touched")

        origin_app.dependency_overrides[get_db] = _no_db
        origin_app.dependency_overrides[get_object_store] = lambda: _NoStore()
        self._assert_404(client.get("/d/" + "A" * 64))


class TestIntegrity:
    def test_bytes_that_no_longer_match_their_address_are_never_sent(
        self,
        client: TestClient,
        seeded: dict[str, str],
        db_session: Session,
        store: LocalObjectStore,
    ) -> None:
        obj = db_session.get(PublicObject, seeded["mask"])
        assert obj is not None
        store.put(obj.storage_key, b"corrupted", obj.mime)
        resp = client.get(f"/d/{obj.sha256}", headers={"Origin": APP})
        assert resp.status_code == 500
        assert resp.headers["cache-control"] == "no-store"
        assert b"corrupted" not in resp.content

    def test_a_store_failure_is_a_500_that_is_never_cached(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        class _Down:
            def get(self, key: str) -> bytes:
                raise ConnectionError("s3://bucket/derived/key unreachable")

        origin_app.dependency_overrides[get_object_store] = lambda: _Down()
        resp = client.get(f"/d/{seeded['mask']}")
        assert resp.status_code == 500
        assert resp.headers["cache-control"] == "no-store"
        assert b"s3://" not in resp.content


def test_the_deployed_store_is_the_configured_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.s3
    from app.object_store import S3ObjectStore

    monkeypatch.setattr(app.s3, "get_s3", lambda: object())
    assert isinstance(get_object_store(), S3ObjectStore)
