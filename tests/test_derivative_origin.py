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

import datetime as dt
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
from app.models import FigureGallery, ImageMask, ImageVersion, PublicObject
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore, derived_key
from app.workers.mask_job import run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import register_public_object, seed_clean_image
from app.workers.matting import MattingBackend, StubMattingBackend
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

APP = "https://figurecollecting.com"


def test_only_the_app_origin_is_allowed_by_default() -> None:
    """The canary origin is a Phase-2 deployment's to add (MEDIA_CORS_ORIGINS),
    never a default."""
    assert Settings(environment="test").media_cors_origins == [APP]


ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/9.jpeg"


def _source_image(db: Session, clean_sha: str) -> ImageModel:
    obj = db.get(PublicObject, clean_sha)
    assert obj is not None
    version = db.get(ImageVersion, obj.image_version_id)
    assert version is not None
    image = db.get(ImageModel, version.image_id)
    assert image is not None
    return image


def _photo(n: int) -> bytes:
    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (31 + 10 * n, 30, 90)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _capture(text: str) -> bytes:
    """The same pixels every time; different bytes per ``text`` (a PNG text chunk)."""
    from PIL import PngImagePlugin

    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (20, 20, 30)
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", text)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _rgba_capture(text: str) -> bytes:
    """An RGBA picture (one transparent pixel: a lossless PNG clean copy) with a text chunk."""
    from PIL import PngImagePlugin

    arr = np.full((60, 40, 4), (245, 245, 245, 255), dtype=np.uint8)
    arr[10:50, 12:28] = (20, 20, 30, 255)
    arr[0, 0, 3] = 0
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", text)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", pnginfo=info)
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
def shop_registered(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """One registered store source, ``shop`` (granted: MASK-G), for the batch."""
    import json

    monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"shop": ["shop.example"]}))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


def _asgi_get(path: str, *, raw_path: bytes | None = None, query: bytes = b"") -> int:
    """The origin's status for a GET whose ASGI scope is built by hand -- the
    shape a real server sends, which the TestClient does not reproduce."""
    import anyio

    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "query_string": query,
        "headers": [],
        "client": ("testclient", 1),
        "server": ("testserver", 80),
        "root_path": "",
    }
    if raw_path is not None:
        scope["raw_path"] = raw_path
    anyio.run(origin_app, scope, receive, send)
    status = sent[0]["status"]
    assert isinstance(status, int)
    return status


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
    @pytest.mark.parametrize("origin", [APP])
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

    def test_a_soft_deleted_image_takes_its_derivative_and_mask_offline(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        _source_image(db_session, seeded["clean"]).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['clean']}"))
        self._assert_404(client.get(f"/d/{seeded['mask']}"))

    def test_a_soft_deleted_clean_version_takes_it_and_its_mask_offline(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        obj = db_session.get(PublicObject, seeded["clean"])
        assert obj is not None
        version = db_session.get(ImageVersion, obj.image_version_id)
        assert version is not None
        version.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['clean']}"))
        self._assert_404(client.get(f"/d/{seeded['mask']}"))

    def test_a_mask_whose_image_gains_a_collector_photo_origin_goes_offline(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        image = _source_image(db_session, seeded["clean"])
        db_session.add(
            FigureGallery(figure_id="mfc:9", source_url=PICTURE, image_id=image.id, source="mfc")
        )
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['mask']}"))
        assert client.get(f"/d/{seeded['clean']}").status_code == 200

    def test_a_mask_whose_origins_are_all_gone_goes_offline(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        image = _source_image(db_session, seeded["clean"])
        for link in db_session.execute(
            select(FigureGallery).where(FigureGallery.image_id == image.id)
        ).scalars():
            link.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['mask']}"))

    def test_a_withdrawn_mask_row_is_not_served(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        row = db_session.execute(
            select(ImageMask).where(ImageMask.mask_sha256 == seeded["mask"])
        ).scalar_one()
        row.status = "withdrawn"
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['mask']}"))

    def test_a_derivative_with_no_version_is_not_served(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        obj = db_session.get(PublicObject, seeded["clean"])
        assert obj is not None
        obj.image_version_id = None
        db_session.commit()
        self._assert_404(client.get(f"/d/{seeded['clean']}"))

    def test_a_mask_shared_by_two_images_stays_while_one_is_live(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        for n in (1, 2):
            seed_clean_image(
                db_session,
                store,
                _photo(n),  # distinct clean copies, the same mask
                origins=[ImageOrigin(url=ITEM.format(n=n), source="mfc")],
                figure_id=f"mfc:{n}",
            )
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        rows = list(db_session.execute(select(ImageMask)).scalars())
        assert len(rows) == 2 and rows[0].mask_sha256 == rows[1].mask_sha256
        mask = rows[0].mask_sha256
        _source_image(db_session, rows[0].source_sha256).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        assert client.get(f"/d/{mask}").status_code == 200
        _source_image(db_session, rows[1].source_sha256).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{mask}"))

    def _seed_capture(self, db: Session, store: LocalObjectStore, i: int, url: str) -> None:
        seed_clean_image(
            db,
            store,
            _capture(f"capture {i}"),
            origins=[ImageOrigin(url=url, source="mfc")],
            figure_id=f"mfc:capture-{i}",
        )

    @pytest.mark.parametrize("deleted", [0, 1], ids=["first-deleted", "second-deleted"])
    def test_a_clean_copy_shared_by_pixel_identical_captures_needs_every_one_live(
        self, client: TestClient, db_session: Session, store: LocalObjectStore, deleted: int
    ) -> None:
        """Two originals, one picture: a soft delete under EITHER capture takes
        the shared clean copy and its mask offline, whichever was seeded first."""
        for i in (0, 1):
            self._seed_capture(db_session, store, i, ITEM.format(n=i))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        row = db_session.execute(select(ImageMask)).scalar_one()
        assert row.status == "ok"
        assert client.get(f"/d/{row.source_sha256}").status_code == 200
        assert client.get(f"/d/{row.mask_sha256}").status_code == 200
        images = list(db_session.execute(select(ImageModel).order_by(ImageModel.id)).scalars())
        assert len(images) == 2
        images[deleted].deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{row.source_sha256}"))
        self._assert_404(client.get(f"/d/{row.mask_sha256}"))

    def test_a_collector_photo_with_the_same_pixels_takes_the_shared_mask_offline(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        self._seed_capture(db_session, store, 0, ITEM.format(n=0))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        row = db_session.execute(select(ImageMask)).scalar_one()
        assert client.get(f"/d/{row.mask_sha256}").status_code == 200
        self._seed_capture(db_session, store, 1, PICTURE)
        self._assert_404(client.get(f"/d/{row.mask_sha256}"))
        assert client.get(f"/d/{row.source_sha256}").status_code == 200

    def test_an_origin_less_capture_of_the_picture_takes_the_mask_not_the_copy_offline(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Pooling fails closed for the MASK (a capture with no live origin
        makes it ineligible); the clean copy is judged by liveness alone."""
        self._seed_capture(db_session, store, 0, ITEM.format(n=0))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        row = db_session.execute(select(ImageMask)).scalar_one()
        assert client.get(f"/d/{row.mask_sha256}").status_code == 200
        seed_clean_image(
            db_session, store, _capture("capture 1"), origins=[], figure_id="mfc:capture-1"
        )
        self._assert_404(client.get(f"/d/{row.mask_sha256}"))
        assert client.get(f"/d/{row.source_sha256}").status_code == 200

    @pytest.mark.parametrize(
        ("second_url", "mask_served"),
        [(ITEM.format(n=2), True), (PICTURE, False)],
        ids=["item-capture", "collector-photo"],
    )
    def test_soft_deleting_the_capture_whose_bytes_are_the_copy_takes_it_offline(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        second_url: str,
        mask_served: bool,
    ) -> None:
        """Challenger i3 D2: the second capture's bytes ARE the first one's
        canonical clean copy. It is linked behind that copy by its pixels, so
        it is judged with it, and its soft delete takes the copy (and any
        mask) offline."""
        from app.workers.mask_seed import make_clean_copy

        first = _rgba_capture("crawl 1")
        again = make_clean_copy(first)[0]
        clean = seed_clean_image(
            db_session,
            store,
            first,
            origins=[ImageOrigin(url=ITEM.format(n=1), source="mfc")],
            figure_id="mfc:1",
        )
        seed_clean_image(
            db_session,
            store,
            again,
            origins=[ImageOrigin(url=second_url, source="mfc")],
            figure_id="mfc:2",
        )
        assert clean.sha256 == hashlib.sha256(again).hexdigest()
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        masks = [
            m.mask_sha256
            for m in db_session.execute(select(ImageMask).where(ImageMask.status == "ok")).scalars()
        ]
        assert len(masks) == (1 if mask_served else 0)
        assert client.get(f"/d/{clean.sha256}").status_code == 200
        for mask in masks:
            assert client.get(f"/d/{mask}").status_code == 200
        captured = db_session.execute(
            select(ImageModel).where(ImageModel.sha256 == clean.sha256)
        ).scalar_one()
        captured.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{clean.sha256}"))
        for mask in masks:
            self._assert_404(client.get(f"/d/{mask}"))

    def _lossy_pair(self) -> tuple[bytes, bytes]:
        """An opaque capture and a capture whose bytes ARE its clean copy, a
        JPEG: the two do not share a pixel digest (challenger i4 J1-J4)."""
        from app.workers.mask_seed import make_clean_copy

        arr = np.full((60, 40, 3), (245, 245, 245), dtype=np.uint8)
        arr[10:50, 12:28] = (20, 20, 30)
        arr[20:30, 15:25] = (200, 40, 40)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        first = buf.getvalue()
        again, mime, _w, _h = make_clean_copy(first)
        assert mime == "image/jpeg"
        return first, again

    def _seed_one(
        self, db: Session, store: LocalObjectStore, data: bytes, url: str, source: str = "mfc"
    ) -> PublicObject:
        return seed_clean_image(
            db,
            store,
            data,
            origins=[ImageOrigin(url=url, source=source)],
            figure_id=f"{source}:{hashlib.sha256(data).hexdigest()[:8]}",
        )

    def _ok_masks(self, db: Session) -> list[str]:
        db.expire_all()
        rows = db.execute(select(ImageMask).where(ImageMask.status == "ok")).scalars()
        return [r.mask_sha256 or "" for r in rows]

    def test_a_collector_photo_whose_bytes_are_the_items_jpeg_copy_takes_its_mask_offline(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """J1: the item's mask is served; then a collector's photo whose bytes
        ARE the item's copy is captured -- the mask is a 404 at once, before
        any re-evaluation; the copy itself stays (display is not decided
        here)."""
        first, again = self._lossy_pair()
        copy = self._seed_one(db_session, store, first, ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (mask,) = self._ok_masks(db_session)
        assert client.get(f"/d/{mask}").status_code == 200
        self._seed_one(db_session, store, again, PICTURE)
        assert copy.sha256 == hashlib.sha256(again).hexdigest()
        self._assert_404(client.get(f"/d/{mask}"))
        assert client.get(f"/d/{copy.sha256}").content == again

    def test_photo_first_the_item_is_seeded_and_no_mask_is_served(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """J2: seeding the item after the collector's photo whose bytes its
        copy turns out to be no longer raises; no mask of the picture is
        made, and every clean copy of it is served."""
        first, again = self._lossy_pair()
        photo_copy = self._seed_one(db_session, store, again, PICTURE)
        item_copy = self._seed_one(db_session, store, first, ITEM.format(n=1))
        assert item_copy.sha256 == hashlib.sha256(again).hexdigest()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert {(i.status, i.reason) for i in report.items} == {("ineligible", "user_photo")}
        assert self._ok_masks(db_session) == []
        for sha in (photo_copy.sha256, item_copy.sha256):
            assert client.get(f"/d/{sha}").status_code == 200

    @pytest.mark.parametrize("again_first", [False, True], ids=["first-first", "again-first"])
    def test_soft_deleting_the_capture_whose_bytes_are_the_jpeg_copy_takes_it_offline(
        self, client: TestClient, db_session: Session, store: LocalObjectStore, again_first: bool
    ) -> None:
        """J3: a takedown under the capture whose bytes ARE the copy takes
        every clean copy of the picture and every mask of it offline."""
        first, again = self._lossy_pair()
        captures = [(first, ITEM.format(n=1)), (again, ITEM.format(n=2))]
        copies = [
            self._seed_one(db_session, store, data, url).sha256
            for data, url in (captures[::-1] if again_first else captures)
        ]
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        masks = self._ok_masks(db_session)
        assert masks
        for sha in [*copies, *masks]:
            assert client.get(f"/d/{sha}").status_code == 200
        captured = db_session.execute(
            select(ImageModel).where(ImageModel.sha256 == hashlib.sha256(again).hexdigest())
        ).scalar_one()
        captured.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        for sha in [*copies, *masks]:
            self._assert_404(client.get(f"/d/{sha}"))

    @pytest.mark.usefixtures("shop_registered")
    @pytest.mark.parametrize("shop_first", [False, True], ids=["item-first", "shop-first"])
    def test_revoking_the_store_whose_capture_is_the_jpeg_copy_takes_every_mask_offline(
        self, client: TestClient, db_session: Session, store: LocalObjectStore, shop_first: bool
    ) -> None:
        """J4: revoke-store reaches the item copy's mask when the store's
        capture IS that copy."""
        from app.workers.mask_job import revoke_store

        first, again = self._lossy_pair()
        captures = [
            (first, ITEM.format(n=1), "mfc"),
            (again, "https://img.shop.example/p/1.jpg", "shop"),
        ]
        for data, url, source in captures[::-1] if shop_first else captures:
            self._seed_one(db_session, store, data, url, source)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        masks = self._ok_masks(db_session)
        assert masks
        granted = Settings(environment="test", mask_store_grants={"shop": ["shop.example"]})
        origin_app.dependency_overrides[get_settings] = lambda: granted
        for mask in masks:
            assert client.get(f"/d/{mask}").status_code == 200
        revoke_store(db_session, "shop", dry_run=False)
        for mask in masks:
            self._assert_404(client.get(f"/d/{mask}"))

    def _register_matted(
        self,
        db: Session,
        store: LocalObjectStore,
        image_id: int,
        derived_from: int | None,
        version_no: int = 3,
    ) -> str:
        out = io.BytesIO()
        Image.new("RGBA", (40, 60), (31, 30, 90, version_no)).save(out, format="PNG")
        data = out.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        key = derived_key(sha, "image/png")
        store.put(key, data, "image/png")
        version = ImageVersion(
            image_id=image_id,
            version_no=version_no,
            derived_from_version=derived_from,
            transform_spec={"derive_kind": "matted"},
            mime="image/png",
            width=40,
            height=60,
            bytes=len(data),
            storage_key=key,
            visibility="public",
            age_rating=0,
            matted=True,
        )
        db.add(version)
        db.flush()
        register_public_object(
            db,
            sha256=sha,
            kind="matted",
            mime="image/png",
            size=len(data),
            width=40,
            height=60,
            storage_key=key,
            image_version_id=version.id,
        )
        db.commit()
        return sha

    @pytest.mark.parametrize(
        ("url", "status"),
        [(ITEM.format(n=7), 200), (PICTURE, 404)],
        ids=["item-image", "collector-photo"],
    )
    def test_a_matted_cut_out_is_judged_like_a_mask(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        url: str,
        status: int,
    ) -> None:
        """A matted cut-out is our derivative work on the picture, like a mask:
        served only while its image is live AND mask-eligible. Nothing
        registers one yet; whatever does cannot skip the permission split."""
        clean = seed_clean_image(
            db_session,
            store,
            _photo(7),
            origins=[ImageOrigin(url=url, source="mfc")],
            figure_id="mfc:7",
        )
        clean_version = db_session.get(ImageVersion, clean.image_version_id)
        assert clean_version is not None
        sha = self._register_matted(db_session, store, clean_version.image_id, 2)
        assert client.get(f"/d/{sha}").status_code == status

    def test_a_matted_cut_out_of_a_shared_picture_is_judged_over_every_capture(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Judged over the images behind the clean copy it was cut from
        (derived_from), not over its own image alone: the same picture is
        also a collector's photo, so the cut-out is refused like the mask."""
        self._seed_capture(db_session, store, 0, ITEM.format(n=0))
        self._seed_capture(db_session, store, 1, PICTURE)
        clean = db_session.execute(
            select(PublicObject).where(PublicObject.kind == "sanitized")
        ).scalar_one()
        assert client.get(f"/d/{clean.sha256}").status_code == 200
        item = _source_image(db_session, clean.sha256)
        sha = self._register_matted(db_session, store, item.id, 2)
        self._assert_404(client.get(f"/d/{sha}"))

    @pytest.mark.parametrize("derived_from", [1, None], ids=["the-original", "nothing"])
    def test_a_matted_cut_out_not_cut_from_a_clean_copy_is_not_served(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        derived_from: int | None,
    ) -> None:
        clean = seed_clean_image(
            db_session,
            store,
            _photo(7),
            origins=[ImageOrigin(url=ITEM.format(n=7), source="mfc")],
            figure_id="mfc:7",
        )
        image = _source_image(db_session, clean.sha256)
        sha = self._register_matted(db_session, store, image.id, derived_from)
        self._assert_404(client.get(f"/d/{sha}"))

    def test_a_cut_out_of_a_cut_out_is_not_served(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        clean = seed_clean_image(
            db_session,
            store,
            _photo(7),
            origins=[ImageOrigin(url=ITEM.format(n=7), source="mfc")],
            figure_id="mfc:7",
        )
        image = _source_image(db_session, clean.sha256)
        first = self._register_matted(db_session, store, image.id, 2)
        assert client.get(f"/d/{first}").status_code == 200
        second = self._register_matted(db_session, store, image.id, 3, version_no=4)
        self._assert_404(client.get(f"/d/{second}"))

    def test_a_cut_out_is_judged_over_its_own_images_clean_copy(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Two pictures, each with its version 2 clean copy: a cut-out of the
        collector's photo is judged over the photo's copy, not the item's."""
        seed_clean_image(
            db_session,
            store,
            _photo(7),
            origins=[ImageOrigin(url=ITEM.format(n=7), source="mfc")],
            figure_id="mfc:7",
        )
        photo = seed_clean_image(
            db_session,
            store,
            _photo(8),
            origins=[ImageOrigin(url=PICTURE, source="mfc")],
            figure_id="mfc:8",
        )
        image = _source_image(db_session, photo.sha256)
        sha = self._register_matted(db_session, store, image.id, 2)
        self._assert_404(client.get(f"/d/{sha}"))

    @pytest.mark.parametrize("deleted", ["matted", "clean"])
    def test_a_matted_cut_out_goes_offline_with_its_version_or_its_clean_copy(
        self, client: TestClient, db_session: Session, store: LocalObjectStore, deleted: str
    ) -> None:
        clean = seed_clean_image(
            db_session,
            store,
            _photo(7),
            origins=[ImageOrigin(url=ITEM.format(n=7), source="mfc")],
            figure_id="mfc:7",
        )
        image = _source_image(db_session, clean.sha256)
        sha = self._register_matted(db_session, store, image.id, 2)
        assert client.get(f"/d/{sha}").status_code == 200
        target = db_session.get(PublicObject, sha if deleted == "matted" else clean.sha256)
        assert target is not None
        version = db_session.get(ImageVersion, target.image_version_id)
        assert version is not None
        version.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        self._assert_404(client.get(f"/d/{sha}"))

    def test_a_store_mask_is_judged_by_the_origins_own_grants(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        seed_clean_image(
            db_session,
            store,
            _photo(3),
            origins=[ImageOrigin(url="https://img.shop.example/p/3.jpg", source="shop")],
            figure_id="shop:3",
        )
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"shop": ["shop.example"]}))
        get_settings.cache_clear()
        try:
            run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        finally:
            monkeypatch.delenv("MASK_STORE_GRANTS")
            get_settings.cache_clear()
        mask = db_session.execute(select(ImageMask)).scalar_one().mask_sha256
        granted = Settings(environment="test", mask_store_grants={"shop": ["shop.example"]})
        origin_app.dependency_overrides[get_settings] = lambda: granted
        assert client.get(f"/d/{mask}").status_code == 200
        origin_app.dependency_overrides[get_settings] = lambda: Settings(environment="test")
        self._assert_404(client.get(f"/d/{mask}"))

    def _store_and_item_masks(self, db: Session, store: LocalObjectStore) -> dict[str, str]:
        grants = {"shop": ["shop.example"]}
        shop = seed_clean_image(
            db,
            store,
            _photo(3),
            origins=[ImageOrigin(url="https://img.shop.example/p/3.jpg", source="shop")],
            figure_id="shop:3",
        )
        arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
        arr[5:55, 6:34] = (20, 120, 30)  # another silhouette: another mask
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
        item = seed_clean_image(
            db,
            store,
            buf.getvalue(),
            origins=[ImageOrigin(url=ITEM.format(n=4), source="mfc")],
            figure_id="mfc:4",
        )
        run_mask_batch(db, store, backend=_Model(), dry_run=False)
        rows = {r.source_sha256: r.mask_sha256 for r in db.execute(select(ImageMask)).scalars()}
        settings = Settings(environment="test", mask_store_grants=grants)
        origin_app.dependency_overrides[get_settings] = lambda: settings
        return {
            "shop_clean": shop.sha256,
            "shop_mask": rows[shop.sha256] or "",
            "item_mask": rows[item.sha256] or "",
        }

    @pytest.mark.usefixtures("shop_registered")
    def test_a_store_mask_is_a_404_the_moment_its_store_is_revoked(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The persisted revocation alone takes the store's masks offline --
        before any row is withdrawn -- and leaves its clean copies and every
        other source's masks alone."""
        from app.models import MaskStoreRevocation

        shas = self._store_and_item_masks(db_session, store)
        assert shas["shop_mask"] != shas["item_mask"]
        assert client.get(f"/d/{shas['shop_mask']}").status_code == 200
        db_session.add(MaskStoreRevocation(source="shop"))
        db_session.commit()
        self._assert_404(client.get(f"/d/{shas['shop_mask']}"))
        assert client.get(f"/d/{shas['item_mask']}").status_code == 200
        assert client.get(f"/d/{shas['shop_clean']}").status_code == 200

    @pytest.mark.usefixtures("shop_registered")
    def test_a_configured_revocation_is_a_404(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        shas = self._store_and_item_masks(db_session, store)
        revoked = Settings(
            environment="test",
            mask_store_grants={"shop": ["shop.example"]},
            mask_store_revoked=["shop"],
        )
        origin_app.dependency_overrides[get_settings] = lambda: revoked
        self._assert_404(client.get(f"/d/{shas['shop_mask']}"))
        assert client.get(f"/d/{shas['item_mask']}").status_code == 200

    @pytest.mark.usefixtures("shop_registered")
    def test_revoke_store_takes_the_masks_offline_and_restore_store_brings_them_back(
        self, client: TestClient, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import restore_store, revoke_store

        shas = self._store_and_item_masks(db_session, store)
        revoke_store(db_session, "shop", dry_run=False)
        self._assert_404(client.get(f"/d/{shas['shop_mask']}"))
        restore_store(db_session, store, "shop", dry_run=False, backend=_Model())
        assert client.get(f"/d/{shas['shop_mask']}").status_code == 200

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

    @pytest.mark.parametrize(
        "path",
        ["/d%2F{sha}", "/d%2f{sha}", "/%64/{sha}", "/d/%{hex0}{rest}", "/healthz%2F", "/%68ealthz"],
    )
    def test_a_percent_encoded_alias_is_not_the_address(
        self, client: TestClient, seeded: dict[str, str], path: str
    ) -> None:
        sha = seeded["mask"]
        url = path.format(sha=sha, hex0=format(ord(sha[0]), "02x"), rest=sha[1:])
        self._assert_404(client.get(url))

    @pytest.mark.parametrize(("path", "status"), [("/d/{sha}", 200), ("/d/{sha}/x", 404)])
    def test_without_a_raw_path_the_decoded_path_is_judged(
        self, client: TestClient, seeded: dict[str, str], path: str, status: int
    ) -> None:
        """ASGI makes ``raw_path`` optional; a server that omits it is judged
        on the decoded path, by the same grammar."""
        assert _asgi_get(path.format(sha=seeded["mask"])) == status

    @pytest.mark.parametrize(
        ("query", "status"),
        [(b"", 200), (b"raw=1", 404), (b"raw", 404), (b"=", 404), (b"&", 404)],
    )
    def test_a_query_is_refused_where_the_server_keeps_it_out_of_raw_path(
        self, client: TestClient, seeded: dict[str, str], query: bytes, status: int
    ) -> None:
        """uvicorn keeps the query out of ``raw_path`` (it splits the target
        at ``?``), so in production the grammar never sees it and only the
        no-query guard stops ``/d/<sha>?raw=1`` from aliasing the address.
        The TestClient leaves the query IN ``raw_path``, where the grammar
        alone refuses it -- so the guard is shown here, on the server's shape."""
        path = f"/d/{seeded['mask']}"
        assert _asgi_get(path, raw_path=path.encode(), query=query) == status

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

    def test_a_judgment_that_raises_is_a_500_that_is_never_cached(
        self, client: TestClient, seeded: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import derivative_main

        def _raises(*args: object, **kwargs: object) -> bool:
            raise RuntimeError("postgresql://user:pw@db/mm unreachable")

        monkeypatch.setattr(derivative_main, "servable", _raises)
        resp = client.get(f"/d/{seeded['mask']}")
        assert resp.status_code == 500
        assert resp.headers["cache-control"] == "no-store"
        _hardening(resp)
        assert b"postgresql" not in resp.content

    def test_a_malformed_origin_is_a_404_not_a_500(
        self, client: TestClient, seeded: dict[str, str], db_session: Session
    ) -> None:
        image = _source_image(db_session, seeded["clean"])
        db_session.add(
            FigureGallery(
                figure_id="shop:bad",
                source_url="http://[evil]/x.jpg",
                image_id=image.id,
                source="shop",
            )
        )
        db_session.commit()
        resp = client.get(f"/d/{seeded['mask']}")
        assert resp.status_code == 404
        assert resp.headers["cache-control"] == "no-store"

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
