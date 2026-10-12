"""Challenger i5 B: the production capture route, the gallery ingest
(``app.workers.tasks.ingest_gallery_images``), links what it captures like
the seed does. A collector photo of the item's picture -- its bytes ARE the
item's clean copy, or the same pixels under other bytes, or our lossy copy's
pixels under other bytes -- refuses the item's mask whichever was captured
first, and a soft delete of an ingested capture takes every clean copy of
its picture, and its mask, offline. Download, S3 and DNS are faked: no
network at all."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import socket
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.config import Settings, get_settings
from app.db import get_db
from app.derivative_main import app as origin_app
from app.derivative_main import get_object_store
from app.models import Image as ImageModel
from app.models import ImageMask, PublicObject, PublicObjectImage
from app.object_store import LocalObjectStore
from app.publication import mask_refusal
from app.workers.mask_job import run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import make_clean_copy, pixel_digest, seed_clean_image
from app.workers.matting import StubMattingBackend
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
from sqlalchemy import select
from sqlalchemy.orm import Session

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/{n}.jpeg"


class _Model(StubMattingBackend):
    name = "stub-model"
    publishes_masks = True


@pytest.fixture()
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


@pytest.fixture()
def origin(db_session: Session, store: LocalObjectStore) -> Generator[TestClient, None, None]:
    def _db() -> Generator[Session, None, None]:
        yield db_session

    origin_app.dependency_overrides[get_db] = _db
    origin_app.dependency_overrides[get_object_store] = lambda: store
    origin_app.dependency_overrides[get_settings] = lambda: Settings(environment="test")
    yield TestClient(origin_app, raise_server_exceptions=False)
    origin_app.dependency_overrides.clear()


def _opaque(text: str | None = None) -> bytes:
    """An opaque picture (its clean copy is a lossy JPEG), with a PNG text
    chunk unless ``text`` is None: other bytes, the very same pixels."""
    arr = np.full((60, 40, 3), (245, 245, 245), dtype=np.uint8)
    arr[10:50, 12:28] = (20, 20, 30)
    arr[20:30, 15:25] = (200, 40, 40)
    info = None
    if text is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", text)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _png_resave(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _photo_of(item: bytes, kind: str) -> bytes:
    """A collector's capture of ``item``'s picture."""
    copy = make_clean_copy(item)[0]
    photo = {
        "bytes-are-the-copy": copy,
        "same-pixels-other-bytes": _opaque("posted"),
        "the-copys-pixels-other-bytes": _png_resave(copy),
    }[kind]
    assert photo != item
    return photo


KINDS = ["bytes-are-the-copy", "same-pixels-other-bytes", "the-copys-pixels-other-bytes"]


def _fake_getaddrinfo(host: str, port: Any, *a: Any, **k: Any) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]


def _ingest(db: Session, data: bytes, url: str, figure_id: str = "mfc:photo") -> ImageModel:
    """The production ingest task body against the test session: the
    download, S3 and DNS are faked. Returns the image it recorded."""
    from app.workers.tasks import ingest_gallery_images

    resp = MagicMock()
    resp.status_code = 200
    resp.content = data
    resp.raise_for_status = MagicMock()
    settings = MagicMock()
    settings.s3_bucket = "test-bucket"

    @contextmanager
    def _worker_session(session_factory: Any = None) -> Generator[Session, None, None]:
        yield db

    with (
        patch("app.workers.tasks.worker_session", _worker_session),
        patch("app.workers.tasks.get_s3", return_value=MagicMock()),
        patch("app.config.get_settings", return_value=settings),
        patch("app.workers.tasks.httpx") as httpx_mock,
        patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo),
    ):
        httpx_mock.get.return_value = resp
        ingest_gallery_images(figure_id=figure_id, images=[{"url": url, "position": 0}])
    db.commit()
    return db.execute(
        select(ImageModel).where(ImageModel.sha256 == hashlib.sha256(data).hexdigest())
    ).scalar_one()


def _seed(db: Session, store: LocalObjectStore, data: bytes, url: str) -> PublicObject:
    return seed_clean_image(
        db,
        store,
        data,
        origins=[ImageOrigin(url=url, source="mfc")],
        figure_id="mfc:item",
    )


def _behind(db: Session, sha256: str) -> set[int]:
    return set(
        db.execute(
            select(PublicObjectImage.image_id).where(PublicObjectImage.sha256 == sha256)
        ).scalars()
    )


def _ok_masks(db: Session) -> list[str]:
    db.expire_all()
    rows = db.execute(select(ImageMask).where(ImageMask.status == "ok")).scalars()
    return [r.mask_sha256 or "" for r in rows]


class TestIngestLinksEveryCapture:
    @pytest.mark.parametrize("kind", KINDS)
    def test_item_first_a_collector_photo_ingested_later_takes_the_mask_offline(
        self, db_session: Session, store: LocalObjectStore, origin: TestClient, kind: str
    ) -> None:
        item = _opaque()
        copy = _seed(db_session, store, item, ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (mask,) = _ok_masks(db_session)
        assert origin.get(f"/d/{mask}").status_code == 200
        photo = _ingest(db_session, _photo_of(item, kind), PICTURE.format(n=1))
        assert origin.get(f"/d/{mask}").status_code == 404
        assert mask_refusal(db_session, copy.sha256) == "user_photo"
        assert photo.id in _behind(db_session, copy.sha256)
        assert origin.get(f"/d/{copy.sha256}").status_code == 200

    @pytest.mark.parametrize("kind", KINDS)
    def test_photo_first_a_collector_photo_ingested_earlier_refuses_the_mask(
        self, db_session: Session, store: LocalObjectStore, kind: str
    ) -> None:
        item = _opaque()
        photo = _ingest(db_session, _photo_of(item, kind), PICTURE.format(n=1))
        copy = _seed(db_session, store, item, ITEM.format(n=1))
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason) for i in report.items] == [("ineligible", "user_photo")]
        assert _ok_masks(db_session) == []
        assert photo.id in _behind(db_session, copy.sha256)

    def test_an_image_another_route_recorded_is_linked_when_ingest_captures_it(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The ingest reuses an image it already has by its bytes (here one
        an upload recorded, with no pixel key): the gallery row it adds makes
        it a capture of this route too, so it is linked like a new one."""
        item = _opaque()
        data = _opaque("posted")
        db_session.add(
            ImageModel(sha256=hashlib.sha256(data).hexdigest(), storage_key="uploads/x.png")
        )
        db_session.commit()
        copy = _seed(db_session, store, item, ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert _ok_masks(db_session)
        photo = _ingest(db_session, data, PICTURE.format(n=1))
        assert mask_refusal(db_session, copy.sha256) == "user_photo"
        assert photo.id in _behind(db_session, copy.sha256)

    def test_an_item_capture_ingested_later_keeps_the_mask(
        self, db_session: Session, store: LocalObjectStore, origin: TestClient
    ) -> None:
        """Linking never refuses on its own: another ITEM capture of the
        picture is behind the copy and the mask stays served."""
        item = _opaque()
        copy = _seed(db_session, store, item, ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (mask,) = _ok_masks(db_session)
        again = _ingest(db_session, _opaque("posted"), ITEM.format(n=2), figure_id="mfc:item")
        assert _behind(db_session, copy.sha256) >= {again.id}
        assert mask_refusal(db_session, copy.sha256) is None
        assert origin.get(f"/d/{mask}").status_code == 200

    @pytest.mark.parametrize("item_first", [True, False], ids=["item-first", "capture-first"])
    def test_soft_deleting_an_ingested_capture_takes_the_copy_and_mask_offline(
        self, db_session: Session, store: LocalObjectStore, origin: TestClient, item_first: bool
    ) -> None:
        item = _opaque()
        if item_first:
            copy = _seed(db_session, store, item, ITEM.format(n=1))
            again = _ingest(db_session, _opaque("posted"), ITEM.format(n=2), figure_id="mfc:item")
        else:
            again = _ingest(db_session, _opaque("posted"), ITEM.format(n=2), figure_id="mfc:item")
            copy = _seed(db_session, store, item, ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (mask,) = _ok_masks(db_session)
        assert origin.get(f"/d/{mask}").status_code == 200
        again.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        assert origin.get(f"/d/{copy.sha256}").status_code == 404
        assert origin.get(f"/d/{mask}").status_code == 404

    def test_an_ingested_capture_records_its_pixel_digest(self, db_session: Session) -> None:
        """The key a clean copy minted later finds it by, before any copy
        of its picture exists."""
        data = _opaque("posted")
        image = _ingest(db_session, data, PICTURE.format(n=1))
        assert image.pixel_sha256 == pixel_digest(data)
        assert db_session.execute(select(PublicObjectImage)).first() is None


class TestACaptureOfOurMaskIsBehindIt:
    """Challenger i5 C: an original whose bytes ARE one of our masks (our
    silhouette captured, after the mask was registered -- before, the
    registration refuses those bytes) is linked behind the mask by those
    bytes, whichever route captured it: while it is soft-deleted (a
    takedown) the mask is not served."""

    @pytest.mark.parametrize("route", ["seed", "ingest"])
    def test_soft_deleting_a_capture_of_the_masks_bytes_takes_the_mask_offline(
        self, db_session: Session, store: LocalObjectStore, origin: TestClient, route: str
    ) -> None:
        _seed(db_session, store, _opaque(), ITEM.format(n=1))
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (mask,) = _ok_masks(db_session)
        obj = db_session.get(PublicObject, mask)
        assert obj is not None
        data = store.get(obj.storage_key)
        if route == "seed":
            _seed(db_session, store, data, ITEM.format(n=2))
            captured = db_session.execute(
                select(ImageModel).where(ImageModel.sha256 == mask)
            ).scalar_one()
        else:
            captured = _ingest(db_session, data, ITEM.format(n=2), figure_id="mfc:item")
        assert captured.sha256 == mask
        assert origin.get(f"/d/{mask}").status_code == 200
        captured.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        assert origin.get(f"/d/{mask}").status_code == 404
        captured.deleted_at = None
        db_session.commit()
        assert origin.get(f"/d/{mask}").status_code == 200
