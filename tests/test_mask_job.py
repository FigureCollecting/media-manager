"""MASK-1 batch job: idempotent, resumable, dry-run by default.

One clean image = one unit of work, committed on its own (a crash loses at
most the image in flight). A result row per (clean image, mask pipeline
version) is the idempotency key: a re-run selects only what has no result
yet, and a version bump selects everything again. Only an ``ok`` mask is
registered for the public /d/<sha256> address; a held mask is kept privately
for review; an empty answer stores nothing.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.models import FigureGallery, ImageMask, MaskBatch, PublicObject, PublicObjectImage
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore, S3ObjectStore, derived_key
from app.workers import mask_job
from app.workers.mask import MASK_PIPELINE_VERSION, decode_mask_png
from app.workers.mask_job import run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import make_clean_copy, register_public_object, seed_clean_image
from app.workers.matting import MattingBackend, StubMattingBackend
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/{n}.jpeg"


def _photo(color: tuple[int, int, int], seed: int = 0) -> bytes:
    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = color
    arr[0, 0] = (250, 250, 248 - seed % 3)  # distinct bytes per seed
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _flat_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (6, 6), (5, 6, 7, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


def _seed(db: Session, store: LocalObjectStore, n: int, *, url: str = ITEM) -> PublicObject:
    return seed_clean_image(
        db,
        store,
        _photo((31 + 10 * n, 30, 90), seed=n),
        origins=[ImageOrigin(url=url.format(n=n), source="mfc")],
        figure_id=f"mfc:{n}",
    )


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask MODEL, so an applied batch
    can run without a GPU. The real stub never publishes."""

    name = "stub-model"
    publishes_masks = True


class _Constant(MattingBackend):
    name = "constant"
    publishes_masks = True

    def __init__(self, value: int) -> None:
        self.value = value

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        return np.full((image.height, image.width), self.value, dtype=np.uint8)


class _Boom(MattingBackend):
    name = "boom"
    publishes_masks = True

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        raise RuntimeError("s3://mindsignals-raw/raw-img/secret-key exploded")


class _Crash(BaseException):
    """Stands in for the worker being killed mid-batch (not an item failure)."""


class _CrashOnSecond(_Model):
    def __init__(self) -> None:
        self.calls = 0

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise _Crash
        return super().predict_alpha(image)


def _masks(db: Session) -> list[ImageMask]:
    return list(db.execute(select(ImageMask).order_by(ImageMask.id)).scalars())


def _public(db: Session, kind: str) -> list[PublicObject]:
    return list(db.execute(select(PublicObject).where(PublicObject.kind == kind)).scalars())


class TestSeed:
    def test_seed_registers_a_clean_copy_distinct_from_its_original(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        original = db_session.execute(select(ImageModel)).scalar_one()
        assert obj.kind == "sanitized"
        assert obj.sha256 != original.sha256
        data = store.get(obj.storage_key)
        assert hashlib.sha256(data).hexdigest() == obj.sha256
        gallery = db_session.execute(select(FigureGallery)).scalar_one()
        assert gallery.source_url == ITEM.format(n=1)

    def test_seed_is_idempotent(self, db_session: Session, store: LocalObjectStore) -> None:
        a = _seed(db_session, store, 1)
        b = _seed(db_session, store, 1)
        assert a.sha256 == b.sha256
        assert len(_public(db_session, "sanitized")) == 1

    def test_clean_copy_keeps_resolution_and_drops_metadata(self) -> None:
        img = Image.new("RGB", (33, 21), (1, 2, 3))
        exif = Image.Exif()
        exif[0x010F] = "SecretCam"  # Make
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        data, mime, w, h = make_clean_copy(buf.getvalue())
        out = Image.open(io.BytesIO(data))
        assert (w, h) == (33, 21) == out.size
        assert mime == "image/jpeg"
        assert not out.getexif()

    def test_clean_copy_of_a_transparent_png_stays_lossless_png(self) -> None:
        img = Image.new("RGBA", (5, 5), (1, 2, 3, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data, mime, _w, _h = make_clean_copy(buf.getvalue())
        assert mime == "image/png"
        assert Image.open(io.BytesIO(data)).mode == "RGBA"

    def test_only_known_kinds_can_be_published(self, db_session: Session) -> None:
        with pytest.raises(ValueError, match="kind"):
            register_public_object(
                db_session,
                sha256="a" * 64,
                kind="original",
                mime="image/png",
                size=1,
                width=1,
                height=1,
                storage_key="derived/x",
            )

    @pytest.mark.parametrize("channels", [3, 4])
    def test_a_clean_copy_differs_from_an_original_our_encoder_wrote(self, channels: int) -> None:
        y, x = np.mgrid[0:64, 0:64]
        planes = [(x * 7 + y * 3) % 256, (x * x + y) % 256, (y * 5) % 256, (x * y) % 256]
        arr = np.dstack(planes[:channels]).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        first, _m, _w, _h = make_clean_copy(buf.getvalue())
        again, _m, _w, _h = make_clean_copy(first)
        assert again != first

    def test_an_original_no_encoding_can_change_is_refused_loudly(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A tiny flat PNG compresses identically at every level: its clean
        copy would BE the original, whose address must never be public."""
        first, _m, _w, _h = make_clean_copy(_flat_png())
        with pytest.raises(ValueError, match="original"):
            seed_clean_image(db_session, store, first, origins=[], figure_id="x")

    def test_clean_copy_of_a_palette_png_with_transparency_keeps_it(self) -> None:
        img = Image.new("P", (4, 4), 0)
        img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
        img.putpixel((1, 1), 1)
        buf = io.BytesIO()
        img.save(buf, format="PNG", transparency=0)
        data, mime, _w, _h = make_clean_copy(buf.getvalue())
        assert mime == "image/png"
        out = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        assert out[0, 0, 3] == 0
        assert out[1, 1, 3] == 255

    def test_an_original_can_never_be_registered_as_public(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        original = db_session.execute(select(ImageModel)).scalar_one()
        with pytest.raises(ValueError, match="original"):
            register_public_object(
                db_session,
                sha256=original.sha256,
                kind="mask",
                mime="image/png",
                size=1,
                width=1,
                height=1,
                storage_key="derived/x",
            )


class TestDryRunByDefault:
    def test_reports_everything_and_writes_nothing(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        keys_before = store.keys()
        report = run_mask_batch(db_session, store, backend=StubMattingBackend())
        assert report.dry_run is True
        assert report.batch_id is None
        assert report.counts["ok"] == 1
        (item,) = report.items
        assert item.source_sha256 == obj.sha256
        assert item.status == "ok"
        assert item.grounding is not None
        assert _masks(db_session) == []
        assert _public(db_session, "mask") == []
        assert db_session.execute(select(MaskBatch)).first() is None
        assert store.keys() == keys_before

    def test_a_dry_run_reports_refusals_and_failures_without_recording_them(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1, url=PICTURE)
        _seed(db_session, store, 2)
        report = run_mask_batch(db_session, store, backend=_Boom())
        assert sorted(i.status for i in report.items) == ["failed", "ineligible"]
        assert (report.counts["failed"], report.counts["ineligible"]) == (1, 1)
        assert _masks(db_session) == []


class TestApply:
    def test_stores_the_mask_and_its_numbers(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert report.batch_id is not None
        (row,) = _masks(db_session)
        assert (row.source_sha256, row.pipeline_version) == (obj.sha256, MASK_PIPELINE_VERSION)
        assert (row.status, row.origin_class, row.backend) == ("ok", "item_main", "stub-model")
        assert row.width == 40 and row.height == 60
        assert row.top_margin_frac == pytest.approx(10 / 60)
        assert row.bottom_margin_frac == pytest.approx(10 / 60)
        assert row.box_left_frac == pytest.approx(12 / 40)
        assert row.box_right_frac == pytest.approx(28 / 40)
        assert row.contact_band_center_x_frac == pytest.approx((12 + 27) / 2 / 40)
        assert row.masked_dominant_color == "#291e5a"
        assert row.plain_dominant_color == "#fafaf8"
        assert row.masked_thumbhash and row.plain_thumbhash
        (pub,) = _public(db_session, "mask")
        assert pub.sha256 == row.mask_sha256
        assert pub.storage_key == derived_key(pub.sha256, "image/png")
        data = store.get(pub.storage_key)
        assert hashlib.sha256(data).hexdigest() == pub.sha256
        assert decode_mask_png(data).shape == (60, 40)
        batch = db_session.get(MaskBatch, report.batch_id)
        assert batch is not None
        assert (batch.status, batch.dry_run) == ("done", False)
        assert batch.counts["ok"] == 1

    def test_a_second_run_is_a_no_op(self, db_session: Session, store: LocalObjectStore) -> None:
        _seed(db_session, store, 1)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        again = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert again.items == []
        assert len(_masks(db_session)) == 1
        assert len(_public(db_session, "mask")) == 1

    def test_two_images_with_the_same_mask_share_one_object(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        _seed(db_session, store, 2)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        rows = _masks(db_session)
        assert len(rows) == 2
        assert rows[0].mask_sha256 == rows[1].mask_sha256
        assert len(_public(db_session, "mask")) == 1

    def test_a_collector_photo_gets_no_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1, url=PICTURE)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason, row.origin_class) == (
            "ineligible",
            "user_photo",
            "user_photo",
        )
        assert row.mask_sha256 is None
        assert report.counts["ineligible"] == 1
        assert _public(db_session, "mask") == []

    def test_a_store_listing_is_masked_only_under_a_grant(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        from app.config import get_settings

        _seed(db_session, store, 1, url="https://img.shop.example/p/{n}.jpg")
        seed_clean_image(
            db_session,
            store,
            _photo((90, 30, 31), seed=2),
            origins=[ImageOrigin(url="https://i.imgur.com/2.jpg", source="shop")],
            figure_id="shop:2",
        )
        for row in db_session.execute(select(FigureGallery)).scalars():
            row.source = "shop"
        db_session.commit()
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"shop": ["shop.example"]}))
        get_settings.cache_clear()
        try:
            run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        finally:
            get_settings.cache_clear()
        by_class = sorted((r.origin_class, r.status, r.reason) for r in _masks(db_session))
        assert by_class == [
            ("not_granted", "ineligible", "not_granted"),
            ("store_listing", "ok", None),
        ]

    def test_a_held_mask_is_kept_privately_and_never_published(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        run_mask_batch(db_session, store, backend=_Constant(255), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason) == ("held", "full_frame")
        assert row.mask_sha256 is not None
        assert row.mask_storage_key == f"held-masks/{row.mask_sha256}.png"
        assert store.get(row.mask_storage_key)
        assert _public(db_session, "mask") == []

    def test_an_empty_answer_stores_nothing(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        keys_before = store.keys()
        run_mask_batch(db_session, store, backend=_Constant(0), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason, row.mask_sha256) == ("empty", "no_opaque_pixels", None)
        assert store.keys() == keys_before

    def test_a_failure_is_recorded_without_leaking_where_the_bytes_were(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        report = run_mask_batch(db_session, store, backend=_Boom(), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason) == ("failed", "RuntimeError")
        assert report.counts["failed"] == 1

    def test_failures_do_not_stop_the_batch_and_can_be_retried(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        run_mask_batch(db_session, store, backend=_Boom(), dry_run=False)
        skipped = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert skipped.items == []
        retried = run_mask_batch(
            db_session, store, backend=_Model(), dry_run=False, retry_failed=True
        )
        assert [i.status for i in retried.items] == ["ok"]
        (row,) = _masks(db_session)
        assert (row.status, row.reason) == ("ok", None)

    def test_a_malformed_origin_row_never_wedges_the_batch(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Challenger i3 D5: a gallery row whose URL the parser rejects is an
        unknown origin for its image, and every other image still settles."""
        from app.models import ImageVersion

        objs = sorted((_seed(db_session, store, n) for n in range(1, 5)), key=lambda o: o.sha256)
        version = db_session.get(ImageVersion, objs[1].image_version_id)
        assert version is not None
        db_session.add(
            FigureGallery(
                figure_id="shop:bad",
                source_url="http://[::1/x.jpg",
                image_id=version.image_id,
                source="shop",
            )
        )
        db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason) for i in report.items] == [
            ("ok", None),
            ("ineligible", "unknown_origin"),
            ("ok", None),
            ("ok", None),
        ]
        batches = list(db_session.execute(select(MaskBatch)).scalars())
        assert [b.status for b in batches] == ["done"]

    @pytest.mark.parametrize("error", ["RuntimeError", "IntegrityError"])
    def test_a_judgment_that_raises_is_a_failure_and_the_batch_goes_on(
        self,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
        error: str,
    ) -> None:
        """Whatever the judgment raises -- a database error that leaves the
        session needing a rollback included -- that image fails, alone."""
        objs = sorted((_seed(db_session, store, n) for n in range(1, 4)), key=lambda o: o.sha256)
        real = mask_job.judged_origins

        def judged(db: Session, sha256: str) -> object:
            if sha256 == objs[1].sha256:
                if error == "RuntimeError":
                    raise RuntimeError("s3://bucket/key exploded")
                _break_the_session(db)
            return real(db, sha256)

        monkeypatch.setattr(mask_job, "judged_origins", judged)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason) for i in report.items] == [
            ("ok", None),
            ("failed", error),
            ("ok", None),
        ]
        assert [(r.status, r.reason) for r in _masks(db_session)] == [
            ("ok", None),
            ("failed", error),
            ("ok", None),
        ]

    def test_a_dry_run_reports_a_judgment_that_raises_and_records_nothing(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db_session, store, 1)

        def judged(db: Session, sha256: str) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(mask_job, "judged_origins", judged)
        report = run_mask_batch(db_session, store, backend=_Model())
        assert [(i.status, i.reason) for i in report.items] == [("failed", "RuntimeError")]
        assert _masks(db_session) == []

    def test_a_store_granted_later_is_masked_on_an_ineligible_recheck(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        from app.config import get_settings

        _seed(db_session, store, 1, url="https://img.shop.example/p/{n}.jpg")
        for row in db_session.execute(select(FigureGallery)).scalars():
            row.source = "shop"
        db_session.commit()
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(r.status, r.reason) for r in _masks(db_session)] == [("ineligible", "not_granted")]
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"shop": ["shop.example"]}))
        get_settings.cache_clear()
        try:
            settled = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
            assert settled.items == []
            rechecked = run_mask_batch(
                db_session, store, backend=_Model(), dry_run=False, recheck_ineligible=True
            )
        finally:
            get_settings.cache_clear()
        assert [i.status for i in rechecked.items] == ["ok"]
        (row,) = _masks(db_session)
        assert (row.status, row.reason, row.origin_class) == ("ok", None, "store_listing")
        assert len(_public(db_session, "mask")) == 1

    def test_an_ineligible_recheck_leaves_failed_and_settled_rows_alone(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        _seed(db_session, store, 2, url=PICTURE)
        run_mask_batch(db_session, store, backend=_Boom(), dry_run=False)
        rechecked = run_mask_batch(
            db_session, store, backend=_Model(), dry_run=False, recheck_ineligible=True
        )
        assert [(i.status, i.reason) for i in rechecked.items] == [("ineligible", "user_photo")]
        assert sorted(r.status for r in _masks(db_session)) == ["failed", "ineligible"]

    def test_a_corrupted_clean_copy_is_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        store.put(obj.storage_key, b"not the bytes the address names", "image/png")
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason) == ("failed", "source_hash_mismatch")

    def test_a_deleted_image_is_skipped(self, db_session: Session, store: LocalObjectStore) -> None:
        import datetime as dt

        _seed(db_session, store, 1)
        original = db_session.execute(select(ImageModel)).scalar_one()
        original.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert report.items == []

    def test_a_deleted_clean_version_is_skipped(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        import datetime as dt

        from app.models import ImageVersion

        obj = _seed(db_session, store, 1)
        version = db_session.get(ImageVersion, obj.image_version_id)
        assert version is not None
        version.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert report.items == []

    def test_only_clean_copies_get_masks_not_a_matted_fallback(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        obj.kind, obj.pixel_sha256 = "matted", None  # only a clean copy carries a digest
        db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert report.items == []

    def test_a_soft_deleted_gallery_link_is_not_an_origin(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        import datetime as dt

        _seed(db_session, store, 1)
        gallery = db_session.execute(select(FigureGallery)).scalar_one()
        gallery.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (row,) = _masks(db_session)
        assert (row.status, row.reason) == ("ineligible", "no_origin")

    def test_limit_bounds_the_batch(self, db_session: Session, store: LocalObjectStore) -> None:
        for n in range(3):
            _seed(db_session, store, n)
        report = run_mask_batch(
            db_session, store, backend=_Model(), dry_run=False, limit=2, page_size=1
        )
        assert len(report.items) == 2
        assert len(_masks(db_session)) == 2
        rest = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert len(rest.items) == 1

    def test_a_new_pipeline_version_selects_everything_again(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db_session, store, 1)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        monkeypatch.setattr(mask_job, "MASK_PIPELINE_VERSION", "mask-v2")
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert len(report.items) == 1
        assert sorted(r.pipeline_version for r in _masks(db_session)) == [
            MASK_PIPELINE_VERSION,
            "mask-v2",
        ]


def _capture(text: str) -> bytes:
    """The same pixels every time; different bytes per ``text`` (a PNG text
    chunk) -- two captures of one picture, as two crawls or a re-save make."""
    from PIL import PngImagePlugin

    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (20, 20, 30)
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", text)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _seed_captures(
    db: Session, store: LocalObjectStore, urls: list[str]
) -> tuple[PublicObject, list[ImageModel]]:
    """One pixel-identical capture per url, seeded in that order; the shared
    clean copy and the images in seeding order."""
    objs = [
        seed_clean_image(
            db,
            store,
            _capture(f"capture {i}"),
            origins=[ImageOrigin(url=url, source="mfc")],
            figure_id=f"mfc:capture-{i}",
        )
        for i, url in enumerate(urls)
    ]
    assert len({o.sha256 for o in objs}) == 1
    images = list(db.execute(select(ImageModel).order_by(ImageModel.id)).scalars())
    assert len(images) == len(urls)
    return objs[0], images


class TestPixelIdenticalCaptures:
    """Two originals with the same pixels and different bytes are two images
    that share ONE clean copy (content-addressed). The copy and its mask are
    judged over EVERY image behind it, whichever was seeded first: the same
    picture captured as an item image and as a collector's photo is a
    collector's photo somewhere, and a picture soft-deleted under any capture
    is offline under all of them."""

    def test_two_captures_are_two_images_with_one_clean_copy(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed_captures(db_session, store, [ITEM.format(n=1), ITEM.format(n=2)])
        assert len(_public(db_session, "sanitized")) == 1

    @pytest.mark.parametrize("photo_first", [False, True], ids=["item-first", "photo-first"])
    def test_a_collector_photo_with_the_same_pixels_refuses_the_mask_in_either_order(
        self, db_session: Session, store: LocalObjectStore, photo_first: bool
    ) -> None:
        urls = [ITEM.format(n=1), PICTURE.format(n=1)]
        _seed_captures(db_session, store, urls[::-1] if photo_first else urls)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason, i.origin_class) for i in report.items] == [
            ("ineligible", "user_photo", "user_photo")
        ]
        assert _public(db_session, "mask") == []

    def test_item_captures_of_the_same_pixels_share_one_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed_captures(db_session, store, [ITEM.format(n=1), ITEM.format(n=2)])
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [i.status for i in report.items] == ["ok"]
        assert len(_public(db_session, "mask")) == 1

    @pytest.mark.parametrize("deleted", [0, 1], ids=["first-deleted", "second-deleted"])
    def test_a_soft_deleted_capture_keeps_the_shared_copy_out_of_the_batch(
        self, db_session: Session, store: LocalObjectStore, deleted: int
    ) -> None:
        import datetime as dt

        _, images = _seed_captures(db_session, store, [ITEM.format(n=1), ITEM.format(n=2)])
        images[deleted].deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert report.items == []

    def test_a_capture_soft_deleted_after_selection_gets_no_mask_and_no_row(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch re-judges what it selected: a capture soft-deleted between
        the candidate query and the judgment is SKIPPED as ``deleted`` and
        nothing is recorded -- as when it is deleted before selection -- so
        a restore makes it a candidate again for a plain batch."""
        import datetime as dt

        clean, images = _seed_captures(db_session, store, [ITEM.format(n=1), ITEM.format(n=2)])
        pages = iter([[clean], []])
        with monkeypatch.context() as m:
            m.setattr(mask_job, "_candidates", lambda *a, **kw: next(pages))
            images[1].deleted_at = dt.datetime.now(dt.UTC)
            db_session.commit()
            report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason) for i in report.items] == [("skipped", "deleted")]
        assert report.counts["skipped"] == 1
        assert _masks(db_session) == []
        assert _public(db_session, "mask") == []
        images[1].deleted_at = None
        db_session.commit()
        again = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [i.status for i in again.items] == ["ok"]

    @pytest.mark.parametrize("second", ["no-gallery-row", "url-none", "row-soft-deleted"])
    def test_a_capture_with_no_live_origin_makes_the_pool_ineligible(
        self, db_session: Session, store: LocalObjectStore, second: str
    ) -> None:
        """Pooling fails closed: a capture behind the shared copy with NO live
        origin would be refused on its own (no_origin), so the pool's mask is
        refused too -- whichever capture lacks it."""
        import datetime as dt

        seed_clean_image(
            db_session,
            store,
            _capture("capture 0"),
            origins=[ImageOrigin(url=ITEM.format(n=1), source="mfc")],
            figure_id="mfc:capture-0",
        )
        origins = {
            "no-gallery-row": [],
            "url-none": [ImageOrigin(url=None, source="mfc")],
            "row-soft-deleted": [ImageOrigin(url=ITEM.format(n=2), source="mfc")],
        }[second]
        seed_clean_image(
            db_session, store, _capture("capture 1"), origins=origins, figure_id="mfc:capture-1"
        )
        if second == "row-soft-deleted":
            row = db_session.execute(
                select(FigureGallery).where(FigureGallery.figure_id == "mfc:capture-1")
            ).scalar_one()
            row.deleted_at = dt.datetime.now(dt.UTC)
            db_session.commit()
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason, i.origin_class) for i in report.items] == [
            ("ineligible", "no_origin", "no_origin")
        ]
        assert _public(db_session, "mask") == []

    def test_a_collector_photo_with_the_same_pixels_seeded_later_withdraws_the_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import reevaluate_masks

        _seed_captures(db_session, store, [ITEM.format(n=1)])
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        seed_clean_image(
            db_session,
            store,
            _capture("a collector's re-save"),
            origins=[ImageOrigin(url=PICTURE.format(n=1), source="mfc")],
            figure_id="mfc:capture-photo",
        )
        report = reevaluate_masks(db_session, dry_run=False)
        (row,) = _masks(db_session)
        assert report["withdrawn"] == [row.source_sha256]
        assert (row.status, row.reason) == ("withdrawn", "user_photo")


def _rgba_capture(text: str | None) -> bytes:
    """An RGBA picture (one transparent pixel, so its clean copy is a lossless
    PNG), with a text chunk unless ``text`` is None."""
    from PIL import PngImagePlugin

    arr = np.full((60, 40, 4), (245, 245, 245, 255), dtype=np.uint8)
    arr[10:50, 12:28] = (20, 20, 30, 255)
    arr[0, 0, 3] = 0
    info = None
    if text is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", text)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _canonical_pair() -> tuple[bytes, bytes]:
    """Challenger i3's D2 pair: a capture with a text chunk, and a capture
    whose bytes ARE that picture's canonical clean encoding (our own clean
    copy captured again). The same pixels, different bytes -- and their
    clean copies do not coincide, because the encoder falls back to another
    setting when the first one reproduces its input."""
    first = _rgba_capture("crawl 1")
    again = make_clean_copy(first)[0]
    assert again != first
    assert make_clean_copy(again)[0] != again
    return first, again


def pixel_digest(original: bytes) -> str:
    from app.workers.mask_seed import pixel_digest as digest

    return digest(original)


def _break_the_session(db: Session) -> None:
    """Fail a flush, which leaves the session refusing every statement until
    it is rolled back -- what a database error mid-judgment does."""
    db.add(
        PublicObject(
            sha256="e" * 64,
            kind="original",
            mime="x",
            bytes=1,
            width=1,
            height=1,
            storage_key="x",
        )
    )
    db.flush()


class TestPixelDigest:
    """The link between captures keys on the PIXELS: the sha-256 of the mode,
    the size and the decoded bytes after EXIF orientation."""

    def test_the_canonical_pair_has_one_digest(self) -> None:
        first, again = _canonical_pair()
        assert pixel_digest(first) == pixel_digest(again)

    def test_other_pixels_have_another_digest(self) -> None:
        assert pixel_digest(_photo((31, 30, 90))) != pixel_digest(_photo((41, 30, 90)))

    def test_the_digest_names_the_size(self) -> None:
        """Same decoded bytes, another shape: another picture."""

        def png(size: tuple[int, int]) -> bytes:
            buf = io.BytesIO()
            Image.new("RGB", size, (9, 9, 9)).save(buf, format="PNG")
            return buf.getvalue()

        assert pixel_digest(png((2, 3))) != pixel_digest(png((3, 2)))

    def test_one_clean_copy_per_picture_is_a_constraint(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        clean = _seed(db_session, store, 1)
        with pytest.raises(IntegrityError):
            register_public_object(
                db_session,
                sha256="b" * 64,
                kind="sanitized",
                mime="image/png",
                size=1,
                width=1,
                height=1,
                storage_key="derived/b",
                pixel_sha256=clean.pixel_sha256,
            )

    def test_the_digest_is_taken_after_exif_orientation(self) -> None:
        arr = np.zeros((20, 10, 3), dtype=np.uint8)
        arr[:5] = (200, 10, 10)
        upright = Image.fromarray(arr, mode="RGB")
        exif = Image.Exif()
        exif[0x0112] = 6  # stored rotated: rotate 90 degrees clockwise to view
        tagged = io.BytesIO()
        upright.transpose(Image.Transpose.ROTATE_90).save(tagged, format="PNG", exif=exif)
        plain = io.BytesIO()
        upright.save(plain, format="PNG")
        assert pixel_digest(tagged.getvalue()) == pixel_digest(plain.getvalue())

    def test_a_seeded_clean_copy_carries_the_digest(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        clean = _seed(db_session, store, 1)
        assert clean.pixel_sha256 == pixel_digest(_photo((41, 30, 90), seed=1))

    @pytest.mark.parametrize(("kind", "digest"), [("sanitized", None), ("mask", "d" * 64)])
    def test_only_a_clean_copy_carries_a_digest(
        self, db_session: Session, kind: str, digest: str | None
    ) -> None:
        with pytest.raises(ValueError, match="pixel digest"):
            register_public_object(
                db_session,
                sha256="a" * 64,
                kind=kind,
                mime="image/png",
                size=1,
                width=1,
                height=1,
                storage_key="derived/x",
                pixel_sha256=digest,
            )


class TestCapturesKeyedOnPixels:
    """Challenger i3 D2: two captures of one picture whose clean copies would
    NOT coincide -- one of them is the canonical clean encoding itself -- still
    share ONE clean copy, in either seeding order, because the link keys on
    the pixel digest and not on the minted clean sha."""

    def _seed_pair(
        self, db: Session, store: LocalObjectStore, urls: tuple[str, str], *, again_first: bool
    ) -> tuple[list[PublicObject], dict[str, ImageModel]]:
        first, again = _canonical_pair()
        order = [("again", again, urls[1]), ("first", first, urls[0])]
        if not again_first:
            order.reverse()
        objs = [
            seed_clean_image(
                db,
                store,
                data,
                origins=[ImageOrigin(url=url, source="mfc")],
                figure_id=f"mfc:{name}",
            )
            for name, data, url in order
        ]
        images = {
            name: db.execute(
                select(ImageModel).where(ImageModel.sha256 == hashlib.sha256(data).hexdigest())
            ).scalar_one()
            for name, data, _ in order
        }
        return objs, images

    @pytest.mark.parametrize("again_first", [False, True], ids=["item-first", "photo-first"])
    def test_a_collector_photo_that_is_the_clean_encoding_refuses_the_mask(
        self, db_session: Session, store: LocalObjectStore, again_first: bool
    ) -> None:
        objs, images = self._seed_pair(
            db_session, store, (ITEM.format(n=1), PICTURE.format(n=1)), again_first=again_first
        )
        assert len({o.sha256 for o in objs}) == 1
        assert len(_public(db_session, "sanitized")) == 1
        linked = db_session.execute(
            select(PublicObjectImage.image_id).where(PublicObjectImage.sha256 == objs[0].sha256)
        ).scalars()
        assert sorted(linked) == sorted(i.id for i in images.values())
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason, i.origin_class) for i in report.items] == [
            ("ineligible", "user_photo", "user_photo")
        ]
        assert _public(db_session, "mask") == []

    @pytest.mark.parametrize("again_first", [False, True], ids=["first-first", "again-first"])
    def test_item_captures_one_of_them_the_clean_encoding_share_one_mask(
        self, db_session: Session, store: LocalObjectStore, again_first: bool
    ) -> None:
        self._seed_pair(
            db_session, store, (ITEM.format(n=1), ITEM.format(n=2)), again_first=again_first
        )
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [i.status for i in report.items] == ["ok"]
        assert len(_public(db_session, "sanitized")) == 1
        assert len(_public(db_session, "mask")) == 1

    def test_soft_deleting_the_capture_whose_bytes_are_the_copy_withdraws_the_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Item first: the shared copy's bytes ARE the second capture's
        original. A soft delete of that capture reaches the copy and its mask."""
        import datetime as dt

        from app.workers.mask_job import reevaluate_masks

        objs, images = self._seed_pair(
            db_session, store, (ITEM.format(n=1), ITEM.format(n=2)), again_first=False
        )
        assert images["again"].sha256 == objs[0].sha256
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        images["again"].deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = reevaluate_masks(db_session, dry_run=False)
        (row,) = _masks(db_session)
        assert report["withdrawn"] == [row.source_sha256]
        assert (row.status, row.reason) == ("withdrawn", "deleted")


def _opaque_capture(text: str | None) -> bytes:
    """An OPAQUE picture (no pixel is transparent), so its clean copy is a
    JPEG -- LOSSY: decoding the copy does not give back these pixels. A text
    chunk unless ``text`` is None."""
    from PIL import PngImagePlugin

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


def _lossy_pair() -> tuple[bytes, bytes]:
    """Challenger i4's J pair: an opaque capture, and a capture whose bytes
    ARE its clean copy (our JPEG captured again). Unlike the lossless pair,
    the two do NOT share a pixel digest."""
    first = _opaque_capture("crawl 1")
    again, mime, _w, _h = make_clean_copy(first)
    assert mime == "image/jpeg"
    assert pixel_digest(first) != pixel_digest(again)
    return first, again


def _jpeg_comment(jpeg: bytes, text: bytes) -> bytes:
    """``jpeg`` with a COM segment after SOI: other bytes, the very same
    pixels (metadata a platform adds without re-encoding)."""
    assert jpeg[:2] == b"\xff\xd8"
    return jpeg[:2] + b"\xff\xfe" + (len(text) + 2).to_bytes(2, "big") + text + jpeg[2:]


def _png_resave(data: bytes) -> bytes:
    """``data``'s pixels re-saved losslessly as a PNG: other bytes, the same pixels."""
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _behind(db: Session, sha256: str) -> set[int]:
    return set(
        db.execute(
            select(PublicObjectImage.image_id).where(PublicObjectImage.sha256 == sha256)
        ).scalars()
    )


def _one_picture(db: Session) -> list[PublicObject]:
    """Every clean copy, each asserted to be judged over EVERY image seeded
    (the test seeded captures of one picture only)."""
    images = set(db.execute(select(ImageModel.id)).scalars())
    copies = _public(db, "sanitized")
    assert copies
    for copy in copies:
        assert _behind(db, copy.sha256) == images, copy.sha256
    return copies


class TestOurLossyCopyCapturedAgain:
    """Challenger i4 J1-J4: an OPAQUE picture's clean copy is a JPEG, so our
    copy captured again -- its very bytes, or its pixels under other bytes (a
    comment a platform adds, a lossless re-save) -- does NOT share the
    original's pixel digest. It is still a capture of the same picture: it
    is linked behind the copy by the copy's bytes or by the copy's OWN pixel
    digest, in either seeding order, and every clean copy of the picture is
    judged over every capture of it."""

    def _seed_in_order(
        self,
        db: Session,
        store: LocalObjectStore,
        captures: list[tuple[bytes, str, str]],
        *,
        reverse: bool,
    ) -> None:
        for data, url, source in captures[::-1] if reverse else captures:
            seed_clean_image(
                db,
                store,
                data,
                origins=[ImageOrigin(url=url, source=source)],
                figure_id=f"{source}:{hashlib.sha256(data).hexdigest()[:8]}",
            )

    def _image(self, db: Session, data: bytes) -> ImageModel:
        sha = hashlib.sha256(data).hexdigest()
        return db.execute(select(ImageModel).where(ImageModel.sha256 == sha)).scalar_one()

    @pytest.mark.parametrize("photo_first", [False, True], ids=["item-first", "photo-first"])
    def test_a_collector_photo_whose_bytes_are_the_items_copy_refuses_the_mask(
        self, db_session: Session, store: LocalObjectStore, photo_first: bool
    ) -> None:
        """J1 (item first: a mask over a copy whose bytes ARE a collector's
        photo) and J2 (photo first: seeding the item raised)."""
        first, again = _lossy_pair()
        self._seed_in_order(
            db_session,
            store,
            [(first, ITEM.format(n=1), "mfc"), (again, PICTURE.format(n=1), "mfc")],
            reverse=photo_first,
        )
        copies = _one_picture(db_session)
        assert hashlib.sha256(again).hexdigest() in {c.sha256 for c in copies}
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert {(i.status, i.reason) for i in report.items} == {("ineligible", "user_photo")}
        assert _public(db_session, "mask") == []

    @pytest.mark.parametrize("photo_first", [False, True], ids=["item-first", "photo-first"])
    @pytest.mark.parametrize("resave", ["jpeg-comment", "png"])
    def test_a_collector_photo_with_the_copys_pixels_under_other_bytes_refuses_the_mask(
        self, db_session: Session, store: LocalObjectStore, photo_first: bool, resave: str
    ) -> None:
        first, again = _lossy_pair()
        photo = _jpeg_comment(again, b"posted") if resave == "jpeg-comment" else _png_resave(again)
        assert photo != again
        assert pixel_digest(photo) == pixel_digest(again)
        self._seed_in_order(
            db_session,
            store,
            [(first, ITEM.format(n=1), "mfc"), (photo, PICTURE.format(n=1), "mfc")],
            reverse=photo_first,
        )
        _one_picture(db_session)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert {(i.status, i.reason) for i in report.items} == {("ineligible", "user_photo")}
        assert _public(db_session, "mask") == []

    @pytest.mark.parametrize("again_first", [False, True], ids=["first-first", "again-first"])
    def test_soft_deleting_the_capture_whose_bytes_are_the_copy_withdraws_every_mask(
        self, db_session: Session, store: LocalObjectStore, again_first: bool
    ) -> None:
        """J3: a takedown under the capture whose bytes ARE the copy reaches
        the copy and every mask of the picture."""
        import datetime as dt

        from app.publication import images_behind
        from app.workers.mask_job import reevaluate_masks

        first, again = _lossy_pair()
        self._seed_in_order(
            db_session,
            store,
            [(first, ITEM.format(n=1), "mfc"), (again, ITEM.format(n=2), "mfc")],
            reverse=again_first,
        )
        copies = _one_picture(db_session)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        ok = sorted(r.source_sha256 for r in _masks(db_session) if r.status == "ok")
        assert ok == sorted(c.sha256 for c in copies)
        self._image(db_session, again).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = reevaluate_masks(db_session, dry_run=False)
        assert sorted(report["withdrawn"]) == ok
        assert {(r.status, r.reason) for r in _masks(db_session)} == {("withdrawn", "deleted")}
        assert [images_behind(db_session, c.sha256) for c in copies] == [None] * len(copies)

    @pytest.mark.usefixtures("stores_registered")
    @pytest.mark.parametrize("shop_first", [False, True], ids=["item-first", "shop-first"])
    def test_revoking_the_store_whose_capture_is_the_copy_withdraws_every_mask(
        self, db_session: Session, store: LocalObjectStore, shop_first: bool
    ) -> None:
        """J4: revoke-store reaches the item's copy whose bytes the store's
        capture IS."""
        first, again = _lossy_pair()
        self._seed_in_order(
            db_session,
            store,
            [(first, ITEM.format(n=1), "mfc"), (again, SHOP.format(n=1), "shop")],
            reverse=shop_first,
        )
        copies = _one_picture(db_session)
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert sorted(r.source_sha256 for r in _masks(db_session) if r.status == "ok") == sorted(
            c.sha256 for c in copies
        )
        report = revoke_store_applied(db_session, "shop")
        assert sorted(report["reevaluate"]["withdrawn"]) == sorted(c.sha256 for c in copies)
        db_session.expire_all()
        assert {(r.status, r.reason) for r in _masks(db_session)} == {
            ("withdrawn", "store_revoked")
        }

    def test_a_copy_is_found_by_its_bytes_when_its_own_digest_is_stale(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The bytes are the key no decoder can change: a copy whose recorded
        own digest no longer matches what the decoder now computes (another
        libjpeg) is still found by a capture of its very bytes."""
        first, again = _lossy_pair()
        copy = seed_clean_image(
            db_session,
            store,
            first,
            origins=[ImageOrigin(url=ITEM.format(n=1), source="mfc")],
            figure_id="mfc:1",
        )
        copy.own_pixel_sha256 = "0" * 64
        db_session.commit()
        seed_clean_image(
            db_session,
            store,
            again,
            origins=[ImageOrigin(url=PICTURE.format(n=1), source="mfc")],
            figure_id="mfc:photo",
        )
        assert [c.sha256 for c in _one_picture(db_session)] == [copy.sha256]

    def test_a_capture_whose_bytes_a_new_copy_turns_out_to_be_is_linked_behind_it(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Photo first, with the photo's own copy keyed on a digest the
        decoder no longer reproduces: the item's new copy is linked to the
        photo by its bytes alone, and the photo's copy joins the picture."""
        first, again = _lossy_pair()
        photo_copy = seed_clean_image(
            db_session,
            store,
            again,
            origins=[ImageOrigin(url=PICTURE.format(n=1), source="mfc")],
            figure_id="mfc:photo",
        )
        photo_copy.pixel_sha256 = "1" * 64
        db_session.commit()
        item_copy = seed_clean_image(
            db_session,
            store,
            first,
            origins=[ImageOrigin(url=ITEM.format(n=1), source="mfc")],
            figure_id="mfc:1",
        )
        assert item_copy.sha256 == hashlib.sha256(again).hexdigest()
        assert sorted(c.sha256 for c in _one_picture(db_session)) == sorted(
            [item_copy.sha256, photo_copy.sha256]
        )

    def test_a_seeded_copy_carries_its_own_digest(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A JPEG copy's own digest is its decoded bytes', not its original's;
        a lossless copy's is its original's."""
        first, again = _lossy_pair()
        lossy = seed_clean_image(db_session, store, first, origins=[], figure_id="x:1")
        assert (lossy.pixel_sha256, lossy.own_pixel_sha256) == (
            pixel_digest(first),
            pixel_digest(again),
        )
        lossless = seed_clean_image(
            db_session, store, _rgba_capture("x"), origins=[], figure_id="x:2"
        )
        assert (
            lossless.own_pixel_sha256 == lossless.pixel_sha256 == pixel_digest(_rgba_capture("x"))
        )


class TestTheGateLinksOurCopyCapturedFirst:
    """``register_public_object`` refuses an original's bytes -- except a
    clean copy of ANOTHER image whose bytes an original already has (our
    copy captured before it was minted): that original is linked behind it
    and judged with it."""

    def _version_of_another_image(self, db: Session, store: LocalObjectStore) -> int:
        from app.models import ImageVersion

        other = _seed(db, store, 2)
        assert other.image_version_id is not None
        version = db.get(ImageVersion, other.image_version_id)
        assert version is not None
        return version.id

    def _register(self, db: Session, sha: str, kind: str, version_id: int | None) -> PublicObject:
        clean = kind == "sanitized"
        return register_public_object(
            db,
            sha256=sha,
            kind=kind,
            mime="image/jpeg",
            size=1,
            width=1,
            height=1,
            storage_key=f"derived/{kind}",
            image_version_id=version_id,
            pixel_sha256="c" * 64 if clean else None,
            own_pixel_sha256="d" * 64 if clean else None,
        )

    def test_a_clean_copy_whose_bytes_another_image_has_is_linked_behind_it(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        captured = db_session.execute(select(ImageModel).order_by(ImageModel.id)).scalars().first()
        assert captured is not None
        version_id = self._version_of_another_image(db_session, store)
        obj = self._register(db_session, captured.sha256, "sanitized", version_id)
        assert captured.id in _behind(db_session, obj.sha256)

    @pytest.mark.parametrize(
        ("kind", "versioned"),
        [("sanitized", False), ("mask", True), ("matted", True)],
        ids=["clean-copy-of-no-version", "mask", "matted"],
    )
    def test_anything_else_with_an_originals_bytes_is_refused(
        self, db_session: Session, store: LocalObjectStore, kind: str, versioned: bool
    ) -> None:
        _seed(db_session, store, 1)
        captured = db_session.execute(select(ImageModel).order_by(ImageModel.id)).scalars().first()
        assert captured is not None
        version_id = self._version_of_another_image(db_session, store) if versioned else None
        with pytest.raises(ValueError, match="original"):
            self._register(db_session, captured.sha256, kind, version_id)

    def test_a_clean_copy_with_its_own_images_bytes_is_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.models import ImageVersion

        own = _seed(db_session, store, 1)
        version = db_session.get(ImageVersion, own.image_version_id)
        assert version is not None
        image = db_session.get(ImageModel, version.image_id)
        assert image is not None
        with pytest.raises(ValueError, match="original"):
            self._register(db_session, image.sha256, "sanitized", version.id)

    @pytest.mark.parametrize(
        ("kind", "pixels", "own"),
        [
            ("sanitized", "c" * 64, None),
            ("sanitized", None, "d" * 64),
            ("mask", None, "d" * 64),
        ],
        ids=["clean-copy-without-own", "clean-copy-without-pixels", "mask-with-own"],
    )
    def test_a_clean_copy_and_only_a_clean_copy_carries_both_digests(
        self, db_session: Session, kind: str, pixels: str | None, own: str | None
    ) -> None:
        with pytest.raises(ValueError, match="pixel digest"):
            register_public_object(
                db_session,
                sha256="a" * 64,
                kind=kind,
                mime="image/png",
                size=1,
                width=1,
                height=1,
                storage_key="derived/x",
                pixel_sha256=pixels,
                own_pixel_sha256=own,
            )

    def test_the_own_digest_is_a_clean_copys_alone_in_the_schema(self, db_session: Session) -> None:
        from sqlalchemy.exc import IntegrityError

        db_session.add(
            PublicObject(
                sha256="f" * 64,
                kind="mask",
                mime="image/png",
                bytes=1,
                width=1,
                height=1,
                storage_key="derived/f",
                own_pixel_sha256="d" * 64,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()


def _image_of(db: Session, clean_sha: str) -> int:
    from app.models import ImageVersion

    clean = db.get(PublicObject, clean_sha)
    assert clean is not None
    version = db.get(ImageVersion, clean.image_version_id)
    assert version is not None
    return version.image_id


def revoke_store_applied(db: Session, source: str) -> dict[str, Any]:
    from app.workers.mask_job import revoke_store

    return revoke_store(db, source, dry_run=False)


SHOP = "https://img.shop.example/p/{n}.jpg"
OTHER = "https://cdn.other.example/p/{n}.jpg"


@pytest.fixture()
def stores_registered(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Two registered store sources: each is granted (MASK-G), no opt-in."""
    import json

    from app.config import get_settings

    grants = {"shop": ["shop.example"], "other": ["other.example"]}
    monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps(grants))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("stores_registered")
class TestStoreRevocation:
    """MASK-G (Ross 2026-10-12): cut-out masks for every store, revocable per
    store. ``revoke_store`` persists the revocation, then withdraws every
    mask of that source -- and judges nothing else; ``restore_store`` lifts
    it, restores the withdrawn masks and re-checks that source's refused
    images (``recheck_ineligible``). Both are dry runs by default."""

    def _seed(self, db: Session, store: LocalObjectStore, n: int, url: str, source: str) -> str:
        obj = seed_clean_image(
            db,
            store,
            _photo((31 + 10 * n, 30, 90), seed=n),
            origins=[ImageOrigin(url=url.format(n=n), source=source)],
            figure_id=f"{source}:{n}",
        )
        return obj.sha256

    def _settle_all(self, db: Session, store: LocalObjectStore) -> dict[str, str]:
        shas = {
            "shop1": self._seed(db, store, 1, SHOP, "shop"),
            "shop2": self._seed(db, store, 2, SHOP, "shop"),
            "other": self._seed(db, store, 3, OTHER, "other"),
            "item": self._seed(db, store, 4, ITEM, "mfc"),
        }
        run_mask_batch(db, store, backend=_Model(), dry_run=False)
        assert {r.status for r in _masks(db)} == {"ok"}
        return shas

    def _status(self, db: Session) -> dict[str, tuple[str, str | None]]:
        db.expire_all()
        return {r.source_sha256: (r.status, r.reason) for r in _masks(db)}

    def _revoked(self, db: Session) -> list[str]:
        from app.models import MaskStoreRevocation

        return list(db.execute(select(MaskStoreRevocation.source)).scalars())

    def test_revoke_is_a_dry_run_unless_applied(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import revoke_store

        shas = self._settle_all(db_session, store)
        report = revoke_store(db_session, "shop")
        assert (report["source"], report["dry_run"]) == ("shop", True)
        assert sorted(report["reevaluate"]["withdrawn"]) == sorted([shas["shop1"], shas["shop2"]])
        assert self._revoked(db_session) == []
        assert set(self._status(db_session).values()) == {("ok", None)}

    def test_revoke_withdraws_every_mask_of_the_source_and_nothing_else(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.models import ImageVersion
        from app.workers.mask_job import revoke_store

        shas = self._settle_all(db_session, store)
        # The MFC mask would fall to a FULL re-evaluation (its image gained a
        # collector's photo origin); a store revocation must not touch it.
        clean = db_session.get(PublicObject, shas["item"])
        assert clean is not None
        version = db_session.get(ImageVersion, clean.image_version_id)
        assert version is not None
        db_session.add(
            FigureGallery(
                figure_id="mfc:9", source_url=PICTURE.format(n=9), image_id=version.image_id
            )
        )
        db_session.commit()
        report = revoke_store(db_session, " Shop ", dry_run=False)
        assert report["source"] == "shop"
        assert sorted(report["reevaluate"]["withdrawn"]) == sorted([shas["shop1"], shas["shop2"]])
        assert report["reevaluate"]["unchanged"] == 0
        assert self._revoked(db_session) == ["shop"]
        status = self._status(db_session)
        assert status[shas["shop1"]] == status[shas["shop2"]] == ("withdrawn", "store_revoked")
        assert status[shas["other"]] == status[shas["item"]] == ("ok", None)
        again = revoke_store(db_session, "shop", dry_run=False)
        assert again["reevaluate"]["withdrawn"] == []
        assert self._revoked(db_session) == ["shop"]

    def test_revoke_reaches_a_copy_the_store_shares_with_another_source(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import revoke_store

        for i, (url, source) in enumerate(((ITEM.format(n=1), "mfc"), (SHOP.format(n=1), "shop"))):
            seed_clean_image(
                db_session,
                store,
                _capture(f"capture {i}"),
                origins=[ImageOrigin(url=url, source=source)],
                figure_id=f"{source}:capture",
            )
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        (row,) = _masks(db_session)
        assert row.status == "ok"
        report = revoke_store(db_session, "shop", dry_run=False)
        assert report["reevaluate"]["withdrawn"] == [row.source_sha256]
        assert self._status(db_session)[row.source_sha256] == ("withdrawn", "store_revoked")

    def test_revoke_judges_only_live_origins_of_the_source(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A copy whose ``shop`` origin was soft-deleted is no longer a shop
        mask: revoke-store leaves its row alone, even though a full
        re-evaluation would withdraw it (it also gained a collector's photo)."""
        import datetime as dt

        shas = self._settle_all(db_session, store)
        image_id = _image_of(db_session, shas["other"])
        db_session.add_all(
            [
                FigureGallery(
                    figure_id="shop:7",
                    source_url=SHOP.format(n=7),
                    image_id=image_id,
                    source="shop",
                    deleted_at=dt.datetime.now(dt.UTC),
                ),
                FigureGallery(figure_id="mfc:7", source_url=PICTURE.format(n=7), image_id=image_id),
            ]
        )
        db_session.commit()
        report = revoke_store_applied(db_session, "shop")
        assert shas["other"] not in report["reevaluate"]["withdrawn"]
        assert self._status(db_session)[shas["other"]] == ("ok", None)

    def test_revoke_finds_a_copy_by_its_own_image_without_a_link_row(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        shas = self._settle_all(db_session, store)
        link = db_session.get(
            PublicObjectImage, (shas["shop1"], _image_of(db_session, shas["shop1"]))
        )
        assert link is not None
        db_session.delete(link)
        db_session.commit()
        report = revoke_store_applied(db_session, "shop")
        assert shas["shop1"] in report["reevaluate"]["withdrawn"]

    def test_revoke_matches_the_source_as_the_policy_does(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A source registered and recorded as ``SHOP`` is the source
        ``revoke-store shop`` names: the origin refuses it, and its row is
        withdrawn too."""
        import json

        from app.config import get_settings

        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"SHOP": ["shop.example"]}))
        get_settings.cache_clear()
        sha = self._seed(db_session, store, 1, SHOP, "SHOP")
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert self._status(db_session)[sha] == ("ok", None)
        report = revoke_store_applied(db_session, "shop")
        assert report["reevaluate"]["withdrawn"] == [sha]

    def test_a_plain_reevaluation_honours_a_persisted_revocation(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.models import MaskStoreRevocation
        from app.workers.mask_job import reevaluate_masks

        shas = self._settle_all(db_session, store)
        db_session.add(MaskStoreRevocation(source="shop"))
        db_session.commit()
        report = reevaluate_masks(db_session, dry_run=False)
        assert sorted(report["withdrawn"]) == sorted([shas["shop1"], shas["shop2"]])

    def test_a_revoked_store_gets_no_new_masks(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import revoke_store

        revoke_store(db_session, "shop", dry_run=False)
        self._seed(db_session, store, 1, SHOP, "shop")
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        assert [(i.status, i.reason) for i in report.items] == [("ineligible", "store_revoked")]
        assert _public(db_session, "mask") == []

    def test_a_blank_source_is_refused(self, db_session: Session, store: LocalObjectStore) -> None:
        from app.workers.mask_job import restore_store, revoke_store

        with pytest.raises(ValueError, match="source"):
            revoke_store(db_session, "  ", dry_run=False)
        with pytest.raises(ValueError, match="source"):
            restore_store(db_session, store, "", dry_run=False, backend=_Model())
        assert self._revoked(db_session) == []

    def test_restore_lifts_the_revocation_restores_and_masks_what_was_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import restore_store, revoke_store

        shas = self._settle_all(db_session, store)
        revoke_store(db_session, "shop", dry_run=False)
        shas["shop5"] = self._seed(db_session, store, 5, SHOP, "shop")
        shas["photo"] = self._seed(db_session, store, 6, PICTURE, "mfc")
        run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        status = self._status(db_session)
        assert status[shas["shop5"]] == ("ineligible", "store_revoked")
        assert status[shas["photo"]] == ("ineligible", "user_photo")
        photo_batch = next(
            r.batch_id for r in _masks(db_session) if r.source_sha256 == shas["photo"]
        )

        dry = restore_store(db_session, store, "shop", backend=_Model())
        assert (dry["source"], dry["dry_run"]) == ("shop", True)
        assert sorted(dry["reevaluate"]["restored"]) == sorted([shas["shop1"], shas["shop2"]])
        assert [(i["source_sha256"], i["status"]) for i in dry["batch"]["items"]] == [
            (shas["shop5"], "ok")
        ]
        assert self._revoked(db_session) == ["shop"]
        assert self._status(db_session) == status

        applied = restore_store(db_session, store, "shop", dry_run=False, backend=_Model())
        assert applied["dry_run"] is False
        assert self._revoked(db_session) == []
        after = self._status(db_session)
        for name in ("shop1", "shop2", "shop5", "other", "item"):
            assert after[shas[name]] == ("ok", None), name
        assert after[shas["photo"]] == ("ineligible", "user_photo")
        assert (
            next(r.batch_id for r in _masks(db_session) if r.source_sha256 == shas["photo"])
            == photo_batch
        )

    def test_restoring_a_store_that_was_never_revoked_changes_nothing(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import restore_store

        shas = self._settle_all(db_session, store)
        before = self._status(db_session)
        report = restore_store(db_session, store, "shop", dry_run=False, backend=_Model())
        assert (report["reevaluate"]["restored"], report["batch"]["items"]) == ([], [])
        assert self._status(db_session) == before
        assert before[shas["shop1"]] == ("ok", None)

    def test_restore_refuses_a_backend_that_is_not_a_mask_model_before_changing_anything(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import restore_store, revoke_store

        shas = self._settle_all(db_session, store)
        revoke_store(db_session, "shop", dry_run=False)
        with pytest.raises(ValueError, match="dry run only"):
            restore_store(db_session, store, "shop", dry_run=False, backend=StubMattingBackend())
        assert self._revoked(db_session) == ["shop"]
        assert self._status(db_session)[shas["shop1"]] == ("withdrawn", "store_revoked")

    def test_a_source_revoked_by_configuration_is_not_restored_here(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.config import get_settings
        from app.workers.mask_job import restore_store

        shas = self._settle_all(db_session, store)
        monkeypatch.setenv("MASK_STORE_REVOKED", '["Shop"]')
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="MASK_STORE_REVOKED"):
            restore_store(db_session, store, "shop", dry_run=False, backend=_Model())
        assert self._status(db_session)[shas["shop1"]] == ("ok", None)

    def test_a_batch_can_be_limited_to_one_source(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        shop = self._seed(db_session, store, 1, SHOP, "shop")
        self._seed(db_session, store, 3, OTHER, "other")
        self._seed(db_session, store, 4, ITEM, "mfc")
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False, source="SHOP")
        assert [(i.source_sha256, i.status) for i in report.items] == [(shop, "ok")]

    def test_the_store_tasks_default_to_a_dry_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.workers import tasks

        seen: dict[str, object] = {}

        def _revoke(db: object, source: str, **kw: object) -> dict[str, object]:
            seen["revoke"] = (source, kw["dry_run"])
            return {}

        def _restore(db: object, store: object, source: str, **kw: object) -> dict[str, object]:
            seen["restore"] = (source, kw["dry_run"])
            return {}

        monkeypatch.setattr(tasks, "revoke_store", _revoke)
        monkeypatch.setattr(tasks, "restore_store", _restore)
        monkeypatch.setattr(tasks, "worker_session", _NullSession)
        monkeypatch.setattr(tasks, "S3ObjectStore", lambda: object())
        tasks.revoke_mask_store("shop")
        tasks.restore_mask_store("shop")
        assert seen == {"revoke": ("shop", True), "restore": ("shop", True)}
        tasks.revoke_mask_store("shop", apply=True)
        tasks.restore_mask_store("shop", apply=True)
        assert seen == {"revoke": ("shop", False), "restore": ("shop", False)}


class TestOnlyAMaskModelPublishes:
    """The stub is a backdrop-distance heuristic, not a mask model: under it
    the black-dress case and white-on-white figures pass the quality gate
    with wrong masks. An APPLIED batch refuses it -- passed or configured
    (the worker image leaves MATTING_BACKEND unset) -- before it writes
    anything, so a stub mask can never be published or settle an image."""

    def test_an_applied_batch_refuses_the_stub_and_writes_nothing(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        keys_before = store.keys()
        with pytest.raises(ValueError, match="dry run only"):
            run_mask_batch(db_session, store, backend=StubMattingBackend(), dry_run=False)
        assert _masks(db_session) == []
        assert _public(db_session, "mask") == []
        assert db_session.execute(select(MaskBatch)).first() is None
        assert store.keys() == keys_before

    def test_the_configured_default_is_the_stub_and_is_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        with pytest.raises(ValueError, match="'stub' .*dry run only"):
            run_mask_batch(db_session, store, dry_run=False)
        assert db_session.execute(select(MaskBatch)).first() is None
        assert _masks(db_session) == []

    def test_resuming_with_the_stub_is_refused_and_the_batch_stays_open(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        for n in range(3):
            _seed(db_session, store, n)
        with pytest.raises(_Crash):
            run_mask_batch(db_session, store, backend=_CrashOnSecond(), dry_run=False)
        (batch,) = db_session.execute(select(MaskBatch)).scalars()
        with pytest.raises(ValueError, match="dry run only"):
            run_mask_batch(
                db_session,
                store,
                backend=StubMattingBackend(),
                dry_run=False,
                resume_batch_id=batch.id,
            )
        db_session.refresh(batch)
        assert (batch.status, batch.counts["ok"]) == ("running", 1)
        assert len(_masks(db_session)) == 1

    def test_a_dry_run_may_still_use_the_stub(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        report = run_mask_batch(db_session, store)
        assert [i.status for i in report.items] == ["ok"]

    def test_publishing_is_opt_in_per_backend(self) -> None:
        from app.workers.basekept import BaseKeptMattingBackend
        from app.workers.matting import BiRefNetMattingBackend

        assert MattingBackend.publishes_masks is False
        assert StubMattingBackend.publishes_masks is False
        assert BiRefNetMattingBackend.publishes_masks is True
        assert BaseKeptMattingBackend.publishes_masks is True


class TestResume:
    def test_a_killed_batch_resumes_where_it_stopped(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        for n in range(3):
            _seed(db_session, store, n)
        with pytest.raises(_Crash):
            run_mask_batch(db_session, store, backend=_CrashOnSecond(), dry_run=False)
        (batch,) = db_session.execute(select(MaskBatch)).scalars()
        assert batch.status == "running"
        assert batch.counts["ok"] == 1
        assert len(_masks(db_session)) == 1
        done_sha = _masks(db_session)[0].source_sha256
        assert batch.cursor_sha256 == done_sha

        resumed = run_mask_batch(
            db_session, store, backend=_Model(), dry_run=False, resume_batch_id=batch.id
        )
        assert resumed.batch_id == batch.id
        assert len(resumed.items) == 2
        db_session.refresh(batch)
        assert batch.status == "done"
        assert batch.counts["ok"] == 3
        assert len(_masks(db_session)) == 3

    def test_a_resumed_batch_keeps_its_limit(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        for n in range(4):
            _seed(db_session, store, n)
        with pytest.raises(_Crash):
            run_mask_batch(db_session, store, backend=_CrashOnSecond(), dry_run=False, limit=3)
        (batch,) = db_session.execute(select(MaskBatch)).scalars()
        resumed = run_mask_batch(
            db_session, store, backend=_Model(), dry_run=False, resume_batch_id=batch.id
        )
        assert len(resumed.items) == 2
        db_session.refresh(batch)
        assert sum(batch.counts.values()) == 3
        assert len(_masks(db_session)) == 3

    def test_resuming_a_finished_or_unknown_batch_is_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        report = run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
        with pytest.raises(ValueError, match="not resumable"):
            run_mask_batch(
                db_session, store, backend=_Model(), dry_run=False, resume_batch_id=report.batch_id
            )
        with pytest.raises(ValueError, match="not resumable"):
            run_mask_batch(db_session, store, backend=_Model(), dry_run=False, resume_batch_id=999)

    def test_a_batch_of_an_older_pipeline_version_is_not_resumed(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for n in range(3):
            _seed(db_session, store, n)
        with pytest.raises(_Crash):
            run_mask_batch(db_session, store, backend=_CrashOnSecond(), dry_run=False)
        (batch,) = db_session.execute(select(MaskBatch)).scalars()
        monkeypatch.setattr(mask_job, "MASK_PIPELINE_VERSION", "mask-v2")
        with pytest.raises(ValueError, match="not resumable"):
            run_mask_batch(
                db_session, store, backend=_Model(), dry_run=False, resume_batch_id=batch.id
            )
        db_session.refresh(batch)
        assert (batch.status, batch.pipeline_version) == ("running", MASK_PIPELINE_VERSION)
        assert len(_masks(db_session)) == 1

    def test_a_dry_run_cannot_resume(self, db_session: Session, store: LocalObjectStore) -> None:
        with pytest.raises(ValueError, match="dry run"):
            run_mask_batch(db_session, store, resume_batch_id=1)


class TestReevaluate:
    """The origin re-checks every mask on every request; ``reevaluate_masks``
    brings the rows in line, so the lane's own record says which masks are
    published: an ``ok`` row whose clean image was soft-deleted or is no
    longer eligible becomes ``withdrawn`` (with the refusal as its reason),
    and a ``withdrawn`` row whose image is live and eligible again becomes
    ``ok``. Dry run by default."""

    def _settle_two(self, db: Session, store: LocalObjectStore) -> list[ImageMask]:
        _seed(db, store, 1)
        _seed(db, store, 2)
        run_mask_batch(db, store, backend=_Model(), dry_run=False)
        rows = _masks(db)
        assert [r.status for r in rows] == ["ok", "ok"]
        return rows

    def test_a_collector_photo_origin_withdraws_the_mask_and_its_removal_restores_it(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.models import ImageVersion
        from app.workers.mask_job import reevaluate_masks

        first, second = self._settle_two(db_session, store)
        clean = db_session.get(PublicObject, first.source_sha256)
        assert clean is not None
        version = db_session.get(ImageVersion, clean.image_version_id)
        assert version is not None
        link = FigureGallery(
            figure_id="mfc:9", source_url=PICTURE.format(n=9), image_id=version.image_id
        )
        db_session.add(link)
        db_session.commit()

        dry = reevaluate_masks(db_session)
        assert dry == {
            "dry_run": True,
            "withdrawn": [first.source_sha256],
            "restored": [],
            "unchanged": 1,
        }
        db_session.commit()  # a dry run left nothing behind to commit
        db_session.refresh(first)
        assert first.status == "ok"

        applied = reevaluate_masks(db_session, dry_run=False)
        assert applied["withdrawn"] == [first.source_sha256]
        db_session.rollback()  # ... and an applied one committed its changes
        db_session.refresh(first)
        db_session.refresh(second)
        assert (first.status, first.reason) == ("withdrawn", "user_photo")
        assert (second.status, second.reason) == ("ok", None)
        assert reevaluate_masks(db_session, dry_run=False)["withdrawn"] == []

        import datetime as dt

        link.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        restored = reevaluate_masks(db_session, dry_run=False)
        assert restored["restored"] == [first.source_sha256]
        db_session.refresh(first)
        assert (first.status, first.reason) == ("ok", None)

    def test_a_soft_deleted_image_withdraws_its_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        import datetime as dt

        from app.models import ImageVersion
        from app.workers.mask_job import reevaluate_masks

        first, _ = self._settle_two(db_session, store)
        clean = db_session.get(PublicObject, first.source_sha256)
        assert clean is not None
        version = db_session.get(ImageVersion, clean.image_version_id)
        assert version is not None
        image = db_session.get(ImageModel, version.image_id)
        assert image is not None
        image.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = reevaluate_masks(db_session, dry_run=False)
        assert (report["withdrawn"], report["unchanged"]) == ([first.source_sha256], 1)
        db_session.refresh(first)
        assert (first.status, first.reason) == ("withdrawn", "deleted")

    @pytest.mark.parametrize("error", ["RuntimeError", "IntegrityError"])
    @pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "apply"])
    def test_a_judgment_that_raises_withdraws_that_row_and_the_rest_are_judged(
        self,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
        dry_run: bool,
        error: str,
    ) -> None:
        """Fail closed, one row at a time: a row whose judgment raises is
        withdrawn (reason: the exception's class), and the batch goes on."""
        from app.workers.mask_job import reevaluate_masks

        first, second = self._settle_two(db_session, store)
        real = mask_job.mask_refusal

        def refusal(db: Session, sha256: str, *args: object, **kwargs: object) -> object:
            if sha256 == first.source_sha256:
                if error == "RuntimeError":
                    raise RuntimeError("boom")
                _break_the_session(db)
            return real(db, sha256, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(mask_job, "mask_refusal", refusal)
        report = reevaluate_masks(db_session, dry_run=dry_run)
        assert (report["withdrawn"], report["unchanged"]) == ([first.source_sha256], 1)
        db_session.refresh(first)
        db_session.refresh(second)
        expected = ("ok", None) if dry_run else ("withdrawn", error)
        assert (first.status, first.reason) == expected
        assert (second.status, second.reason) == ("ok", None)

    def test_a_malformed_origin_withdraws_only_its_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.models import ImageVersion
        from app.workers.mask_job import reevaluate_masks

        first, second = self._settle_two(db_session, store)
        clean = db_session.get(PublicObject, first.source_sha256)
        assert clean is not None
        version = db_session.get(ImageVersion, clean.image_version_id)
        assert version is not None
        db_session.add(
            FigureGallery(
                figure_id="shop:bad",
                source_url="http://[evil]/x.jpg",
                image_id=version.image_id,
                source="shop",
            )
        )
        db_session.commit()
        report = reevaluate_masks(db_session, dry_run=False)
        assert (report["withdrawn"], report["unchanged"]) == ([first.source_sha256], 1)
        db_session.refresh(first)
        assert (first.status, first.reason) == ("withdrawn", "unknown_origin")

    def test_held_and_ineligible_rows_are_left_alone(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        from app.workers.mask_job import reevaluate_masks

        _seed(db_session, store, 1)
        _seed(db_session, store, 2, url=PICTURE)
        run_mask_batch(db_session, store, backend=_Constant(255), dry_run=False)
        before = [(r.status, r.reason) for r in _masks(db_session)]
        assert sorted(before) == [("held", "full_frame"), ("ineligible", "user_photo")]
        report = reevaluate_masks(db_session, dry_run=False)
        assert report == {"dry_run": False, "withdrawn": [], "restored": [], "unchanged": 0}
        assert [(r.status, r.reason) for r in _masks(db_session)] == before

    def test_the_task_defaults_to_a_dry_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.workers import tasks

        seen: dict[str, object] = {}

        def _fake(db: object, **kw: object) -> dict[str, object]:
            seen.update(kw)
            return {"dry_run": kw["dry_run"]}

        monkeypatch.setattr(tasks, "reevaluate_masks", _fake)
        monkeypatch.setattr(tasks, "worker_session", _NullSession)
        assert tasks.reevaluate_published_masks() == {"dry_run": True}
        assert tasks.reevaluate_published_masks(apply=True) == {"dry_run": False}


class TestObjectStores:
    def test_local_store_refuses_keys_that_leave_its_root(self, tmp_path: Path) -> None:
        s = LocalObjectStore(tmp_path)
        for key in ("../x", "/etc/passwd", "a/../../x", ""):
            with pytest.raises(ValueError):
                s.put(key, b"x", "image/png")

    def test_local_store_round_trip(self, tmp_path: Path) -> None:
        s = LocalObjectStore(tmp_path)
        s.put("derived/ab/abc.png", b"bytes", "image/png")
        assert s.get("derived/ab/abc.png") == b"bytes"
        assert s.keys() == {"derived/ab/abc.png"}

    def test_an_empty_local_store_has_no_keys(self, tmp_path: Path) -> None:
        assert LocalObjectStore(tmp_path / "missing").keys() == set()

    def test_s3_store_defaults_to_the_configured_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.s3

        sentinel = object()
        monkeypatch.setattr(app.s3, "get_s3", lambda: sentinel)
        s = S3ObjectStore()
        assert s._client is sentinel
        assert s._bucket == "images"

    def test_s3_store_round_trip(self) -> None:
        from moto import mock_aws

        with mock_aws():
            import boto3

            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="images")
            s = S3ObjectStore(client=client, bucket="images")
            s.put("derived/ab/abc.png", b"bytes", "image/png")
            assert s.get("derived/ab/abc.png") == b"bytes"
            head = client.head_object(Bucket="images", Key="derived/ab/abc.png")
            assert head["ContentType"] == "image/png"

    def test_derived_keys_are_content_addressed(self) -> None:
        sha = "ab" + "0" * 62
        assert derived_key(sha, "image/png") == f"derived/ab/{sha}.png"
        assert derived_key(sha, "image/jpeg") == f"derived/ab/{sha}.jpg"
        with pytest.raises(ValueError):
            derived_key("AB" + "0" * 62, "image/png")


class TestCeleryTask:
    def test_the_task_defaults_to_a_dry_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.workers import tasks

        seen: dict[str, object] = {}

        def _fake(db: object, store: object, **kw: object) -> mask_job.BatchReport:
            seen.update(kw)
            return mask_job.BatchReport(batch_id=None, dry_run=True)

        monkeypatch.setattr(tasks, "run_mask_batch", _fake)
        monkeypatch.setattr(tasks, "worker_session", _NullSession)
        monkeypatch.setattr(tasks, "S3ObjectStore", lambda: object())
        out = tasks.generate_masks()
        assert seen["dry_run"] is True
        assert out["dry_run"] is True


class TestCeleryTaskRefusesTheStub:
    def test_an_applied_task_on_the_default_config_refuses_the_stub(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextlib import nullcontext

        from app.workers import tasks

        _seed(db_session, store, 1)
        monkeypatch.setattr(tasks, "worker_session", lambda: nullcontext(db_session))
        monkeypatch.setattr(tasks, "S3ObjectStore", lambda: store)
        with pytest.raises(ValueError, match="dry run only"):
            tasks.generate_masks(apply=True)
        assert _masks(db_session) == []
        assert _public(db_session, "mask") == []


class _NullSession:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *exc: object) -> None:
        return None
