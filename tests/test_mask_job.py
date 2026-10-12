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
from pathlib import Path

import numpy as np
import pytest
from app.models import FigureGallery, ImageMask, MaskBatch, PublicObject
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
        obj.kind = "matted"
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
