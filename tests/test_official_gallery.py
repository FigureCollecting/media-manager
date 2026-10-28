"""GAL3: cut-outs for MFC's official gallery list, decided by PROVENANCE.

The gallery-capture lane records every photo it takes from MFC's official
pictures list as ``manufacturer_press`` (the spine's product_image.source_class);
the work list carries that class, the seed keeps it on the gallery link and
the mask policy reads it (app.workers.mask_policy). A collector's own upload
under the same /upload/pictures/ path carries no such class and stays refused.

``rejudge_official_gallery`` is the one-off step for the photos settled
before the class was carried: it records the class on their links and clears
an ``ineligible`` row only where the full judgment now allows a mask, so the
next mask run mints it. Dry run unless applied; nothing else is touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.cli import masks as masks_cli
from app.config import get_settings
from app.models import (
    FigureGallery,
    ImageMask,
    MaskStoreRevocation,
    PublicObject,
    PublicObjectImage,
)
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.publication import mask_refusal, servable
from app.workers import gallery_rejudge
from app.workers.gallery_rejudge import GalleryCapture, rejudge_official_gallery
from app.workers.mask import MASK_PIPELINE_VERSION
from app.workers.mask_job import reevaluate_masks, revoke_store, run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image
from app.workers.matting import StubMattingBackend
from PIL import Image
from sqlalchemy import event, select
from sqlalchemy.orm import Session

PICTURE = "https://static.myfigurecollection.net/upload/pictures/2026/10/27/{n}.jpeg"
ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PRESS = "manufacturer_press"


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask MODEL (publishes)."""

    name = "stub-model"
    publishes_masks = True


def _photo(n: int) -> bytes:
    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (31 + 10 * n % 200, 30, 90)
    arr[0, 0] = (250, 250, 248 - n % 3)
    arr[1, 0] = (n % 251, 250, 250)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fresh_settings() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


@pytest.fixture()
def setting_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASK_MFC_OFFICIAL_GALLERY", "false")
    get_settings.cache_clear()


def _seed(
    db: Session,
    store: LocalObjectStore,
    n: int,
    *,
    source_class: str | None = PRESS,
    url: str = PICTURE,
    figure: str | None = None,
    photo: int | None = None,
) -> PublicObject:
    return seed_clean_image(
        db,
        store,
        _photo(n if photo is None else photo),
        origins=[ImageOrigin(url=url.format(n=n), source="mfc", source_class=source_class)],
        figure_id=figure or f"p{n}",
    )


def _link(db: Session, n: int, url: str = PICTURE) -> FigureGallery:
    return db.execute(
        select(FigureGallery).where(FigureGallery.source_url == url.format(n=n))
    ).scalar_one()


def _row(db: Session, sha: str) -> ImageMask | None:
    return db.execute(
        select(ImageMask).where(
            ImageMask.source_sha256 == sha, ImageMask.pipeline_version == MASK_PIPELINE_VERSION
        )
    ).scalar_one_or_none()


def _apply(db: Session, store: LocalObjectStore, **kw: object) -> dict[str, str]:
    report = run_mask_batch(db, store, backend=_Model(), dry_run=False, **kw)  # type: ignore[arg-type]
    return {i.source_sha256: f"{i.status}:{i.reason}:{i.origin_class}" for i in report.items}


def _original(n: int) -> str:
    import hashlib

    return hashlib.sha256(_photo(n)).hexdigest()


def _capture(n: int, *, source_class: str | None = PRESS, **kw: str) -> GalleryCapture:
    return GalleryCapture(
        original_sha256=kw.get("original", _original(n)),
        figure_id=kw.get("figure", f"p{n}"),
        url=kw.get("url", PICTURE.format(n=n)),
        source=kw.get("source", "mfc"),
        source_class=source_class,
    )


# --------------------------------------------------------------------------
# the provenance, from the work list to the link
# --------------------------------------------------------------------------


class TestSeedKeepsTheProvenance:
    def test_a_new_link_records_the_class(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1)
        assert _link(db_session, 1).source_class == PRESS

    def test_a_link_without_a_class_gains_one_from_a_later_capture(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _seed(db_session, store, 1, source_class=None)
        assert _link(db_session, 1).source_class is None
        _seed(db_session, store, 1)
        assert _link(db_session, 1).source_class == PRESS

    @pytest.mark.parametrize("later", [None, "unknown"])
    def test_a_recorded_class_is_never_overwritten(
        self, db_session: Session, store: LocalObjectStore, later: str | None
    ) -> None:
        _seed(db_session, store, 1)
        _seed(db_session, store, 1, source_class=later)
        assert _link(db_session, 1).source_class == PRESS

    def test_the_judgment_reads_the_class_from_the_link(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        gallery = _seed(db_session, store, 1)
        collector = _seed(db_session, store, 2, source_class=None)
        assert mask_refusal(db_session, gallery.sha256) is None
        assert mask_refusal(db_session, collector.sha256) == "user_photo"


# --------------------------------------------------------------------------
# the mask lane
# --------------------------------------------------------------------------


class TestTheMaskLane:
    def test_a_gallery_list_photo_gets_a_published_mask_by_default(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        assert _apply(db_session, store) == {obj.sha256: "ok:None:official_gallery"}
        mask = db_session.execute(
            select(PublicObject).where(PublicObject.kind == "mask")
        ).scalar_one()
        assert servable(db_session, mask)

    def test_with_the_setting_off_it_is_refused_as_before(
        self, db_session: Session, store: LocalObjectStore, setting_off: None
    ) -> None:
        obj = _seed(db_session, store, 1)
        assert _apply(db_session, store) == {obj.sha256: "ineligible:user_photo:user_photo"}

    def test_a_collectors_own_upload_is_still_refused_with_the_setting_on(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1, source_class=None)
        assert _apply(db_session, store) == {obj.sha256: "ineligible:user_photo:user_photo"}

    def test_turning_the_setting_off_withdraws_the_published_masks(
        self, db_session: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _seed(db_session, store, 1)
        _apply(db_session, store)
        mask = db_session.execute(
            select(PublicObject).where(PublicObject.kind == "mask")
        ).scalar_one()
        monkeypatch.setenv("MASK_MFC_OFFICIAL_GALLERY", "false")
        get_settings.cache_clear()
        assert not servable(db_session, mask)  # the origin refuses it at once
        report = reevaluate_masks(db_session, dry_run=False)
        assert report["withdrawn"] == [obj.sha256]
        assert _row(db_session, obj.sha256).status == "withdrawn"  # type: ignore[union-attr]

    def test_revoking_mfc_withdraws_a_gallery_mask_and_refuses_new_ones(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        first = _seed(db_session, store, 1)
        _apply(db_session, store)
        report = revoke_store(db_session, "mfc", dry_run=False)
        assert report["reevaluate"]["withdrawn"] == [first.sha256]
        mask = db_session.execute(
            select(PublicObject).where(PublicObject.kind == "mask")
        ).scalar_one()
        assert not servable(db_session, mask)
        second = _seed(db_session, store, 2)
        assert _apply(db_session, store) == {
            second.sha256: "ineligible:store_revoked:store_revoked"
        }

    def test_a_soft_deleted_gallery_photo_loses_its_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        _apply(db_session, store)
        image = db_session.execute(select(ImageModel)).scalar_one()
        image.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        mask = db_session.execute(
            select(PublicObject).where(PublicObject.kind == "mask")
        ).scalar_one()
        assert not servable(db_session, mask)
        assert reevaluate_masks(db_session, dry_run=False)["withdrawn"] == [obj.sha256]
        assert _row(db_session, obj.sha256).reason == "deleted"  # type: ignore[union-attr]

    def test_a_deleted_gallery_link_leaves_no_origin(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1)
        _link(db_session, 1).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        assert mask_refusal(db_session, obj.sha256) == "no_origin"


# --------------------------------------------------------------------------
# re-judging the photos settled before the class was carried
# --------------------------------------------------------------------------


def _settled_refused(
    db: Session, store: LocalObjectStore, n: int, source_class: str | None = None
) -> PublicObject:
    """A gallery-list photo as the 10-27 capture left it: seeded without its
    class, judged a collector's photo, its ineligible row recorded."""
    obj = _seed(db, store, n, source_class=source_class)
    _apply(db, store)
    assert _row(db, obj.sha256).status == "ineligible"  # type: ignore[union-attr]
    return obj


class TestRejudge:
    def test_a_dry_run_reports_the_requeue_and_writes_nothing(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        report = rejudge_official_gallery(db_session, [_capture(1)])
        assert report["dry_run"] is True and report["official_gallery"] is True
        assert report["requeued"] == [obj.sha256]
        assert report["counts"]["provenance_recorded"] == 1
        assert _link(db_session, 1).source_class is None
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_applied_it_records_the_class_and_the_next_run_mints_the_mask(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["requeued"] == [obj.sha256]
        assert _link(db_session, 1).source_class == PRESS
        assert _row(db_session, obj.sha256) is None  # a candidate again
        assert _apply(db_session, store) == {obj.sha256: "ok:None:official_gallery"}
        again = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert again["requeued"] == []
        assert again["counts"]["settled"] == 1 and again["counts"]["provenance_recorded"] == 0

    @pytest.mark.parametrize(
        ("source_class", "source"),
        [(None, "mfc"), ("unknown", "mfc"), ("user_photo", "mfc"), (PRESS, "hpoi")],
    )
    def test_only_gallery_list_captures_are_considered(
        self,
        db_session: Session,
        store: LocalObjectStore,
        source_class: str | None,
        source: str,
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        capture = _capture(1, source_class=source_class, source=source)
        report = rejudge_official_gallery(db_session, [capture], dry_run=False)
        assert report["counts"]["not_gallery"] == 1 and report["requeued"] == []
        assert _link(db_session, 1).source_class is None
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_collectors_photo_not_on_the_list_is_left_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        gallery = _settled_refused(db_session, store, 1)
        collector = _settled_refused(db_session, store, 2)
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["requeued"] == [gallery.sha256]
        assert _link(db_session, 2).source_class is None
        assert _row(db_session, collector.sha256).status == "ineligible"  # type: ignore[union-attr]
        assert _apply(db_session, store) == {gallery.sha256: "ok:None:official_gallery"}

    def test_the_same_picture_as_a_collectors_photo_elsewhere_stays_refused(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        other = PICTURE.replace("{n}", "{n}-elsewhere")
        _seed(db_session, store, 1, source_class=None, url=other, figure="elsewhere", photo=1)
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["requeued"] == []
        assert report["still_refused"] == [{"clean_sha256": obj.sha256, "reason": "user_photo"}]
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_every_listed_capture_of_one_picture_counts_in_a_dry_run_too(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """Both links get the class before the picture is judged, once."""
        obj = _settled_refused(db_session, store, 1)
        other = PICTURE.replace("{n}", "{n}-second")
        _seed(db_session, store, 1, source_class=None, url=other, figure="second", photo=1)
        captures = [_capture(1), _capture(1, figure="second", url=other.format(n=1)), _capture(1)]
        dry = rejudge_official_gallery(db_session, captures)
        assert (dry["requeued"], dry["still_refused"]) == ([obj.sha256], [])
        counts = dry["counts"]
        assert (counts["provenance_recorded"], counts["provenance_present"]) == (2, 1)
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]
        applied = rejudge_official_gallery(db_session, captures, dry_run=False)
        assert (applied["requeued"], applied["counts"]) == (dry["requeued"], dry["counts"])
        assert _row(db_session, obj.sha256) is None

    def test_a_revoked_mfc_is_honoured(self, db_session: Session, store: LocalObjectStore) -> None:
        obj = _settled_refused(db_session, store, 1)
        db_session.add(MaskStoreRevocation(source="mfc"))
        db_session.commit()
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["requeued"] == []
        assert report["still_refused"] == [{"clean_sha256": obj.sha256, "reason": "store_revoked"}]
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_soft_deleted_photo_is_honoured(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        image = db_session.execute(select(ImageModel)).scalar_one()
        image.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["still_refused"] == [{"clean_sha256": obj.sha256, "reason": "deleted"}]
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_withdrawn_mask_is_never_touched(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A withdrawal is the record behind the spine's tombstone: only
        reevaluate_masks or restore-store may change it."""
        obj = _seed(db_session, store, 1, source_class=None)
        _link(db_session, 1).source_class = PRESS
        db_session.commit()
        _apply(db_session, store)
        row = _row(db_session, obj.sha256)
        assert row is not None
        row.status, row.reason = "withdrawn", "store_revoked"
        db_session.commit()
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["counts"]["settled"] == 1 and report["requeued"] == []
        assert (row.status, row.reason) == ("withdrawn", "store_revoked")

    def test_the_setting_off_requeues_nothing_but_keeps_the_class(
        self, db_session: Session, store: LocalObjectStore, setting_off: None
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["official_gallery"] is False
        assert report["requeued"] == []
        assert report["still_refused"] == [{"clean_sha256": obj.sha256, "reason": "user_photo"}]
        assert _link(db_session, 1).source_class == PRESS

    def test_what_is_not_there_is_named_and_skipped(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        _settled_refused(db_session, store, 1)
        deleted = _settled_refused(db_session, store, 3)
        _link(db_session, 3).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        report = rejudge_official_gallery(
            db_session,
            [
                _capture(2),  # never seeded: the next mask run seeds it with its class
                _capture(1, figure="not-linked"),
                _capture(3),
            ],
            dry_run=False,
        )
        counts = report["counts"]
        assert (counts["not_seeded"], counts["no_link"], counts["link_deleted"]) == (1, 1, 1)
        assert report["requeued"] == []
        assert _link(db_session, 3).source_class is None
        assert _row(db_session, deleted.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_conflicting_class_is_kept_and_not_requeued(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1, source_class="retailer_studio")
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["counts"]["provenance_conflict"] == 1 and report["requeued"] == []
        assert _link(db_session, 1).source_class == "retailer_studio"
        assert _row(db_session, obj.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_copy_with_no_result_yet_is_left_to_the_next_run(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _seed(db_session, store, 1, source_class=None)
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["counts"]["unsettled"] == 1 and report["requeued"] == []
        assert _link(db_session, 1).source_class == PRESS
        assert _apply(db_session, store) == {obj.sha256: "ok:None:official_gallery"}

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_a_judgment_that_raises_is_that_pictures_failure(
        self,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
        dry_run: bool,
    ) -> None:
        first = _settled_refused(db_session, store, 1)
        second = _settled_refused(db_session, store, 2)
        real = gallery_rejudge.mask_refusal

        def flaky(db: Session, sha: str, **kw: Any) -> str | None:
            if sha == first.sha256:
                raise RuntimeError("s3://bucket/secret-key")
            return real(db, sha, **kw)

        monkeypatch.setattr(gallery_rejudge, "mask_refusal", flaky)
        report = rejudge_official_gallery(db_session, [_capture(1), _capture(2)], dry_run=dry_run)
        assert report["failed"] == [{"clean_sha256": first.sha256, "reason": "RuntimeError"}]
        assert report["requeued"] == [second.sha256]  # judged with the class, after the rollback
        assert _row(db_session, first.sha256).status == "ineligible"  # type: ignore[union-attr]
        expected = None if dry_run else PRESS  # the class is a fact; a rerun re-judges
        assert _link(db_session, 1).source_class == expected
        assert (_row(db_session, second.sha256) is None) is not dry_run

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_a_capture_that_raises_before_its_write_is_its_own_failure(
        self,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
        dry_run: bool,
    ) -> None:
        _settled_refused(db_session, store, 1)
        second = _settled_refused(db_session, store, 2)
        real = gallery_rejudge._copies_behind
        broken = db_session.execute(
            select(ImageModel.id).where(ImageModel.sha256 == _original(1))
        ).scalar_one()

        def flaky(db: Session, image_id: int) -> list[str]:
            if image_id == broken:
                raise RuntimeError("s3://bucket/secret-key")
            return real(db, image_id)

        monkeypatch.setattr(gallery_rejudge, "_copies_behind", flaky)
        report = rejudge_official_gallery(db_session, [_capture(1), _capture(2)], dry_run=dry_run)
        assert report["failed"] == [{"original_sha256": _original(1), "reason": "RuntimeError"}]
        assert report["requeued"] == [second.sha256]
        assert _link(db_session, 1).source_class is None  # it raised before the class was set
        assert _link(db_session, 2).source_class == (None if dry_run else PRESS)

    def test_a_capture_whose_write_fails_is_rolled_back_alone(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The class is set, then its commit's flush fails: that capture is
        rolled back and named, and the next capture is still recorded."""
        first = _settled_refused(db_session, store, 1)
        second = _settled_refused(db_session, store, 2)
        broken = _link(db_session, 1).id
        fired: list[int] = []

        def refuse(_mapper: object, _connection: object, target: FigureGallery) -> None:
            if target.id == broken:
                fired.append(target.id)
                raise RuntimeError("s3://bucket/secret-key")

        event.listen(FigureGallery, "before_update", refuse)
        try:
            report = rejudge_official_gallery(db_session, [_capture(1), _capture(2)], dry_run=False)
        finally:
            event.remove(FigureGallery, "before_update", refuse)
        assert fired == [broken]
        assert report["failed"] == [{"original_sha256": _original(1), "reason": "RuntimeError"}]
        assert report["requeued"] == [second.sha256]
        assert _link(db_session, 1).source_class is None  # rolled back
        assert _link(db_session, 2).source_class == PRESS
        assert _row(db_session, first.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_link_at_that_url_to_other_bytes_is_not_recorded(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The URL's link names other bytes (the picture there changed): the
        class is not stamped on bytes the lane never reported there."""
        _settled_refused(db_session, store, 1)
        other = _settled_refused(db_session, store, 2)
        report = rejudge_official_gallery(
            db_session, [_capture(1, original=_original(2))], dry_run=False
        )
        assert report["counts"]["no_link"] == 1 and report["requeued"] == []
        assert {_link(db_session, n).source_class for n in (1, 2)} == {None}
        assert _row(db_session, other.sha256).status == "ineligible"  # type: ignore[union-attr]

    def test_a_link_at_that_url_from_another_source_is_not_recorded(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        url = PICTURE.format(n=1)
        seed_clean_image(
            db_session, store, _photo(1), origins=[ImageOrigin(url=url, source="hpoi")],
            figure_id="p1",
        )  # fmt: skip
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["counts"]["no_link"] == 1
        assert _link(db_session, 1).source_class is None

    def test_only_the_clean_copies_behind_the_image_are_judged(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A mask object whose very bytes were captured sits behind that
        capture too; it is no clean copy and has no result to clear."""
        obj = _settled_refused(db_session, store, 1)
        image_id = db_session.execute(
            select(ImageModel.id).where(ImageModel.sha256 == _original(1))
        ).scalar_one()
        mask = "f" * 64
        db_session.add(
            PublicObject(
                sha256=mask, kind="mask", mime="image/png", bytes=1, width=1, height=1,
                storage_key=f"masks/{mask}.png",
            )
        )  # fmt: skip
        db_session.flush()
        db_session.add(PublicObjectImage(sha256=mask, image_id=image_id))
        db_session.commit()
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert report["requeued"] == [obj.sha256]
        assert (report["counts"]["unsettled"], report["counts"]["settled"]) == (0, 0)

    def test_a_result_from_another_pipeline_version_is_left_alone(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        obj = _settled_refused(db_session, store, 1)
        db_session.add(
            ImageMask(
                source_sha256=obj.sha256, pipeline_version="older", status="ineligible",
                reason="user_photo", origin_class="user_photo",
            )
        )  # fmt: skip
        db_session.commit()
        report = rejudge_official_gallery(db_session, [_capture(1)], dry_run=False)
        assert (report["requeued"], report["failed"]) == ([obj.sha256], [])
        older = db_session.execute(
            select(ImageMask.status).where(ImageMask.pipeline_version == "older")
        ).scalar_one()
        assert older == "ineligible"

    def test_a_dry_run_writes_nothing_so_it_holds_no_row_locks(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """A mask batch or a seed running beside a dry run never waits on it:
        the classes it would record are held in memory, not written."""
        first = _settled_refused(db_session, store, 1)
        second = _settled_refused(db_session, store, 2)
        other = PICTURE.replace("{n}", "{n}-second")
        _seed(db_session, store, 1, source_class=None, url=other, figure="second", photo=1)
        writes: list[str] = []

        def watch(_c: object, _cur: object, statement: str, *_rest: object) -> None:
            if statement.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                writes.append(statement)

        captures = [_capture(1), _capture(1, figure="second", url=other.format(n=1)), _capture(2)]
        engine = db_session.get_bind()
        event.listen(engine, "before_cursor_execute", watch)
        try:
            report = rejudge_official_gallery(db_session, captures)
        finally:
            event.remove(engine, "before_cursor_execute", watch)
        assert writes == []
        assert sorted(report["requeued"]) == sorted([first.sha256, second.sha256])
        assert report["counts"]["provenance_recorded"] == 3


# --------------------------------------------------------------------------
# the command: every work-list row is one gallery link
# --------------------------------------------------------------------------


class TestTheCommand:
    @staticmethod
    def _line(n: int, figure: str, url: str) -> str:
        sha = _original(n)
        return json.dumps(
            {
                "blob_sha256": sha, "storage_key": f"raw-img/sha256/{sha[:2]}/{sha}.png",
                "url": url, "source": "mfc", "product_id": figure, "rank": 9,
                "source_class": PRESS,
            }
        )  # fmt: skip

    @pytest.mark.parametrize("apply", [False, True])
    def test_every_listed_link_of_one_picture_gains_the_class(
        self, db_session: Session, store: LocalObjectStore, tmp_path: Path, apply: bool
    ) -> None:
        """One official picture listed on two products: both rows reach the
        re-judge, else the second link (no class) keeps the copy refused."""
        obj = _settled_refused(db_session, store, 1)
        other = PICTURE.replace("{n}", "{n}-second").format(n=1)
        _seed(db_session, store, 1, source_class=None, url=other, figure="second", photo=1)
        worklist = tmp_path / "gallery.jsonl"
        lines = [self._line(1, "p1", PICTURE.format(n=1)), self._line(1, "second", other)]
        worklist.write_text("\n".join(lines) + "\n")
        args = argparse.Namespace(worklist=[str(worklist)], apply=apply)
        report = masks_cli._official_gallery(db_session, args)
        assert (report["requeued"], report["still_refused"]) == ([obj.sha256], [])
        assert report["counts"]["provenance_recorded"] == 2
        assert (_row(db_session, obj.sha256) is None) is apply
