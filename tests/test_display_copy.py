"""SHOW1: the display-copy lane's SEEDER (app.workers.display_copy).

Per queued original: claim it (the attempt committed before any decode),
read its bytes from the raw bucket (read only, never from the store it was
captured at), check them, seed exactly as the mask batch seeds (original
private, clean copy public, a gallery link per depiction), and record the
copy the replay would report (app.spine.copies.reported_copy) with its own
appearance, for the reporter. FAILED is only ever the image's own fault
(a hash mismatch, a broken limit, an undecodable image); everything else
leaves no FAILED behind."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import socket
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.models import (
    DisplayDepiction,
    DisplayOriginal,
    FigureGallery,
    ImageMask,
    ImageVersion,
    PublicObject,
)
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.publication import mask_refusal, servable
from app.spine import copies
from app.spine.replay import load_plan
from app.workers import display_copy as dc
from app.workers import mask as mask_artifact
from app.workers.mask_job import open_mask_batch, settle_clean_copy
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image, seed_with_retry
from app.workers.matting import StubMattingBackend
from app.workers.raw_source import ReadOnlyStore
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

T0 = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)
MFC_ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
STORE = "https://img.amiami.example/images/product/main/{n}.jpg"
GRANTS = {"amiami": ["amiami.example"]}


def _photo(n: int, *, alpha: bool = False) -> bytes:
    arr = np.full((60, 40, 4 if alpha else 3), 250, dtype=np.uint8)
    arr[10:50, 12:28, :3] = (31 + 10 * n % 200, 30, 90)
    arr[0, 0, 0] = n % 251
    if alpha:
        arr[..., 3] = 255
        arr[:5, :, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA" if alpha else "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _pid(n: int) -> str:
    return str(uuid.UUID(int=n))


class _Raw:
    """A raw bucket on disk, keyed the sha256-v1 way."""

    def __init__(self, root: Path) -> None:
        self.store = LocalObjectStore(root)

    def put(self, data: bytes, *, codec: str = "identity") -> tuple[str, str]:
        sha = hashlib.sha256(data).hexdigest()
        key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
        self.store.put(key, gzip.compress(data) if codec == "gzip" else data, "image/png")
        return sha, key


def _queue(
    db: Session,
    sha: str,
    key: str,
    *,
    depictions: list[tuple[str, str, str, str]],
    codec: str = "identity",
    seen: dt.datetime = T0,
    source_class: str | None = None,
) -> DisplayOriginal:
    """One queued original and its depictions: (product_image_id, product_id, url, site)."""
    original = DisplayOriginal(
        origin_sha256=sha, storage_key=key, codec=codec, first_seen_at=seen, state="pending"
    )
    db.add(original)
    db.flush()
    for pid, product, url, site in depictions:
        db.add(
            DisplayDepiction(
                product_image_id=pid,
                origin_sha256=sha,
                product_id=product,
                origin_url=url,
                source_site=site,
                source_class=source_class,
                first_seen_at=seen,
            )
        )
    db.commit()
    return original


@pytest.fixture()
def lane(db_session: Session, tmp_path: Path) -> dict[str, Any]:
    return {
        "db": db_session,
        "raw": _Raw(tmp_path / "raw"),
        "media": LocalObjectStore(tmp_path / "media"),
    }


def _seed_all(
    lane: dict[str, Any], *, at: dt.datetime = T0, **config: Any
) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: lane["db"],
        ReadOnlyStore(lane["raw"].store),
        lane["media"],
        dc.SeedConfig(threads=0, grants=GRANTS, **config),
        emit=lines.append,
        now=lambda: at,
    )
    return code, lines


def _done(lines: list[dict[str, Any]]) -> dict[str, Any]:
    return next(line for line in lines if line["event"] == "done")


# --------------------------------------------------------------------------
# the seed
# --------------------------------------------------------------------------


def test_one_depiction_seeds_a_private_original_a_public_copy_and_its_gallery_link(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    data = _photo(1)
    sha, key = lane["raw"].put(data)
    _queue(db, sha, key, depictions=[(_pid(1), _pid(101), MFC_ITEM.format(n=1), "mfc")])

    code, lines = _seed_all(lane)

    assert code == dc.EXIT_DONE
    assert _done(lines)["seeded"] == 1
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == sha)).scalar_one()
    versions = {v.version_no: v for v in db.execute(select(ImageVersion)).scalars()}
    assert versions[1].visibility == "private" and versions[2].visibility == "public"
    link = db.execute(select(FigureGallery)).scalar_one()
    assert (link.figure_id, link.source_url, link.source, link.image_id) == (
        _pid(101),
        MFC_ITEM.format(n=1),
        "mfc",
        image.id,
    )
    original = db.get(DisplayOriginal, sha)
    assert original is not None and original.state == "seeded" and original.attempts == 0
    # the copy the replay would report, by its own plan
    assert original.copy_sha256 == load_plan(db).copy_of[sha] == copies.reported_copy(db, sha)
    obj = db.get(PublicObject, original.copy_sha256)
    assert obj is not None and obj.kind == "sanitized"
    assert (original.copy_mime, original.copy_width, original.copy_height) == (
        obj.mime,
        obj.width,
        obj.height,
    )
    # its own appearance, exactly as the mask lane measures it unmasked
    clean = lane["media"].get(obj.storage_key)
    unmasked = mask_artifact.build_mask_artifact(clean, backend=StubMattingBackend()).unmasked
    assert (original.thumbhash, original.dominant_color) == (
        unmasked.thumbhash,
        unmasked.dominant_color,
    )
    assert original.own_grounding is None
    depiction = db.get(DisplayDepiction, _pid(1))
    assert depiction is not None and depiction.linked_at is not None
    assert depiction.reported_at is None  # the reporter's, never the seeder's


def test_a_gzip_original_is_inflated_and_a_transparent_copy_carries_its_own_grounding(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    data = _photo(2, alpha=True)
    sha, key = lane["raw"].put(data, codec="gzip")
    _queue(
        db, sha, key, codec="gzip", depictions=[(_pid(2), _pid(102), STORE.format(n=2), "amiami")]
    )
    code, _ = _seed_all(lane)
    assert code == dc.EXIT_DONE
    original = db.get(DisplayOriginal, sha)
    assert original is not None and original.state == "seeded"
    clean = lane["media"].get(db.get(PublicObject, original.copy_sha256).storage_key)
    outcome = mask_artifact.build_mask_artifact(clean, backend=StubMattingBackend())
    assert outcome.own_alpha_grounding is not None
    assert original.own_grounding == dc.grounding_dict(outcome.own_alpha_grounding)
    assert original.thumbhash == outcome.unmasked.thumbhash


def test_two_products_depicting_one_original_make_one_image_and_two_links(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(3))
    _queue(
        db,
        sha,
        key,
        depictions=[
            (_pid(3), _pid(103), STORE.format(n=3), "amiami"),
            (_pid(4), _pid(104), STORE.format(n=3), "amiami"),
        ],
    )
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.scalar(select(func.count()).select_from(ImageModel)) == 1
    links = db.execute(select(FigureGallery).order_by(FigureGallery.figure_id)).scalars().all()
    assert [g.figure_id for g in links] == [_pid(103), _pid(104)]
    assert {d.linked_at is not None for d in db.execute(select(DisplayDepiction)).scalars()} == {
        True
    }


def test_a_capture_of_a_picture_that_already_has_a_copy_mints_no_new_copy(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    first = _photo(5)
    seed_clean_image(
        db,
        lane["media"],
        first,
        origins=[ImageOrigin(url=MFC_ITEM.format(n=5), source="mfc")],
        figure_id=_pid(105),
    )
    existing = {o.sha256 for o in db.execute(select(PublicObject)).scalars()}
    # the same pixels in other bytes: a PNG re-save at another compression level
    with Image.open(io.BytesIO(first)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    again = buf.getvalue()
    assert again != first
    sha, key = lane["raw"].put(again)
    _queue(db, sha, key, depictions=[(_pid(5), _pid(106), STORE.format(n=5), "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert {o.sha256 for o in db.execute(select(PublicObject)).scalars()} == existing
    original = db.get(DisplayOriginal, sha)
    assert original is not None and original.copy_sha256 in existing
    assert original.copy_sha256 == load_plan(db).copy_of[sha]


def test_a_new_depiction_of_a_seeded_original_links_it_without_a_second_copy(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(6))
    _queue(db, sha, key, depictions=[(_pid(6), _pid(106), STORE.format(n=6), "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    objects = db.scalar(select(func.count()).select_from(PublicObject))
    original = db.get(DisplayOriginal, sha)
    assert original is not None
    original.state = "pending"  # what the feed does when a new depiction arrives
    db.add(
        DisplayDepiction(
            product_image_id=_pid(7),
            origin_sha256=sha,
            product_id=_pid(107),
            origin_url=STORE.format(n=6),
            source_site="amiami",
            first_seen_at=T0,
        )
    )
    db.commit()
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.scalar(select(func.count()).select_from(PublicObject)) == objects
    assert {g.figure_id for g in db.execute(select(FigureGallery)).scalars()} == {
        _pid(106),
        _pid(107),
    }


# --------------------------------------------------------------------------
# beside the mask batch
# --------------------------------------------------------------------------


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask MODEL (publishes)."""

    name = "stub-model"
    publishes_masks = True


def _mask(db: Session, media: LocalObjectStore, copy: str) -> Any:
    backend = _Model()
    batch = open_mask_batch(db, backend_name=backend.name)
    return settle_clean_copy(db, media, copy, backend=backend, batch=batch)


def test_lane_then_batch_the_mask_lands_on_the_lanes_copy_and_nothing_is_duplicated(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    data = _photo(8)
    sha, key = lane["raw"].put(data)
    _queue(db, sha, key, depictions=[(_pid(8), _pid(108), MFC_ITEM.format(n=8), "mfc")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    lane_copy = db.get(DisplayOriginal, sha).copy_sha256
    counts = {
        t: db.scalar(select(func.count()).select_from(t))
        for t in (ImageModel, ImageVersion, PublicObject, FigureGallery)
    }
    # the batch's own seed, then its settle, exactly as _mint calls them
    again = seed_clean_image(
        db,
        lane["media"],
        data,
        origins=[ImageOrigin(url=MFC_ITEM.format(n=8), source="mfc")],
        figure_id=_pid(108),
    )
    assert again.sha256 == lane_copy
    assert {
        t: db.scalar(select(func.count()).select_from(t))
        for t in (ImageModel, ImageVersion, PublicObject, FigureGallery)
    } == counts
    settled = _mask(db, lane["media"], lane_copy)
    assert settled is not None
    assert db.execute(select(ImageMask.source_sha256)).scalars().all() == [lane_copy]


def test_batch_then_lane_the_lane_only_records_the_batchs_copy_and_its_measured_look(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    data = _photo(9)
    clean = seed_clean_image(
        db,
        lane["media"],
        data,
        origins=[ImageOrigin(url=MFC_ITEM.format(n=9), source="mfc")],
        figure_id=_pid(109),
    )
    _mask(db, lane["media"], clean.sha256)
    row = db.execute(select(ImageMask)).scalar_one()
    counts = {t: db.scalar(select(func.count()).select_from(t)) for t in (ImageModel, PublicObject)}
    sha, key = lane["raw"].put(data)
    _queue(db, sha, key, depictions=[(_pid(9), _pid(109), MFC_ITEM.format(n=9), "mfc")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert {
        t: db.scalar(select(func.count()).select_from(t)) for t in (ImageModel, PublicObject)
    } == counts
    original = db.get(DisplayOriginal, sha)
    assert original.copy_sha256 == clean.sha256
    # the appearance the replay sends: the mask row's plain values
    assert (original.thumbhash, original.dominant_color) == (
        row.plain_thumbhash,
        row.plain_dominant_color,
    )


def test_a_published_mask_stays_servable_when_the_lane_seeds_a_granted_capture_of_its_picture(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: the lane always writes the gallery link, so a published mask is
    never judged no_origin because the lane seeded another capture."""
    monkeypatch.setattr("app.workers.mask_policy.configured_store_grants", lambda: GRANTS)
    db = lane["db"]
    data = _photo(10)
    clean = seed_clean_image(
        db,
        lane["media"],
        data,
        origins=[ImageOrigin(url=MFC_ITEM.format(n=10), source="mfc")],
        figure_id=_pid(110),
    )
    _mask(db, lane["media"], clean.sha256)
    mask = db.execute(select(ImageMask)).scalar_one()
    assert mask.status == "ok" and mask_refusal(db, clean.sha256, GRANTS, frozenset()) is None
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    sha, key = lane["raw"].put(buf.getvalue())
    _queue(db, sha, key, depictions=[(_pid(10), _pid(111), STORE.format(n=10), "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.get(DisplayOriginal, sha).state == "seeded"
    assert mask_refusal(db, clean.sha256, GRANTS, frozenset()) is None
    assert servable(db, db.get(PublicObject, mask.mask_sha256), GRANTS, frozenset())


def test_a_capture_that_would_withdraw_a_published_mask_is_held_not_seeded(
    lane: dict[str, Any],
) -> None:
    """The critique's case: a store outside the mask grants. Seeding it would
    turn the pooled verdict to not_granted and take the published mask
    offline at the origin while the spine still advertises it, so the lane
    HOLDS it (state held, never reported) until an operator forces it."""
    db = lane["db"]
    data = _photo(11)
    clean = seed_clean_image(
        db,
        lane["media"],
        data,
        origins=[ImageOrigin(url=MFC_ITEM.format(n=11), source="mfc")],
        figure_id=_pid(112),
    )
    _mask(db, lane["media"], clean.sha256)
    mask = db.execute(select(ImageMask)).scalar_one()
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    sha, key = lane["raw"].put(buf.getvalue())
    ungranted = "https://cdn.unlisted.example/p/11.jpg"
    _queue(db, sha, key, depictions=[(_pid(11), _pid(113), ungranted, "unlisted")])

    code, lines = _seed_all(lane)

    assert code == dc.EXIT_DONE
    original = db.get(DisplayOriginal, sha)
    assert (original.state, original.reason) == ("held", "would_withdraw_mask:not_granted")
    assert _done(lines)["held"] == 1
    assert (
        db.execute(select(ImageModel).where(ImageModel.sha256 == sha)).scalar_one_or_none() is None
    )
    assert mask_refusal(db, clean.sha256, GRANTS, frozenset()) is None
    assert servable(db, db.get(PublicObject, mask.mask_sha256), GRANTS, frozenset())
    # forced, it seeds -- and the pooled policy then takes the mask offline, as designed
    assert dc.requeue(db, state="held", reason_prefix=None, force=True, apply=True) == 1
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.get(DisplayOriginal, sha).state == "seeded"
    assert mask_refusal(db, clean.sha256, GRANTS, frozenset()) == "not_granted"


def test_an_ungranted_capture_of_a_picture_with_no_mask_is_seeded(lane: dict[str, Any]) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(12))
    _queue(
        db, sha, key, depictions=[(_pid(12), _pid(114), "https://cdn.unlisted.example/12.jpg", "x")]
    )
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.get(DisplayOriginal, sha).state == "seeded"


def test_the_same_url_with_new_bytes_gets_no_second_gallery_link(lane: dict[str, Any]) -> None:
    """Existing behaviour (uq_figure_source_url), recorded: a product's URL
    re-captured with new bytes keeps its first link, so the new image has no
    origin of its own and its copy is no_origin for a mask."""
    db = lane["db"]
    url = STORE.format(n=13)
    a, ka = lane["raw"].put(_photo(13))
    _queue(db, a, ka, depictions=[(_pid(13), _pid(115), url, "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    b, kb = lane["raw"].put(_photo(14))
    _queue(db, b, kb, depictions=[(_pid(14), _pid(115), url, "amiami")], seen=T0 - dt.timedelta(1))
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.scalar(select(func.count()).select_from(FigureGallery)) == 1
    copy_b = db.get(DisplayOriginal, b).copy_sha256
    assert mask_refusal(db, copy_b, GRANTS, frozenset()) == "no_origin"


# --------------------------------------------------------------------------
# faults: FAILED only for the image's own
# --------------------------------------------------------------------------


def test_a_hash_mismatch_is_the_images_own_fault(lane: dict[str, Any]) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(15))
    lane["raw"].store.put(key, _photo(16), "image/png")  # the object is not what the spine recorded
    _queue(db, sha, key, depictions=[(_pid(15), _pid(116), STORE.format(n=15), "amiami")])
    code, lines = _seed_all(lane)
    assert code == dc.EXIT_DONE
    assert (db.get(DisplayOriginal, sha).state, db.get(DisplayOriginal, sha).reason) == (
        "failed",
        "raw_hash_mismatch",
    )
    assert _done(lines)["failed"] == 1


def test_an_undecodable_original_and_a_broken_limit_are_the_images_own(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = lane["db"]
    junk = b"not an image at all" * 10
    a, ka = lane["raw"].put(junk)
    _queue(db, a, ka, depictions=[(_pid(17), _pid(117), STORE.format(n=17), "amiami")])
    big = _photo(18)
    b, kb = lane["raw"].put(big)
    _queue(db, b, kb, depictions=[(_pid(18), _pid(118), STORE.format(n=18), "amiami")])
    monkeypatch.setenv("MM_MAX_IMAGE_PIXELS", "100")
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert (db.get(DisplayOriginal, a).state, db.get(DisplayOriginal, a).reason) == (
        "failed",
        "undecodable",
    )
    assert db.get(DisplayOriginal, b).state == "failed"
    assert db.get(DisplayOriginal, b).reason.startswith("refused:")


def test_a_missing_raw_object_is_not_failed_and_gives_up_as_stuck_after_three_runs(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha = hashlib.sha256(b"never stored").hexdigest()
    key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
    _queue(db, sha, key, depictions=[(_pid(19), _pid(119), STORE.format(n=19), "amiami")])
    for run, expected in enumerate(("pending", "pending", "stuck")):
        code, lines = _seed_all(lane, at=T0 + dt.timedelta(hours=run))
        assert _done(lines)["error" if expected == "pending" else "stuck"] == 1
        assert code == dc.EXIT_DONE
        assert db.get(DisplayOriginal, sha).state == expected
    assert db.get(DisplayOriginal, sha).reason == "gave_up:missing_raw"


def test_an_unknown_codec_is_stuck_never_failed(lane: dict[str, Any]) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(20))
    _queue(
        db, sha, key, codec="zstd", depictions=[(_pid(20), _pid(120), STORE.format(n=20), "amiami")]
    )
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert (db.get(DisplayOriginal, sha).state, db.get(DisplayOriginal, sha).reason) == (
        "stuck",
        "unsupported_codec:zstd",
    )


class _Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def get(self, key: str) -> bytes:
        raise self.exc

    def put(self, key: str, data: bytes, content_type: str) -> None:
        raise self.exc


def _client_error(code: str, status: int) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject"
    )


def test_a_bad_credential_stops_the_run_at_once_with_exit_2_and_fails_nothing(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    for n in (21, 22):
        sha, key = lane["raw"].put(_photo(n))
        _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: db,
        _Failing(_client_error("InvalidAccessKeyId", 403)),
        lane["media"],
        dc.SeedConfig(threads=0, grants=GRANTS),
        emit=lines.append,
        now=lambda: T0,
    )
    assert code == dc.EXIT_CONFIG
    states = {o.state for o in db.execute(select(DisplayOriginal)).scalars()}
    assert states == {"pending"}
    assert {o.attempts for o in db.execute(select(DisplayOriginal)).scalars()} == {0}
    assert next(line for line in lines if line["event"] == "stopped")["why"] == "configuration"


def test_an_outage_gives_the_attempt_back_and_stops_after_consecutive_ones(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    for n in range(23, 30):
        sha, key = lane["raw"].put(_photo(n))
        _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: db,
        _Failing(_client_error("ServiceUnavailable", 503)),
        lane["media"],
        dc.SeedConfig(threads=0, grants=GRANTS, max_outages=3),
        emit=lines.append,
        now=lambda: T0,
    )
    assert code == dc.EXIT_STOPPED
    rows = db.execute(select(DisplayOriginal)).scalars().all()
    assert {r.state for r in rows} == {"pending"} and {r.attempts for r in rows} == {0}


def test_a_flood_of_own_faults_trips_the_breaker_with_exit_1(lane: dict[str, Any]) -> None:
    db = lane["db"]
    for n in range(30, 40):
        sha, key = lane["raw"].put(_photo(n))
        lane["raw"].store.put(key, b"x" + _photo(n), "image/png")  # every object altered
        _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    code, lines = _seed_all(lane, max_failed=3, max_failed_fraction=0.05)
    assert code == dc.EXIT_BREAKER
    assert next(line for line in lines if line["event"] == "stopped")["why"] == "failed_rate"
    # it stopped: not every original was judged
    assert db.scalar(select(func.count()).where(DisplayOriginal.state == "failed")) == 4
    assert db.scalar(select(func.count()).where(DisplayOriginal.state == "pending")) == 6


def test_unexpected_errors_stop_the_run_with_exit_1_and_fail_nothing(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = lane["db"]
    for n in range(40, 45):
        sha, key = lane["raw"].put(_photo(n))
        _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a bug, not the image")

    monkeypatch.setattr(dc, "seed_with_retry", broken)
    code, lines = _seed_all(lane, max_errors=2)
    assert code == dc.EXIT_BREAKER
    assert next(line for line in lines if line["event"] == "stopped")["why"] == "errors"
    assert db.scalar(select(func.count()).where(DisplayOriginal.state == "failed")) == 0


def test_the_poison_guard_counts_claims_not_depictions(lane: dict[str, Any]) -> None:
    """A shared press photo on three products is ONE original, claimed once
    per run: three depictions never make it look like three crashes."""
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(45))
    _queue(
        db,
        sha,
        key,
        depictions=[(_pid(45 + i), _pid(145 + i), STORE.format(n=45), "amiami") for i in range(3)],
    )
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.get(DisplayOriginal, sha).state == "seeded"


def test_an_original_whose_claims_never_settle_is_stuck_as_crashed_repeatedly(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(48))
    row = _queue(db, sha, key, depictions=[(_pid(48), _pid(148), STORE.format(n=48), "amiami")])
    row.attempts = 3  # three pods died in its decode
    db.commit()
    code, lines = _seed_all(lane)
    assert code == dc.EXIT_DONE
    assert (db.get(DisplayOriginal, sha).state, db.get(DisplayOriginal, sha).reason) == (
        "stuck",
        "crashed_repeatedly",
    )
    assert _done(lines)["stuck"] == 1


def test_a_claim_is_committed_before_the_bytes_are_read(lane: dict[str, Any]) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(49))
    _queue(db, sha, key, depictions=[(_pid(49), _pid(149), STORE.format(n=49), "amiami")])
    seen: list[tuple[int, Any]] = []
    other = sessionmaker(bind=db.get_bind())()

    class _Spy:
        def get(self, k: str) -> bytes:
            row = other.get(DisplayOriginal, sha)
            other.refresh(row)
            seen.append((row.attempts, row.claimed_until))
            return lane["raw"].store.get(k)

        def put(self, *a: Any) -> None:
            raise PermissionError

    dc.run_seed(
        lambda: db,
        _Spy(),
        lane["media"],
        dc.SeedConfig(threads=0, grants=GRANTS),
        emit=lambda e: None,
        now=lambda: T0,
    )
    assert seen and seen[0][0] == 1 and seen[0][1] is not None
    other.close()


def test_a_claimed_original_is_not_claimed_again_until_its_claim_lapses(
    lane: dict[str, Any],
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(50))
    _queue(db, sha, key, depictions=[(_pid(50), _pid(150), STORE.format(n=50), "amiami")])
    first = dc.claim_next(db, owner="a", claim_seconds=900, now=T0)
    assert first is not None and first.origin_sha256 == sha
    assert (
        dc.claim_next(db, owner="b", claim_seconds=900, now=T0 + dt.timedelta(seconds=60)) is None
    )
    lapsed = dc.claim_next(db, owner="b", claim_seconds=900, now=T0 + dt.timedelta(seconds=901))
    assert lapsed is not None and lapsed.claimed_by == "b" and lapsed.attempts == 2


def test_the_newest_original_is_claimed_first(lane: dict[str, Any]) -> None:
    db = lane["db"]
    old, ko = lane["raw"].put(_photo(51))
    new, kn = lane["raw"].put(_photo(52))
    _queue(
        db,
        old,
        ko,
        seen=T0 - dt.timedelta(days=9),
        depictions=[(_pid(51), _pid(151), STORE.format(n=51), "a")],
    )
    _queue(db, new, kn, seen=T0, depictions=[(_pid(52), _pid(152), STORE.format(n=52), "a")])
    assert dc.claim_next(db, owner="x", claim_seconds=900, now=T0).origin_sha256 == new


def test_the_budget_stops_the_run_cleanly(lane: dict[str, Any]) -> None:
    db = lane["db"]
    for n in range(53, 56):
        sha, key = lane["raw"].put(_photo(n))
        _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: db,
        ReadOnlyStore(lane["raw"].store),
        lane["media"],
        dc.SeedConfig(threads=0, grants=GRANTS, budget_seconds=50),
        emit=lines.append,
        now=lambda: T0,
        monotonic=lambda: next(ticks, 100.0),
    )
    assert code == dc.EXIT_DONE
    assert _done(lines)["seeded"] == 1
    assert _done(lines)["why"] == "budget"


def test_the_lane_never_opens_a_network_connection_of_its_own(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes come from the raw bucket only: no request ever reaches the
    store a photo was captured at (MFC and every hands-off host included)."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the display-copy lane opened a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(56))
    _queue(db, sha, key, depictions=[(_pid(56), _pid(156), MFC_ITEM.format(n=56), "mfc")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert db.get(DisplayOriginal, sha).state == "seeded"
    import app.workers.display_copy as module

    source = Path(module.__file__).read_text()
    for client in ("httpx", "requests", "urllib.request", "aiohttp"):
        assert f"import {client}" not in source


def test_the_raw_bucket_refuses_writes() -> None:
    with pytest.raises(PermissionError):
        ReadOnlyStore(LocalObjectStore(Path("/nonexistent"))).put("k", b"x", "image/png")


def test_threads_seed_every_original_once(tmp_path: Path) -> None:
    from app.models import Base
    from sqlalchemy import create_engine

    engine = create_engine(
        f"sqlite:///{tmp_path / 'queue.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    raw = _Raw(tmp_path / "raw")
    media = LocalObjectStore(tmp_path / "media")
    with factory() as db:
        for n in range(60, 66):
            sha, key = raw.put(_photo(n))
            _queue(db, sha, key, depictions=[(_pid(n), _pid(100 + n), STORE.format(n=n), "amiami")])
    lock = threading.Lock()

    def session() -> Session:
        with lock:
            return factory()

    code = dc.run_seed(
        session,
        ReadOnlyStore(raw.store),
        media,
        dc.SeedConfig(threads=2, grants=GRANTS),
        emit=lambda e: None,
        now=lambda: T0,
    )
    assert code == dc.EXIT_DONE
    with factory() as db:
        assert {o.state for o in db.execute(select(DisplayOriginal)).scalars()} == {"seeded"}
        assert db.scalar(select(func.count()).select_from(ImageModel)) == 6


# --------------------------------------------------------------------------
# the seed's retry (shared with the mask batch)
# --------------------------------------------------------------------------


def test_a_concurrent_first_seed_is_retried_once_and_finds_the_winners_rows(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = lane["db"]
    data = _photo(70)
    calls: list[int] = []
    real = seed_clean_image

    def racing(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise IntegrityError("INSERT", {}, Exception("duplicate key value"))
        return real(*args, **kwargs)

    monkeypatch.setattr("app.workers.mask_seed.seed_clean_image", racing)
    obj = seed_with_retry(
        db,
        lane["media"],
        data,
        origins=[ImageOrigin(url=STORE.format(n=70), source="amiami")],
        figure_id=_pid(170),
    )
    assert len(calls) == 2 and obj.kind == "sanitized"


def test_a_persistent_integrity_error_is_raised_after_the_retry(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def always(*args: Any, **kwargs: Any) -> Any:
        raise IntegrityError("INSERT", {}, Exception("duplicate key value"))

    monkeypatch.setattr("app.workers.mask_seed.seed_clean_image", always)
    with pytest.raises(IntegrityError):
        seed_with_retry(lane["db"], lane["media"], _photo(71), origins=[], figure_id=_pid(171))


def test_in_the_lane_a_persistent_integrity_error_is_an_interruption_not_a_failure(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(72))
    _queue(db, sha, key, depictions=[(_pid(72), _pid(172), STORE.format(n=72), "amiami")])

    def always(*args: Any, **kwargs: Any) -> Any:
        raise IntegrityError("INSERT", {}, Exception("duplicate key value"))

    monkeypatch.setattr(dc, "seed_with_retry", always)
    code, lines = _seed_all(lane)
    assert code == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.attempts) == ("pending", 0)
    assert _done(lines)["interrupted"] == 1


def test_a_database_outage_mid_seed_gives_the_attempt_back(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(73))
    _queue(db, sha, key, depictions=[(_pid(73), _pid(173), STORE.format(n=73), "amiami")])

    def down(*args: Any, **kwargs: Any) -> Any:
        raise OperationalError("SELECT", {}, Exception("server closed the connection"))

    monkeypatch.setattr(dc, "seed_with_retry", down)
    code, _ = _seed_all(lane, max_outages=1)
    assert code == dc.EXIT_STOPPED
    assert (db.get(DisplayOriginal, sha).state, db.get(DisplayOriginal, sha).attempts) == (
        "pending",
        0,
    )


# --------------------------------------------------------------------------
# operator commands
# --------------------------------------------------------------------------


def test_requeue_is_a_dry_run_unless_applied_and_filters_by_reason(lane: dict[str, Any]) -> None:
    db = lane["db"]
    for n, (state, reason) in enumerate(
        [
            ("stuck", "gave_up:missing_raw"),
            ("stuck", "crashed_repeatedly"),
            ("failed", "undecodable"),
        ]
    ):
        db.add(
            DisplayOriginal(
                origin_sha256=f"{n:064x}",
                storage_key="k",
                codec="identity",
                first_seen_at=T0,
                state=state,
                reason=reason,
                attempts=3,
            )
        )
    db.commit()
    assert dc.requeue(db, state="stuck", reason_prefix="gave_up:", force=False, apply=False) == 1
    assert db.scalar(select(func.count()).where(DisplayOriginal.state == "stuck")) == 2
    assert dc.requeue(db, state="stuck", reason_prefix="gave_up:", force=False, apply=True) == 1
    row = db.get(DisplayOriginal, f"{0:064x}")
    assert (row.state, row.attempts, row.reason, row.force_seed) == ("pending", 0, None, False)
    assert dc.status(db)["originals"] == {"failed": 1, "pending": 1, "stuck": 1}


def test_requeue_refuses_a_state_it_does_not_own(lane: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="requeue"):
        dc.requeue(lane["db"], state="seeded", reason_prefix=None, force=False, apply=True)


# --------------------------------------------------------------------------
# reported_copy is the replay's plan, per origin
# --------------------------------------------------------------------------


def test_reported_copy_matches_the_replays_plan_for_owned_linked_and_dead_copies(
    db_session: Session,
) -> None:
    from spine_fakes import seed_copy

    db = db_session
    live = seed_copy(db, "a", extra_originals=("a2",))
    dead = seed_copy(db, "b", deleted=True)
    plan = load_plan(db)
    for origin in (live["origin"], *live["others"], dead["origin"], "f" * 64):
        assert copies.reported_copy(db, origin) == plan.copy_of.get(origin), origin


def test_reported_copy_takes_the_lowest_live_copy_when_an_image_minted_two(
    db_session: Session,
) -> None:
    from spine_fakes import seed_copy

    db = db_session
    first = seed_copy(db, "c")
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == first["origin"])).scalar_one()
    version = ImageVersion(
        image_id=image.id, version_no=3, derived_from_version=1, storage_key="d/x"
    )
    db.add(version)
    db.flush()
    low = "0" * 64
    db.add(
        PublicObject(
            sha256=low,
            kind="sanitized",
            mime="image/jpeg",
            bytes=1,
            width=1,
            height=1,
            storage_key="d/low",
            image_version_id=version.id,
            pixel_sha256="1" * 64,
            own_pixel_sha256="2" * 64,
        )
    )
    db.commit()
    assert (
        copies.reported_copy(db, first["origin"]) == low == load_plan(db).copy_of[first["origin"]]
    )


def test_plain_appearance_is_the_mask_lanes_unmasked_appearance_on_the_sample() -> None:
    sample = Path(__file__).parent / "fixtures" / "mask_sample" / "originals"
    files = sorted(p for p in sample.iterdir() if p.is_file())
    assert files
    for path in files:
        clean = path.read_bytes()
        look, grounding = mask_artifact.plain_appearance(clean)
        outcome = mask_artifact.build_mask_artifact(clean, backend=StubMattingBackend())
        assert look == outcome.unmasked, path.name
        assert grounding == outcome.own_alpha_grounding, path.name


# --------------------------------------------------------------------------
# the memory bound (critique 7: the decode units cover the appearance step)
# --------------------------------------------------------------------------


def test_an_original_past_the_lanes_memory_bound_is_stuck_never_failed(
    lane: dict[str, Any],
) -> None:
    """Above decode_units x unit_pixels the pod's memory limit would be at
    risk: the lane leaves it to the GPU batch (more memory), no report."""
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(70))  # 40 x 60 = 2,400 pixels
    _queue(db, sha, key, depictions=[(_pid(70), _pid(170), STORE.format(n=70), "amiami")])
    code, lines = _seed_all(lane, decode_units=2, unit_pixels=1000)
    assert code == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.reason) == ("stuck", "too_large_for_lane:2400")
    assert db.scalar(select(func.count()).select_from(ImageModel)) == 0


def test_a_claim_a_dead_pod_left_behind_is_seeded_alone(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critique 1: a pod killed by ONE poison image counts an attempt on every
    original it held. The next claim of each runs with every decode unit, so
    a second crash is its own and an innocent neighbour is never stuck."""
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(71))
    row = _queue(db, sha, key, depictions=[(_pid(71), _pid(171), STORE.format(n=71), "amiami")])
    row.attempts, row.claimed_by = 1, "display-seed/dead-pod/1"
    row.claimed_until = T0 - dt.timedelta(seconds=1)
    db.commit()
    taken: list[int] = []
    real = dc._Units.acquire

    def spy(self: Any, n: int) -> int:
        taken.append(n)
        return real(self, n)

    monkeypatch.setattr(dc._Units, "acquire", spy)
    code, lines = _seed_all(lane, decode_units=6)
    assert code == dc.EXIT_DONE and db.get(DisplayOriginal, sha).state == "seeded"
    assert taken == [6]
    assert any(line.get("kind") == "isolated" for line in lines)


# --------------------------------------------------------------------------
# the remaining failure classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"threads": -1}, "threads"),
        ({"decode_units": 0}, "decode_units"),
        ({"budget_seconds": 0}, "budget"),
    ],
)
def test_a_bad_seed_config_is_refused(kwargs: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        dc.SeedConfig(**kwargs)


def test_decode_units_make_a_large_original_wait_for_the_others() -> None:
    import threading

    units = dc._Units(2)
    assert units.acquire(9) == 2  # never more than all
    got: list[int] = []
    waiter = threading.Thread(target=lambda: got.append(units.acquire(1)))
    waiter.start()
    waiter.join(0.2)
    assert got == []  # waiting: every unit is taken
    units.release(2)
    waiter.join(5)
    assert got == [1]


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        ("limit", ("failed", "refused:too_many_pixels")),
        ("unidentified", ("failed", "undecodable")),
    ],
)
def test_an_image_fault_raised_inside_the_seed_is_failed(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch, raised: str, expected: tuple[str, str]
) -> None:
    from app.workers.image_limits import ImageLimitError
    from PIL import UnidentifiedImageError

    db = lane["db"]
    sha, key = lane["raw"].put(_photo(80))
    _queue(db, sha, key, depictions=[(_pid(80), _pid(180), STORE.format(n=80), "amiami")])

    def seed(*a: Any, **k: Any) -> Any:
        if raised == "limit":
            raise ImageLimitError("too_many_pixels", "past the cap")
        raise UnidentifiedImageError("not an image")

    monkeypatch.setattr(dc, "seed_with_retry", seed)
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.reason) == expected


def test_an_original_with_no_depictions_is_stuck(lane: dict[str, Any]) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(81))
    _queue(db, sha, key, depictions=[])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert (db.get(DisplayOriginal, sha).state, db.get(DisplayOriginal, sha).reason) == (
        "stuck",
        "no_depictions",
    )


def test_a_corrupt_stored_object_is_an_error_never_the_images_fault(lane: dict[str, Any]) -> None:
    db = lane["db"]
    data = _photo(82)
    sha = hashlib.sha256(data).hexdigest()
    key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
    lane["raw"].store.put(key, b"not gzip at all", "image/png")
    _queue(db, sha, key, codec="gzip", depictions=[(_pid(82), _pid(182), STORE.format(n=82), "a")])
    code, lines = _seed_all(lane)
    assert code == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.reason, row.attempts) == ("pending", "decode_stored:BadGzipFile", 1)


@pytest.mark.parametrize(
    ("codec", "limit"), [("identity", "too_large_bytes"), ("gzip", "gzip_too_large")]
)
def test_a_stored_object_past_the_byte_cap_is_refused(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch, codec: str, limit: str
) -> None:
    db = lane["db"]
    sha, key = lane["raw"].put(_photo(83), codec=codec)
    _queue(db, sha, key, codec=codec, depictions=[(_pid(83), _pid(183), STORE.format(n=83), "a")])
    monkeypatch.setenv("MM_MAX_IMAGE_BYTES", "100")
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.reason) == ("failed", f"refused:{limit}")


def test_a_soft_deleted_original_seeds_no_reportable_copy(lane: dict[str, Any]) -> None:
    """Critique 11: the seed hands back the dead image's copy; the replay's
    rule (reported_copy) names none, so nothing is recorded to report."""
    db = lane["db"]
    data = _photo(84)
    sha, key = lane["raw"].put(data)
    _queue(db, sha, key, depictions=[(_pid(84), _pid(184), STORE.format(n=84), "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == sha)).scalar_one()
    image.deleted_at = T0
    row = db.get(DisplayOriginal, sha)
    row.state = "pending"
    db.add(
        DisplayDepiction(
            product_image_id=_pid(85),
            origin_sha256=sha,
            product_id=_pid(185),
            origin_url=STORE.format(n=85),
            source_site="amiami",
            first_seen_at=T0,
        )
    )
    db.commit()
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.copy_sha256) == ("seeded", None)


def test_a_database_outage_outside_the_seed_gives_the_attempt_back_and_stops_3(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    db = lane["db"]
    sha, key = lane["raw"].put(_photo(86))
    _queue(db, sha, key, depictions=[(_pid(86), _pid(186), STORE.format(n=86), "amiami")])

    def down(*a: Any, **k: Any) -> Any:
        raise OperationalError("SELECT", {}, Exception("server closed the connection"))

    monkeypatch.setattr(dc, "seed_one", down)
    code, lines = _seed_all(lane, max_outages=1)
    assert code == dc.EXIT_STOPPED
    assert next(x for x in lines if x["event"] == "stopped")["why"] == "outages"
    row = db.get(DisplayOriginal, sha)
    assert (row.state, row.attempts) == ("pending", 0)
