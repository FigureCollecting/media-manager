"""MEDIADB: the seed's transactions are short, its rows written once.

The display seeder used to hold one database transaction from its read of
an original's depictions through the raw bucket's GET, the decode, the wait
for decode units and both media PUTs (40-85 s in prod, 134 s seen): every
idle second of it counted against an idle-in-transaction limit and held
back DDL on the tables it read. Now each read is committed before the next
bucket call or CPU stretch, and the seed's writes are one short transaction
after its PUTs (M-1, as the verifier corrected it: the withdraw check's
reads committed too, and a clean copy encoded and PUT only when the picture
has none, so the reorder writes no object no row names).

An image is inserted with its pixel digest, never inserted and then updated
(M-2: ``pixel_sha256`` is indexed, so the update was never HOT).

A capture whose pixels differ from an earlier original's but whose clean
copy is byte-identical to that original's copy is linked behind the copy
that exists, never minted again (the verifier's MISSED finding: the second
v2 row with the same storage key failed the unique key twice, and the
original was ``interrupted`` on every run, forever).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.models import (
    DisplayOriginal,
    FigureGallery,
    ImageVersion,
    PublicObject,
    PublicObjectImage,
)
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore, ObjectStore
from app.workers import display_copy as dc
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import make_clean_copy, pixel_digest, seed_clean_image
from app.workers.raw_source import ReadOnlyStore
from PIL import Image
from sqlalchemy import event, func, select, text
from sqlalchemy.orm import Session

from test_display_copy import GRANTS, MFC_ITEM, STORE, T0, _photo, _pid, _queue, _Raw
from test_mask_ingest_capture import _ingest


class _Watched:
    """A bucket that records, for each GET and PUT, whether ``db`` had a
    transaction open when it was made."""

    def __init__(self, store: ObjectStore, db: Session) -> None:
        self.store, self.db = store, db
        self.calls: list[tuple[str, str, bool]] = []

    def get(self, key: str) -> bytes:
        self.calls.append(("get", key, self.db.in_transaction()))
        return self.store.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.calls.append(("put", key, self.db.in_transaction()))
        self.store.put(key, data, content_type)

    def in_transaction(self) -> list[tuple[str, str]]:
        return [(op, key) for op, key, open_ in self.calls if open_]


def _run(db: Session, raw: _Watched, media: _Watched) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: db,
        raw,
        media,
        dc.SeedConfig(threads=0, grants=GRANTS),
        emit=lines.append,
        now=lambda: T0,
    )
    return code, lines


def _done(lines: list[dict[str, Any]]) -> dict[str, Any]:
    return next(line for line in lines if line["event"] == "done")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _recaptured(first: bytes) -> bytes:
    """The same pixels in other bytes (a PNG re-save at another level)."""
    with Image.open(io.BytesIO(first)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    again = buf.getvalue()
    assert again != first
    return again


def _twins() -> tuple[bytes, bytes]:
    """Two originals one bit apart in one pixel channel: other bytes and
    other pixel digests, yet byte-identical clean copies (the JPEG at
    quality 95 rounds the bit away). Neither is a capture of the other's
    copy by any key ``_copies_showing`` reads."""
    arr = np.full((60, 40, 3), 250, dtype=np.uint8)
    arr[10:50, 12:28] = (131, 30, 90)
    twin = arr.copy()
    twin[0, 0, 0] ^= 1
    a, b = io.BytesIO(), io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(a, format="PNG")
    Image.fromarray(twin, mode="RGB").save(b, format="PNG")
    first, second = a.getvalue(), b.getvalue()
    clean = make_clean_copy(first)[0]
    assert clean == make_clean_copy(second)[0]
    assert pixel_digest(first) != pixel_digest(second) != pixel_digest(clean)
    return first, second


def _unreferenced(db: Session, root: Path) -> set[str]:
    """Media objects no images, image_versions or public_objects row names."""
    keys = {
        *db.execute(select(ImageModel.storage_key)).scalars(),
        *db.execute(select(ImageVersion.storage_key)).scalars(),
        *db.execute(select(PublicObject.storage_key)).scalars(),
    }
    stored = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    return stored - keys


@pytest.fixture()
def bucket(tmp_path: Path) -> dict[str, Any]:
    return {"raw": _Raw(tmp_path / "raw"), "root": tmp_path / "media"}


def _stores(db: Session, bucket: dict[str, Any]) -> tuple[_Watched, _Watched]:
    raw = _Watched(ReadOnlyStore(bucket["raw"].store), db)
    return raw, _Watched(LocalObjectStore(bucket["root"]), db)


# --------------------------------------------------------------------------
# M-1: no transaction open across a bucket call
# --------------------------------------------------------------------------


def test_a_new_picture_is_read_and_written_with_no_transaction_open(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    """A granted store's capture (the withdraw check needs no decode): the
    raw GET, the original's PUT and the clean copy's PUT are each made with
    the session's last read committed."""
    db = db_session
    sha, key = bucket["raw"].put(_photo(1))
    _queue(db, sha, key, depictions=[(_pid(1), _pid(101), STORE.format(n=1), "amiami")])
    raw, media = _stores(db, bucket)

    code, lines = _run(db, raw, media)

    assert code == dc.EXIT_DONE and _done(lines)["seeded"] == 1
    assert [op for op, _key, _open in raw.calls] == ["get"]
    assert [op for op, _key, _open in media.calls] == ["put", "put"]
    assert raw.in_transaction() == [] and media.in_transaction() == []


def test_the_withdraw_check_commits_its_reads_before_the_seed_writes(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    """An MFC capture (not granted: the withdraw check judges the picture,
    reading the revocations, the origins and the copies): the seed's PUTs
    still come with no transaction open (the verifier's V1 left the check's
    reads open across them)."""
    db = db_session
    sha, key = bucket["raw"].put(_photo(2))
    _queue(db, sha, key, depictions=[(_pid(2), _pid(102), MFC_ITEM.format(n=2), "mfc")])
    raw, media = _stores(db, bucket)

    code, lines = _run(db, raw, media)

    assert code == dc.EXIT_DONE and _done(lines)["seeded"] == 1
    assert len(media.calls) == 2
    assert raw.in_transaction() == [] and media.in_transaction() == []


def test_a_copy_read_back_for_its_appearance_is_read_with_no_transaction_open(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    """A capture of a picture that has a copy already, not written by this
    seed: its appearance is measured from the copy read back from the media
    bucket, after the reads that found it are committed."""
    db = db_session
    first = _photo(3)
    seed_clean_image(
        db,
        LocalObjectStore(bucket["root"]),
        first,
        origins=[ImageOrigin(url=STORE.format(n=3), source="amiami")],
        figure_id=_pid(103),
    )
    copy = db.execute(select(PublicObject)).scalar_one()
    again = _recaptured(first)
    sha, key = bucket["raw"].put(again)
    _queue(db, sha, key, depictions=[(_pid(4), _pid(104), STORE.format(n=4), "amiami")])
    raw, media = _stores(db, bucket)

    code, lines = _run(db, raw, media)

    assert code == dc.EXIT_DONE and _done(lines)["seeded"] == 1
    assert ("get", copy.storage_key, False) in media.calls
    assert [op for op, _key, _open in media.calls if op == "put"] == ["put"]  # its original only
    assert raw.in_transaction() == [] and media.in_transaction() == []
    original = db.get(DisplayOriginal, sha)
    assert original is not None and original.copy_sha256 == copy.sha256


def test_the_seed_commits_a_read_its_caller_left_open_before_any_put(
    db_session: Session, tmp_path: Path
) -> None:
    """The mask batch reads (its footprint, the settled check) before the
    seed: the seed's PUTs come after a commit, its writes in one short
    transaction, committed when it returns."""
    db = db_session
    db.execute(select(func.count()).select_from(ImageModel)).scalar_one()  # the caller's read
    assert db.in_transaction()
    media = _Watched(LocalObjectStore(tmp_path / "media"), db)

    obj = seed_clean_image(
        db,
        media,
        _photo(5),
        origins=[ImageOrigin(url=STORE.format(n=5), source="amiami")],
        figure_id=_pid(105),
    )

    assert obj.kind == "sanitized"
    assert [op for op, _key, _open in media.calls] == ["put", "put"]
    assert media.in_transaction() == []
    assert not db.in_transaction()


# --------------------------------------------------------------------------
# M-1 corrected: the reorder writes no object without a row
# --------------------------------------------------------------------------


def test_our_own_copy_captured_again_writes_no_object_no_row_names(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    """Seed A, then seed B whose bytes ARE A's clean copy (found by its
    sha): B gets no copy of its own, so none is encoded and PUT (the
    audit's literal reorder PUT one, an object no takedown would find)."""
    db = db_session
    first = _photo(6)
    sha, key = bucket["raw"].put(first)
    _queue(db, sha, key, depictions=[(_pid(6), _pid(106), STORE.format(n=6), "amiami")])
    raw, media = _stores(db, bucket)
    assert _run(db, raw, media)[0] == dc.EXIT_DONE
    copy = db.execute(select(PublicObject)).scalar_one()

    again = LocalObjectStore(bucket["root"]).get(copy.storage_key)
    sha2, key2 = bucket["raw"].put(again)
    _queue(db, sha2, key2, depictions=[(_pid(7), _pid(107), STORE.format(n=7), "amiami")])
    raw, media = _stores(db, bucket)
    assert _run(db, raw, media)[0] == dc.EXIT_DONE

    assert db.scalar(select(func.count()).select_from(PublicObject)) == 1
    assert _unreferenced(db, bucket["root"]) == set()
    assert [op for op, _key, _open in media.calls if op == "put"] == ["put"]  # B's original


def test_a_capture_of_a_picture_with_a_copy_writes_no_object_no_row_names(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    """The same pixels in other bytes: linked behind the copy, its own
    original PUT, no second clean copy encoded or PUT."""
    db = db_session
    first = _photo(8)
    seed_clean_image(
        db,
        LocalObjectStore(bucket["root"]),
        first,
        origins=[ImageOrigin(url=STORE.format(n=8), source="amiami")],
        figure_id=_pid(108),
    )
    sha, key = bucket["raw"].put(_recaptured(first))
    _queue(db, sha, key, depictions=[(_pid(9), _pid(109), STORE.format(n=9), "amiami")])
    raw, media = _stores(db, bucket)

    assert _run(db, raw, media)[0] == dc.EXIT_DONE
    assert db.scalar(select(func.count()).select_from(PublicObject)) == 1
    assert _unreferenced(db, bucket["root"]) == set()


def test_a_copy_gone_between_the_read_and_the_write_is_minted_in_the_write(
    db_session: Session, tmp_path: Path
) -> None:
    """The picture had a copy when the seed read, none when it wrote (the
    copy's rows removed by hand meanwhile): the seed mints one, as it would
    have had it read then."""
    db = db_session
    store = LocalObjectStore(tmp_path / "media")
    first = _photo(10)
    old_sha = seed_clean_image(db, store, first, origins=[], figure_id=_pid(110)).sha256

    class _RemovesTheCopy:
        def get(self, key: str) -> bytes:
            return store.get(key)

        def put(self, key: str, data: bytes, content_type: str) -> None:
            store.put(key, data, content_type)
            with db.get_bind().connect() as other:  # another session's delete, committed
                other.execute(text("DELETE FROM public_object_images"))
                other.execute(text("DELETE FROM public_objects"))
                other.execute(text("DELETE FROM image_versions WHERE version_no = 2"))
                other.commit()

    again = _recaptured(first)
    obj = seed_clean_image(db, _RemovesTheCopy(), again, origins=[], figure_id=_pid(111))

    assert obj.sha256 == old_sha  # the same pixels: the same clean copy, minted again
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == _sha(again))).scalar_one()
    assert db.get(PublicObjectImage, (obj.sha256, image.id)) is not None


# --------------------------------------------------------------------------
# M-2: inserted with its digest, never updated
# --------------------------------------------------------------------------


def _statements(db: Session, run: Callable[[], Any]) -> list[str]:
    seen: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_: Any) -> None:
        seen.append(" ".join(statement.split()))

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        run()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return seen


def test_a_seeded_image_is_inserted_with_its_pixel_digest_and_never_updated(
    db_session: Session, tmp_path: Path
) -> None:
    db = db_session
    data = _photo(12)
    statements = _statements(
        db,
        lambda: seed_clean_image(
            db,
            LocalObjectStore(tmp_path / "media"),
            data,
            origins=[ImageOrigin(url=STORE.format(n=12), source="amiami")],
            figure_id=_pid(112),
        ),
    )
    assert [s for s in statements if s.startswith("UPDATE images")] == []
    inserted = [s for s in statements if s.startswith("INSERT INTO images ")]
    assert len(inserted) == 1 and "pixel_sha256" in inserted[0]
    image = db.execute(select(ImageModel)).scalar_one()
    assert image.pixel_sha256 == pixel_digest(data)


def test_an_ingested_image_is_inserted_with_its_pixel_digest_and_never_updated(
    db_session: Session,
) -> None:
    db = db_session
    data = _photo(13)
    captured: list[ImageModel] = []
    statements = _statements(
        db, lambda: captured.append(_ingest(db, data, "https://shop.example/13.png"))
    )
    assert [s for s in statements if s.startswith("UPDATE images")] == []
    assert captured[0].pixel_sha256 == pixel_digest(data)


def test_an_image_recorded_without_a_digest_still_gets_it(
    db_session: Session, tmp_path: Path
) -> None:
    """An image another route recorded with no digest (before MASK-1): the
    seed of its bytes records it, the one update left."""
    db = db_session
    data = _photo(14)
    store = LocalObjectStore(tmp_path / "media")
    seed_clean_image(db, store, data, origins=[], figure_id=_pid(114))
    db.execute(text("UPDATE images SET pixel_sha256 = NULL"))
    db.commit()
    seed_clean_image(db, store, data, origins=[], figure_id=_pid(115))
    assert db.execute(select(ImageModel.pixel_sha256)).scalar_one() == pixel_digest(data)


# --------------------------------------------------------------------------
# MISSED: a pixel-distinct twin of an original's clean copy
# --------------------------------------------------------------------------


def test_a_capture_whose_clean_copy_is_another_originals_copy_is_seeded_behind_it(
    db_session: Session, bucket: dict[str, Any]
) -> None:
    db = db_session
    first, second = _twins()
    sha1, key1 = bucket["raw"].put(first)
    _queue(db, sha1, key1, depictions=[(_pid(15), _pid(115), STORE.format(n=15), "amiami")])
    raw, media = _stores(db, bucket)
    assert _run(db, raw, media)[0] == dc.EXIT_DONE
    copy = db.execute(select(PublicObject)).scalar_one()

    sha2, key2 = bucket["raw"].put(second)
    _queue(db, sha2, key2, depictions=[(_pid(16), _pid(116), STORE.format(n=16), "amiami")])
    raw, media = _stores(db, bucket)
    code, lines = _run(db, raw, media)

    assert code == dc.EXIT_DONE
    assert _done(lines)["seeded"] == 1 and _done(lines)["interrupted"] == 0
    original = db.get(DisplayOriginal, sha2)
    assert original is not None
    assert (original.state, original.copy_sha256) == ("seeded", copy.sha256)
    image = db.execute(select(ImageModel).where(ImageModel.sha256 == sha2)).scalar_one()
    assert image.pixel_sha256 == pixel_digest(second)
    assert db.get(PublicObjectImage, (copy.sha256, image.id)) is not None
    # one copy, one public version naming it, no duplicate-key row
    assert db.scalar(select(func.count()).select_from(PublicObject)) == 1
    assert list(
        db.execute(
            select(ImageVersion.storage_key).where(ImageVersion.storage_key == copy.storage_key)
        ).scalars()
    ) == [copy.storage_key]
    link = db.execute(select(FigureGallery).where(FigureGallery.figure_id == _pid(116)))
    assert link.scalar_one().image_id == image.id
    assert _unreferenced(db, bucket["root"]) == set()
