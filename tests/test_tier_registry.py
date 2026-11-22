"""IMGREND (IR-MM-1): registering a tier, minting a parent's tiers into the
store and the ledger, and serving a tier at ``/d/<its own sha>``.

A TIER HAS NO STATUS OF ITS OWN. It is served exactly while its parent is
(app.publication): every case that takes a clean copy or a mask offline is
re-run here against their rungs. A tier's bytes are our work product and
nothing else: an address an original or a public object already has is
refused.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from app.config import Settings, get_settings
from app.db import get_db
from app.derivative_main import app as origin_app
from app.derivative_main import get_object_store
from app.models import (
    FigureGallery,
    ImageMask,
    ImageVersion,
    MaskStoreRevocation,
    PublicObject,
    PublicTier,
    TierParent,
)
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.publication import tier_servable
from app.workers import tiers
from app.workers.mask_job import restore_store, revoke_store, run_mask_batch
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import register_public_object, seed_clean_image
from app.workers.matting import StubMattingBackend
from app.workers.tiers import (
    TIER_PIPELINE_VERSION,
    TierStoreError,
    mint_parent,
    refused_parents,
    register_public_tier,
    rung_size,
    tier_key,
)
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/9.jpeg"
GRANTS = {"shop": ["shop.example"]}


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask model (the real stub never publishes)."""

    name = "stub-model"
    publishes_masks = True


def _photo(n: int, width: int = 300, height: int = 400) -> bytes:
    """A figure on a near-white backdrop, 300 x 400: rungs 128, 256 and 400. Each
    ``n`` is another silhouette, so another mask."""
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.full((height, width, 3), (250, 250, 248), dtype=np.uint8)
    wide = width * (0.3 - 0.02 * n)
    body = ((xx - width / 2) / wide) ** 2 + ((yy - height / 2) / (height * 0.4)) ** 2 < 1
    arr[body] = (31 + 10 * n, 30 + 3 * n, 90)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _webp(n: int, size: tuple[int, int] = (256, 192)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (n, 2 * n % 256, 77)).save(buf, format="WEBP", quality=80)
    return buf.getvalue()


@pytest.fixture()
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


def _seed(db: Session, store: LocalObjectStore, n: int, source: str = "mfc") -> PublicObject:
    url = ITEM.format(n=n) if source == "mfc" else f"https://img.shop.example/p/{n}.jpg"
    return seed_clean_image(
        db, store, _photo(n), origins=[ImageOrigin(url=url, source=source)], figure_id=f"f:{n}"
    )


@pytest.fixture()
def seeded(db_session: Session, store: LocalObjectStore) -> dict[str, str]:
    """One item photo with its clean copy and its published mask."""
    clean = _seed(db_session, store, 1)
    run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
    mask = db_session.execute(select(ImageMask)).scalar_one()
    assert mask.status == "ok" and mask.mask_sha256
    original = db_session.execute(select(ImageModel)).scalars().first()
    assert original is not None
    db_session.commit()
    return {"clean": clean.sha256, "mask": mask.mask_sha256, "original": original.sha256}


@pytest.fixture()
def minted(db_session: Session, store: LocalObjectStore, seeded: dict[str, str]) -> dict[str, str]:
    """``seeded`` with every rung of both parents minted: ``clean@256`` etc."""
    shas = dict(seeded)
    for which in ("clean", "mask"):
        outcome = mint_parent(db_session, store, seeded[which])
        assert outcome.state == "done", outcome
        for row in outcome.tiers:
            shas[f"{which}@{row.long_edge_px}"] = row.sha256
    db_session.commit()
    assert {f"{which}@{edge}" for which in ("clean", "mask") for edge in (128, 256, 400)} <= set(
        shas
    )
    return shas


def _source_image(db: Session, clean_sha: str) -> ImageModel:
    obj = db.get(PublicObject, clean_sha)
    assert obj is not None
    version = db.get(ImageVersion, obj.image_version_id)
    assert version is not None
    image = db.get(ImageModel, version.image_id)
    assert image is not None
    return image


def _register(db: Session, parent: str, data: bytes, **over: object) -> PublicTier:
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
    fields: dict[str, object] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "parent_sha256": parent,
        "long_edge_px": max(width, height),
        "size": len(data),
        "width": width,
        "height": height,
    }
    fields.update(over)
    return register_public_tier(db, **fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


class TestTierKey:
    def test_a_tier_lives_under_its_own_prefix_by_its_sha(self) -> None:
        sha = "ab" + "c" * 62
        assert tier_key(sha) == f"tiers/ab/{sha}.webp"

    @pytest.mark.parametrize("bad", ["", "AB" + "c" * 62, "c" * 63, "c" * 65, "../" + "c" * 61])
    def test_only_a_lowercase_hex_sha_names_a_key(self, bad: str) -> None:
        with pytest.raises(ValueError):
            tier_key(bad)


class TestRegister:
    def test_a_tier_is_registered_under_its_parent_version_and_edge(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        data = _webp(1)
        row = _register(db_session, seeded["clean"], data)
        db_session.commit()
        got = db_session.get(PublicTier, (seeded["clean"], TIER_PIPELINE_VERSION, 256))
        assert got is row
        sha = hashlib.sha256(data).hexdigest()
        assert (got.sha256, got.parent_kind, got.mime) == (sha, "sanitized", "image/webp")
        assert (got.width, got.height, got.bytes) == (256, 192, len(data))
        assert got.storage_key == f"tiers/{sha[:2]}/{sha}.webp"
        assert got.created_at is not None

    def test_a_masks_tier_is_registered_as_a_masks(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        assert _register(db_session, seeded["mask"], _webp(2)).parent_kind == "mask"

    def test_the_first_tier_of_a_parent_version_and_edge_is_kept(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        first = _register(db_session, seeded["clean"], _webp(1))
        again = _register(db_session, seeded["clean"], _webp(1))
        other = _register(db_session, seeded["clean"], _webp(9))  # other bytes, the same edge
        assert first is again is other
        assert other.sha256 == hashlib.sha256(_webp(1)).hexdigest()
        assert len(db_session.execute(select(PublicTier)).all()) == 1

    def test_another_version_or_edge_is_another_tier(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        _register(db_session, seeded["clean"], _webp(1))
        _register(db_session, seeded["clean"], _webp(2), pipeline_version="mm-tier-v2")
        _register(db_session, seeded["clean"], _webp(3, (128, 96)))
        assert len(db_session.execute(select(PublicTier)).all()) == 3

    def test_an_originals_address_is_refused(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        """The public origin must never be able to name an original."""
        with pytest.raises(ValueError, match="original"):
            _register(db_session, seeded["clean"], _webp(1), sha256=seeded["original"])
        assert db_session.execute(select(PublicTier)).all() == []

    @pytest.mark.parametrize("which", ["clean", "mask"])
    def test_a_public_objects_address_is_refused(
        self, db_session: Session, seeded: dict[str, str], which: str
    ) -> None:
        with pytest.raises(ValueError, match="public object"):
            _register(db_session, seeded["clean"], _webp(1), sha256=seeded[which])
        assert db_session.execute(select(PublicTier)).all() == []

    def test_a_soft_deleted_originals_address_is_refused_too(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        _source_image(db_session, seeded["clean"]).deleted_at = dt.datetime.now(dt.UTC)
        db_session.flush()
        with pytest.raises(ValueError, match="original"):
            _register(db_session, seeded["clean"], _webp(1), sha256=seeded["original"])

    def test_a_parent_that_is_not_registered_is_refused(self, db_session: Session) -> None:
        with pytest.raises(ValueError, match="parent"):
            _register(db_session, "0" * 64, _webp(1))

    def test_only_a_clean_copy_or_a_mask_has_tiers(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        version = db_session.execute(select(ImageVersion)).scalars().first()
        assert version is not None
        cut = register_public_object(
            db_session,
            sha256="7" * 64,
            kind="matted",
            mime="image/png",
            size=10,
            width=300,
            height=400,
            storage_key="derived/77/x.png",
            image_version_id=version.id,
        )
        with pytest.raises(ValueError, match="parent"):
            _register(db_session, cut.sha256, _webp(1))

    @pytest.mark.parametrize(
        "over",
        [
            {"sha256": "A" * 64},
            {"sha256": "abc"},
            {"long_edge_px": 0},
            {"long_edge_px": 200},  # not the long side of 256 x 192
            {"long_edge_px": 4097, "width": 4097, "height": 5},  # past the contract's cap
            {"width": 0},
            {"height": 0},
            {"height": 300},  # then 256 is not the long side
            {"size": 0},
            {"mime": "image/png"},
            {"mime": "image/jpeg"},
            {"pipeline_version": ""},
            {"pipeline_version": "x" * 33},
        ],
    )
    def test_a_tier_that_breaks_the_contracts_shape_is_refused(
        self, db_session: Session, seeded: dict[str, str], over: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            _register(db_session, seeded["clean"], _webp(1), **over)
        assert db_session.execute(select(PublicTier)).all() == []

    def test_a_4096_px_tier_is_the_largest_registered(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        row = _register(
            db_session, seeded["clean"], _webp(1), long_edge_px=4096, width=4096, height=7
        )
        assert row.long_edge_px == 4096

    def test_two_parents_may_share_one_tiers_bytes(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        """Two near-identical pictures can scale down to the same bytes: the
        tier is each parent's, at one address and one key."""
        other = _seed(db_session, store, 2)
        a = _register(db_session, seeded["clean"], _webp(1))
        b = _register(db_session, other.sha256, _webp(1))
        db_session.commit()
        assert a is not b and a.sha256 == b.sha256 and a.storage_key == b.storage_key
        assert {a.parent_sha256, b.parent_sha256} == {seeded["clean"], other.sha256}

    def test_the_database_holds_the_shape_too(
        self, db_session: Session, seeded: dict[str, str]
    ) -> None:
        def row(**over: object) -> PublicTier:
            fields: dict[str, object] = {
                "sha256": "1" * 64,
                "parent_sha256": seeded["clean"],
                "parent_kind": "sanitized",
                "pipeline_version": "v",
                "long_edge_px": 256,
                "mime": "image/webp",
                "bytes": 10,
                "width": 256,
                "height": 192,
                "storage_key": "tiers/11/x.webp",
            }
            fields.update(over)
            return PublicTier(**fields)

        for bad in (
            row(parent_kind="matted"),
            row(long_edge_px=0, width=0, height=0),
            row(long_edge_px=4097, width=4097),
            row(width=100),  # neither side is the long edge
            row(height=300),
            row(mime="image/png"),
            row(parent_sha256="0" * 64),  # no such public object
        ):
            db_session.add(bad)
            with pytest.raises(IntegrityError):
                db_session.flush()
            db_session.rollback()
        db_session.add(row())
        db_session.flush()

    def test_a_parents_tiers_and_ledger_go_with_its_row(
        self, db_session: Session, store: LocalObjectStore, minted: dict[str, str]
    ) -> None:
        parent = db_session.get(PublicObject, minted["mask"])
        assert parent is not None
        db_session.delete(parent)
        db_session.commit()
        left = {t.parent_kind for t in db_session.execute(select(PublicTier)).scalars()}
        assert left == {"sanitized"}
        assert [p.parent_sha256 for p in db_session.execute(select(TierParent)).scalars()] == [
            minted["clean"]
        ]


# --------------------------------------------------------------------------
# minting a parent: store, registry, ledger
# --------------------------------------------------------------------------


class _Counting:
    """The store, counting what is read and written."""

    def __init__(self, inner: LocalObjectStore) -> None:
        self.inner, self.gets, self.puts = inner, [], []

    def get(self, key: str) -> bytes:
        self.gets.append(key)
        return self.inner.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.puts.append((key, content_type))
        self.inner.put(key, data, content_type)


class TestMintParent:
    def test_a_clean_copys_rungs_are_stored_then_registered_and_the_ledger_says_done(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        counting = _Counting(store)
        outcome = mint_parent(db_session, counting, seeded["clean"])
        assert (outcome.state, outcome.reason, outcome.minted) == ("done", None, 3)
        assert [t.long_edge_px for t in outcome.tiers] == [128, 256, 400]
        for row in outcome.tiers:
            data = store.get(row.storage_key)
            assert hashlib.sha256(data).hexdigest() == row.sha256 and len(data) == row.bytes
            assert row.storage_key == tier_key(row.sha256)
            assert (row.width, row.height) == rung_size(300, 400, row.long_edge_px)
            assert Image.open(io.BytesIO(data)).size == (row.width, row.height)
            assert (row.parent_kind, row.mime) == ("sanitized", "image/webp")
        assert counting.puts == [(t.storage_key, "image/webp") for t in outcome.tiers]
        ledger = db_session.get(TierParent, (seeded["clean"], TIER_PIPELINE_VERSION))
        assert ledger is not None
        assert (ledger.state, ledger.reason, ledger.max_edge_px, ledger.attempts) == (
            "done",
            None,
            2048,
            1,
        )

    def test_a_masks_rungs_are_the_size_of_its_photos(
        self, db_session: Session, store: LocalObjectStore, minted: dict[str, str]
    ) -> None:
        rows = db_session.execute(select(PublicTier)).scalars().all()
        size = {(r.parent_kind, r.long_edge_px): (r.width, r.height) for r in rows}
        assert size[("mask", 128)] == size[("sanitized", 128)] == (96, 128)
        assert size[("mask", 256)] == size[("sanitized", 256)] == (192, 256)
        assert size[("mask", 400)] == size[("sanitized", 400)] == (300, 400)
        mask = Image.open(io.BytesIO(store.get(tier_key(minted["mask@256"]))))
        assert mask.mode == "RGBA" and mask.size == (192, 256)

    def test_the_bytes_are_in_the_store_before_any_row_names_them(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        class _FailsSecondPut(_Counting):
            def put(self, key: str, data: bytes, content_type: str) -> None:
                if self.puts:
                    raise ConnectionError("bucket down")
                super().put(key, data, content_type)

        with pytest.raises(ConnectionError):
            mint_parent(db_session, _FailsSecondPut(store), seeded["clean"])
        # an outage is not the parent's fault: nothing registered, nothing in the ledger
        assert db_session.execute(select(PublicTier)).all() == []
        assert db_session.execute(select(TierParent)).all() == []
        assert mint_parent(db_session, store, seeded["clean"]).state == "done"

    def test_a_settled_parent_is_not_read_again(
        self, db_session: Session, store: LocalObjectStore, minted: dict[str, str]
    ) -> None:
        counting = _Counting(store)
        outcome = mint_parent(db_session, counting, minted["clean"])
        assert (outcome.state, outcome.minted) == ("done", 0)
        assert [t.sha256 for t in outcome.tiers] == [
            minted[f"clean@{edge}"] for edge in (128, 256, 400)
        ]
        assert counting.gets == [] and counting.puts == []

    def test_phase_a_mints_the_small_rungs_and_phase_b_the_rest(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        """The backfill's order: 128, 256 and 512 for everything (max_edge
        512), then 1,024 and 2,048 (the default), going on from the ledger."""
        big = seed_clean_image(db_session, store, _photo(5, 900, 1200), origins=[], figure_id="f:5")
        a = mint_parent(db_session, store, big.sha256, max_edge=512)
        assert [t.long_edge_px for t in a.tiers] == [128, 256, 512] and a.minted == 3
        ledger = db_session.get(TierParent, (big.sha256, TIER_PIPELINE_VERSION))
        assert ledger is not None and (ledger.state, ledger.max_edge_px) == ("done", 512)
        counting = _Counting(store)
        assert mint_parent(db_session, counting, big.sha256, max_edge=512).minted == 0
        assert counting.gets == []
        b = mint_parent(db_session, counting, big.sha256)
        assert [t.long_edge_px for t in b.tiers] == [128, 256, 512, 1024, 1200] and b.minted == 2
        assert [t.sha256 for t in b.tiers[:3]] == [t.sha256 for t in a.tiers]
        assert [key for key, _ in counting.puts] == [t.storage_key for t in b.tiers[3:]]
        assert (ledger.state, ledger.max_edge_px, ledger.attempts) == ("done", 2048, 2)
        # a smaller phase after a larger one asks for nothing and lowers nothing
        again = mint_parent(db_session, counting, big.sha256, max_edge=512)
        assert again.minted == 0 and [t.long_edge_px for t in again.tiers] == [128, 256, 512]
        assert ledger.max_edge_px == 2048

    def test_a_phase_edge_that_is_not_a_rung_is_an_error(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        for bad in (0, 300, 4096):
            with pytest.raises(ValueError, match="max_edge"):
                mint_parent(db_session, store, seeded["clean"], max_edge=bad)

    def test_an_unknown_parent_or_one_of_another_kind_is_an_error(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        with pytest.raises(LookupError):
            mint_parent(db_session, store, "0" * 64)
        obj = db_session.get(PublicObject, seeded["clean"])
        assert obj is not None
        cut = register_public_object(
            db_session,
            sha256="7" * 64,
            kind="matted",
            mime="image/png",
            size=10,
            width=300,
            height=400,
            storage_key="derived/77/x.png",
            image_version_id=obj.image_version_id,
        )
        with pytest.raises(LookupError):
            mint_parent(db_session, store, cut.sha256)
        assert db_session.execute(select(TierParent)).all() == []

    def test_bytes_that_are_not_the_parents_are_a_store_fault_not_a_refusal(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        obj = db_session.get(PublicObject, seeded["clean"])
        assert obj is not None
        good = store.get(obj.storage_key)
        store.put(obj.storage_key, good[:-9], obj.mime)
        with pytest.raises(TierStoreError):
            mint_parent(db_session, store, seeded["clean"])
        assert db_session.execute(select(TierParent)).all() == []
        store.put(obj.storage_key, good, obj.mime)
        assert mint_parent(db_session, store, seeded["clean"]).state == "done"

    def test_a_missing_parent_object_is_the_stores_error(
        self, db_session: Session, tmp_path: Path, seeded: dict[str, str]
    ) -> None:
        with pytest.raises(FileNotFoundError):
            mint_parent(db_session, LocalObjectStore(tmp_path / "empty"), seeded["clean"])
        assert db_session.execute(select(TierParent)).all() == []

    def test_the_parents_own_fault_is_refused_once_and_listed(
        self, db_session: Session, store: LocalObjectStore
    ) -> None:
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        Image.new("RGB", (300, 400), (9, 9, 9)).save(buf, format="JPEG", exif=exif)
        data = buf.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        tagged = PublicObject(
            sha256=sha,
            kind="sanitized",
            mime="image/jpeg",
            bytes=len(data),
            width=300,
            height=400,
            storage_key=f"derived/{sha[:2]}/{sha}.jpg",
            pixel_sha256="a" * 64,
            own_pixel_sha256="b" * 64,
        )
        db_session.add(tagged)
        db_session.flush()
        store.put(tagged.storage_key, data, "image/jpeg")
        counting = _Counting(store)
        outcome = mint_parent(db_session, counting, tagged.sha256)
        assert (outcome.state, outcome.reason, outcome.tiers, outcome.minted) == (
            "refused",
            "parent_has_orientation",
            (),
            0,
        )
        assert counting.puts == []
        ledger = db_session.get(TierParent, (tagged.sha256, TIER_PIPELINE_VERSION))
        assert ledger is not None
        assert (ledger.state, ledger.reason, ledger.attempts) == (
            "refused",
            "parent_has_orientation",
            1,
        )
        # the next sweep does not read it again
        again = mint_parent(db_session, counting, tagged.sha256)
        assert (again.state, again.reason) == ("refused", "parent_has_orientation")
        assert len(counting.gets) == 1 and ledger.attempts == 1
        assert [p.parent_sha256 for p in refused_parents(db_session)] == [tagged.sha256]

    def test_a_row_whose_size_is_not_its_pixels_is_refused(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        obj = db_session.get(PublicObject, seeded["clean"])
        assert obj is not None
        obj.width, obj.height = 400, 300
        outcome = mint_parent(db_session, store, seeded["clean"])
        assert (outcome.state, outcome.reason) == ("refused", "parent_size_mismatch")

    def test_a_mask_that_is_not_its_derivatives_size_is_refused_never_resized(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        clean = db_session.get(PublicObject, seeded["clean"])
        assert clean is not None
        clean.width, clean.height = 600, 800
        db_session.flush()
        counting = _Counting(store)
        outcome = mint_parent(db_session, counting, seeded["mask"])
        assert (outcome.state, outcome.reason) == ("refused", "mask_size_mismatch")
        assert counting.puts == [] and db_session.execute(select(PublicTier)).all() == []

    def test_a_mask_no_row_names_is_refused(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        for row in db_session.execute(select(ImageMask)).scalars():
            db_session.delete(row)
        db_session.flush()
        outcome = mint_parent(db_session, store, seeded["mask"])
        assert (outcome.state, outcome.reason) == ("refused", "mask_without_source")

    def test_a_withdrawn_masks_rungs_are_still_minted(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        """Status follows the parent: a withdrawn mask's rungs exist, unserved."""
        db_session.execute(select(ImageMask)).scalar_one().status = "withdrawn"
        assert mint_parent(db_session, store, seeded["mask"]).state == "done"

    def test_a_rung_at_an_originals_address_refuses_the_parent(
        self, db_session: Session, store: LocalObjectStore, seeded: dict[str, str]
    ) -> None:
        """Our rung captured as an original (its row since removed from the
        registry): its address is an original's now, never a tier's again."""
        first = mint_parent(db_session, store, seeded["clean"])
        taken = first.tiers[0].sha256
        for row in db_session.execute(select(PublicTier)).scalars():
            db_session.delete(row)
        ledger = db_session.get(TierParent, (seeded["clean"], TIER_PIPELINE_VERSION))
        db_session.delete(ledger)
        db_session.add(ImageModel(sha256=taken, mime="image/webp", storage_key="orig/captured"))
        db_session.flush()
        counting = _Counting(store)
        outcome = mint_parent(db_session, counting, seeded["clean"])
        assert (outcome.state, outcome.reason, outcome.minted) == (
            "refused",
            "tier_address_taken",
            0,
        )
        assert counting.puts == [] and db_session.execute(select(PublicTier)).all() == []

    def test_an_unexpected_failure_is_recorded_and_tried_again(
        self,
        db_session: Session,
        store: LocalObjectStore,
        seeded: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> list[tiers.Tier]:
            raise MemoryError("the encoder ran out")

        real = tiers.mint_photo_tiers
        monkeypatch.setattr(tiers, "mint_photo_tiers", _boom)
        for attempt in (1, 2):
            outcome = mint_parent(db_session, store, seeded["clean"])
            assert (outcome.state, outcome.reason, outcome.tiers) == ("failed", "MemoryError", ())
            ledger = db_session.get(TierParent, (seeded["clean"], TIER_PIPELINE_VERSION))
            assert ledger is not None and (ledger.state, ledger.attempts) == ("failed", attempt)
        monkeypatch.setattr(tiers, "mint_photo_tiers", real)
        assert mint_parent(db_session, store, seeded["clean"]).state == "done"
        assert ledger.attempts == 3 and ledger.reason is None

    def test_a_database_error_is_not_the_parents_failure(
        self,
        db_session: Session,
        store: LocalObjectStore,
        seeded: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _down(*args: object, **kwargs: object) -> tuple[int, int]:
            raise OperationalError("SELECT", {}, ConnectionError("server closed the connection"))

        monkeypatch.setattr(tiers, "_derivative_size", _down)
        with pytest.raises(OperationalError):
            mint_parent(db_session, store, seeded["mask"])
        assert db_session.execute(select(TierParent)).all() == []

    def test_a_repeat_keeps_the_first_tier_of_each_rung(
        self, db_session: Session, store: LocalObjectStore, minted: dict[str, str]
    ) -> None:
        """A parent minted again (its ledger row gone) registers nothing new."""
        ledger = db_session.get(TierParent, (minted["clean"], TIER_PIPELINE_VERSION))
        db_session.delete(ledger)
        db_session.flush()
        before = len(db_session.execute(select(PublicTier)).all())
        outcome = mint_parent(db_session, store, minted["clean"])
        assert (outcome.state, outcome.minted) == ("done", 0)
        assert len(db_session.execute(select(PublicTier)).all()) == before


class TestRefusedParents:
    def test_refused_parents_are_listed_in_pages_by_sha(self, db_session: Session) -> None:
        shas = sorted(hashlib.sha256(bytes([n])).hexdigest() for n in range(5))
        for n, sha in enumerate(shas):
            db_session.add(
                PublicObject(
                    sha256=sha,
                    kind="mask",
                    mime="image/png",
                    bytes=1,
                    width=1,
                    height=1,
                    storage_key=f"k{n}",
                )
            )
            db_session.add(
                TierParent(
                    parent_sha256=sha,
                    pipeline_version=TIER_PIPELINE_VERSION if n != 4 else "other",
                    state="done" if n == 1 else "refused",
                    reason=None if n == 1 else "parent_too_large",
                    max_edge_px=0,
                    attempts=1,
                )
            )
        db_session.flush()
        first = refused_parents(db_session, limit=2)
        assert [p.parent_sha256 for p in first] == [shas[0], shas[2]]
        rest = refused_parents(db_session, after=first[-1].parent_sha256, limit=2)
        assert [p.parent_sha256 for p in rest] == [shas[3]]
        assert [p.reason for p in first + rest] == ["parent_too_large"] * 3
        other = refused_parents(db_session, pipeline_version="other")
        assert [p.parent_sha256 for p in other] == [shas[4]]

    def test_the_ledger_takes_only_its_states(self, db_session: Session) -> None:
        db_session.add(
            PublicObject(
                sha256="1" * 64,
                kind="mask",
                mime="image/png",
                bytes=1,
                width=1,
                height=1,
                storage_key="k",
            )
        )
        db_session.add(
            TierParent(
                parent_sha256="1" * 64,
                pipeline_version="v",
                state="skipped",
                max_edge_px=0,
                attempts=0,
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()


# --------------------------------------------------------------------------
# the origin: a tier is served exactly while its parent is
# --------------------------------------------------------------------------

SERVED = Settings(environment="test", media_tiers_served=True)


@pytest.fixture()
def client(db_session: Session, store: LocalObjectStore) -> Generator[TestClient, None, None]:
    def _db() -> Generator[Session, None, None]:
        yield db_session

    origin_app.dependency_overrides[get_db] = _db
    origin_app.dependency_overrides[get_object_store] = lambda: store
    origin_app.dependency_overrides[get_settings] = lambda: SERVED
    yield TestClient(origin_app, raise_server_exceptions=False)
    origin_app.dependency_overrides.clear()


def _assert_404(resp: object) -> None:
    assert resp.status_code == 404  # type: ignore[attr-defined]
    assert resp.headers["cache-control"] == "no-store"  # type: ignore[attr-defined]
    assert resp.content == b""  # type: ignore[attr-defined]
    assert resp.headers["x-content-type-options"] == "nosniff"  # type: ignore[attr-defined]


PHOTO_RUNGS = ("clean@128", "clean@256", "clean@400")
MASK_RUNGS = ("mask@128", "mask@256", "mask@400")


class TestOrigin:
    @pytest.mark.parametrize("which", PHOTO_RUNGS + MASK_RUNGS)
    def test_a_tier_is_served_at_its_own_sha_like_its_parent(
        self, client: TestClient, minted: dict[str, str], which: str
    ) -> None:
        sha = minted[which]
        resp = client.get(f"/d/{sha}")
        assert resp.status_code == 200
        assert hashlib.sha256(resp.content).hexdigest() == sha
        assert resp.headers["content-type"] == "image/webp"
        assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert resp.headers["etag"] == f'"{sha}"'
        assert resp.headers["access-control-allow-origin"] == "*"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["cross-origin-resource-policy"] == "cross-origin"
        assert resp.headers["content-security-policy"] == "default-src 'none'; sandbox"
        assert "vary" not in resp.headers
        head = client.head(f"/d/{sha}")
        assert head.status_code == 200 and head.content == b""
        assert head.headers["content-length"] == str(len(resp.content))

    def test_the_parents_still_answer_as_before(
        self, client: TestClient, minted: dict[str, str]
    ) -> None:
        assert client.get(f"/d/{minted['clean']}").headers["content-type"] == "image/jpeg"
        assert client.get(f"/d/{minted['mask']}").headers["content-type"] == "image/png"
        _assert_404(client.get(f"/d/{minted['original']}"))
        _assert_404(client.get("/d/" + "0" * 64))

    def test_tiers_are_not_served_until_the_deployment_says_so(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        """MEDIA_TIERS_SERVED is off by default: the origin does not read
        public_tiers at all (a role without the grant, a database without the
        table), and answers exactly as it did."""
        assert Settings(environment="test").media_tiers_served is False
        origin_app.dependency_overrides[get_settings] = lambda: Settings(environment="test")
        statements: list[str] = []

        def record(*args: object) -> None:
            statements.append(str(args[2]))

        engine = db_session.get_bind()
        event.listen(engine, "before_cursor_execute", record)
        try:
            for which in PHOTO_RUNGS + MASK_RUNGS:
                _assert_404(client.get(f"/d/{minted[which]}"))
        finally:
            event.remove(engine, "before_cursor_execute", record)
        assert statements and not [s for s in statements if "public_tiers" in s]
        assert client.get(f"/d/{minted['clean']}").status_code == 200

    def test_a_soft_deleted_image_takes_every_rung_offline(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        _source_image(db_session, minted["clean"]).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        for which in PHOTO_RUNGS + MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))
            _assert_404(client.head(f"/d/{minted[which]}"))

    def test_a_soft_deleted_clean_version_takes_every_rung_offline(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        obj = db_session.get(PublicObject, minted["clean"])
        assert obj is not None
        version = db_session.get(ImageVersion, obj.image_version_id)
        assert version is not None
        version.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        for which in PHOTO_RUNGS + MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))

    def test_a_withdrawn_mask_withholds_its_rungs_and_not_the_photos(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        db_session.execute(select(ImageMask)).scalar_one().status = "withdrawn"
        db_session.commit()
        for which in MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))
        for which in PHOTO_RUNGS:
            assert client.get(f"/d/{minted[which]}").status_code == 200
        db_session.execute(select(ImageMask)).scalar_one().status = "ok"
        db_session.commit()
        for which in MASK_RUNGS:
            assert client.get(f"/d/{minted[which]}").status_code == 200

    def test_a_collector_photo_origin_withholds_the_masks_rungs(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        image = _source_image(db_session, minted["clean"])
        db_session.add(
            FigureGallery(figure_id="mfc:9", source_url=PICTURE, image_id=image.id, source="mfc")
        )
        db_session.commit()
        for which in MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))
        for which in PHOTO_RUNGS:
            assert client.get(f"/d/{minted[which]}").status_code == 200

    def test_a_mask_with_no_live_origin_withholds_its_rungs(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        for link in db_session.execute(select(FigureGallery)).scalars():
            link.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        for which in MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))

    def test_a_clean_copy_with_no_version_withholds_its_rungs(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        obj = db_session.get(PublicObject, minted["clean"])
        assert obj is not None
        obj.image_version_id = None
        db_session.commit()
        for which in PHOTO_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))

    def test_a_tier_whose_parent_row_is_gone_is_a_404(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        parent = db_session.get(PublicObject, minted["mask"])
        db_session.delete(parent)
        db_session.commit()
        for which in MASK_RUNGS:
            _assert_404(client.get(f"/d/{minted[which]}"))

    def _shop(
        self, db: Session, store: LocalObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, str]:
        """A store's photo and an item photo, masks published, rungs minted."""
        shop = _seed(db, store, 3, source="shop")
        item = _seed(db, store, 4)
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps(GRANTS))
        get_settings.cache_clear()
        try:
            run_mask_batch(db, store, backend=_Model(), dry_run=False)
        finally:
            get_settings.cache_clear()
        masks = {r.source_sha256: r.mask_sha256 for r in db.execute(select(ImageMask)).scalars()}
        assert masks[shop.sha256] and masks[item.sha256]
        shas: dict[str, str] = {}
        for name, parent in (
            ("shop_clean", shop.sha256),
            ("shop_mask", masks[shop.sha256]),
            ("item_mask", masks[item.sha256]),
        ):
            outcome = mint_parent(db, store, parent or "")
            assert outcome.state == "done"
            shas[name] = outcome.tiers[0].sha256
        db.commit()
        granted = Settings(environment="test", media_tiers_served=True, mask_store_grants=GRANTS)
        origin_app.dependency_overrides[get_settings] = lambda: granted
        return shas

    def test_a_revoked_store_withholds_its_masks_rungs_at_once(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The persisted revocation alone -- before any row is withdrawn --
        and it never touches a clean copy's rungs or another source's."""
        shas = self._shop(db_session, store, monkeypatch)
        for name in shas:
            assert client.get(f"/d/{shas[name]}").status_code == 200, name
        db_session.add(MaskStoreRevocation(source="shop"))
        db_session.commit()
        _assert_404(client.get(f"/d/{shas['shop_mask']}"))
        assert client.get(f"/d/{shas['item_mask']}").status_code == 200
        assert client.get(f"/d/{shas['shop_clean']}").status_code == 200

    @pytest.mark.parametrize("written", ["shop", " Shop "])
    def test_a_configured_revocation_withholds_them_too(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
        written: str,
    ) -> None:
        shas = self._shop(db_session, store, monkeypatch)
        revoked = Settings(
            environment="test",
            media_tiers_served=True,
            mask_store_grants=GRANTS,
            mask_store_revoked=[written],
        )
        origin_app.dependency_overrides[get_settings] = lambda: revoked
        _assert_404(client.get(f"/d/{shas['shop_mask']}"))
        assert client.get(f"/d/{shas['item_mask']}").status_code == 200

    def test_a_store_without_its_grant_has_no_mask_rungs(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shas = self._shop(db_session, store, monkeypatch)
        origin_app.dependency_overrides[get_settings] = lambda: SERVED  # no grants
        _assert_404(client.get(f"/d/{shas['shop_mask']}"))
        assert client.get(f"/d/{shas['item_mask']}").status_code == 200

    def test_revoke_store_and_restore_store_carry_the_rungs_with_the_mask(
        self,
        client: TestClient,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shas = self._shop(db_session, store, monkeypatch)
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps(GRANTS))
        get_settings.cache_clear()
        try:
            revoke_store(db_session, "shop", dry_run=False)
            _assert_404(client.get(f"/d/{shas['shop_mask']}"))
            restore_store(db_session, store, "shop", dry_run=False, backend=_Model())
            assert client.get(f"/d/{shas['shop_mask']}").status_code == 200
        finally:
            get_settings.cache_clear()

    def test_a_soft_deleted_capture_of_a_tiers_own_bytes_takes_it_offline(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        """Our rung captured again as someone's original: served while that
        capture is live, never once it is soft-deleted (a takedown of it)."""
        sha = minted["clean@256"]
        capture = ImageModel(sha256=sha, mime="image/webp", storage_key="orig/again")
        db_session.add(capture)
        db_session.commit()
        assert client.get(f"/d/{sha}").status_code == 200
        capture.deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        _assert_404(client.get(f"/d/{sha}"))
        assert client.get(f"/d/{minted['clean@400']}").status_code == 200
        assert client.get(f"/d/{minted['clean']}").status_code == 200

    def test_one_parent_holding_the_same_bytes_twice_is_judged_once(
        self, client: TestClient, minted: dict[str, str], db_session: Session
    ) -> None:
        """The same bytes under two pipeline versions of one parent: served
        with the parent, gone with it."""
        shared = db_session.get(PublicTier, (minted["clean"], TIER_PIPELINE_VERSION, 256))
        assert shared is not None
        register_public_tier(
            db_session,
            sha256=shared.sha256,
            parent_sha256=minted["clean"],
            pipeline_version="mm-tier-v2",
            long_edge_px=256,
            size=shared.bytes,
            width=shared.width,
            height=shared.height,
        )
        db_session.commit()
        assert client.get(f"/d/{shared.sha256}").status_code == 200
        _source_image(db_session, minted["clean"]).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        _assert_404(client.get(f"/d/{shared.sha256}"))

    def test_a_tier_two_parents_share_stays_while_one_of_them_is_served(
        self,
        client: TestClient,
        minted: dict[str, str],
        db_session: Session,
        store: LocalObjectStore,
    ) -> None:
        other = _seed(db_session, store, 2)
        shared = db_session.get(PublicTier, (minted["clean"], TIER_PIPELINE_VERSION, 256))
        assert shared is not None
        register_public_tier(
            db_session,
            sha256=shared.sha256,
            parent_sha256=other.sha256,
            long_edge_px=256,
            size=shared.bytes,
            width=shared.width,
            height=shared.height,
        )
        _source_image(db_session, minted["clean"]).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        assert client.get(f"/d/{shared.sha256}").status_code == 200  # the other parent's rung
        _assert_404(client.get(f"/d/{minted['clean@400']}"))
        _source_image(db_session, other.sha256).deleted_at = dt.datetime.now(dt.UTC)
        db_session.commit()
        _assert_404(client.get(f"/d/{shared.sha256}"))

    def test_bytes_that_are_not_the_tiers_are_never_sent(
        self, client: TestClient, minted: dict[str, str], store: LocalObjectStore
    ) -> None:
        sha = minted["mask@256"]
        store.put(tier_key(sha), b"corrupted", "image/webp")
        resp = client.get(f"/d/{sha}")
        assert resp.status_code == 500 and resp.headers["cache-control"] == "no-store"
        assert b"corrupted" not in resp.content

    def test_a_tier_judgment_that_raises_is_a_500_that_is_never_cached(
        self, client: TestClient, minted: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import derivative_main

        def _raises(*args: object, **kwargs: object) -> None:
            raise RuntimeError("postgresql://user:pw@db/mm: permission denied for public_tiers")

        monkeypatch.setattr(derivative_main, "tier_servable", _raises)
        resp = client.get(f"/d/{minted['clean@256']}")
        assert resp.status_code == 500 and resp.headers["cache-control"] == "no-store"
        assert b"postgresql" not in resp.content
        # a registered public object is found first: its answer does not depend on tiers
        assert client.get(f"/d/{minted['clean']}").status_code == 200

    def test_a_tier_store_failure_is_a_500_that_is_never_cached(
        self, client: TestClient, minted: dict[str, str]
    ) -> None:
        class _Down:
            def get(self, key: str) -> bytes:
                raise ConnectionError("s3://bucket/tiers/key unreachable")

        origin_app.dependency_overrides[get_object_store] = lambda: _Down()
        resp = client.get(f"/d/{minted['clean@256']}")
        assert resp.status_code == 500 and resp.headers["cache-control"] == "no-store"
        assert b"s3://" not in resp.content


class TestTierServable:
    def test_it_answers_the_row_to_serve_or_none(
        self, db_session: Session, minted: dict[str, str]
    ) -> None:
        row = tier_servable(db_session, minted["mask@256"])
        assert row is not None
        assert (row.parent_sha256, row.long_edge_px) == (minted["mask"], 256)
        assert tier_servable(db_session, "0" * 64) is None
        assert tier_servable(db_session, minted["mask"]) is None  # a public object is no tier

    def test_a_tier_left_without_its_parent_row_is_not_served(
        self, db_session: Session, minted: dict[str, str]
    ) -> None:
        """The foreign key takes a parent's tiers with it; a row that outlived
        its parent all the same (a restore, a load without constraints) is
        still no address."""
        db_session.execute(text("PRAGMA foreign_keys=OFF"))
        db_session.execute(
            text("DELETE FROM public_objects WHERE sha256 = :sha"), {"sha": minted["mask"]}
        )
        db_session.expire_all()
        assert db_session.get(PublicTier, (minted["mask"], TIER_PIPELINE_VERSION, 256)) is not None
        assert tier_servable(db_session, minted["mask@256"]) is None
        assert tier_servable(db_session, minted["clean@256"]) is not None

    def test_a_tier_of_anything_but_a_clean_copy_or_a_mask_is_not_served(
        self, db_session: Session, minted: dict[str, str]
    ) -> None:
        parent = db_session.get(PublicObject, minted["mask"])
        assert parent is not None
        parent.kind = "matted"
        db_session.flush()
        assert tier_servable(db_session, minted["mask@256"]) is None

    def test_the_revocations_default_to_the_stored_ones(
        self,
        db_session: Session,
        store: LocalObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shop = _seed(db_session, store, 3, source="shop")
        monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps(GRANTS))
        get_settings.cache_clear()
        try:
            run_mask_batch(db_session, store, backend=_Model(), dry_run=False)
            mask = db_session.execute(select(ImageMask)).scalar_one().mask_sha256
            assert mask and shop
            rung = mint_parent(db_session, store, mask).tiers[0].sha256
            assert tier_servable(db_session, rung, GRANTS) is not None
            db_session.add(MaskStoreRevocation(source="shop"))
            db_session.flush()
            assert tier_servable(db_session, rung, GRANTS) is None
            assert tier_servable(db_session, rung, GRANTS, frozenset()) is not None
        finally:
            get_settings.cache_clear()
