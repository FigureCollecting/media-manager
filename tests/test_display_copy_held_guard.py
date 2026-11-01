# ruff: noqa: F811  (the lane fixture is imported from test_display_copy)
"""SHOW1: the held guard judges the pooled verdict, not a class list --
three refusals the class list missed. Each test publishes a mask, lets the
lane seed another capture of the same picture, and asks whether the guard
held it or the published mask went offline at the origin."""

from __future__ import annotations

import io
from typing import Any

import pytest
from app.models import DisplayOriginal, FigureGallery, ImageMask, MaskStoreRevocation, PublicObject
from app.publication import mask_refusal, servable
from app.workers import display_copy as dc
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image
from PIL import Image
from sqlalchemy import select

from test_display_copy import (  # type: ignore[import-not-found]
    GRANTS,
    MFC_ITEM,
    STORE,
    _mask,
    _photo,
    _pid,
    _queue,
    _seed_all,
    lane,  # noqa: F401 -- fixture
)

MFC_USER = "https://static.myfigurecollection.net/upload/pictures/2026/10/31/{n}.jpeg"


def _resaved(data: bytes) -> bytes:
    """The same pixels, other bytes (another capture of the picture)."""
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _published(
    lane: dict[str, Any], n: int, *, url: str, source: str, figure: str
) -> tuple[str, str]:
    db = lane["db"]
    data = _photo(n)
    clean = seed_clean_image(
        db, lane["media"], data, origins=[ImageOrigin(url=url, source=source)], figure_id=figure
    )
    _mask(db, lane["media"], clean.sha256)
    mask = db.execute(select(ImageMask).where(ImageMask.source_sha256 == clean.sha256)).scalar_one()
    assert mask.status == "ok"
    assert mask_refusal(db, clean.sha256, GRANTS, frozenset()) is None
    return clean.sha256, mask.mask_sha256


def _verdict(lane: dict[str, Any], sha: str, clean: str, mask_sha: str) -> dict[str, Any]:
    db = lane["db"]
    original = db.get(DisplayOriginal, sha)
    return {
        "state": original.state,
        "reason": original.reason,
        "refusal": mask_refusal(db, clean, GRANTS, None),
        "mask_servable": servable(db, db.get(PublicObject, mask_sha), GRANTS, None),
    }


def test_an_mfc_collector_photo_of_a_masked_picture_is_held(lane: dict[str, Any]) -> None:
    """An MFC collector photo (/upload/pictures/, no manufacturer_press class)
    of an item-main picture whose mask is published. classify_origin says
    ``user_photo``, which _WITHDRAWING does not list."""
    clean, mask_sha = _published(
        lane, 41, url=MFC_ITEM.format(n=41), source="mfc", figure=_pid(941)
    )
    sha, key = lane["raw"].put(_resaved(_photo(41)))
    _queue(lane["db"], sha, key, depictions=[(_pid(41), _pid(942), MFC_USER.format(n=41), "mfc")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    v = _verdict(lane, sha, clean, mask_sha)
    assert v["state"] == "held", f"guard did not hold; mask now {v}"


def test_a_revoked_stores_capture_of_a_masked_picture_is_held(lane: dict[str, Any]) -> None:
    """A capture from a GRANTED store that `masks revoke-store` revoked:
    classify_origin says ``store_revoked``; _WITHDRAWING does not list it,
    and the guard never reads the persisted revocations."""
    db = lane["db"]
    clean, mask_sha = _published(
        lane, 42, url=MFC_ITEM.format(n=42), source="mfc", figure=_pid(943)
    )
    db.add(MaskStoreRevocation(source="amiami"))
    db.commit()
    sha, key = lane["raw"].put(_resaved(_photo(42)))
    _queue(db, sha, key, depictions=[(_pid(42), _pid(944), STORE.format(n=42), "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    v = _verdict(lane, sha, clean, mask_sha)
    assert v["state"] == "held", f"guard did not hold; mask now {v}"


def test_a_recapture_at_a_url_the_product_already_links_is_held(
    lane: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store re-serves the SAME picture at the SAME URL with new bytes
    (metadata, re-encode to the same pixels). The product already links that
    URL to the first image (uq_figure_source_url), so the seed writes NO
    gallery link for the new image: judged_origins pools it as an originless
    capture -> ``no_origin`` -> the published mask goes offline. The guard
    only classifies the URL (granted), so it does not hold."""
    monkeypatch.setattr("app.workers.mask_policy.configured_store_grants", lambda: GRANTS)
    db = lane["db"]
    url = STORE.format(n=43)
    clean, mask_sha = _published(lane, 43, url=url, source="amiami", figure=_pid(945))
    sha, key = lane["raw"].put(_resaved(_photo(43)))
    _queue(db, sha, key, depictions=[(_pid(43), _pid(945), url, "amiami")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    assert len(db.execute(select(FigureGallery)).scalars().all()) == 1  # no second link
    v = _verdict(lane, sha, clean, mask_sha)
    assert v["state"] == "held" or v["mask_servable"], f"mask went offline: {v}"


def test_a_held_capture_leaves_the_mask_servable(lane: dict[str, Any]) -> None:
    """RT1 again, end to end: held, so the published mask stays servable."""
    clean, mask_sha = _published(
        lane, 44, url=MFC_ITEM.format(n=44), source="mfc", figure=_pid(946)
    )
    sha, key = lane["raw"].put(_resaved(_photo(44)))
    _queue(lane["db"], sha, key, depictions=[(_pid(44), _pid(947), MFC_USER.format(n=44), "mfc")])
    assert _seed_all(lane)[0] == dc.EXIT_DONE
    v = _verdict(lane, sha, clean, mask_sha)
    assert v == {
        "state": "held",
        "reason": "would_withdraw_mask:user_photo",
        "refusal": None,
        "mask_servable": True,
    }
