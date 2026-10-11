"""MASK-1: which clean images may get a mask, keyed on where the bytes came from.

A mask is our derivative work on an image, so it is made only for imagery
that represents the ITEM: a store's listing photos and MFC's item-main
images. A collector's own photo (MFC ``/upload/pictures/``, a ``user`` role
or a ``user_photo`` source class) is never masked; it may still be shown,
unmasked, wherever display allows it. Display itself is not decided here:
the spine's rights gate and the coordinator's display restrictions own that.

WHY THE ORIGIN AND NOT THE ROLE. Measured 2026-10-09 (SIZE-MEASURE): every
captured MFC image row carries role ``gallery``, whether it came from
``/upload/items/`` or would come from ``/upload/pictures/``; only the URL the
bytes were captured from separates the two lanes.

FAIL CLOSED. An MFC URL that matches neither rule, an unreadable URL, and
bytes with no known origin get no mask; so do bytes seen under several
origins when ANY of them is ineligible (the same picture captured as an item
image and as a collector's photo is a collector's photo somewhere).
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, unquote, urlsplit

OriginClass = Literal["item_main", "store_listing", "user_photo", "unknown_origin", "no_origin"]

#: The origin classes a mask may be made for.
MASKABLE: frozenset[str] = frozenset({"item_main", "store_listing"})

_MFC_DOMAIN = "myfigurecollection.net"
_MFC_ITEM_PREFIX = "/upload/items/"
_MFC_PICTURE_PREFIX = "/upload/pictures/"


@dataclass(frozen=True)
class ImageOrigin:
    """One place the clean image's original bytes were captured from."""

    url: str | None
    source: str | None = None
    role: str | None = None
    source_class: str | None = None


@dataclass(frozen=True)
class MaskEligibility:
    eligible: bool
    origin_class: OriginClass
    #: ``"eligible"``, or the origin class that refused the mask.
    reason: str


def _is_mfc(host: str) -> bool:
    return host == _MFC_DOMAIN or host.endswith("." + _MFC_DOMAIN)


def _classify_mfc(path: str, query: str) -> OriginClass:
    if path.startswith(_MFC_ITEM_PREFIX):
        return "item_main"
    if path.startswith(_MFC_PICTURE_PREFIX):
        return "user_photo"
    if path == "/":
        # The item image endpoint: /?_tb=commit&...&objectType=item&objectId=N
        object_types = parse_qs(query).get("objectType", [])
        if "picture" in object_types:
            return "user_photo"
        if object_types == ["item"]:
            return "item_main"
    return "unknown_origin"


def classify_origin(origin: ImageOrigin) -> OriginClass:
    if origin.source_class == "user_photo" or origin.role == "user":
        return "user_photo"
    if not origin.url:
        return "no_origin"
    parts = urlsplit(origin.url)
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host:
        return "unknown_origin"
    if not _is_mfc(host):
        return "store_listing"
    # Judge the path where it actually lands: percent-decode, then resolve
    # any ../ so /upload/items/../pictures/ is a picture.
    path = posixpath.normpath(unquote(parts.path or "/"))
    return _classify_mfc(path, parts.query)


def mask_eligibility(origins: Sequence[ImageOrigin]) -> MaskEligibility:
    if not origins:
        return MaskEligibility(eligible=False, origin_class="no_origin", reason="no_origin")
    classes = [classify_origin(o) for o in origins]
    for cls in classes:
        if cls not in MASKABLE:
            return MaskEligibility(eligible=False, origin_class=cls, reason=cls)
    return MaskEligibility(eligible=True, origin_class=classes[0], reason="eligible")
