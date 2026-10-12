"""MASK-1: which clean images may get a mask, keyed on where the bytes came from.

A mask is our derivative work on an image, so it is made only for imagery
that represents the ITEM, and only where its source permits it:

- MFC (source ``mfc`` on an MFC host): the item-main images under
  ``/upload/items/`` (or the item image endpoint) get a mask; a collector's
  own photo (``/upload/pictures/``) never does -- it may still be shown,
  unmasked, wherever display allows it.
- A store's listing photo gets a mask when its source is a REGISTERED store
  source and the bytes came from one of its domains
  (``Settings.mask_store_grants``, env ``MASK_STORE_GRANTS`` =
  ``{"<source>": ["<domain>", ...]}``, a domain covering its subdomains).
  MASK-G (Ross 2026-10-12, option b): every registered store source is
  granted -- cut-outs for every store, no per-store opt-in -- and revocable
  per store: a source on the deny list (``MASK_STORE_REVOKED`` plus the
  revocations ``masks revoke-store`` persists) is refused whatever its grant,
  MFC included (``store_revoked``).

Display itself is not decided here: the spine's rights gate and the
coordinator's display restrictions own that.

WHY THE ORIGIN AND NOT THE ROLE. Measured 2026-10-09 (SIZE-MEASURE): every
captured MFC image row carries role ``gallery``, whether it came from
``/upload/items/`` or would come from ``/upload/pictures/``; only the URL the
bytes were captured from separates the two lanes. media-manager records an
origin as (source, url) only (``figure_galleries``), so that is all this
policy reads.

FAIL CLOSED. No mask for: a revoked source (``store_revoked``); an
unregistered source or a host outside its domains (``not_granted``); an
MFC URL that matches neither rule, an MFC host recorded under another source
or an ``mfc`` row on another host, a URL carrying credentials or a
backslash, a host that is not a plain ASCII name (whitespace, a percent
escape, a non-ASCII letter), a port that is not a port, a URL the parser
rejects (which host served it is ambiguous), a URL that is not http(s),
schemeless included (``unknown_origin``); bytes with no origin
(``no_origin``); and bytes seen
under several origins when ANY of them is ineligible (the same picture
captured as an item image and as a collector's photo is a collector's photo
somewhere). Hosts are compared lower-cased without trailing dots.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, unquote, urlsplit

OriginClass = Literal[
    "item_main",
    "store_listing",
    "user_photo",
    "store_revoked",
    "not_granted",
    "unknown_origin",
    "no_origin",
]

#: The origin classes a mask may be made for.
MASKABLE: frozenset[str] = frozenset({"item_main", "store_listing"})

#: Per registered store source, the domains its listing photos come from.
StoreGrants = Mapping[str, Sequence[str]]

#: The revoked sources, as ``source_key`` names.
Revoked = frozenset[str]

MFC_SOURCE = "mfc"
_MFC_DOMAIN = "myfigurecollection.net"
_MFC_ITEM_PREFIX = "/upload/items/"
_MFC_PICTURE_PREFIX = "/upload/pictures/"

#: A host as a fetcher resolves it: a DNS name, an IPv4 or an IPv6 literal
#: (urlsplit lower-cases it and drops the brackets). Anything else -- a
#: space, a percent escape, a non-ASCII letter -- is ambiguous.
_PLAIN_HOST = re.compile(r"[a-z0-9._:-]+")


@dataclass(frozen=True)
class ImageOrigin:
    """One place the clean image's original bytes were captured from."""

    url: str | None
    source: str | None = None


@dataclass(frozen=True)
class MaskEligibility:
    eligible: bool
    origin_class: OriginClass
    #: ``"eligible"``, or the origin class that refused the mask.
    reason: str


def _domain(name: str) -> str:
    return name.strip().lower().rstrip(".")


def source_key(name: str) -> str:
    """A source as the deny list compares it: stripped, lower-cased."""
    return name.strip().lower()


def _within(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


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


def configured_store_grants() -> StoreGrants:
    from ..config import get_settings

    return get_settings().mask_store_grants


def configured_revocations() -> Revoked:
    """MASK_STORE_REVOKED alone; app.publication.store_revocations adds the
    persisted revocations."""
    from ..config import get_settings

    return frozenset(source_key(s) for s in get_settings().mask_store_revoked)


def classify_origin(
    origin: ImageOrigin, store_grants: StoreGrants | None = None, revoked: Revoked | None = None
) -> OriginClass:
    """``store_grants`` defaults to the configured grants and ``revoked`` to
    the configured deny list."""
    if not origin.url:
        return "no_origin"
    if source_key(origin.source or "") in (
        configured_revocations() if revoked is None else revoked
    ):
        return "store_revoked"
    try:
        parts = urlsplit(origin.url)
        parts.port  # noqa: B018 -- raises on a port that is not one
    except ValueError:
        # Unparseable (an unclosed or non-IP bracket, a netloc that changes
        # under NFKC) or a port that is not a port: which host served the
        # bytes is ambiguous.
        return "unknown_origin"
    if parts.scheme not in ("http", "https") or "@" in parts.netloc or "\\" in origin.url:
        return "unknown_origin"
    host = (parts.hostname or "").rstrip(".")
    if not _PLAIN_HOST.fullmatch(host):
        return "unknown_origin"
    if _within(host, _MFC_DOMAIN) or origin.source == MFC_SOURCE:
        if not (_within(host, _MFC_DOMAIN) and origin.source == MFC_SOURCE):
            return "unknown_origin"
        # Judge the path where it actually lands: percent-decode, then resolve
        # any ../ so /upload/items/../pictures/ is a picture.
        path = posixpath.normpath(unquote(parts.path or "/"))
        return _classify_mfc(path, parts.query)
    grants = configured_store_grants() if store_grants is None else store_grants
    granted = grants.get(origin.source or "", ())
    if any(_within(host, _domain(d)) for d in granted):
        return "store_listing"
    return "not_granted"


def mask_eligibility(
    origins: Sequence[ImageOrigin],
    store_grants: StoreGrants | None = None,
    revoked: Revoked | None = None,
) -> MaskEligibility:
    if not origins:
        return MaskEligibility(eligible=False, origin_class="no_origin", reason="no_origin")
    classes = [classify_origin(o, store_grants, revoked) for o in origins]
    for cls in classes:
        if cls not in MASKABLE:
            return MaskEligibility(eligible=False, origin_class=cls, reason=cls)
    return MaskEligibility(eligible=True, origin_class=classes[0], reason="eligible")
