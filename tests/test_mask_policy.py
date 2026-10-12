"""MASK-1 per-source permission split, keyed on ORIGIN, fail closed.

Measured 2026-10-09 (SIZE-MEASURE): every captured MFC image row carries role
'gallery' whether it came from /upload/items/ (the item's own main images) or
would come from /upload/pictures/ (collectors' photos), so only the URL tells
them apart. Item-main images are eligible for a mask; user photos are not --
they stay displayable, unmasked.

A store's listing photo gets a mask when its source is a registered store
source and the bytes came from one of its domains (Settings.mask_store_grants,
env MASK_STORE_GRANTS = {"<source>": ["<domain>", ...]}; none is registered
by default). MASK-G: every registered source is granted, and a revoked one
(MASK_STORE_REVOKED, ``masks revoke-store``) is refused whatever its grant.
Everything else is refused: an unregistered source or host, a host that does
not match the recorded source, a URL carrying credentials or a backslash, an
IP literal. media-manager records an origin as (source, url) only, so nothing
else about the image is consulted.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from app.config import Settings, get_settings
from app.workers.mask_policy import ImageOrigin, classify_origin, mask_eligibility

ITEM = "https://static.myfigurecollection.net/upload/items/2/1154594-1a2b3.jpg"
ITEM_ENDPOINT = (
    "https://myfigurecollection.net/?_tb=commit&commit=nsp&objectType=item&objectId=1154594&size=1"
)
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/123456.jpeg"
AMIAMI = "https://img.amiami.com/images/product/main/241/FIGURE-123456.jpg"


def _grant(monkeypatch: pytest.MonkeyPatch, grants: dict[str, list[str]]) -> None:
    monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps(grants))
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_settings() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def granted(monkeypatch: pytest.MonkeyPatch) -> None:
    _grant(monkeypatch, {"amiami": ["amiami.com"], "shop": ["shop.example"]})


class TestMfc:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (ITEM, "item_main"),
            (ITEM_ENDPOINT, "item_main"),
            ("https://images.myfigurecollection.net/upload/items/0/9-9.jpg", "item_main"),
            ("https://MyFigureCollection.NET/upload/items/0/9-9.jpg", "item_main"),
            (PICTURE, "user_photo"),
            (
                "https://myfigurecollection.net/?_tb=commit&objectType=picture&objectId=3",
                "user_photo",
            ),
            ("https://static.myfigurecollection.net/upload/users/1/avatar.png", "unknown_origin"),
            ("https://myfigurecollection.net/item/1154594", "unknown_origin"),
            ("https://myfigurecollection.net/?objectType=item", "item_main"),
            ("https://myfigurecollection.net/?objectType=Item", "unknown_origin"),
            ("https://myfigurecollection.net/?objectType=item&objectType=x", "unknown_origin"),
            ("https://myfigurecollection.net/?objectType=item&objectType=picture", "user_photo"),
            ("https://myfigurecollection.net/item/1?objectType=item", "unknown_origin"),
        ],
    )
    def test_mfc_is_classified_by_path(self, url: str, expected: str) -> None:
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://static.myfigurecollection.net/upload/items/../pictures/2024/1.jpg",
            "https://static.myfigurecollection.net/upload/items/%2e%2e/pictures/2024/1.jpg",
            "https://static.myfigurecollection.net/upload/items/%2E%2E%2Fpictures/2024/1.jpg",
        ],
    )
    def test_a_path_that_climbs_out_of_items_is_judged_where_it_lands(self, url: str) -> None:
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == "user_photo"

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("static.myfigurecollection.net.", "user_photo"),
            ("myfigurecollection.net..", "user_photo"),
        ],
    )
    def test_a_trailing_dot_host_is_still_mfc(self, host: str, expected: str) -> None:
        url = f"https://{host}/upload/pictures/2024/01/02/1.jpeg"
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == expected

    def test_an_mfc_host_recorded_under_another_source_is_unknown(self) -> None:
        assert classify_origin(ImageOrigin(url=ITEM, source="amiami")) == "unknown_origin"

    def test_an_mfc_row_on_a_host_that_is_not_mfc_is_unknown(self) -> None:
        url = "https://cdn.example-cdn.com/upload/items/2/1154594-1a2b3.jpg"
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == "unknown_origin"

    def test_a_lookalike_host_is_not_mfc(self) -> None:
        url = "https://evilmyfigurecollection.net/upload/items/1/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == "unknown_origin"

    def test_no_url_is_no_origin(self) -> None:
        assert classify_origin(ImageOrigin(url=None, source="mfc")) == "no_origin"
        assert classify_origin(ImageOrigin(url="", source="mfc")) == "no_origin"

    @pytest.mark.parametrize(
        "url",
        [
            "not a url",
            "ftp://static.myfigurecollection.net/upload/items/0/9-9.jpg",
            "https:///upload/items/0/9-9.jpg",
            "//static.myfigurecollection.net/upload/items/0/9-9.jpg",
        ],
    )
    def test_an_unreadable_or_non_http_url_is_unknown(self, url: str) -> None:
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == "unknown_origin"


class TestStores:
    def test_a_granted_store_host_is_a_store_listing(self, granted: None) -> None:
        assert classify_origin(ImageOrigin(url=AMIAMI, source="amiami")) == "store_listing"

    def test_the_granted_domain_itself_is_a_store_listing(self, granted: None) -> None:
        url = "https://amiami.com/p/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="amiami")) == "store_listing"

    def test_a_trailing_dot_granted_host_is_normalised(self, granted: None) -> None:
        url = "https://img.amiami.com./images/product/main/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="amiami")) == "store_listing"

    def test_with_no_grant_a_store_listing_gets_no_mask(self) -> None:
        assert classify_origin(ImageOrigin(url=AMIAMI, source="amiami")) == "not_granted"

    @pytest.mark.parametrize(
        ("url", "source"),
        [
            ("https://i.imgur.com/abc.jpg", "amiami"),
            ("https://evilamiami.com/p/1.jpg", "amiami"),
            ("https://amiami.com.evil.test/p/1.jpg", "amiami"),
            ("https://img.amiami.com/p/1.jpg", "shop"),
            ("https://r.hpoi.net/gk/user/album/1.jpg", "hpoi"),
            ("https://i.imgur.com/abc.jpg", "imgur"),
            ("https://pbs.twimg.com/media/abc.jpg", "twitter"),
            ("http://203.0.113.7/p/1.jpg", "shop"),
            ("http://[2001:db8::1]/p/1.jpg", "shop"),
            ("https://shop.example/p/1.jpg", None),
        ],
    )
    def test_anything_not_granted_gets_no_mask(
        self, granted: None, url: str, source: str | None
    ) -> None:
        assert classify_origin(ImageOrigin(url=url, source=source)) == "not_granted"

    @pytest.mark.parametrize(
        ("url", "source"),
        [
            ("https://evil.test@shop.example/p/1.jpg", "shop"),
            ("https://user:pw@shop.example/p/1.jpg", "shop"),
            ("https://evil.test\\@static.myfigurecollection.net/upload/items/1/1-a.jpg", "mfc"),
            ("https://shop.example\\.evil.test/p/1.jpg", "shop"),
            ("https:///p/1.jpg", "shop"),
        ],
    )
    def test_credentials_or_a_backslash_make_the_host_ambiguous(
        self, granted: None, url: str, source: str
    ) -> None:
        assert classify_origin(ImageOrigin(url=url, source=source)) == "unknown_origin"

    @pytest.mark.parametrize(
        ("url", "source"),
        [
            ("https://evil.test img.shop.example/p/1.jpg", "shop"),
            ("https://evil.test%20img.shop.example/p/1.jpg", "shop"),
            ("https:// img.shop.example/p/1.jpg", "shop"),
            ("https://evil.test static.myfigurecollection.net/upload/items/1/1-a.jpg", "mfc"),
            ("https://evil.test%2f.static.myfigurecollection.net/upload/items/1/1-a.jpg", "mfc"),
        ],
    )
    def test_a_host_that_is_not_a_plain_name_is_ambiguous(self, url: str, source: str) -> None:
        """Whitespace or a percent escape in the host: which host served the
        bytes is as ambiguous as with a backslash, so no mask."""
        grants = {"shop": ["shop.example"]}
        assert classify_origin(ImageOrigin(url=url, source=source), grants) == "unknown_origin"

    @pytest.mark.parametrize(
        ("url", "source"),
        [
            ("http://[::1/p/1.jpg", "shop"),
            ("https://[::1]evil/p/1.jpg", "shop"),
            ("http://[evil]/p/1.jpg", "shop"),
            ("https://ex\uff0fample.shop.example/p/1.jpg", "shop"),
            ("http://[::1/upload/items/1/1-a.jpg", "mfc"),
        ],
    )
    def test_a_url_the_parser_rejects_is_unknown_not_an_error(self, url: str, source: str) -> None:
        """urlsplit raises on these; the policy fails closed instead, so one
        such row can never abort a batch or a re-evaluation."""
        grants = {"shop": ["shop.example"]}
        assert classify_origin(ImageOrigin(url=url, source=source), grants) == "unknown_origin"
        assert mask_eligibility([ImageOrigin(url=url, source=source)], grants).reason == (
            "unknown_origin"
        )

    @pytest.mark.parametrize(
        ("url", "source"),
        [
            ("https://img.shop.example:1.evil.test/p/1.jpg", "shop"),
            ("https://img.shop.example:99999/p/1.jpg", "shop"),
            ("https://static.myfigurecollection.net:x.evil.test/upload/items/1/1-a.jpg", "mfc"),
        ],
    )
    def test_a_port_that_is_not_a_port_makes_the_host_ambiguous(
        self, url: str, source: str
    ) -> None:
        grants = {"shop": ["shop.example"]}
        assert classify_origin(ImageOrigin(url=url, source=source), grants) == "unknown_origin"

    def test_a_real_port_is_still_the_host(self) -> None:
        o = ImageOrigin(url="https://img.shop.example:8443/p/1.jpg", source="shop")
        assert classify_origin(o, {"shop": ["shop.example"]}) == "store_listing"

    def test_a_non_ascii_host_is_ambiguous_even_when_granted(self) -> None:
        """A fetcher reaches an IDN host by its punycode; the letters alone are
        ambiguous (and a lookalike), so even a grant naming them gives no mask."""
        o = ImageOrigin(url="https://img.sh\u00f6p.example/p/1.jpg", source="shop")
        assert classify_origin(o, {"shop": ["sh\u00f6p.example"]}) == "unknown_origin"

    def test_a_schemeless_url_is_unknown_even_on_a_granted_host(self) -> None:
        o = ImageOrigin(url="//cdn.shop.example/p/1.jpg", source="shop")
        assert classify_origin(o, store_grants={"shop": ["shop.example"]}) == "unknown_origin"

    def test_a_grant_domain_is_read_without_surrounding_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MASK_STORE_GRANTS is hand-written JSON: whitespace around a domain
        neither voids the grant nor, alone, grants anything."""
        url = "https://cdn.shop.example/p/1.jpg"
        _grant(monkeypatch, {"shop": ["  shop.example \n"]})
        assert classify_origin(ImageOrigin(url=url, source="shop")) == "store_listing"
        _grant(monkeypatch, {"shop": ["   "]})
        assert classify_origin(ImageOrigin(url=url, source="shop")) == "not_granted"

    def test_grants_are_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _grant(monkeypatch, {"shop": ["Shop.Example.", ""]})
        url = "https://cdn.shop.example/p/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="shop")) == "store_listing"
        other = "https://shop.examplex/p/1.jpg"
        assert classify_origin(ImageOrigin(url=other, source="shop")) == "not_granted"

    def test_explicit_grants_override_the_settings(self, granted: None) -> None:
        o = ImageOrigin(url="https://x.example/p.jpg", source="x")
        assert classify_origin(o, store_grants={"x": ["x.example"]}) == "store_listing"
        assert classify_origin(ImageOrigin(url=AMIAMI, source="amiami"), store_grants={}) == (
            "not_granted"
        )


class TestMaskEligibility:
    def test_item_main_is_eligible(self) -> None:
        e = mask_eligibility([ImageOrigin(url=ITEM, source="mfc")])
        assert (e.eligible, e.origin_class, e.reason) == (True, "item_main", "eligible")

    def test_a_granted_store_listing_is_eligible(self, granted: None) -> None:
        e = mask_eligibility([ImageOrigin(url="https://shop.example/p/1.jpg", source="shop")])
        assert (e.eligible, e.origin_class) == (True, "store_listing")

    def test_an_ungranted_store_listing_is_refused_by_name(self) -> None:
        e = mask_eligibility([ImageOrigin(url="https://shop.example/p/1.jpg", source="shop")])
        assert (e.eligible, e.origin_class, e.reason) == (False, "not_granted", "not_granted")

    def test_user_photo_is_not(self) -> None:
        e = mask_eligibility([ImageOrigin(url=PICTURE, source="mfc")])
        assert (e.eligible, e.origin_class, e.reason) == (False, "user_photo", "user_photo")

    def test_no_origins_at_all_is_refused(self) -> None:
        e = mask_eligibility([])
        assert (e.eligible, e.origin_class, e.reason) == (False, "no_origin", "no_origin")

    def test_one_ineligible_origin_refuses_the_bytes(self) -> None:
        """The same bytes seen as an item image AND as a collector's photo
        are a collector's photo somewhere: no mask (fail closed)."""
        e = mask_eligibility(
            [
                ImageOrigin(url=ITEM, source="mfc"),
                ImageOrigin(url=PICTURE, source="mfc"),
            ]
        )
        assert (e.eligible, e.origin_class) == (False, "user_photo")

    def test_unknown_origin_is_refused(self) -> None:
        e = mask_eligibility([ImageOrigin(url="https://myfigurecollection.net/x", source="mfc")])
        assert (e.eligible, e.reason) == (False, "unknown_origin")

    def test_two_eligible_origins_report_the_first(self, granted: None) -> None:
        e = mask_eligibility(
            [
                ImageOrigin(url=ITEM, source="mfc"),
                ImageOrigin(url="https://shop.example/p/1.jpg", source="shop"),
            ]
        )
        assert (e.eligible, e.origin_class) == (True, "item_main")


def _revoke(monkeypatch: pytest.MonkeyPatch, sources: list[str]) -> None:
    monkeypatch.setenv("MASK_STORE_REVOKED", json.dumps(sources))
    get_settings.cache_clear()


class TestStoreRevocation:
    """MASK-G (Ross 2026-10-12, option b): every REGISTERED store source --
    one named in MASK_STORE_GRANTS with the domains its listing photos come
    from -- is granted, with no per-store opt-in; a per-source deny list
    (MASK_STORE_REVOKED, plus the revocations ``masks revoke-store``
    persists) wins over any grant."""

    def test_nothing_is_revoked_by_default(self) -> None:
        assert Settings(environment="test").mask_store_revoked == []

    def test_every_registered_store_source_is_granted(self) -> None:
        grants = {"amiami": ["amiami.com"], "shop": ["shop.example"]}
        for url, source in ((AMIAMI, "amiami"), ("https://cdn.shop.example/p.jpg", "shop")):
            assert classify_origin(ImageOrigin(url=url, source=source), grants) == "store_listing"

    def test_a_revoked_source_is_refused_whatever_its_grant(self, granted: None) -> None:
        o = ImageOrigin(url=AMIAMI, source="amiami")
        assert classify_origin(o, revoked=frozenset({"amiami"})) == "store_revoked"
        e = mask_eligibility([o], revoked=frozenset({"amiami"}))
        assert (e.eligible, e.origin_class, e.reason) == (False, "store_revoked", "store_revoked")

    def test_revoking_one_source_leaves_the_others_granted(self, granted: None) -> None:
        o = ImageOrigin(url="https://shop.example/p/1.jpg", source="shop")
        assert classify_origin(o, revoked=frozenset({"amiami"})) == "store_listing"
        item = ImageOrigin(url=ITEM, source="mfc")
        assert classify_origin(item, revoked=frozenset({"amiami"})) == "item_main"

    def test_the_deny_list_is_per_source_mfc_included(self) -> None:
        item = ImageOrigin(url=ITEM, source="mfc")
        assert classify_origin(item, revoked=frozenset({"mfc"})) == "store_revoked"

    def test_the_configured_deny_list_is_read_normalised(
        self, granted: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MASK_STORE_REVOKED is hand-written JSON: case and surrounding
        whitespace never let a revoked source through."""
        _revoke(monkeypatch, [" AmiAmi\n"])
        assert classify_origin(ImageOrigin(url=AMIAMI, source="amiami")) == "store_revoked"
        upper = ImageOrigin(url=AMIAMI, source="AMIAMI ")
        assert classify_origin(upper, revoked=frozenset({"amiami"})) == "store_revoked"

    def test_an_explicit_deny_list_overrides_the_configured_one(
        self, granted: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _revoke(monkeypatch, ["amiami"])
        o = ImageOrigin(url=AMIAMI, source="amiami")
        assert classify_origin(o, revoked=frozenset()) == "store_listing"
