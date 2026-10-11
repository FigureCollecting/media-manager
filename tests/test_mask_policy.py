"""MASK-1 per-source permission split, keyed on ORIGIN.

Measured 2026-10-09 (SIZE-MEASURE): every captured MFC image row carries role
'gallery' whether it came from /upload/items/ (the item's own main images) or
would come from /upload/pictures/ (collectors' photos), so the role cannot
tell them apart. Item-main images are eligible for a mask; user photos are
not -- they stay displayable, unmasked. Anything the rules cannot place is
refused (fail closed).
"""

from __future__ import annotations

import pytest
from app.workers.mask_policy import ImageOrigin, classify_origin, mask_eligibility

ITEM = "https://static.myfigurecollection.net/upload/items/2/1154594-1a2b3.jpg"
ITEM_ENDPOINT = (
    "https://myfigurecollection.net/?_tb=commit&commit=nsp&objectType=item&objectId=1154594&size=1"
)
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/123456.jpeg"


class TestClassifyOrigin:
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
        ],
    )
    def test_mfc_is_classified_by_path(self, url: str, expected: str) -> None:
        assert classify_origin(ImageOrigin(url=url, source="mfc", role="gallery")) == expected

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

    def test_a_lookalike_host_is_not_mfc(self) -> None:
        url = "https://evilmyfigurecollection.net/upload/pictures/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="other")) == "store_listing"

    def test_a_store_listing_image_is_eligible_kind(self) -> None:
        url = "https://img.amiami.com/images/product/main/241/FIGURE-123456.jpg"
        assert classify_origin(ImageOrigin(url=url, source="amiami")) == "store_listing"

    def test_a_user_class_wins_over_an_item_path(self) -> None:
        o = ImageOrigin(url=ITEM, source="mfc", role="gallery", source_class="user_photo")
        assert classify_origin(o) == "user_photo"

    def test_a_user_role_wins_over_any_host(self) -> None:
        url = "https://www.hpoi.net/gallery/user/1.jpg"
        assert classify_origin(ImageOrigin(url=url, source="hpoi", role="user")) == "user_photo"

    def test_no_url_is_no_origin(self) -> None:
        assert classify_origin(ImageOrigin(url=None, source="mfc", role="gallery")) == "no_origin"
        assert classify_origin(ImageOrigin(url="", source="mfc")) == "no_origin"

    def test_no_url_but_a_user_class_is_still_a_user_photo(self) -> None:
        o = ImageOrigin(url=None, source_class="user_photo")
        assert classify_origin(o) == "user_photo"

    @pytest.mark.parametrize("url", ["not a url", "ftp://static.myfigurecollection.net/x"])
    def test_an_unreadable_or_non_http_url_is_unknown(self, url: str) -> None:
        assert classify_origin(ImageOrigin(url=url, source="mfc")) == "unknown_origin"


class TestMaskEligibility:
    def test_item_main_is_eligible(self) -> None:
        e = mask_eligibility([ImageOrigin(url=ITEM, source="mfc", role="gallery")])
        assert (e.eligible, e.origin_class, e.reason) == (True, "item_main", "eligible")

    def test_store_listing_is_eligible(self) -> None:
        e = mask_eligibility([ImageOrigin(url="https://shop.example/p/1.jpg", source="shop")])
        assert (e.eligible, e.origin_class) == (True, "store_listing")

    def test_user_photo_is_not(self) -> None:
        e = mask_eligibility([ImageOrigin(url=PICTURE, source="mfc", role="gallery")])
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

    def test_two_eligible_origins_report_the_first(self) -> None:
        e = mask_eligibility(
            [
                ImageOrigin(url=ITEM, source="mfc"),
                ImageOrigin(url="https://shop.example/p/1.jpg", source="shop"),
            ]
        )
        assert (e.eligible, e.origin_class) == (True, "item_main")
