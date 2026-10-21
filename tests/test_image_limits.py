"""MM-CAPS-1: the explicit limits every original passes before anything decodes it.

An original reaches the clean-copy step from the raw bucket as stored bytes (maybe gzip), and until now the only
guards between those bytes and a full decode were Pillow's DEFAULT pixel guard and nothing at all on the gzip step
(``gzip.decompress`` of a bomb allocates whatever it expands to). These tests pin: a byte cap before any decode, a
pixel cap read from the HEADER (no pixel is decoded to learn the size), a decoded-size cap for deep modes, and a gzip
step that stops at its output cap. A refusal is an :class:`ImageLimitError` with a stable ``code``; an image the
limits do not concern (including one Pillow cannot read) is left to fail where it always did.
"""

from __future__ import annotations

import gzip
import io
import struct
import time
import zlib

import pytest
from app.workers import image_limits as il
from app.workers.image_limits import (
    ImageLimitError,
    Limits,
    check_image,
    current_limits,
    gunzip_capped,
)
from PIL import Image


def _png(width: int, height: int, mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), 0).save(out, format="PNG")
    return out.getvalue()


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png_claiming(width: int, height: int, *, color_type: int = 2, bit_depth: int = 8) -> bytes:
    """A PNG whose HEADER says width x height and whose pixel data is a few bytes: what a decompression bomb looks like
    to a header read. Nothing in it can be decoded to that size."""
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b""))
        + _chunk(b"IEND", b"")
    )


SMALL = Limits(max_bytes=1_000_000, max_pixels=10_000, max_decoded_bytes=1_000_000)


def test_the_defaults_are_the_scrapers_byte_cap_and_a_generous_pixel_cap() -> None:
    lim = current_limits({})
    assert (
        lim.max_bytes == 32 * 1024 * 1024
    )  # the scraper's DEFAULT_MAX_IMAGE_BYTES: nothing it stored is refused for size
    assert lim.max_pixels == 64_000_000
    assert lim.max_decoded_bytes == 256 * 1024 * 1024


def test_the_limits_can_be_set_by_environment_and_a_bad_value_keeps_the_default() -> None:
    lim = current_limits(
        {
            "MM_MAX_IMAGE_BYTES": "1000",
            "MM_MAX_IMAGE_PIXELS": "2500",
            "MM_MAX_DECODED_BYTES": "9000",
        }
    )
    assert (lim.max_bytes, lim.max_pixels, lim.max_decoded_bytes) == (1000, 2500, 9000)
    bad = current_limits(
        {"MM_MAX_IMAGE_BYTES": "lots", "MM_MAX_IMAGE_PIXELS": "-4", "MM_MAX_DECODED_BYTES": "0"}
    )
    assert bad == current_limits({})


def test_an_ordinary_image_passes_and_reports_its_size() -> None:
    assert check_image(_png(40, 30), limits=SMALL) == (40, 30)


def test_a_jpeg_and_a_webp_pass_too() -> None:
    for fmt in ("JPEG", "WEBP"):
        out = io.BytesIO()
        Image.new("RGB", (50, 20), (10, 20, 30)).save(out, format=fmt)
        assert check_image(out.getvalue(), limits=SMALL) == (50, 20)


def test_bytes_over_the_cap_are_refused_before_pillow_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*a: object, **k: object) -> None:
        raise AssertionError("the image must not be opened")

    monkeypatch.setattr(il.Image, "open", explode)
    with pytest.raises(ImageLimitError) as err:
        check_image(
            b"x" * 101, limits=Limits(max_bytes=100, max_pixels=10**9, max_decoded_bytes=10**12)
        )
    assert err.value.code == "too_large_bytes"


def test_the_byte_cap_is_inclusive() -> None:
    data = _png(8, 8)
    assert check_image(
        data, limits=Limits(max_bytes=len(data), max_pixels=10_000, max_decoded_bytes=10**9)
    ) == (8, 8)
    with pytest.raises(ImageLimitError):
        check_image(
            data, limits=Limits(max_bytes=len(data) - 1, max_pixels=10_000, max_decoded_bytes=10**9)
        )


def test_too_many_pixels_is_read_from_the_header_and_refused_without_decoding() -> None:
    # 60000 x 60000 = 3.6 billion pixels claimed by a ~70 byte file: no decode could even start
    started = time.monotonic()
    with pytest.raises(ImageLimitError) as err:
        check_image(
            _png_claiming(60_000, 60_000),
            limits=Limits(max_bytes=10**6, max_pixels=64_000_000, max_decoded_bytes=10**12),
        )
    assert err.value.code == "too_many_pixels"
    assert time.monotonic() - started < 1.0


def test_the_pixel_cap_is_inclusive() -> None:
    data = _png(100, 100)
    assert check_image(
        data, limits=Limits(max_bytes=10**6, max_pixels=10_000, max_decoded_bytes=10**9)
    ) == (100, 100)
    with pytest.raises(ImageLimitError) as err:
        check_image(data, limits=Limits(max_bytes=10**6, max_pixels=9_999, max_decoded_bytes=10**9))
    assert err.value.code == "too_many_pixels"


def test_a_pillow_bomb_error_is_a_refusal_not_a_crash() -> None:
    # past twice Pillow's own bound (set to our cap) Pillow raises DecompressionBombError while OPENING; same refusal
    with pytest.raises(ImageLimitError) as err:
        check_image(
            _png_claiming(30_000, 30_000),
            limits=Limits(max_bytes=10**6, max_pixels=1_000_000, max_decoded_bytes=10**12),
        )
    assert err.value.code == "too_many_pixels"


def test_decoded_size_is_at_least_the_rgba_working_copy_so_it_can_bind_inside_the_pixel_cap() -> (
    None
):
    # the clean-copy step converts to RGBA and copies to numpy: 4 bytes a pixel, whatever the file's own depth
    rgb = _png_claiming(
        1000, 1000
    )  # 1 million pixels, 3 bytes a pixel in the file, 4 once it is worked on
    with pytest.raises(ImageLimitError) as err:
        check_image(
            rgb, limits=Limits(max_bytes=10**6, max_pixels=2_000_000, max_decoded_bytes=3_999_999)
        )
    assert err.value.code == "decoded_too_large"
    assert check_image(
        rgb, limits=Limits(max_bytes=10**6, max_pixels=2_000_000, max_decoded_bytes=4_000_000)
    ) == (1000, 1000)


def test_what_pillow_cannot_read_is_not_this_modules_business() -> None:
    assert check_image(b"not an image at all", limits=SMALL) is None
    assert check_image(b"", limits=SMALL) is None
    assert check_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5, limits=SMALL) is None


def test_the_global_pillow_guard_is_set_to_the_pixel_cap_on_import() -> None:
    assert current_limits().max_pixels == Image.MAX_IMAGE_PIXELS


class TestGunzipCapped:
    def test_inflates_a_gzip_member_exactly(self) -> None:
        raw = bytes(range(256)) * 50
        assert gunzip_capped(gzip.compress(raw), 100_000) == raw

    def test_inflates_concatenated_members_like_gzip_decompress_does(self) -> None:
        assert gunzip_capped(gzip.compress(b"abc") + gzip.compress(b"def"), 100) == b"abcdef"

    def test_the_cap_is_inclusive(self) -> None:
        packed = gzip.compress(b"a" * 1000)
        assert len(gunzip_capped(packed, 1000)) == 1000
        with pytest.raises(ImageLimitError) as err:
            gunzip_capped(packed, 999)
        assert err.value.code == "gzip_too_large"

    def test_a_bomb_stops_at_the_cap_without_expanding_it(self) -> None:
        bomb = gzip.compress(
            b"\x00" * (200 * 1024 * 1024), compresslevel=9
        )  # ~200 KB packed, 200 MB expanded
        assert len(bomb) < 1_000_000
        started = time.monotonic()
        with pytest.raises(ImageLimitError) as err:
            gunzip_capped(bomb, 1024 * 1024)
        assert err.value.code == "gzip_too_large"
        assert time.monotonic() - started < 2.0

    def test_a_bomb_split_across_members_is_capped_in_total(self) -> None:
        member = gzip.compress(b"\x00" * 600_000)
        with pytest.raises(ImageLimitError):
            gunzip_capped(member * 3, 1_000_000)

    def test_not_gzip_fails_as_gzip_decompress_would(self) -> None:
        with pytest.raises(gzip.BadGzipFile):
            gunzip_capped(b"this is not gzip at all", 1000)

    def test_a_truncated_member_fails_as_gzip_decompress_would(self) -> None:
        with pytest.raises(EOFError):
            gunzip_capped(gzip.compress(b"hello world" * 100)[:-9], 100_000)

    def test_empty_input_is_an_error_not_an_empty_image(self) -> None:
        with pytest.raises(EOFError):
            gunzip_capped(b"", 1000)
