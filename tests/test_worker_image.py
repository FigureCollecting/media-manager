"""Dockerfile.worker carries exactly the matting weights the mask lane was
measured on (MASK-1 live acceptance, 2026-10-11): rembg's BiRefNet-general
export plus the two IS-Net base donors of the ported July base-keep pass,
each pinned by sha-256, at the paths Settings reads. The image never picks a
backend itself (MATTING_BACKEND is a deployment's decision)."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = (Path(__file__).resolve().parent.parent / "Dockerfile.worker").read_text()

WEIGHTS = {
    "MATTING_MODEL_PATH": "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
    "MATTING_ISNET_GENERAL_PATH": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    "MATTING_ISNET_ANIME_PATH": "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
}


def _env(name: str) -> str | None:
    m = re.search(rf"\b{name}=(\S+)", DOCKERFILE)
    return m.group(1).rstrip("\\").strip() if m else None


def test_every_weight_the_base_keep_backend_reads_is_baked_at_its_setting() -> None:
    for name in WEIGHTS:
        assert _env(name), name


def test_every_weight_is_pinned_by_its_measured_sha256() -> None:
    for name, sha in WEIGHTS.items():
        assert re.search(rf"{sha}\s+\"?\$\{{?{name}\}}?\"?", DOCKERFILE), name
    assert "sha256sum -c" in DOCKERFILE


def test_the_image_never_selects_a_backend() -> None:
    assert not re.search(r"^\s*(ENV|ARG)\b.*\bMATTING_BACKEND=", DOCKERFILE, re.MULTILINE)
