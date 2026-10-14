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


def test_onnxruntime_gpu_is_the_release_the_batch_was_measured_on() -> None:
    """MASK-BATCH-1 (2026-10-14): onnxruntime-gpu 1.31.0, the latest release,
    measured against 1.27.0 on this lane's seven July figures: the same IoU
    against the July cut-outs to three places, no alpha more than 2 of 255
    apart, the same speed and memory."""
    assert re.search(r"pip install --no-cache-dir onnxruntime-gpu==1\.31\.0\s", DOCKERFILE)


def test_the_base_is_the_cuda_13_cudnn_9_runtime_it_was_measured_on() -> None:
    """onnxruntime-gpu 1.27+ is built against CUDA 13.0 and cuDNN 9 and runs on
    any later CUDA 13 minor; nvidia/cuda 13.4.2 is the latest 13.x runtime."""
    m = re.search(
        r"^FROM nvidia/cuda:(\d+)\.(\d+)\.(\d+)-cudnn-runtime-ubuntu24\.04\b", DOCKERFILE, re.M
    )
    assert m is not None
    assert (int(m[1]), int(m[2]), int(m[3])) == (13, 4, 2)


def test_the_image_never_selects_a_device() -> None:
    assert not re.search(r"^\s*(ENV|ARG)\b.*\bMATTING_DEVICE=", DOCKERFILE, re.MULTILINE)
