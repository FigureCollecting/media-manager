"""IMGREND (IR-MM-1): the largest parents are minted inside a memory bound.

HEAVY, AND ONLY IN A CAPPED SCOPE. These decode 64 to 107 megapixel pictures
for real, so they run only when MM_TIER_HEAVY=1 says the process is inside a
memory cap (a systemd scope with MemoryMax; never an uncapped developer
shell or CI):

    MM_TIER_HEAVY=1 systemd-run --user --scope -q -p MemoryMax=6G -p TasksMax=2000 -- \
        timeout 5400 nice -n 10 poetry run pytest tests/test_tier_memory.py

Each case builds its parent here, then mints it in a CHILD process that
reports its own peak resident size, so the bound is on minting alone, not on
making the fixture. The peak is VmHWM from /proc/self/status, never
getrusage's ru_maxrss: Linux carries that one over fork and exec, so a child
would report its PARENT's size (three cases here once read the same 842 MiB,
pytest's own). The bounds are what a 2 Gi pod can carry with room to spare:
the lane's pods are the consumers (fc-infra, IR-INFRA).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from app.workers.mask import encode_mask_png
from app.workers.tiers import rung_size
from PIL import Image

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("MM_TIER_HEAVY") != "1",
        reason="heavy: set MM_TIER_HEAVY=1 inside a memory-capped scope",
    ),
    pytest.mark.timeout(900),
]

MIB = 1024 * 1024

#: Mints the parent at argv[1] (argv[2]: photo or mask; argv[3], argv[4]: its
#: size) and prints its rungs and this process's own peak resident size.
CHILD = """
import json, re, sys, time
from pathlib import Path
from app.workers.tiers import DecodedBytes, mint_mask_tiers, mint_photo_tiers
data = Path(sys.argv[1]).read_bytes()
gate = DecodedBytes()
start = time.perf_counter()
if sys.argv[2] == "mask":
    minted = mint_mask_tiers(data, width=int(sys.argv[3]), height=int(sys.argv[4]), gate=gate)
else:
    minted = mint_photo_tiers(data, gate=gate)
print(json.dumps({
    "seconds": round(time.perf_counter() - start, 2),
    "peak_rss_kib": int(re.search(r"VmHWM:\\s+(\\d+) kB", Path("/proc/self/status").read_text())[1]),
    "gate_peak": gate.peak,
    "rungs": [[t.long_edge_px, t.width, t.height, len(t.data)] for t in minted],
}))
"""


def _scene(width: int, height: int) -> np.ndarray:
    """A graded frame with a body in it, built a band at a time."""
    arr = np.empty((height, width, 3), dtype=np.uint8)
    xs = np.arange(width)
    for top in range(0, height, 512):
        ys = np.arange(top, min(top + 512, height))[:, None]
        arr[top : top + 512, :, 0] = (xs[None, :] * 255 // width + ys * 0) % 256
        arr[top : top + 512, :, 1] = (ys * 255 // height + xs[None, :] * 0) % 256
        arr[top : top + 512, :, 2] = ((xs[None, :] + ys) // 16) % 256
    arr[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = (200, 60, 90)
    return arr


def _alpha(width: int, height: int) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = 255
    alpha[height // 4 - 40 : height // 4, width // 3 : 2 * width // 3] = 128
    return alpha


def _parent(kind: str, width: int, height: int) -> bytes:
    out = io.BytesIO()
    if kind == "jpeg":
        Image.fromarray(_scene(width, height)).save(out, format="JPEG", quality=95, subsampling=0)
    elif kind == "png-rgba":
        rgba = np.dstack([_scene(width, height), _alpha(width, height)])
        Image.fromarray(rgba).save(out, format="PNG", compress_level=1)
    else:
        return encode_mask_png(_alpha(width, height))
    return out.getvalue()


@pytest.mark.parametrize(
    ("kind", "width", "height", "bound_mib", "gate_bytes"),
    [
        # 64 Mpx, the pixel bound itself: decoded whole
        ("jpeg", 8000, 8000, 512, 8000 * 8000 * 4),
        # the largest live clean copy (U0, 106.7 Mpx): decoded at 1/2 by the JPEG decoder
        ("jpeg", 14174, 7525, 384, 7087 * 3763 * 4),
        # its mask: decoded whole (LA plus the alpha plane)
        ("mask", 14174, 7525, 768, 14174 * 7525 * 5),
        # a 64 Mpx transparent PNG clean copy: RGBA plus its premultiplied working copy
        ("png-rgba", 8000, 8000, 1024, 8000 * 8000 * 8),
    ],
)
def test_a_giant_parent_is_minted_inside_the_memory_bound(
    tmp_path: Path, kind: str, width: int, height: int, bound_mib: int, gate_bytes: int
) -> None:
    path = tmp_path / "parent"
    path.write_bytes(_parent(kind, width, height))
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(path), "mask" if kind == "mask" else "photo"]
        + [str(width), str(height)],
        capture_output=True,
        text=True,
        timeout=800,
        env={**os.environ, "ENVIRONMENT": "test"},
    )
    assert child.returncode == 0, child.stderr[-2000:]
    report = json.loads(child.stdout.strip().splitlines()[-1])
    print(f"\n{kind} {width}x{height} ({path.stat().st_size / MIB:.1f} MiB): {report}")
    assert [tuple(r[:3]) for r in report["rungs"]] == [
        (edge, *rung_size(width, height, edge)) for edge in (128, 256, 512, 1024, 2048)
    ]
    assert report["gate_peak"] == gate_bytes
    assert report["peak_rss_kib"] / 1024 <= bound_mib, report
