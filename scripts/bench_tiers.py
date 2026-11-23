"""IMGREND bench: what minting tiers costs, on the sandbox sample.

    poetry run python scripts/bench_tiers.py [--sample tests/fixtures/mask_sample] [--giants]

Two parts, both local (a temporary directory; no bucket, no database server):

1. THE SAMPLE, END TO END: every sample picture seeded into a sandbox
   (SQLite + a local object store), its reference matte registered as its
   mask, then ``mint_parent`` for the clean copy and for the mask: wall time
   per parent and the bytes of each rung.
2. BY PARENT SIZE: one sample picture enlarged to each long edge (a bench
   INPUT, not a rendition: nothing minted is ever larger than its parent)
   as a JPEG clean copy and a PNG mask. Each (parent, kind, rungs up to an
   edge) is minted in its own child process, which reports its wall time and
   its own peak resident size; a rung's cost is the step from the edge below.
   ``--giants`` adds 64 Mpx and the largest live parent's size (14,174 x
   7,525): run those only inside a memory-capped scope.

Synthetic sample pictures are flat art: real product photos encode larger
(the plan's storage estimate is not this script's to confirm).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
from app.cli.masks import open_stores  # noqa: E402
from app.models import ImageMask  # noqa: E402
from app.workers.mask import MASK_PIPELINE_VERSION, encode_mask_png  # noqa: E402
from app.workers.mask_seed import register_public_object, seed_clean_image  # noqa: E402
from app.workers.tiers import (  # noqa: E402
    LADDER,
    DecodedBytes,
    mint_mask_tiers,
    mint_parent,
    mint_photo_tiers,
)
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # the bench's own inputs, made here


def child(path: str, kind: str, width: int, height: int, max_edge: int) -> None:
    data = Path(path).read_bytes()
    gate = DecodedBytes()
    start = time.perf_counter()
    if kind == "mask":
        minted = mint_mask_tiers(data, width=width, height=height, max_edge=max_edge, gate=gate)
    else:
        minted = mint_photo_tiers(data, max_edge=max_edge, gate=gate)
    status = Path("/proc/self/status").read_text()
    report = {
        "seconds": round(time.perf_counter() - start, 3),
        # VmHWM, not ru_maxrss: Linux carries that one over from the parent process
        "peak_rss_mib": round(int(re.search(r"VmHWM:\s+(\d+) kB", status)[1]) / 1024, 1),  # type: ignore[index]
        "decoded_mib": round(gate.peak / 2**20, 1),
        "rungs": {t.long_edge_px: len(t.data) for t in minted},
    }
    print(json.dumps(report))


def _sample(sample: Path) -> list[tuple[str, bytes, np.ndarray]]:
    entries = json.loads((sample / "manifest.json").read_text())["entries"]
    out = []
    for entry in entries:
        original = (sample / "originals" / f"{entry['name']}.png").read_bytes()
        matte = Image.open(sample / "reference" / f"{entry['reference']}.png").convert("RGBA")
        out.append((entry["name"], original, np.asarray(matte)[..., 3].copy()))
    return out


def end_to_end(sample: Path, work: Path) -> list[dict[str, object]]:
    db, store = open_stores(str(work / "sandbox"))  # the masks CLI's own sandbox
    rows: list[dict[str, object]] = []
    with db:
        for name, original, alpha in _sample(sample):
            clean = seed_clean_image(db, store, original, origins=[], figure_id=f"bench:{name}")
            mask = encode_mask_png(alpha)
            sha = hashlib.sha256(mask).hexdigest()
            key = f"derived/{sha[:2]}/{sha}.png"
            store.put(key, mask, "image/png")
            if db.get(type(clean), sha) is None:
                register_public_object(
                    db,
                    sha256=sha,
                    kind="mask",
                    mime="image/png",
                    size=len(mask),
                    width=clean.width,
                    height=clean.height,
                    storage_key=key,
                )
            db.add(
                ImageMask(
                    source_sha256=clean.sha256,
                    pipeline_version=MASK_PIPELINE_VERSION,
                    status="ok",
                    mask_sha256=sha,
                    mask_storage_key=key,
                )
            )
            db.flush()
            for kind, parent, full in (
                ("photo", clean.sha256, clean.bytes),
                ("mask", sha, len(mask)),
            ):
                start = time.perf_counter()
                outcome = mint_parent(db, store, parent)
                rows.append(
                    {
                        "name": name,
                        "kind": kind,
                        "size": f"{clean.width}x{clean.height}",
                        "state": outcome.state,
                        "ms": round((time.perf_counter() - start) * 1000, 1),
                        "full_bytes": full,
                        "rungs": {t.long_edge_px: t.bytes for t in outcome.tiers},
                    }
                )
            db.commit()
    return rows


def _enlarged(original: bytes, alpha: np.ndarray, size: tuple[int, int]) -> tuple[bytes, bytes]:
    """A clean copy (JPEG quality 95, 4:4:4) and a mask of ``size`` from one
    sample picture."""
    photo = Image.open(io.BytesIO(original)).convert("RGB")
    photo = photo.resize(size, Image.Resampling.BICUBIC)
    out = io.BytesIO()
    photo.save(out, format="JPEG", quality=95, subsampling=0)
    del photo
    mask = Image.fromarray(alpha).resize(size, Image.Resampling.BILINEAR)
    return out.getvalue(), encode_mask_png(np.asarray(mask))


def by_size(sample: Path, work: Path, sizes: list[tuple[int, int]]) -> list[dict[str, object]]:
    _name, original, alpha = next(s for s in _sample(sample) if s[0] == "rem")
    rows: list[dict[str, object]] = []
    for width, height in sizes:
        photo, mask = _enlarged(original, alpha, (width, height))
        for kind, data in (("photo", photo), ("mask", mask)):
            path = work / f"{kind}-{width}x{height}"
            path.write_bytes(data)
            for max_edge in LADDER:
                run = subprocess.run(
                    [sys.executable, __file__, "child", str(path), kind]
                    + [str(width), str(height), str(max_edge)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                report = json.loads(run.stdout.strip().splitlines()[-1])
                rows.append(
                    {
                        "size": f"{width}x{height}",
                        "mpx": round(width * height / 1e6, 1),
                        "kind": kind,
                        "parent_bytes": len(data),
                        "up_to": max_edge,
                        **report,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
            path.unlink()
    return rows


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", default=str(REPO / "tests" / "fixtures" / "mask_sample"))
    parser.add_argument("--giants", action="store_true", help="add 64 Mpx and 14,174 x 7,525")
    parser.add_argument("--json", help="also write every row to this file")
    args = parser.parse_args()
    sizes = [(550, 800), (1000, 1455), (2000, 2909), (4000, 5818), (6000, 8000)]
    if args.giants:
        sizes += [(8000, 8000), (14174, 7525)]
    with tempfile.TemporaryDirectory(prefix="bench-tiers-") as tmp:
        sample_rows = end_to_end(Path(args.sample), Path(tmp))
        for row in sample_rows:
            print(json.dumps(row), flush=True)
        size_rows = by_size(Path(args.sample), Path(tmp), sizes)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"sample": sample_rows, "by_size": size_rows}, indent=1)
        )


if __name__ == "__main__":
    main()
