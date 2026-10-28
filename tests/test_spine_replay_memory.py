"""GKPH-S3-2: the replay's memory grows by a small, bounded amount per original.

The media DB held ~282k originals on 2026-10-26 and the mask batch is heading
for ~927k. The mm-report Job runs with a 512 MiB limit; a plan that keeps
every copy's record and every mask row in memory for the whole run measured
2.37 GB peak at 282k originals (Postgres, local), so the Job could never get
past loading its plan. Per batch, the replay needs the details of that batch's
copies only; what it keeps for the whole run is the index (which original
maps to which copy, which copies are live, which mask bytes are published).

Measured as the SLOPE of tracemalloc's peak between two sizes, so the fixed
cost of imports, compiled SQL and the session cancels out.
"""

from __future__ import annotations

import datetime as dt
import gc
import sys
import tracemalloc
from typing import Any

import grpc
from app.models import Image, ImageMask, ImageVersion, PublicObject, PublicObjectImage
from app.spine import replay as rp
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from app.spine.media_v1 import media_pb2
from sqlalchemy import insert
from sqlalchemy.orm import Session

from spine_fakes import AT, FakeSpine, serve, sha

#: Bytes of peak memory per original the replay may add. The plan before
#: GKPH-S3-2 measured ~7,400 here; the index alone needs well under half.
MAX_BYTES_PER_ORIGINAL = 3_000

#: Live allocator blocks a REAL run may gain per original it has passed.
#: Holding every landed copy for the whole run measured over 1 here.
MAX_BLOCKS_PER_PASSED_ORIGINAL = 0.25


def _bulk_copies(db: Session, start: int, count: int) -> None:
    """``count`` originals, each with its clean copy and one ok mask, by bulk
    insert (seed_copy's flush per row is too slow for thousands)."""
    ids = range(start, start + count)
    images = [
        {"id": i + 1, "sha256": sha(f"orig:{i}"), "mime": "image/jpeg", "created_at": AT}
        for i in ids
    ]
    versions = [
        {
            "id": 2 * i + 1 + k,
            "image_id": i + 1,
            "version_no": k + 1,
            "transform_spec": {},
            "storage_key": f"v/{i}/{k}",
            "visibility": "private",
            "age_rating": 0,
            "created_at": AT,
            "matted": False,
        }
        for i in ids
        for k in (0, 1)
    ]
    copies = [
        {
            "sha256": sha(f"clean:{i}"),
            "kind": "sanitized",
            "mime": "image/webp",
            "bytes": 10,
            "width": 100,
            "height": 200,
            "storage_key": f"d/{sha(f'clean:{i}')}",
            "image_version_id": 2 * i + 2,
            "pixel_sha256": sha(f"px:{i}"),
            "own_pixel_sha256": sha(f"own:{i}"),
            "created_at": AT,
        }
        for i in ids
    ]
    masks = [
        {
            "sha256": sha(f"mask:{i}"),
            "kind": "mask",
            "mime": "image/png",
            "bytes": 5,
            "width": 100,
            "height": 200,
            "storage_key": f"d/{sha(f'mask:{i}')}",
            "created_at": AT,
        }
        for i in ids
    ]
    rows = [
        {
            "id": i + 1,
            "source_sha256": sha(f"clean:{i}"),
            "pipeline_version": "mask-v1",
            "status": "ok",
            "origin_class": "store_listing",
            "backend": "test",
            "width": 100,
            "height": 200,
            "mask_sha256": sha(f"mask:{i}"),
            "mask_storage_key": f"d/{sha(f'mask:{i}')}",
            "bottom_margin_frac": 0.01,
            "top_margin_frac": 0.02,
            "box_top_frac": 0.1,
            "box_bottom_frac": 0.9,
            "box_left_frac": 0.2,
            "box_right_frac": 0.8,
            "contact_band_center_x_frac": 0.5,
            "contact_band_width_frac": 0.3,
            "masked_thumbhash": "x" * 35,
            "masked_dominant_color": "#a1b2c3",
            "plain_thumbhash": "y" * 30,
            "plain_dominant_color": "#c3b2a1",
            "created_at": AT,
            "updated_at": AT,
        }
        for i in ids
    ]
    db.execute(insert(Image), images)
    db.execute(insert(ImageVersion), versions)
    db.execute(insert(PublicObject), copies + masks)
    db.execute(
        insert(PublicObjectImage),
        [{"sha256": c["sha256"], "image_id": i + 1} for i, c in zip(ids, copies, strict=True)],
    )
    db.execute(insert(ImageMask), rows)
    db.commit()


def _peak_of_a_dry_run(db: Session) -> int:
    """tracemalloc's peak over one dry run, every line dropped as it is emitted."""
    db.expunge_all()
    gc.collect()
    tracemalloc.start()
    try:
        code = rp.run_replay(
            db,
            rp.ReplayConfig(dry_run=True),
            connect=lambda: (_ for _ in ()).throw(AssertionError("a dry run opens no channel")),
            emit=lambda line: None,
            now=lambda: AT + dt.timedelta(hours=1),
            sleep=lambda s: None,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert code == rp.EXIT_DONE
    return peak


def test_the_replay_adds_a_bounded_few_kilobytes_per_original(db_session: Session) -> None:
    small, large = 400, 2_400
    _bulk_copies(db_session, 0, small)
    low = _peak_of_a_dry_run(db_session)
    _bulk_copies(db_session, small, large - small)
    high = _peak_of_a_dry_run(db_session)
    per_original = (high - low) / (large - small)
    assert per_original < MAX_BYTES_PER_ORIGINAL, f"{per_original:.0f} bytes per original"


class _ForgetfulSpine(FakeSpine):
    """Answers every call as a healthy spine would and keeps nothing per
    call, so what the process gains along a run is the replay's."""

    reports = 0
    masks_sent = 0

    def _enter(self, rpc: str, request: Any, context: grpc.ServicerContext) -> None:
        pass

    def ReportDerivative(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.depictions[request.product_image_id]["state"] = "derived"
        self.held.discard(request.product_image_id)
        self.reports += 1
        return media_pb2.DerivativeAck(stored=True, product_image_state="derived")

    def ReportMask(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.masks_sent += 1
        return media_pb2.MaskAck(stored=1, matched=1)


def test_a_real_run_holds_nothing_more_for_the_originals_it_has_passed(
    db_session: Session,
) -> None:
    """A copy's derivative is remembered until its masks are due (a
    FAILED_PRECONDITION is then judged unregistered, not orphan), not for the
    rest of the run. Counted in live allocator blocks, not tracemalloc: with
    the spine in the same process, CPython's free lists hand blocks first
    traced to the replay over to the spine's state."""
    count, size = 2_400, 100
    _bulk_copies(db_session, 0, count)
    fake = _ForgetfulSpine()
    for i in range(count):
        fake.depict(sha(f"orig:{i}"))
    first, last = 2, count // size - 1
    blocks: dict[int, int] = {}

    def emit(line: dict[str, Any]) -> None:
        if line.get("event") == "batch" and line["batch"] in (first, last):
            gc.collect()
            blocks[line["batch"]] = sys.getallocatedblocks()

    with serve(fake) as url:
        code = rp.run_replay(
            db_session,
            rp.ReplayConfig(batch_size=size),
            connect=lambda: SpineMediaClient(
                open_channel(url), policy=RetryPolicy(max_attempts=1), sleep=lambda s: None
            ),
            emit=emit,
            now=lambda: AT + dt.timedelta(hours=1),
            sleep=lambda s: None,
        )
    assert code == rp.EXIT_DONE
    assert fake.reports == count and fake.masks_sent == count
    per_original = (blocks[last] - blocks[first]) / ((last - first) * size)
    assert per_original < MAX_BLOCKS_PER_PASSED_ORIGINAL, f"{per_original:.2f} blocks per original"
