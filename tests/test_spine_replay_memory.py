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
import tracemalloc

from app.models import Image, ImageMask, ImageVersion, PublicObject, PublicObjectImage
from app.spine import replay as rp
from sqlalchemy import insert
from sqlalchemy.orm import Session

from spine_fakes import AT, sha

#: Bytes of peak memory per original the replay may add. The plan before
#: GKPH-S3-2 measured ~7,400 here; the index alone needs well under half.
MAX_BYTES_PER_ORIGINAL = 3_000


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
