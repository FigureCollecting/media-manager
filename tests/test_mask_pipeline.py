"""MASK-PERF-2: the mask batch pipelined, so the I/O overlaps the GPU.

With ``writers`` above 0 the images of a chunk run on that many workers, each
with a database session of its own, and the next ``prefetch`` originals are
read, decoded and hash-checked ahead; the GPU stays ONE stream, on the
driver's own thread. Each image's work runs in the same order as before;
images of one picture (the ones whose seed or settle could see each other's
rows) wait for each other in work-list order, and the batch row is written in
that order too. So the pipeline writes what one image at a time writes: the
same rows (save the surrogate ids concurrent inserts draw in another order),
the same bucket bytes, the same records. A pause or a stop lets no further
image onto the GPU; the images it held are not recorded and run again.

SQLite on a file here (a pool of real connections, as on Postgres), local
directories, fake models; the clock is the real one where seconds are
measured.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from app.cli import mask_batch as mb
from app.models import Base, ImageMask, ImageVersion, MaskBatch, PublicObject
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.workers import mask_job
from app.workers import mask_pipeline as mp
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import picture_keys, seed_clean_image, seed_footprint
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.stage_timing import StageTimings
from PIL import Image, PngImagePlugin
from sqlalchemy import Engine, create_engine, select
from sqlalchemy import event as sa_event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/{n}.jpeg"
H = 200
NOON = dt.datetime(2026, 10, 14, 15, 30, tzinfo=dt.UTC)

#: The chunk event's fields that measure time: they differ run to run.
_CHUNK_TIMING = {
    "seconds_per_image",
    "median_seconds",
    "seconds_per_masked_image",
    "mask_seconds_per_image",
    "stage_seconds_per_image",
    "stage_median_seconds",
    "stage_images",
    "elapsed_seconds",
    "throughput_seconds_per_image",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _photo(width: int, n: int, note: str | None = None) -> bytes:
    arr = np.full((H, width, 3), (250, 250, 248), dtype=np.uint8)
    arr[20:160, int(width * 0.4) : int(width * 0.6)] = (31 + 10 * n % 200, 30, 90)
    arr[0, 0] = (n % 251, 250, 250)
    info = None
    if note:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", note)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


class _Raw:
    """A raw bucket on disk, counting its reads (and able to slow them)."""

    def __init__(self, root: Path, cost: float = 0.0) -> None:
        self.store = _Counted(root, get_cost=cost)

    def add(
        self,
        width: int,
        n: int,
        *,
        codec: str = "identity",
        url: str = ITEM,
        pixels_of: int | None = None,
        corrupt: bool = False,
    ) -> mb.WorkItem:
        data = _photo(width, pixels_of or n, note="again" if pixels_of else None)
        sha = hashlib.sha256(data).hexdigest()
        key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
        stored = gzip.compress(data, mtime=0) if codec == "gzip" else data
        LocalObjectStore.put(self.store, key, _photo(width, 99) if corrupt else stored, "image/png")
        return mb.WorkItem(
            blob_sha256=sha,
            storage_key=key,
            url=url.format(n=n),
            product_id=f"00000000-0000-0000-0000-{n:012d}",
            head_id=None,
            rank=0,
            codec=codec,
        )


class _Counted(LocalObjectStore):
    """A local bucket that counts its GETs and PUTs (thread-safe), each
    costing real seconds when asked, and can be told to slow a PUT."""

    def __init__(self, root: Path, *, get_cost: float = 0.0, put_cost: float = 0.0) -> None:
        super().__init__(root)
        self.get_cost, self.put_cost = get_cost, put_cost
        self.lock = threading.Lock()
        self.gets: list[str] = []
        self.puts: list[str] = []
        self.slow: dict[str, float] = {}  # a GET's key, or a PUT's bytes' sha: its delay
        self.fail_once: set[str] = set()  # a GET's key: it times out, once

    def get(self, key: str) -> bytes:
        with self.lock:
            self.gets.append(key)
            failing = key in self.fail_once
            self.fail_once.discard(key)
        delay = self.get_cost + self.slow.get(key, 0.0)
        if delay:
            time.sleep(delay)
        if failing:
            raise TimeoutError("read timed out")
        return super().get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        delay = self.put_cost + self.slow.get(hashlib.sha256(data).hexdigest(), 0.0)
        if delay:
            time.sleep(delay)
        super().put(key, data, content_type)
        with self.lock:
            self.puts.append(key)


class _Shapes(MattingBackend):
    """A mask model by the photo's width: 101 a full frame (held), 102
    nothing (empty), 103 a crash (failed), any other the figure (ok). It
    notes each call's thread and how many ran at once."""

    name = "shapes"
    publishes_masks = True

    def __init__(self, cost: float = 0.0, gate: Callable[[int], None] | None = None) -> None:
        self.cost, self.gate = cost, gate
        self.lock = threading.Lock()
        self.calls: list[tuple[int, int]] = []
        self.running = 0
        self.most = 0

    @property
    def active_provider(self) -> str | None:
        return "FakeExecutionProvider"

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - never asked
        raise AssertionError("the batch asks for the alpha alone")

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        width = image.width
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.calls.append((width, threading.get_ident()))
        try:
            if self.gate is not None:
                self.gate(width)
            if self.cost:
                time.sleep(self.cost)
            if width == 101:
                return np.full((H, width), 255, dtype=np.uint8)
            if width == 102:
                return np.zeros((H, width), dtype=np.uint8)
            if width == 103:
                raise RuntimeError("CUDA error: an illegal memory access")
            alpha = np.zeros((H, width), dtype=np.uint8)
            alpha[20:160, int(width * 0.4) : int(width * 0.6)] = 255
            return alpha
        finally:
            with self.lock:
                self.running -= 1


class _Sampler:
    peak_mib = 3210

    def __enter__(self) -> _Sampler:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _engine(path: Path, **pool: Any) -> Engine:
    """A SQLite file with a pool of real connections (foreign keys on)."""
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30}, **pool
    )

    @sa_event.listens_for(engine, "connect")
    def _fk(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def engines(tmp_path: Path) -> Iterator[Callable[[str], Engine]]:
    made: list[Engine] = []

    def make(name: str) -> Engine:
        engine = _engine(tmp_path / f"{name}.db")
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.dispose()


def _runtime(
    main: Session,
    root: Path,
    raw: _Raw,
    backend: MattingBackend,
    events: list[dict[str, Any]],
    *,
    media: LocalObjectStore | None = None,
    progress: mb.Progress | None = None,
    stop: Callable[[], bool] | None = None,
    gates: mb.Gates | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> mb.Runtime:
    return mb.Runtime(
        gates=gates or mb.Gates(windows=(), gpu=None, residential=None, now=lambda: NOON),
        open_db=lambda: main,
        media_store=media if media is not None else LocalObjectStore(root / "media"),
        raw_store=mb.ReadOnlyStore(raw.store),
        make_backend=lambda: backend,
        progress=progress or mb.Progress(root / "progress.jsonl"),
        vram=lambda: _Sampler(),
        sleep=sleep,
        monotonic=time.monotonic,
        emit=events.append,
        stop_requested=stop or (lambda: False),
    )


def _mixed(raw: _Raw) -> list[mb.WorkItem]:
    """Every path an image can take, two captures of one picture adjacent."""
    return [
        raw.add(100, 1),  # ok
        raw.add(100, 2, pixels_of=1),  # the same picture, other bytes: already
        raw.add(101, 3),  # held: a full frame
        raw.add(102, 4),  # empty
        raw.add(103, 5),  # the model crashes: failed
        raw.add(107, 6, url=PICTURE),  # ineligible
        raw.add(108, 7, corrupt=True),  # raw_hash_mismatch
        raw.add(109, 8, codec="gzip"),  # ok, gzip
        raw.add(110, 9, codec="zstd"),  # unsupported_codec
        raw.add(111, 10),  # the raw read times out once: interrupted
        raw.add(112, 11),  # its mask row's INSERT is dropped once: interrupted
        raw.add(113, 12),  # ok: the batch row read again after that rollback
    ]


def _faults(items: list[mb.WorkItem], raw: _Raw, engine: Engine) -> None:
    """Arm ``_mixed``'s two outages for one run: item 10's raw read times
    out, item 11's image_masks INSERT loses the connection, each once."""
    raw.store.fail_once = {items[9].storage_key}
    dropped: list[bool] = []

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _insert(conn: Any, cursor: Any, statement: str, params: Any, *rest: Any) -> None:
        if statement.startswith("INSERT INTO image_masks") and 112 in params and not dropped:
            dropped.append(True)
            raise OperationalError(statement, params, Exception("server closed the connection"))


def _canonical(engine: Engine) -> dict[str, list[str]]:
    """Every row of every table, the surrogate ids replaced by natural keys
    (an image by its sha, a version by its image and number), timestamps
    dropped: what concurrent inserts can change is only the ids they draw."""
    with Session(engine) as db:
        images = dict(db.execute(select(ImageModel.id, ImageModel.sha256)).all())
        versions = {
            vid: f"{images[iid]}#{no}"
            for vid, iid, no in db.execute(
                select(ImageVersion.id, ImageVersion.image_id, ImageVersion.version_no)
            )
        }
        out: dict[str, list[str]] = {}
        for table in Base.metadata.sorted_tables:
            rows = []
            for mapping in db.execute(select(table)).mappings():
                row = {k: v for k, v in mapping.items() if not k.endswith("_at")}
                if table.name in ("images", "figure_galleries", "image_masks"):
                    row.pop("id")
                if table.name == "image_versions":
                    row["id"] = versions[row["id"]]
                if row.get("image_id") is not None:
                    row["image_id"] = images[row["image_id"]]
                if row.get("image_version_id") is not None:
                    row["image_version_id"] = versions[row["image_version_id"]]
                rows.append(json.dumps(row, sort_keys=True, default=str))
            out[table.name] = sorted(rows)
    return out


def _settled(tables: dict[str, list[str]]) -> dict[str, list[str]]:
    """The tables without the batch rows and each mask row's batch: a rerun
    opens batches of its own."""
    out = {name: rows for name, rows in tables.items() if name != "mask_batches"}
    masks = [json.loads(r) for r in out["image_masks"]]
    out["image_masks"] = sorted(
        json.dumps({k: v for k, v in m.items() if k != "batch_id"}, sort_keys=True) for m in masks
    )
    return out


def _media(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _untimed(line: dict[str, Any]) -> dict[str, Any]:
    drop = _CHUNK_TIMING if line["event"] == "chunk" else {"seconds", "mask_seconds"}
    return {k: v for k, v in line.items() if k not in drop}


def _run(
    tmp_path: Path,
    name: str,
    items: list[mb.WorkItem],
    raw: _Raw,
    config: mb.BatchConfig,
    *,
    backend: MattingBackend | None = None,
    engine: Engine | None = None,
    faults: bool = False,
) -> dict[str, Any]:
    """One batch on a fresh database and media bucket; everything it left."""
    root = tmp_path / name
    root.mkdir()
    engine = engine or _engine(root / "media.db")
    if faults:
        _faults(items, raw, engine)
    batch_row: list[tuple[str, Any]] = []  # its UPDATEs, in order: their statements and values

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _batch_row(conn: Any, cursor: Any, statement: str, params: Any, *rest: Any) -> None:
        if statement.startswith("UPDATE mask_batches") and "finished_at" not in statement:
            batch_row.append((statement, params))

    main = Session(engine, expire_on_commit=False)  # autoflush on, as the sandbox's
    events: list[dict[str, Any]] = []
    try:
        code = mb.run_batch(items, config, _runtime(main, root, raw, backend or _Shapes(), events))
    finally:
        main.close()
    outcome = {
        "exit": code,
        "tables": _canonical(engine),
        "media": _media(root / "media"),
        "progress": [_untimed(line) for line in _lines(root / "progress.jsonl")],
        "events": [_untimed(e) for e in events],
        "batch_row": batch_row,
    }
    engine.dispose()
    return outcome


# --------------------------------------------------------------------------
# the pipeline writes what one image at a time writes
# --------------------------------------------------------------------------


def test_the_pipeline_writes_what_one_image_at_a_time_writes(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = _mixed(raw)
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=4), faults=True)
    records = [r for r in alone["progress"] if r["event"] == "item"]
    assert [r["status"] for r in records] == [
        *("ok", "already", "held", "empty", "failed", "ineligible", "failed", "ok"),
        *("failed", "interrupted", "interrupted", "ok"),
    ]
    assert [r["reason"] for r in records[9:11]] == ["fetch:TimeoutError", "settle:OperationalError"]
    for attempt in range(3):  # three runs: three interleavings
        config = mb.BatchConfig(chunk_size=4, writers=4)
        piped = _run(tmp_path, f"piped{attempt}", items, raw, config, faults=True)
        assert piped == alone


def test_a_rerun_after_the_pipeline_is_idempotent_and_reads_no_original_again(
    tmp_path: Path,
) -> None:
    """Every image of a settled run is 'already' on a rerun with a fresh
    record (idempotent per image sha); its prefetched read is not staged."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(104 + n, n) for n in range(1, 6)]
    root = tmp_path / "run"
    root.mkdir()
    engine = _engine(root / "media.db")
    main = Session(engine, expire_on_commit=False)
    config = mb.BatchConfig(chunk_size=5, writers=3)
    events: list[dict[str, Any]] = []
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events)) == 0
    again: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), again, progress=mb.Progress(root / "fresh.jsonl"))
    assert mb.run_batch(items, config, rt) == 0
    chunk = next(e for e in again if e["event"] == "chunk")
    assert chunk["counts"] == {"already": 5}
    assert "fetch" not in chunk["stage_seconds_per_image"]
    main.close()
    engine.dispose()


# --------------------------------------------------------------------------
# the overlap, the one GPU stream, the bounds
# --------------------------------------------------------------------------


def test_the_next_images_io_overlaps_the_model(tmp_path: Path) -> None:
    """While the first image is on the GPU, another image's seed stores its
    original in the media bucket (one image at a time, nothing else runs)."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(103 + n, n) for n in range(1, 5)]
    root = tmp_path / "run"
    media = _Counted(root / "media")
    media.slow[items[3].blob_sha256] = 0.5  # the last image's original: still uploading
    overlapped = threading.Event()
    first: list[int] = []

    def the_first_waits_for_another_upload(width: int) -> None:
        if first:
            return
        first.append(width)
        before = len(media.puts)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(media.puts) == before:
            time.sleep(0.01)
        if len(media.puts) > before:
            overlapped.set()

    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    backend = _Shapes(gate=the_first_waits_for_another_upload)
    rt = _runtime(main, root, raw, backend, [], media=media)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=4, writers=4), rt) == 0
    assert overlapped.is_set()
    main.close()
    engine.dispose()


def test_the_gpu_is_one_stream_on_the_drivers_thread(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw", cost=0.01)
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 13)]
    backend = _Shapes(cost=0.02)
    _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=6, writers=4), backend=backend)
    assert len(backend.calls) == 12
    assert backend.most == 1
    assert {thread for _w, thread in backend.calls} == {threading.get_ident()}


def test_images_in_flight_and_read_ahead_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the GPU held up, no more than ``writers`` images are on workers
    and no more than ``prefetch`` originals are read past them."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 5 * 8, n) for n in range(1, 15)]
    reads_while_held: list[int] = []

    def first_holds_the_gpu(width: int) -> None:
        if not reads_while_held:
            time.sleep(0.5)
            reads_while_held.append(len(raw.store.gets))

    lock = threading.Lock()
    minting = [0, 0]  # now, most
    real_mint = mb._mint

    def counted_mint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with lock:
            minting[0] += 1
            minting[1] = max(minting[1], minting[0])
        try:
            return real_mint(*args, **kwargs)
        finally:
            with lock:
                minting[0] -= 1

    monkeypatch.setattr(mb, "_mint", counted_mint)
    backend = _Shapes(gate=first_holds_the_gpu)
    config = mb.BatchConfig(chunk_size=14, writers=2, prefetch=3)
    _run(tmp_path, "run", items, raw, config, backend=backend)
    assert reads_while_held == [2 + 3]
    assert minting[1] == 2
    assert len(raw.store.gets) == 14


def test_no_more_than_three_copies_are_decoded_for_the_model_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image's decoded copy is held from its decode, through its wait for
    the GPU, to its mask: whatever the writers, only the one on the GPU and
    two waiting are decoded at once, so a chunk's memory stays near one
    image at a time's (MASK-PERF-2 at 4000x3000: ~0.55 GB each)."""
    lock = threading.Lock()
    decoded = [0, 0]  # now, most
    real = mask_job.build_mask_artifact

    def counted(*args: Any, **kwargs: Any) -> Any:
        with lock:
            decoded[0] += 1
            decoded[1] = max(decoded[1], decoded[0])
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                decoded[0] -= 1

    monkeypatch.setattr(mask_job, "build_mask_artifact", counted)
    first: list[bool] = []

    def first_holds_the_gpu(width: int) -> None:
        if not first:
            first.append(True)
            time.sleep(0.5)  # every other image is ready for the GPU meanwhile

    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 2 * 8, n) for n in range(1, 13)]
    engine = _engine(tmp_path / "media.db", pool_size=9)
    config = mb.BatchConfig(chunk_size=12, writers=8)
    backend = _Shapes(gate=first_holds_the_gpu)
    outcome = _run(tmp_path, "run", items, raw, config, backend=backend, engine=engine)
    assert outcome["exit"] == 0
    assert [r["status"] for r in outcome["progress"] if r["event"] == "item"] == ["ok"] * 12
    assert decoded[1] == 3
    assert mb.DECODED_AT_ONCE == 3


def test_decode_slots_go_to_the_earliest_image_waiting() -> None:
    slots = mp.Slots(1)
    order: list[int] = []

    def take(index: int) -> None:
        slots.acquire(index)
        order.append(index)
        slots.release()

    slots.acquire(0)
    threads = [threading.Thread(target=take, args=(i,)) for i in (5, 3, 4)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while slots.waiting() < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    slots.release()
    for thread in threads:
        thread.join(5)
    assert order == [3, 4, 5]
    two = mp.Slots(2)
    two.acquire(7)
    two.acquire(9)  # a second slot: no wait
    assert two.waiting() == 0


# --------------------------------------------------------------------------
# records, related pictures, the batch row
# --------------------------------------------------------------------------


class _Checked(mb.Progress):
    """A progress file that, as each image's record is written, reads the
    image's settled row from the database on a connection of its own."""

    def __init__(self, path: Path, engine: Engine) -> None:
        super().__init__(path)
        self.engine = engine
        self.seen: list[tuple[str, str | None]] = []

    def append(self, record: dict[str, Any]) -> None:
        if record.get("event") == "item" and record.get("clean_sha256"):
            with Session(self.engine) as db:
                status = db.execute(
                    select(ImageMask.status).where(
                        ImageMask.source_sha256 == record["clean_sha256"]
                    )
                ).scalar_one_or_none()
            self.seen.append((record["status"], status))
        super().append(record)


def test_a_record_is_written_only_after_its_writes_commit(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = _mixed(raw)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    progress = _Checked(root / "progress.jsonl", engine)
    rt = _runtime(main, root, raw, _Shapes(), [], progress=progress)
    mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=4), rt)
    by_record = {"already": "ok"}  # its picture's copy, settled by the first capture
    assert progress.seen
    for status, row in progress.seen:
        assert row == by_record.get(status, status)
    main.close()
    engine.dispose()


def test_two_captures_of_one_picture_never_overlap(tmp_path: Path) -> None:
    """The second capture's work starts only once the first's has ended,
    however slow the first: it finds the copy settled ('already')."""
    raw = _Raw(tmp_path / "raw")
    first, second = raw.add(100, 1), raw.add(100, 2, pixels_of=1)
    media = _Counted(tmp_path / "run" / "media")
    media.slow[first.blob_sha256] = 0.4  # its original's PUT
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [], media=media)
    assert mb.run_batch([first, second], mb.BatchConfig(chunk_size=2, writers=2), rt) == 0
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert [r["status"] for r in records] == ["ok", "already"]
    assert records[1]["mask_status"] == "ok"
    second_original = f"{second.blob_sha256[:2]}/{second.blob_sha256[2:4]}/{second.blob_sha256}"
    # the first's original, copy and mask, and only then the second's original
    assert len(media.puts) == 4
    assert media.puts[-1].startswith(second_original)
    assert not any(k.startswith(second_original) for k in media.puts[:3])
    main.close()
    engine.dispose()


def test_the_batch_rows_counts_and_cursor_are_one_image_at_a_times(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 4, n) for n in range(1, 11)]  # ok, held, empty, crash, ...
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=10))
    engine = _engine(tmp_path / "piped.db", pool_size=7)  # a connection per worker and the batch
    config = mb.BatchConfig(chunk_size=10, writers=6)
    piped = _run(tmp_path, "piped", items, raw, config, engine=engine)
    assert piped["tables"]["mask_batches"] == alone["tables"]["mask_batches"]
    (row,) = (json.loads(r) for r in piped["tables"]["mask_batches"])
    assert sum(row["counts"].values()) == 10


# --------------------------------------------------------------------------
# a pause, a stop, an outage
# --------------------------------------------------------------------------


def test_a_stop_lets_no_further_image_onto_the_gpu_and_a_rerun_finishes(
    tmp_path: Path,
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 9)]
    raw.store.slow[items[3].storage_key] = 0.5  # still reading when the stop comes
    stop = threading.Event()
    backend = _Shapes(gate=lambda width: stop.set())
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=8, writers=4)
    rt = _runtime(main, root, raw, backend, events, stop=stop.is_set)
    assert mb.run_batch(items, config, rt) == 130
    assert len(backend.calls) == 1  # the image already on the GPU, no other
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert len(records) == 1
    assert events[-1] == {"event": "stopped", "counts": {"ok": 1}, "pending": 7}
    rerun: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), rerun)
    assert mb.run_batch(items, config, rt) == 0
    assert rerun[0]["pending"] == 7
    main.close()
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=8))
    assert _settled(_canonical(engine)) == _settled(alone["tables"])
    assert _media(root / "media") == alone["media"]
    engine.dispose()


def test_a_busy_window_mid_chunk_yields_before_the_next_gpu_image(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    wall = {"minute": 30}
    minutes: list[int] = []

    def opens_the_window(width: int) -> None:
        minutes.append(wall["minute"])
        wall["minute"] = 5  # a scraper roll window begins

    def sleep(seconds: float) -> None:
        wall["minute"] = 30  # the window passed while the batch waited

    gates = mb.Gates(
        windows=mb.parse_windows("03-26"),
        gpu=None,
        residential=None,
        now=lambda: NOON.replace(minute=wall["minute"]),
    )
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    backend = _Shapes(gate=opens_the_window)
    rt = _runtime(main, root, raw, backend, events, gates=gates, sleep=sleep)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=3), rt) == 0
    assert set(minutes) == {30}  # no image on the GPU inside the window
    chunks = [e for e in events if e["event"] == "chunk"]
    assert chunks[0]["yielded"] is True
    assert chunks[0]["count"] == 1
    assert sum(c["count"] for c in chunks) == 6
    assert [e["event"] for e in events].count("paused") == len(chunks) - 1
    main.close()
    engine.dispose()


class _Outage(_Counted):
    """The media bucket's connection drops after ``healthy`` PUTs."""

    def __init__(self, root: Path, healthy: int) -> None:
        super().__init__(root)
        self.healthy = healthy

    def put(self, key: str, data: bytes, content_type: str) -> None:
        with self.lock:
            down = len(self.puts) >= self.healthy
        if down:
            raise ConnectionError("connection reset by peer")
        super().put(key, data, content_type)


def test_a_bucket_outage_opens_the_circuit_and_a_plain_rerun_finishes(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 13)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=12, writers=4, max_consecutive_failures=3)
    media = _Outage(root / "media", healthy=6)
    assert (
        mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events, media=media)) == 3
    )
    circuit = next(e for e in events if e["event"] == "circuit_open")
    assert circuit["consecutive_failures"] >= 3
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert {r["status"] for r in records} <= {"ok", "interrupted"}
    assert len(records) < 12
    media.healthy = 10**6
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), [], media=media)) == 0
    main.close()
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=12))
    assert _settled(_canonical(engine)) == _settled(alone["tables"])
    assert _media(root / "media") == alone["media"]
    engine.dispose()


def test_a_dropped_database_connection_on_the_workers_is_an_outage(tmp_path: Path) -> None:
    """The port-forward drops: every image's query fails on its own
    connection (OperationalError), each image is interrupted, the circuit
    opens (exit 3) and a plain rerun retries them."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 9)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    driver = threading.get_ident()
    down = threading.Event()

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _dropped(*args: Any) -> None:
        if down.is_set() and threading.get_ident() != driver:
            raise OperationalError("SELECT", {}, Exception("server closed the connection"))

    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=8, writers=4, max_consecutive_failures=3)
    down.set()
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events)) == 3
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert {r["status"] for r in records} == {"interrupted"}
    assert {r["reason"] for r in records} == {"db:OperationalError"}
    down.clear()
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), [])) == 0
    main.close()
    engine.dispose()


def test_an_unexpected_error_on_a_worker_stops_the_batch_without_hanging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    real_mint = mb._mint

    def mint(db: Session, item: mb.WorkItem, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if item is items[2]:
            raise LookupError("a bug")
        return real_mint(db, item, *args, **kwargs)

    monkeypatch.setattr(mb, "_mint", mint)
    with pytest.raises(LookupError):
        _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=6, writers=3))
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []


class _Unwritable(mb.Progress):
    def append(self, record: dict[str, Any]) -> None:
        if record.get("event") == "item":
            raise OSError(28, "No space left on device")
        super().append(record)


def test_a_record_that_cannot_be_written_stops_the_batch_and_lets_every_worker_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first record fails to write while another image waits for the
    GPU: the driver refuses that image, lets every worker go and raises.
    The order is forced: image 1 holds the GPU until image 0 is done and
    image 2 has asked for the GPU after it."""
    posted: list[str] = []
    seen = threading.Condition()

    class _Watched(queue.Queue[Any]):
        def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
            super().put(item, block, timeout)
            with seen:
                posted.append(type(item).__name__)
                seen.notify_all()

    def until(check: Callable[[], bool]) -> None:
        with seen:
            assert seen.wait_for(check, timeout=5)

    monkeypatch.setattr(mb, "queue", SimpleNamespace(Queue=_Watched))
    rgba = Image.new("RGBA", (100, H))
    on_gpu = threading.Event()

    def model(width: int) -> None:
        on_gpu.set()
        until(lambda: posted[-2:] == ["_Done", "GpuJob"])

    def mint(db: Session, item: mb.WorkItem, rt: mb.Runtime, gpu: MattingBackend, *a: Any) -> Any:
        n = int(item.product_id[-1])
        if n == 1:
            gpu.predict_alpha(rgba)
        else:
            assert on_gpu.wait(timeout=5)
            if n == 2:
                until(lambda: "_Done" in posted)
                gpu.predict_alpha(rgba)  # refused: Held
        return {"event": "item", "status": "ok", "blob_sha256": item.blob_sha256, "rank": 0}

    monkeypatch.setattr(mb, "_mint", mint)
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, n) for n in range(3)]
    backend = _Shapes(gate=model)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    progress = _Unwritable(root / "progress.jsonl")
    rt = _runtime(main, root, raw, backend, [], progress=progress)
    with pytest.raises(OSError):
        mb.run_batch(items, mb.BatchConfig(chunk_size=3, writers=3), rt)
    assert len(backend.calls) == 1  # image 2 never got the GPU
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []
    main.close()
    engine.dispose()


def test_a_read_ahead_that_breaks_stops_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 5)]
    real = mb._read_ahead

    def read_ahead(item: mb.WorkItem, rt: mb.Runtime) -> Any:
        if item is items[1]:
            raise MemoryError
        return real(item, rt)

    monkeypatch.setattr(mb, "_read_ahead", read_ahead)
    with pytest.raises(MemoryError):
        _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=4, writers=2))
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []


# --------------------------------------------------------------------------
# refusals: a pipeline needs a connection per worker
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pool",
    [
        {"poolclass": StaticPool},  # one connection for every session
        {"poolclass": QueuePool, "pool_size": 2, "max_overflow": 0},  # fewer than writers + 1
        # enough only with its overflow, whose connections are closed when they come back
        {"poolclass": QueuePool, "pool_size": 4, "max_overflow": 20},
    ],
)
def test_the_pipeline_is_refused_without_a_connection_per_worker(
    tmp_path: Path, pool: dict[str, Any]
) -> None:
    raw = _Raw(tmp_path / "raw")
    engine = _engine(tmp_path / "media.db", **pool)
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, tmp_path / "run", raw, _Shapes(), events)
    assert mb.run_batch([raw.add(100, 1)], mb.BatchConfig(writers=4), rt) == 2
    assert events[-1]["event"] == "refused"
    assert "--writers 0" in events[-1]["detail"]
    assert main.execute(select(MaskBatch)).first() is None
    main.close()
    engine.dispose()


def test_a_pool_that_opens_a_connection_per_checkout_runs_the_pipeline(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 5)]
    engine = _engine(tmp_path / "media.db", poolclass=NullPool)
    outcome = _run(tmp_path, "run", items, raw, mb.BatchConfig(writers=4), engine=engine)
    assert outcome["exit"] == 0


def test_a_chunk_at_the_command_lines_defaults_closes_no_pooled_connection(
    tmp_path: Path,
) -> None:
    """The live run reaches its database through a kubectl port-forward,
    which any closed connection kills: a chunk at the command line's
    defaults, on a pool like the media db's (10 kept, 20 more lent past
    them and closed when they come back), closes none of its connections."""
    raw = _Raw(tmp_path / "raw", cost=0.01)
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 25)]
    engine = _engine(tmp_path / "media.db", poolclass=QueuePool, pool_size=10, max_overflow=20)
    closed: list[str] = []
    for name in ("close", "close_detached"):
        sa_event.listen(engine, name, lambda *_, name=name: closed.append(name))
    args = mb._parser().parse_args(["run", "--worklist", "w"])
    config = mb.BatchConfig(
        chunk_size=args.chunk_size, writers=args.writers, prefetch=args.prefetch
    )
    root = tmp_path / "run"
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(cost=0.03), [])
    assert mb.run_batch(items, config, rt) == 0
    main.close()
    assert closed == []
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert len(records) == 24
    engine.dispose()


# --------------------------------------------------------------------------
# the seconds: each stage the image's own wall time
# --------------------------------------------------------------------------


def test_each_stage_is_the_images_own_wall_and_the_chunk_tells_the_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[StageTimings] = []

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            captured.append(self)

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    raw = _Raw(tmp_path / "raw", cost=0.05)
    items = [raw.add(100 + n % 2 * 8, n) for n in range(1, 7)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(cost=0.08), events)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=3), rt) == 0
    records = {r["blob_sha256"]: r for r in _lines(root / "progress.jsonl") if "status" in r}
    assert len(captured) == 6
    for timings in captured:
        seconds = timings.seconds
        # each its own: six images queue 0.48 s of model calls, which no
        # image's fetch or inference holds (that is gpu_wait's)
        assert 0.05 <= seconds["fetch"] < 0.2
        assert 0.08 <= seconds["inference"] < 0.2
        assert seconds["gpu_wait"] >= 0.0
        assert {"claim", "seed", "readback", "upload", "db", "turn_wait"} <= set(seconds)
    for record in records.values():
        assert record["seconds"] >= 0.05 + 0.08
    staged = sorted(round(sum(t.seconds.values()), 2) for t in captured)
    assert staged == pytest.approx(sorted(r["seconds"] for r in records.values()), abs=0.1)
    chunk = next(e for e in events if e["event"] == "chunk")
    # from the unrounded elapsed seconds: within a rounding of the reported ones
    assert chunk["throughput_seconds_per_image"] == pytest.approx(
        chunk["elapsed_seconds"] / 6, abs=0.001
    )
    assert chunk["throughput_seconds_per_image"] < chunk["seconds_per_image"]  # overlapped
    main.close()
    engine.dispose()


# --------------------------------------------------------------------------
# a picture's keys and its neighbourhood in the database
# --------------------------------------------------------------------------


def test_picture_keys_are_the_keys_the_seed_registers(db_session: Session, tmp_path: Path) -> None:
    original = _photo(100, 1)
    keys = picture_keys(original)
    store = LocalObjectStore(tmp_path / "media")
    copy = seed_clean_image(
        db_session, store, original, origins=[ImageOrigin(url=ITEM.format(n=1))], figure_id="f1"
    )
    assert (keys.digest, keys.clean_sha256, keys.own_digest) == (
        copy.pixel_sha256,
        copy.sha256,
        copy.own_pixel_sha256,
    )


def test_the_footprint_reaches_every_capture_and_copy_of_a_picture(
    db_session: Session, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "media")
    one, again, other = _photo(100, 1), _photo(100, 1, note="again"), _photo(100, 7)
    for n, data in ((1, one), (2, again), (3, other)):
        seed_clean_image(
            db_session,
            store,
            data,
            origins=[ImageOrigin(url=ITEM.format(n=n))],
            figure_id=f"f{n}",
        )
    copy = db_session.execute(
        select(PublicObject.sha256).where(PublicObject.pixel_sha256 == picture_keys(one).digest)
    ).scalar_one()
    sha = {d: hashlib.sha256(d).hexdigest() for d in (one, again, other)}
    # the second capture, by its own sha alone, reaches the first and the copy
    found = seed_footprint(db_session, shas={sha[again]}, digests=set())
    assert {sha[one], sha[again], copy} <= found
    assert sha[other] not in found
    # a new picture reaches nothing but its own keys
    assert seed_footprint(db_session, shas={"f" * 64}, digests={"e" * 64}) == {"f" * 64, "e" * 64}


# --------------------------------------------------------------------------
# the pipeline's parts
# --------------------------------------------------------------------------


def test_claims_are_registered_in_order_and_join_related_images() -> None:
    claims = mp.Claims()
    since = claims.snapshot()
    order: list[int] = []

    def register(index: int, keys: set[str]) -> None:
        claims.register(index, keys, since)
        order.append(index)

    late = threading.Thread(target=register, args=(1, {"b"}))
    late.start()
    time.sleep(0.05)
    assert order == []  # 1 waits for 0
    first, waits = claims.register(0, {"a", "x"}, since)
    late.join(timeout=5)
    assert order == [1] and waits == []
    third, waits = claims.register(2, {"x"}, since)
    assert waits == [first]
    assert third.keys == {"a", "x"}
    claims.finish(first)
    assert first.done.is_set()
    claims.skip(3)
    fifth, waits = claims.register(4, {"a"}, since)
    assert waits == [third]
    assert fifth.keys == {"a", "x"}


def test_a_claim_finished_after_the_snapshot_still_lends_its_keys() -> None:
    """Its writes may be half visible to a footprint read meanwhile: its
    keys join (no wait), so later related images still wait for this one."""
    claims = mp.Claims()
    since = claims.snapshot()
    first, _ = claims.register(0, {"a", "b"}, since)
    claims.finish(first)
    second, waits = claims.register(1, {"b"}, since)
    assert waits == [] and second.keys == {"a", "b"}
    later = claims.snapshot()
    third, waits = claims.register(2, {"a"}, later)
    assert waits == [second] and third.keys == {"a", "b"}


def test_turns_are_taken_in_order_whoever_ends_first() -> None:
    turns = mp.Turns()
    taken: list[int] = []

    def take(index: int) -> None:
        turns.take(index)
        taken.append(index)

    second = threading.Thread(target=take, args=(2,))
    second.start()
    turns.take(0)
    turns.end(1)  # ended without writing
    time.sleep(0.05)
    assert taken == []
    turns.end(0)
    second.join(timeout=5)
    assert taken == [2]


def test_a_gpu_job_runs_on_the_serving_thread_and_returns_its_answer_or_error() -> None:
    jobs = [mp.GpuJob(threading.get_ident), mp.GpuJob(lambda: 1 / 0), mp.GpuJob(lambda: "x")]
    answers: list[Any] = [None, None, None]

    def wait(index: int) -> None:
        try:
            answers[index] = jobs[index].wait()
        except BaseException as exc:  # the test looks at it
            answers[index] = exc

    callers = [threading.Thread(target=wait, args=(i,)) for i in range(3)]
    for caller in callers:
        caller.start()
    jobs[0].run()
    jobs[1].run()
    jobs[2].refuse()
    for caller in callers:
        caller.join(timeout=5)
    assert answers[0] == threading.get_ident()  # run here, on the serving thread
    assert isinstance(answers[1], ZeroDivisionError)
    assert isinstance(answers[2], mp.Held)


def test_a_gpu_job_the_serving_thread_dies_in_lets_its_caller_go() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    job = mp.GpuJob(interrupted)
    with pytest.raises(KeyboardInterrupt):
        job.run()
    with pytest.raises(mp.Held):
        job.wait()


def test_held_is_no_exception_a_lane_could_catch() -> None:
    assert not issubclass(mp.Held, Exception)


def test_the_gpu_proxy_is_the_model_by_name_and_waits_as_gpu_wait() -> None:
    served: list[Callable[[], Any]] = []

    def submit(call: Callable[[], Any]) -> Any:
        served.append(call)
        return call()

    model = _Shapes()
    proxy = mp.on_the_gpu(model, submit)
    assert (proxy.name, proxy.publishes_masks, proxy.active_provider) == (
        "shapes",
        True,
        "FakeExecutionProvider",
    )
    rgba = Image.new("RGBA", (100, H))
    timings = StageTimings(time.monotonic)
    assert proxy.predict_alpha_timed(rgba, timings).shape == (H, 100)
    assert set(timings.seconds) == {"gpu_wait", "inference"}
    assert proxy.predict_alpha(rgba).shape == (H, 100)
    assert len(served) == 2
    stub = mp.on_the_gpu(StubMattingBackend(), submit)
    assert stub.matte(rgba).size == (100, H)
    assert stub.name == StubMattingBackend.name and not stub.publishes_masks
    assert len(served) == 3
