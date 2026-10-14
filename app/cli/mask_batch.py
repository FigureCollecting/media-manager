"""MASK-BATCH-1: mint the cut-out masks in gated, resumable chunks on a GPU box.

    python -m app.cli.mask_batch [--sandbox DIR] run --worklist FILE [--worklist FILE ...]
        [--progress FILE] [--chunk-size N] [--limit N] [--retry-failed] [--dry-run]
        [--backend birefnet-basekept|birefnet] [--device cuda|auto|cpu] [--cpu-threads N]
        [--gpu-index N] [--gpu-util-max PCT] [--gpu-free-min-mib MIB] [--gpu-samples N]
        [--busy-windows SPEC] [--residential-ip IP | --no-residential-check]
        [--ip-echo-url URL] [--residential-ttl S] [--poll-seconds S]
        [--max-wait-seconds S] [--max-consecutive-failures N] [--raw-dir DIR]
    python -m app.cli.mask_batch sql ross-heads --user UUID
    python -m app.cli.mask_batch sql ross-worklist --heads FILE
    python -m app.cli.mask_batch sql catalog-worklist [--shard I/N]

THE WORK. A work list is JSON lines, one per ORIGINAL the spine captured
(``blob_sha256``, its ``storage_key`` in the raw bucket, ``codec``, the origin
``url`` and ``source``, ``product_id``, ``head_id``, ``rank``): Ross's
holdings ranked 0 (owned), 1 (ordered), 2 (wished), the rest of the catalog 9.
The ``sql`` commands print the read-only queries that produce it (run by the
operator against the coordinator and spine databases; psql output is fine,
non-JSON lines are skipped). Work lists are merged in rank order, each original
once at its lowest rank.

PER ORIGINAL, in one chunk: skipped when a clean copy behind it already has a
mask result (``original_settled``: idempotent per image sha, its bytes are not
even read); else its bytes are read from the raw bucket (READ ONLY: the store
refuses a write) and checked against its sha; the clean copy is seeded
(``seed_clean_image``: original private, copy public) and its mask settled
(``settle_clean_copy``) in the media database and bucket. A failure is that
image's alone and is named (``fetch:<class>``, ``raw_hash_mismatch``,
``unsupported_codec:<codec>``, ``seed:<class>``, ``settle:<class>``, or the
lane's own reason); ``--max-consecutive-failures`` in a row stops the batch
(an outage, not an image).

THE GATE, before EVERY chunk (Ross 2026-10-12): not inside a scraper roll
window (``--busy-windows``, minutes of the hour, default :03-:26, :41-:43,
:50-:56; a chunk that runs into one stops before its next image); the GPU idle
(utilisation < 5 %) OR at least 9 GiB of it free, on every one of
``--gpu-samples`` readings; and the box on its residential address (a VPN
window changes it). While the gate fails the batch waits, polling every
``--poll-seconds``, with the model released (a game gets the GPU back; it is
loaded again, ~4 s, when the gate passes).

THE RECORD. Every image's result is appended to ``--progress`` (JSON lines,
flushed per line: a power cut loses at most the image in flight), and a run
skips what it lists (``--retry-failed`` re-drives its failures). Per chunk one
JSON line on stdout: count, statuses, seconds per image (and the model's
share), peak GPU memory used (whole card, sampled), the failures. ``--dry-run``
reads the work list and the progress file, checks the gate once and prints the
plan: no database, no bucket, no model, nothing written.

Exit status: 0 done, 1 done with failures, 2 refused, 3 too many consecutive
failures, 4 waited longer than ``--max-wait-seconds``, 130 stopped (SIGINT or
SIGTERM: the image in flight is finished first).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
from sqlalchemy.orm import Session

from ..object_store import ObjectStore, S3ObjectStore
from ..workers.mask_job import (
    finish_mask_batch,
    open_mask_batch,
    original_settled,
    settle_clean_copy,
)
from ..workers.mask_policy import ImageOrigin
from ..workers.mask_seed import seed_clean_image
from ..workers.matting import MattingBackend

#: The scraper's roll windows (minutes of the hour): no chunk runs in them.
DEFAULT_BUSY_WINDOWS = "03-26,41-43,50-56"
#: Ross's gate: the GPU is idle below this utilisation (percent)...
DEFAULT_GPU_UTIL_MAX = 5
#: ...or has at least this much memory free (MiB; 9 GiB, "the comfortable mark").
DEFAULT_GPU_FREE_MIN_MIB = 9 * 1024
DEFAULT_IP_ECHO_URL = "https://api.ipify.org"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MAX_RANK = 99


# --------------------------------------------------------------------------
# the work list
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One original the spine captured, as the work-list SQL emits it."""

    blob_sha256: str
    storage_key: str
    url: str
    product_id: str
    source: str = "mfc"
    head_id: str | None = None
    rank: int = 9
    codec: str = "identity"

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _field(row: dict[str, Any], key: str, where: str, *, required: bool = True) -> str | None:
    value = row.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value


def _item(row: dict[str, Any], where: str) -> WorkItem:
    sha = row.get("blob_sha256")
    if not isinstance(sha, str) or not _SHA256.match(sha):
        raise ValueError(f"{where}: blob_sha256 must be 64 lowercase hex characters")
    rank = row.get("rank", 9)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= _MAX_RANK:
        raise ValueError(f"{where}: rank must be an integer 0-{_MAX_RANK}")
    codec = row.get("codec", "identity")
    if not isinstance(codec, str):
        raise ValueError(f"{where}: codec must be a string")
    return WorkItem(
        blob_sha256=sha,
        storage_key=str(_field(row, "storage_key", where)),
        url=str(_field(row, "url", where)),
        product_id=str(_field(row, "product_id", where)),
        source=str(_field(row, "source", where)),
        head_id=_field(row, "head_id", where, required=False),
        rank=rank,
        codec=codec,
    )


def _parse(entries: Iterable[tuple[str, int, str]]) -> tuple[list[WorkItem], int]:
    best: dict[str, tuple[int, WorkItem]] = {}
    skipped = 0
    for order, (label, n, line) in enumerate(entries):
        text = line.strip()
        if not text:
            continue
        if not text.startswith("{"):
            skipped += 1  # psql headers, footers, the runner's banner
            continue
        where = f"{label} line {n}"
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"{where}: not a JSON object") from None
        if not isinstance(row, dict):  # pragma: no cover - "{" always decodes to a dict
            raise ValueError(f"{where}: not a JSON object")
        item = _item(row, where)
        held = best.get(item.blob_sha256)
        if held is None or item.rank < held[1].rank:
            best[item.blob_sha256] = (order, item)
    ordered = sorted(best.values(), key=lambda kept: (kept[1].rank, kept[0]))
    return [item for _, item in ordered], skipped


def parse_worklist(lines: Iterable[str], label: str = "work list") -> tuple[list[WorkItem], int]:
    """``(items, skipped)``: the originals in rank order (file order within a
    rank), each once at its lowest rank; ``skipped`` counts the non-blank
    lines that are not JSON objects. A JSON row that is not a valid item is
    refused (ValueError naming its line)."""
    return _parse((label, n, line) for n, line in enumerate(lines, 1))


def read_worklists(paths: Sequence[Path]) -> tuple[list[WorkItem], int]:
    def entries() -> Iterable[tuple[str, int, str]]:
        for path in paths:
            for n, line in enumerate(Path(path).read_text().splitlines(), 1):
                yield str(path), n, line

    return _parse(entries())


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    ok: bool
    gate: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GpuReading:
    util: int
    free_mib: int
    used_mib: int
    total_mib: int


def nvidia_smi_query(index: int) -> Callable[[], str]:
    """A reader of one GPU's utilisation and memory, as nvidia-smi prints them."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]

    def query() -> str:
        done = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        return str(done.stdout).strip()

    return query


class GpuGate:
    """Ross's rule: the GPU is free for a chunk when its utilisation is below
    ``util_max`` percent OR at least ``free_min_mib`` of its memory is free,
    on each of ``samples`` readings ``interval`` seconds apart (one failing
    reading fails the gate). An unreadable GPU fails it (closed); with
    ``absent_ok`` (``--device auto``) a box with no nvidia-smi at all passes:
    the model then runs on the CPU, and there is no GPU to protect."""

    def __init__(
        self,
        *,
        query: Callable[[], str],
        util_max: int = DEFAULT_GPU_UTIL_MAX,
        free_min_mib: int = DEFAULT_GPU_FREE_MIN_MIB,
        samples: int = 3,
        interval: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        absent_ok: bool = False,
    ) -> None:
        self._query = query
        self.absent_ok = absent_ok
        self.util_max = util_max
        self.free_min_mib = free_min_mib
        self.samples = max(1, samples)
        self.interval = interval
        self._sleep = sleep
        self.last: GpuReading | None = None

    def read(self) -> GpuReading:
        parts = [p.strip() for p in self._query().split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi output: {parts!r}")
        util, free, used, total = (int(p) for p in parts)
        return GpuReading(util=util, free_mib=free, used_mib=used, total_mib=total)

    def check(self) -> GateResult:
        detail = ""
        for n in range(self.samples):
            if n:
                self._sleep(self.interval)
            try:
                reading = self.read()
            except FileNotFoundError:
                if self.absent_ok:
                    return GateResult(True, "gpu", "no GPU on this box (device auto: CPU)")
                return GateResult(False, "gpu", "GPU unreadable: no nvidia-smi")
            except Exception as exc:
                return GateResult(False, "gpu", f"GPU unreadable: {type(exc).__name__}")
            self.last = reading
            detail = (
                f"util {reading.util}% (passes below {self.util_max}%), "
                f"free {reading.free_mib} MiB (passes at {self.free_min_mib} MiB)"
            )
            if not (reading.util < self.util_max or reading.free_mib >= self.free_min_mib):
                return GateResult(False, "gpu", detail)
        return GateResult(True, "gpu", detail)


def parse_windows(spec: str) -> tuple[tuple[int, int], ...]:
    """``"03-26,41-43"`` -> ``((3, 26), (41, 43))``: inclusive minute ranges."""
    windows: list[tuple[int, int]] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", part)
        if not m or not 0 <= int(m[1]) <= int(m[2]) <= 59:
            raise ValueError(f"bad busy window {part!r}: want MM-MM within 00-59, start <= end")
        windows.append((int(m[1]), int(m[2])))
    return tuple(windows)


def busy_window(minute: int, windows: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    for start, end in windows:
        if start <= minute <= end:
            return (start, end)
    return None


def public_ip(url: str) -> Callable[[], str]:
    def fetch() -> str:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return str(response.text)

    return fetch


class ResidentialCheck:
    """The box is on its residential exit when its public address is the
    expected one (a VPN window moves it). A pass is trusted for ``ttl``
    seconds; a failure is never cached."""

    def __init__(
        self,
        *,
        expected_ip: str,
        fetch: Callable[[], str],
        ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.expected_ip = expected_ip.strip()
        self._fetch = fetch
        self.ttl = ttl
        self._clock = clock
        self._ok_until: float | None = None

    def check(self) -> GateResult:
        now = self._clock()
        if self._ok_until is not None and now < self._ok_until:
            return GateResult(True, "residential", "residential address (checked recently)")
        try:
            seen = self._fetch().strip()
        except Exception as exc:
            return GateResult(
                False, "residential", f"public address unreadable: {type(exc).__name__}"
            )
        if seen != self.expected_ip:
            return GateResult(
                False, "residential", f"public address {seen} is not the residential exit (VPN?)"
            )
        self._ok_until = now + self.ttl
        return GateResult(True, "residential", "residential address")


@dataclass
class Gates:
    windows: tuple[tuple[int, int], ...]
    gpu: GpuGate | None
    residential: ResidentialCheck | None
    now: Callable[[], dt.datetime]

    def busy(self) -> tuple[int, int] | None:
        return busy_window(self.now().minute, self.windows)

    def check(self) -> GateResult:
        """Window first (no reading needed), then the GPU, then the address."""
        window = self.busy()
        if window is not None:
            return GateResult(
                False,
                "window",
                f"scraper roll window :{window[0]:02d}-:{window[1]:02d} "
                f"(now :{self.now().minute:02d})",
            )
        for gate in (self.gpu, self.residential):
            if gate is not None:
                result = gate.check()
                if not result.ok:
                    return result
        return GateResult(True, "all", "quiet window, GPU free, residential address")


# --------------------------------------------------------------------------
# the record, the VRAM sampler, the raw bucket
# --------------------------------------------------------------------------


class Progress:
    """The batch's own record: JSON lines, one per image (and per chunk),
    each flushed to disk as it is written."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> dict[str, dict[str, Any]]:
        """Per original sha, its last recorded result. A torn line (the power
        went out mid-write) is ignored."""
        if self.path is None or not self.path.exists():
            return {}
        found: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("event") == "item":
                sha = record.get("blob_sha256")
                if isinstance(sha, str):
                    found[sha] = record
        return found

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")  # close a torn line so it cannot swallow this one
            fh.write((json.dumps(record, sort_keys=True, default=str) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())


def vram_command(index: int) -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
        "-lms",
        "200",
    ]


class VramSampler:
    """The peak of a GPU's memory in use while a chunk runs (whole card, MiB:
    under WSL nvidia-smi cannot attribute memory to a process). No command,
    or one that cannot start, samples nothing (``peak_mib`` None)."""

    def __init__(self, command: Sequence[str] | None, *, grace: float = 0.0) -> None:
        self._command = list(command) if command else None
        self._grace = grace
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.peak_mib: int | None = None

    def _read(self, stream: Iterable[str]) -> None:
        for line in stream:
            try:
                value = int(line.strip())
            except ValueError:
                continue
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def __enter__(self) -> VramSampler:
        if self._command is None:
            return self
        try:
            self._proc = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return self
        assert self._proc.stdout is not None
        self._thread = threading.Thread(target=self._read, args=(self._proc.stdout,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        assert self._thread is not None  # started with the process
        self._thread.join(timeout=10)


class ReadOnlyStore:
    """The raw bucket as the batch sees it: it reads originals, never writes."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def get(self, key: str) -> bytes:
        return self._store.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        raise PermissionError("the raw bucket is read only to the mask batch")


class RecentWrites:
    """The media bucket with its last few writes kept in memory: the lane
    re-reads the clean copy it has just stored (to hash it before the mask),
    and over the WAN that read is a round trip per image. Every write still
    goes to the bucket; keys are content addresses, so a kept copy is never
    stale."""

    def __init__(self, store: ObjectStore, *, keep: int = 4) -> None:
        self._store = store
        self._keep = keep
        self._recent: dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        if key in self._recent:
            return self._recent[key]
        return self._store.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store.put(key, data, content_type)
        self._recent.pop(key, None)
        self._recent[key] = data
        while len(self._recent) > self._keep:
            del self._recent[next(iter(self._recent))]


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


@dataclass
class BatchConfig:
    chunk_size: int = 20
    limit: int | None = None
    retry_failed: bool = False
    dry_run: bool = False
    poll_seconds: float = 60.0
    max_wait_seconds: float | None = None
    max_consecutive_failures: int = 5


@dataclass
class Runtime:
    gates: Gates
    open_db: Callable[[], Session]
    media_store: ObjectStore | None
    raw_store: ObjectStore | None
    make_backend: Callable[[], MattingBackend]
    progress: Progress
    vram: Callable[[], Any]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    emit: Callable[[dict[str, Any]], None]
    stop_requested: Callable[[], bool]


class _Stopped(Exception):
    pass


class _GaveUp(Exception):
    def __init__(self, result: GateResult) -> None:
        self.result = result


@dataclass
class _Chunk:
    number: int
    batch_id: int | None
    started: float
    baseline_vram_mib: int | None
    provider: str | None
    records: list[dict[str, Any]] = field(default_factory=list)
    yielded: bool = False

    def report(self, peak_vram_mib: int | None, elapsed: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        minted = [r for r in self.records if r["status"] != "already" and "seconds" in r]
        seconds = [r["seconds"] for r in minted]
        mask_seconds = [r["mask_seconds"] for r in minted if r.get("mask_seconds") is not None]
        return {
            "event": "chunk",
            "chunk": self.number,
            "batch_id": self.batch_id,
            "count": len(self.records),
            "counts": counts,
            "ranks": [r["rank"] for r in self.records],
            "seconds_per_image": round(statistics.fmean(seconds), 3) if seconds else None,
            "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
            "mask_seconds_per_image": (
                round(statistics.fmean(mask_seconds), 3) if mask_seconds else None
            ),
            "peak_vram_mib": peak_vram_mib,
            "baseline_vram_mib": self.baseline_vram_mib,
            "provider": self.provider,
            "failures": [
                {"blob_sha256": r["blob_sha256"], "reason": r["reason"]}
                for r in self.records
                if r["status"] == "failed"
            ],
            "yielded": self.yielded,
            "elapsed_seconds": round(elapsed, 3),
        }


def _decode(stored: bytes, codec: str) -> bytes:
    if codec == "gzip":
        return gzip.decompress(stored)
    return stored


def _mint(
    db: Session,
    item: WorkItem,
    rt: Runtime,
    backend: MattingBackend,
    batch: Any,
    retry_failed: bool,
) -> dict[str, Any]:
    """One original, start to finish; the result as its progress record."""
    assert rt.raw_store is not None and rt.media_store is not None
    start = rt.monotonic()
    record: dict[str, Any] = {
        "event": "item",
        "blob_sha256": item.blob_sha256,
        "product_id": item.product_id,
        "head_id": item.head_id,
        "rank": item.rank,
        "batch_id": getattr(batch, "id", None),
        "clean_sha256": None,
        "mask_sha256": None,
        "reason": None,
    }

    def failed(reason: str) -> dict[str, Any]:
        record.update(status="failed", reason=reason, seconds=round(rt.monotonic() - start, 3))
        return record

    try:
        settled = original_settled(db, item.blob_sha256, retry_failed=retry_failed)
    except Exception as exc:
        db.rollback()
        return failed(f"db:{type(exc).__name__}")
    if settled is not None:
        record.update(status="already", mask_status=settled)
        return record
    if item.codec not in ("identity", "gzip"):
        return failed(f"unsupported_codec:{item.codec}")
    try:
        data = _decode(rt.raw_store.get(item.storage_key), item.codec)
    except Exception as exc:
        return failed(f"fetch:{type(exc).__name__}")
    if hashlib.sha256(data).hexdigest() != item.blob_sha256:
        return failed("raw_hash_mismatch")
    try:
        clean = seed_clean_image(
            db,
            rt.media_store,
            data,
            origins=[ImageOrigin(url=item.url, source=item.source)],
            figure_id=item.product_id,
        )
        clean_sha = clean.sha256
    except Exception as exc:
        db.rollback()
        return failed(f"seed:{type(exc).__name__}")
    record["clean_sha256"] = clean_sha
    mask_start = rt.monotonic()
    try:
        report = settle_clean_copy(
            db, rt.media_store, clean_sha, backend=backend, batch=batch, retry_failed=retry_failed
        )
    except Exception as exc:
        db.rollback()
        return failed(f"settle:{type(exc).__name__}")
    record["mask_seconds"] = round(rt.monotonic() - mask_start, 3)
    if report is None:
        # another original's copy of the same picture was settled first
        record.update(status="already", mask_status=original_settled(db, item.blob_sha256))
    else:
        record.update(status=report.status, reason=report.reason, mask_sha256=report.mask_sha256)
    record["seconds"] = round(rt.monotonic() - start, 3)
    return record


def _wait_for_gate(rt: Runtime, config: BatchConfig, backend: MattingBackend | None) -> bool:
    """Return once the gate passes -- True when the model was released on the
    way -- meanwhile poll and say so once per gate. Any pause releases the
    model: a game gets the GPU back, and a scraper window (up to 24 minutes,
    during which no GPU reading is taken) does not hold ~9.5 GB idle."""
    paused: str | None = None
    released = False
    waited = 0.0
    while True:
        if rt.stop_requested():
            raise _Stopped
        result = rt.gates.check()
        if result.ok:
            if paused is not None:
                rt.emit({"event": "resumed", "waited_seconds": waited})
            return released
        if result.gate != paused:
            paused = result.gate
            rt.emit({"event": "paused", "gate": result.gate, "detail": result.detail})
            if backend is not None and not released:
                backend.release()
                released = True
        if config.max_wait_seconds is not None and waited >= config.max_wait_seconds:
            raise _GaveUp(result)
        rt.sleep(config.poll_seconds)
        waited += config.poll_seconds


def _plan(pending: list[WorkItem], config: BatchConfig, rt: Runtime) -> int:
    result = rt.gates.check()
    rt.emit(
        {
            "event": "plan",
            "pending": len(pending),
            "chunks": math.ceil(len(pending) / config.chunk_size),
            "ranks": {
                str(r): sum(1 for i in pending if i.rank == r)
                for r in sorted({i.rank for i in pending})
            },
            "gate": result.as_dict(),
            "first_chunk": [
                {"blob_sha256": i.blob_sha256, "product_id": i.product_id, "rank": i.rank}
                for i in pending[: config.chunk_size]
            ],
        }
    )
    return 0


def run_batch(items: Sequence[WorkItem], config: BatchConfig, rt: Runtime) -> int:
    """Mint ``items`` chunk by chunk under the gate, lowest rank first (a
    stable sort: work-list order within a rank). Returns the exit status
    (module docstring)."""
    items = sorted(items, key=lambda i: i.rank)
    recorded = rt.progress.load()
    done = {sha for sha, r in recorded.items() if r.get("status") != "failed"}
    failed_before = {sha for sha, r in recorded.items() if r.get("status") == "failed"}
    pending = [
        i
        for i in items
        if i.blob_sha256 not in done and (config.retry_failed or i.blob_sha256 not in failed_before)
    ]
    if config.limit is not None:
        pending = pending[: config.limit]
    rt.emit(
        {
            "event": "start",
            "items": len(items),
            "done": len(done & {i.blob_sha256 for i in items}),
            "failed_before": len(failed_before & {i.blob_sha256 for i in items}),
            "pending": len(pending),
            "chunk_size": config.chunk_size,
            "dry_run": config.dry_run,
        }
    )
    if config.dry_run:
        return _plan(pending, config, rt)

    backend = rt.make_backend()
    if not backend.publishes_masks:
        rt.emit({"event": "refused", "detail": f"backend {backend.name!r} is not a mask model"})
        return 2
    db = rt.open_db()
    totals: dict[str, int] = {}
    streak = 0
    position = 0
    chunk_no = 0
    loaded = False
    try:
        while position < len(pending):
            if _wait_for_gate(rt, config, backend if loaded else None):
                loaded = False
            if not loaded:
                try:
                    backend.load()
                except Exception as exc:
                    rt.emit({"event": "refused", "detail": str(exc)})
                    return 2
                loaded = True
            chunk_no += 1
            gpu = rt.gates.gpu
            chunk = _Chunk(
                number=chunk_no,
                batch_id=None,
                started=rt.monotonic(),
                baseline_vram_mib=gpu.last.used_mib if gpu is not None and gpu.last else None,
                provider=backend.active_provider,
            )
            batch = open_mask_batch(db, backend_name=backend.name)
            chunk.batch_id = batch.id
            with rt.vram() as sampler:
                for item in pending[position : position + config.chunk_size]:
                    if rt.stop_requested():
                        break
                    if rt.gates.busy() is not None:
                        chunk.yielded = True
                        break
                    record = _mint(db, item, rt, backend, batch, config.retry_failed)
                    record["chunk"] = chunk_no
                    rt.progress.append(record)
                    chunk.records.append(record)
                    position += 1
                    streak = streak + 1 if record["status"] == "failed" else 0
                    if streak >= config.max_consecutive_failures:
                        break
            finish_mask_batch(db, batch)
            report = chunk.report(sampler.peak_mib, rt.monotonic() - chunk.started)
            rt.emit(report)
            rt.progress.append(report)
            for status, n in report["counts"].items():
                totals[status] = totals.get(status, 0) + n
            if streak >= config.max_consecutive_failures:
                rt.emit({"event": "circuit_open", "consecutive_failures": streak})
                return 3
            if rt.stop_requested():
                raise _Stopped
    except _Stopped:
        rt.emit({"event": "stopped", "counts": totals, "pending": len(pending) - position})
        return 130
    except _GaveUp as gave_up:
        rt.emit({"event": "gave_up", **gave_up.result.as_dict()})
        return 4
    rt.emit({"event": "done", "counts": totals, "pending": len(pending) - position})
    return 1 if totals.get("failed") else 0


# --------------------------------------------------------------------------
# the work-list SQL (run by the operator, read only)
# --------------------------------------------------------------------------

_SQL_HEAD = """BEGIN TRANSACTION READ ONLY;
\\pset tuples_only on
\\pset footer off
"""
_SQL_TAIL = "ROLLBACK;\n"

_IMAGE_COLUMNS = """json_build_object(
    'blob_sha256', encode(pi.blob_sha256, 'hex'),
    'storage_key', b.storage_key,
    'codec', b.codec,
    'url', u.url,
    'source', s.site,
    'product_id', pi.product_id,
    'head_id', {head},
    'rank', {rank},
    'position', pi.position)"""

_IMAGE_JOINS = """JOIN source s ON s.id = pi.source_id
JOIN raw.blob b ON b.sha256 = pi.blob_sha256
JOIN raw.capture c ON c.id = pi.origin_capture_id AND c.fetched_at = pi.origin_captured_at
JOIN raw.url u ON u.id = c.url_id
WHERE s.site = 'mfc'
  AND pi.state <> 'suppressed' AND pi.suppressed_at IS NULL
  AND b.visibility = 'visible'"""


def _uuid(value: object, what: str) -> str:
    if not isinstance(value, str) or not _UUID.match(value):
        raise ValueError(f"{what} must be a uuid, got {value!r}")
    return value.lower()


def ross_heads_sql(user_id: str) -> str:
    """Step 1 of Ross's work list, on the COORDINATOR database: the head of
    every live copy of one user, ranked by the copy's status (owned 0,
    ordered 1, wished 2, anything else 3), one JSON line each."""
    user = _uuid(user_id, "the user")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: the heads behind one user's live copies (coordinator db).
SELECT json_build_object('head_id', head_id, 'rank', rank) FROM (
  SELECT h.payload::jsonb->>'head_id' AS head_id,
         min(CASE s.payload::jsonb->>'status'
               WHEN 'owned' THEN 0 WHEN 'ordered' THEN 1 WHEN 'wished' THEN 2 ELSE 3 END) AS rank
  FROM facet_state h
  JOIN facet_state s
    ON s.user_id = h.user_id AND s.op = 'upsert'
   AND s.facet_key = 'occ/' || split_part(h.facet_key, '/', 2) || '/status'
  WHERE h.user_id = '{user}'::uuid AND h.op = 'upsert' AND h.facet_key LIKE 'occ/%/head'
  GROUP BY 1
) heads
WHERE head_id IS NOT NULL
ORDER BY rank, head_id;
"""
        + _SQL_TAIL
    )


def ross_worklist_sql(head_lines: Iterable[str]) -> str:
    """Step 2, on the SPINE database: every live MFC item image of those heads
    (and of the products merged into them), the heads' ranks carried."""
    heads: dict[str, int] = {}
    for line in head_lines:
        text = line.strip()
        if not text.startswith("{"):
            continue
        row = json.loads(text)
        head = _uuid(row.get("head_id"), "a head")
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= 9:
            raise ValueError(f"a head's rank must be an integer 0-9, got {rank!r}")
        heads[head] = min(rank, heads.get(head, rank))
    if not heads:
        raise ValueError("no heads in the input")
    values = ",\n  ".join(f"('{h}'::uuid, {r})" for h, r in sorted(heads.items()))
    columns = _IMAGE_COLUMNS.format(head="m.head_id", rank="m.rank")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: Ross's holdings, their MFC item images (spine db).
WITH RECURSIVE heads(head_id, rank) AS (VALUES
  {values}
),
members(product_id, head_id, rank) AS (
  SELECT head_id, head_id, rank FROM heads
  UNION
  SELECT p.id, m.head_id, m.rank FROM product p JOIN members m ON p.redirect_to_product_id = m.product_id
)
SELECT {columns}
FROM members m
JOIN product_image pi ON pi.product_id = m.product_id
{_IMAGE_JOINS}
ORDER BY m.rank, m.head_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


def catalog_worklist_sql(shard: str | None = None) -> str:
    """The whole catalog's live MFC item images (spine db), rank 9; ``shard``
    ``I/N`` keeps the originals whose sha's first byte is I modulo N, so the
    list can be fetched in pieces."""
    where = ""
    if shard is not None:
        m = re.fullmatch(r"(\d+)/(\d+)", shard)
        if not m or not 0 <= int(m[1]) < int(m[2]):
            raise ValueError(f"bad shard {shard!r}: want I/N with 0 <= I < N")
        where = f"\n  AND get_byte(pi.blob_sha256, 0) % {int(m[2])} = {int(m[1])}"
    columns = _IMAGE_COLUMNS.format(head="NULL", rank="9")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: the catalog's MFC item images (spine db).
SELECT {columns}
FROM product_image pi
{_IMAGE_JOINS}{where}
ORDER BY pi.product_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


class Refused(Exception):
    """A request refused before anything was touched (exit status 2)."""


def build_backend(args: argparse.Namespace) -> MattingBackend:
    from .masks import make_backend

    backend = make_backend(
        args.backend, None, None, None, device=args.device, cpu_threads=args.cpu_threads
    )
    assert backend is not None
    return backend


def _raw_store(args: argparse.Namespace) -> ObjectStore:
    if args.raw_dir:
        from ..object_store import LocalObjectStore

        return ReadOnlyStore(LocalObjectStore(args.raw_dir))
    names = ("RAW_S3_ENDPOINT_URL", "RAW_S3_BUCKET", "RAW_S3_ACCESS_KEY", "RAW_S3_SECRET_KEY")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise Refused(f"the raw bucket needs {', '.join(missing)} (or --raw-dir)")
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["RAW_S3_ENDPOINT_URL"],
        region_name=os.environ.get("RAW_S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["RAW_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RAW_S3_SECRET_KEY"],
    )
    return ReadOnlyStore(S3ObjectStore(client=client, bucket=os.environ["RAW_S3_BUCKET"]))


def _emit(event: dict[str, Any]) -> None:
    event.setdefault("at", dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    print(json.dumps(event, sort_keys=True, default=str), flush=True)


def _run(args: argparse.Namespace) -> int:
    try:
        items, skipped = read_worklists(args.worklist)
        windows = parse_windows(args.busy_windows)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    residential_ip = args.residential_ip or os.environ.get("MASK_BATCH_RESIDENTIAL_IP")
    if not args.no_residential_check and not residential_ip:
        raise Refused(
            "the residential check needs --residential-ip (or MASK_BATCH_RESIDENTIAL_IP), "
            "or --no-residential-check for a local run"
        )
    on_gpu = args.device != "cpu"
    gates = Gates(
        windows=windows,
        gpu=(
            GpuGate(
                query=nvidia_smi_query(args.gpu_index),
                util_max=args.gpu_util_max,
                free_min_mib=args.gpu_free_min_mib,
                samples=args.gpu_samples,
                absent_ok=args.device == "auto",
            )
            if on_gpu
            else None
        ),
        residential=(
            None
            if args.no_residential_check
            else ResidentialCheck(
                expected_ip=str(residential_ip),
                fetch=public_ip(args.ip_echo_url),
                ttl=args.residential_ttl,
            )
        ),
        now=lambda: dt.datetime.now(dt.UTC),
    )
    config = BatchConfig(
        chunk_size=args.chunk_size,
        limit=args.limit,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    stop = threading.Event()
    previous: dict[int, Any] = {}
    if not args.dry_run:

        def on_signal(signum: int, frame: FrameType | None) -> None:
            stop.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, on_signal)

    media_store: ObjectStore | None = None
    raw_store: ObjectStore | None = None
    sessions: list[Session] = []
    if not args.dry_run:
        from .masks import open_stores

        raw_store = _raw_store(args)
        db, bucket = open_stores(args.sandbox)
        media_store = RecentWrites(bucket)
        sessions.append(db)
    rt = Runtime(
        gates=gates,
        open_db=lambda: sessions[0],
        media_store=media_store,
        raw_store=raw_store,
        make_backend=lambda: build_backend(args),
        progress=Progress(Path(args.progress)),
        vram=lambda: VramSampler(vram_command(args.gpu_index) if on_gpu else None),
        sleep=time.sleep,
        monotonic=time.monotonic,
        emit=_emit,
        stop_requested=stop.is_set,
    )
    if skipped:
        _emit({"event": "worklist", "items": len(items), "skipped_lines": skipped})
    try:
        return run_batch(items, config, rt)
    finally:
        for db in sessions:
            db.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


def _sql(args: argparse.Namespace) -> int:
    try:
        if args.query == "ross-heads":
            sql = ross_heads_sql(args.user)
        elif args.query == "ross-worklist":
            sql = ross_worklist_sql(Path(args.heads).read_text().splitlines())
        else:
            sql = catalog_worklist_sql(args.shard)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    sys.stdout.write(sql)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.mask_batch", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("--sandbox", help="SQLite + local objects in DIR instead of DATABASE_URL/S3")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="mint masks for a work list (gated, chunked, resumable)")
    run.add_argument("--worklist", action="append", required=True, type=Path)
    run.add_argument("--progress", default="mask-batch-progress.jsonl")
    run.add_argument("--chunk-size", type=int, default=20)
    run.add_argument("--limit", type=int)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--dry-run", action="store_true", help="plan only: nothing is opened or written"
    )
    run.add_argument(
        "--backend", choices=["birefnet-basekept", "birefnet"], default="birefnet-basekept"
    )
    run.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    run.add_argument("--cpu-threads", type=int)
    run.add_argument("--gpu-index", type=int, default=0)
    run.add_argument("--gpu-util-max", type=int, default=DEFAULT_GPU_UTIL_MAX)
    run.add_argument("--gpu-free-min-mib", type=int, default=DEFAULT_GPU_FREE_MIN_MIB)
    run.add_argument("--gpu-samples", type=int, default=3)
    run.add_argument("--busy-windows", default=DEFAULT_BUSY_WINDOWS)
    run.add_argument("--residential-ip")
    run.add_argument("--no-residential-check", action="store_true")
    run.add_argument("--ip-echo-url", default=DEFAULT_IP_ECHO_URL)
    run.add_argument("--residential-ttl", type=float, default=300.0)
    run.add_argument("--poll-seconds", type=float, default=60.0)
    run.add_argument("--max-wait-seconds", type=float)
    run.add_argument("--max-consecutive-failures", type=int, default=5)
    run.add_argument("--raw-dir", help="read originals from this directory instead of RAW_S3_*")

    sql = sub.add_parser("sql", help="print a read-only work-list query")
    queries = sql.add_subparsers(dest="query", required=True)
    heads = queries.add_parser("ross-heads", help="coordinator db: one user's heads, ranked")
    heads.add_argument("--user", required=True)
    ross = queries.add_parser("ross-worklist", help="spine db: those heads' MFC item images")
    ross.add_argument("--heads", required=True)
    catalog = queries.add_parser("catalog-worklist", help="spine db: every MFC item image")
    catalog.add_argument("--shard")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "sql":
            return _sql(args)
        return _run(args)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
