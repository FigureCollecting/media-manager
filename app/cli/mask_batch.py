"""MASK-BATCH-1: mint the cut-out masks in gated, resumable chunks on a GPU box.

    python -m app.cli.mask_batch [--sandbox DIR] run --worklist FILE [--worklist FILE ...]
        [--progress FILE] [--chunk-size N] [--limit N] [--retry-failed] [--dry-run]
        [--backend birefnet-basekept|birefnet] [--device cuda|auto|cpu] [--cpu-threads N]
        [--gpu-index N] [--gpu-util-max PCT] [--gpu-free-min-mib MIB] [--gpu-samples N]
        [--busy-windows SPEC] [--residential-ip IP|CIDR | --no-residential-check]
        [--ip-echo-url URL] [--residential-ttl S] [--poll-seconds S]
        [--max-wait-seconds S] [--max-consecutive-failures N] [--raw-dir DIR]
        [--writers N] [--prefetch N]
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
lane's own reason). An OUTAGE-class failure (``is_outage``: a dropped
connection, a timeout, the database's OperationalError, the bucket's 5xx) is
recorded ``interrupted``: a plain rerun retries it (its failed media-db row
too), and ``--max-consecutive-failures`` of them in a row stop the batch. An
image's OWN fault (its codec, its hash, NoSuchKey, an undecodable picture,
the model's error) stays ``failed``: neither counted toward that streak nor
resetting it, and retried only with ``--retry-failed``.

THE GATE, before EVERY chunk (Ross 2026-10-12): not inside a scraper roll
window (``--busy-windows``, minutes of the hour, default :03-:26, :41-:43,
:50-:56; a chunk that runs into one stops before its next image); the GPU idle
(utilisation < 5 %) OR at least 9 GiB of it free, on every one of
``--gpu-samples`` readings -- while the model is held, no other consumer
instead: the card's memory not grown more than 512 MiB since the model was
measured and its utilisation not at or above 5 % on every reading, taken 2 s
after the chunk (``GpuGate``); and the box on its residential address (a VPN
window moves it; ``--residential-ip`` takes an address or the ISP's block).
While the gate fails the batch waits, polling every
``--poll-seconds``, with the model released (a game gets the GPU back; it is
loaded again, ~4 s, when the gate passes).

THE RECORD. Every image's result is appended to ``--progress`` (JSON lines,
flushed per line: a power cut loses at most the image in flight), and a run
skips what it lists (``--retry-failed`` re-drives its failures). Per chunk one
JSON line on stdout: count, statuses, seconds per image (and the model's
share, and per MASKED image -- ok or held -- apart from the cheap ineligible ones), peak GPU memory used (whole card, sampled), the failures,
and where the seconds went: each stage's mean and median per image over the
images that ran it (``stage_seconds_per_image``, ``stage_median_seconds``;
app.workers.stage_timing: fetch, seed, readback, inference, base_keep,
analysis, encode, upload, db; a stage no image ran is absent) and how many
did (``stage_images``): each mean is over those images only, so the means
add up to the seconds per image only when every image ran every stage. ``--dry-run``
reads the work list and the progress file, checks the gate once and prints the
plan: no database, no bucket, no model, nothing written.

THE PIPELINE (MASK-PERF-2). The images of a chunk run ``--writers`` at a
time (default 12), each on a worker with a database session of its own, and
the next ``--prefetch`` originals (default 4) are read, decoded and
hash-checked ahead; the model runs on the driver's thread alone, one image
at a time, so one image's I/O (its read, its seed, its judgment, its
uploads and rows) overlaps another's model run. Each image's own work runs
in the same order as before, and what it writes is what one image at a
time writes: images of one picture (their originals, captures, copies and
gallery link meet: app.workers.mask_pipeline) wait for each other in
work-list order, and the batch row is written in that order. A record is
written, in work-list order, once its image's writes committed. A stop, a
scraper window or the circuit lets no further image start or onto the GPU:
the images past the GPU finish, the others are let go unrecorded and come
back first. Each image's stages are its own wall time, the pipeline's waits
among them (``claim``, ``gpu_wait``, ``turn_wait``); the chunk's
``throughput_seconds_per_image`` is its wall time over its images.
``--writers 0`` runs one image at a time on the driver's own session.

Exit status: 0 done, 1 done with failures, 2 refused, 3 too many consecutive
failures, 4 waited longer than ``--max-wait-seconds``, 130 stopped (SIGINT or
SIGTERM: the image in flight is finished first).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import functools
import gzip
import hashlib
import ipaddress
import json
import math
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
from sqlalchemy import inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.pool import QueuePool, SingletonThreadPool, StaticPool

from ..models import MaskBatch
from ..object_store import ObjectStore, S3ObjectStore
from ..workers.mask_job import (
    finish_mask_batch,
    open_mask_batch,
    original_settled,
    settle_clean_copy,
)
from ..workers.mask_pipeline import Claim, Claims, GpuJob, Held, Turns, on_the_gpu
from ..workers.mask_policy import ImageOrigin
from ..workers.mask_seed import PictureKeys, picture_keys, seed_clean_image, seed_footprint
from ..workers.matting import BiRefNetMattingBackend, IsNetMattingBackend, MattingBackend
from ..workers.stage_timing import STAGES, DatabaseSeconds, StageTimings, timed

#: The scraper's roll windows (minutes of the hour): no chunk runs in them.
DEFAULT_BUSY_WINDOWS = "03-26,41-43,50-56"
#: Ross's gate: the GPU is idle below this utilisation (percent)...
DEFAULT_GPU_UTIL_MAX = 5
#: ...or has at least this much memory free (MiB; 9 GiB, "the comfortable mark").
DEFAULT_GPU_FREE_MIN_MIB = 9 * 1024
#: While the model is held: the seconds the gate waits after a chunk before
#: its readings (the batch's own compute has left them by then)...
DEFAULT_GPU_SETTLE_SECONDS = 2.0
#: ...and the growth of the card's used memory (MiB) that means another consumer.
DEFAULT_GPU_GROWTH_MAX_MIB = 512
#: The most of the card's growth that counts as the batch's own model (MiB).
#: Measured on the RTX 4070 Ti: ~9,470 MiB (9.27-9.48 GiB) with BiRefNet plus its
#: two IS-Net donors loaded. The caps are far higher (BiRefNet 8 GiB + two IS-Net
#: 2 GiB each = 12 GiB), so this is an empirical bound (8 GiB + 2 GiB + 512 MiB,
#: ~1.3 GB over the measurement), not the sum of the caps; growth beyond it counts
#: as another consumer's.
OWN_MAX_MIB = (
    (BiRefNetMattingBackend.GPU_MEM_LIMIT + IsNetMattingBackend.GPU_MEM_LIMIT) >> 20
) + 512
DEFAULT_IP_ECHO_URL = "https://api.ipify.org"
#: MASK-PERF-2: the images a chunk runs at once, each on a worker with its own
#: database session (0: one image at a time, the driver's own session). 12 by
#: the simulation of the live costs: 5.5 s an image one at a time, 1.6 s with
#: 6, 1.23 s with 12, 1.22 s with 16. The media db's pool lends 30
#: connections, the sandbox's 15 (a run needs writers + 1)...
DEFAULT_WRITERS = 12
MAX_WRITERS = 16
#: ...and the originals read, decoded and hash-checked ahead of them.
DEFAULT_PREFETCH = 4
MAX_PREFETCH = 16

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MAX_RANK = 99
#: The recorded statuses a run does not count as done.
_REDO = ("failed", "interrupted")
#: The statuses of an image that got a mask (published, or held for review).
_MASKED = ("ok", "held")
#: The statuses of an image the model ran on (its arena is full afterwards).
_RAN_MODEL = ("ok", "held", "empty")


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
    """Ross's rule (2026-10-12): the GPU is free for a chunk when its
    utilisation is below ``util_max`` percent OR at least ``free_min_mib`` of
    its memory is free, on each of ``samples`` readings ``interval`` seconds
    apart (one failing reading fails the gate). An unreadable GPU fails it
    (closed); with ``absent_ok`` (``--device auto``) a box with no nvidia-smi
    at all passes: the model then runs on the CPU, and there is no GPU to
    protect.

    That rule holds until the model is loaded. WHILE it is held, the model's
    own memory (9,453-9,481 MiB measured: more than the 9 GiB mark) would fail
    it on every check, and crediting that memory as free would pass every
    check, a game included. So while held the gate looks for ANOTHER consumer
    instead, after waiting ``settle`` seconds (the chunk has ended: the batch
    is idle, so its own compute is not in the readings): it pauses when the
    card's used memory has grown more than ``growth_max_mib`` since the model
    was measured, or when the utilisation is at or above ``util_max`` on EVERY
    one of the samples. Under WSL nvidia-smi cannot attribute memory to a
    process, so the model is measured: the memory in use after the first image
    it runs (its arena is full by then: both models take fixed 1024 px inputs)
    less the reading before it was loaded (``mark_loading``), once per load;
    ``forget_own`` on release. A consumer that starts while the model loads is
    measured as ours, so no more than ``OWN_MAX_MIB`` is taken as ours: the
    rest counts as growth. Until the model has run (a chunk of ineligible
    photos only), or when the measurement fails, Ross's rule applies, after
    the settle."""

    def __init__(
        self,
        *,
        query: Callable[[], str],
        util_max: int = DEFAULT_GPU_UTIL_MAX,
        free_min_mib: int = DEFAULT_GPU_FREE_MIN_MIB,
        samples: int = 3,
        interval: float = 0.5,
        settle: float = DEFAULT_GPU_SETTLE_SECONDS,
        growth_max_mib: int = DEFAULT_GPU_GROWTH_MAX_MIB,
        sleep: Callable[[float], None] = time.sleep,
        absent_ok: bool = False,
    ) -> None:
        self._query = query
        self.absent_ok = absent_ok
        self.util_max = util_max
        self.free_min_mib = free_min_mib
        self.samples = max(1, samples)
        self.interval = interval
        self.settle = settle
        self.growth_max_mib = growth_max_mib
        self._sleep = sleep
        self.last: GpuReading | None = None
        self.own_mib = 0
        self._held = False
        self._baseline_mib: int | None = None
        self._reference_mib: int | None = None

    def mark_loading(self) -> None:
        """The model is about to load: the last reading is the card without it."""
        self._held = True
        self._baseline_mib = self.last.used_mib if self.last is not None else None
        self._reference_mib = None
        self.own_mib = 0

    def measure_own(self) -> None:
        """The model has run once: what it added since ``mark_loading`` is
        ours (up to ``OWN_MAX_MIB``), and the card as it is now, the reference
        another consumer is measured against."""
        if self._baseline_mib is None:
            return
        try:
            reading = self.read()
        except Exception:
            return  # not measured: Ross's rule stays in force
        self.own_mib = max(0, reading.used_mib - self._baseline_mib)
        self._reference_mib = self._baseline_mib + min(self.own_mib, OWN_MAX_MIB)

    def forget_own(self) -> None:
        self.own_mib = 0
        self._held = False
        self._baseline_mib = None
        self._reference_mib = None

    def read(self) -> GpuReading:
        parts = [p.strip() for p in self._query().split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi output: {parts!r}")
        util, free, used, total = (int(p) for p in parts)
        return GpuReading(util=util, free_mib=free, used_mib=used, total_mib=total)

    def check(self) -> GateResult:
        if self._held:
            self._sleep(self.settle)
        detail = ""
        busy = 0
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
            if self._reference_mib is None:
                detail = (
                    f"util {reading.util}% (passes below {self.util_max}%), "
                    f"free {reading.free_mib} MiB (passes at {self.free_min_mib} MiB)"
                )
                if not (reading.util < self.util_max or reading.free_mib >= self.free_min_mib):
                    return GateResult(False, "gpu", detail)
                continue
            # memory another process freed lowers the reference: what it takes next shows
            self._reference_mib = min(self._reference_mib, reading.used_mib)
            grown = reading.used_mib - self._reference_mib
            detail = (
                f"util {reading.util}%, used {reading.used_mib} MiB, {grown:+d} MiB since "
                f"our model ({self.own_mib} MiB) was measured "
                f"(another consumer above +{self.growth_max_mib} MiB)"
            )
            if grown > self.growth_max_mib:
                return GateResult(False, "gpu", detail)
            busy += reading.util >= self.util_max
        if self._reference_mib is not None and busy == self.samples:
            return GateResult(
                False,
                "gpu",
                f"{detail}; util at or above {self.util_max}% on every one of "
                f"{self.samples} readings with our model idle: another consumer",
            )
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
    expected one, or inside the expected block (``198.51.100.0/24``: a DHCP
    renewal stays in the ISP's block, a VPN window leaves it). A pass is
    trusted for ``ttl`` seconds; a failure is never cached."""

    def __init__(
        self,
        *,
        expected_ip: str,
        fetch: Callable[[], str],
        ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            self.expected = ipaddress.ip_network(expected_ip.strip(), strict=False)
        except ValueError:
            raise ValueError(
                f"the residential address {expected_ip!r} is not an IP or a CIDR"
            ) from None
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
        try:
            home = ipaddress.ip_address(seen) in self.expected
        except ValueError:
            home = False
        if not home:
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
    and over the WAN that read is a GET of the whole image per image. Every
    write still goes to the bucket, and is kept only once the PUT returned.
    A kept copy is served only when the bucket's ETag (a HEAD, no body) is
    the md5 of the kept bytes, so the lane's hash check still covers what the
    bucket holds: a PUT the bucket stored wrong, a store with no ETag, an ETag
    that is no md5 (multipart) or a HEAD that fails all read the bucket.
    Pipelined (MASK-PERF-2) its workers share it: ``keep`` covers each
    one's last original, copy and mask, and the kept copies are locked."""

    def __init__(self, store: ObjectStore, *, keep: int = 4) -> None:
        self._store = store
        self.keep = keep
        self._recent: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes:
        with self._lock:
            kept = self._recent.get(key)
        if kept is not None and self._stored_as(key, kept):
            return kept
        return self._store.get(key)

    def _stored_as(self, key: str, data: bytes) -> bool:
        etag = getattr(self._store, "etag", None)
        if etag is None:
            return False
        try:
            stored = etag(key)
        except Exception:
            return False  # the GET that follows succeeds or fails on its own
        return bool(stored == hashlib.md5(data, usedforsecurity=False).hexdigest())

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store.put(key, data, content_type)
        with self._lock:
            self._recent.pop(key, None)
            self._recent[key] = data
            while len(self._recent) > self.keep:
                del self._recent[next(iter(self._recent))]


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """``writers`` above 0 runs each chunk's images that many at a time, each
    on a worker with a database session of its own, ``prefetch`` originals
    read ahead (MASK-PERF-2); 0 runs one image at a time on the driver's own
    session, as before (the command line's default is the pipeline)."""

    chunk_size: int = 20
    limit: int | None = None
    retry_failed: bool = False
    dry_run: bool = False
    poll_seconds: float = 60.0
    max_wait_seconds: float | None = None
    max_consecutive_failures: int = 5
    writers: int = 0
    prefetch: int = 4


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
    #: per record, its seconds by stage (app.workers.stage_timing)
    stages: list[dict[str, float]] = field(default_factory=list)
    yielded: bool = False

    def report(self, peak_vram_mib: int | None, elapsed: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        minted = [r for r in self.records if r["status"] != "already" and "seconds" in r]
        seconds = [r["seconds"] for r in minted]
        mask_seconds = [r["mask_seconds"] for r in minted if r.get("mask_seconds") is not None]
        # an ineligible photo costs next to nothing (no model): the real rate
        masked = [r["seconds"] for r in minted if r["status"] in _MASKED]
        # each stage over the minted images that ran it: one that never ran is absent
        by_stage: dict[str, list[float]] = {name: [] for name in STAGES}
        for r, stages in zip(self.records, self.stages, strict=True):
            if r["status"] != "already" and "seconds" in r:
                for name, took in stages.items():
                    by_stage.setdefault(name, []).append(took)
        ran = {name: per_image for name, per_image in by_stage.items() if per_image}
        return {
            "event": "chunk",
            "chunk": self.number,
            "batch_id": self.batch_id,
            "count": len(self.records),
            "counts": counts,
            "ranks": [r["rank"] for r in self.records],
            "seconds_per_image": round(statistics.fmean(seconds), 3) if seconds else None,
            "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
            "masked": len(masked),
            "seconds_per_masked_image": round(statistics.fmean(masked), 3) if masked else None,
            "mask_seconds_per_image": (
                round(statistics.fmean(mask_seconds), 3) if mask_seconds else None
            ),
            "stage_seconds_per_image": {n: round(statistics.fmean(t), 3) for n, t in ran.items()},
            "stage_median_seconds": {n: round(statistics.median(t), 3) for n, t in ran.items()},
            "stage_images": {n: len(t) for n, t in ran.items()},
            "peak_vram_mib": peak_vram_mib,
            "baseline_vram_mib": self.baseline_vram_mib,
            "provider": self.provider,
            "failures": [
                {"blob_sha256": r["blob_sha256"], "reason": r["reason"]}
                for r in self.records
                if r["status"] in _REDO
            ],
            "yielded": self.yielded,
            "elapsed_seconds": round(elapsed, 3),
            # the chunk's wall seconds over its images: what overlapping them gains
            "throughput_seconds_per_image": (
                round(elapsed / len(self.records), 3) if self.records else None
            ),
        }


#: The exception classes (by name, anywhere in the class's ancestry) of an
#: OUTAGE -- the network, the database or the bucket -- not of the image: the
#: builtin and botocore ConnectionError, timeouts, SQLAlchemy's
#: OperationalError and InterfaceError (a dropped connection), botocore's
#: HTTPClientError (a connection closed mid-response).
_OUTAGE_CLASSES = frozenset(
    {"ConnectionError", "TimeoutError", "Timeout", "OperationalError", "InterfaceError",
     "HTTPClientError"}
)  # fmt: skip
#: The same, as the mask lane records a failure: its class name alone.
_OUTAGE_REASONS = _OUTAGE_CLASSES | {
    "ConnectionResetError", "ConnectionRefusedError", "ConnectionAbortedError",
    "BrokenPipeError", "EndpointConnectionError", "ConnectionClosedError",
    "ConnectTimeoutError", "ReadTimeoutError", "ProxyConnectionError",
}  # fmt: skip


def is_outage(exc: BaseException) -> bool:
    """An outage-class failure (``_OUTAGE_CLASSES``, or the bucket answering
    5xx): the image is not at fault, a rerun should try it again."""
    if any(cls.__name__ in _OUTAGE_CLASSES for cls in type(exc).__mro__):
        return True
    response = getattr(exc, "response", None)  # botocore's ClientError
    if isinstance(response, dict):
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return isinstance(status, int) and status >= 500
    return False


def is_outage_reason(reason: str | None) -> bool:
    return reason in _OUTAGE_REASONS


class _Witness:
    """The media bucket as one image's lane sees it, keeping the last error
    it raised (the lane records only its class: a 503 and a 403 are both
    ``ClientError`` there). With ``timings``, each PUT is an "upload" and
    each GET a "readback", inside whichever stage made it (the seed's too)."""

    def __init__(self, store: ObjectStore, timings: StageTimings | None = None) -> None:
        self._store = store
        self._timings = timings
        self.error: Exception | None = None

    def get(self, key: str) -> bytes:
        try:
            with timed(self._timings, "readback"):
                return self._store.get(key)
        except Exception as exc:
            self.error = exc
            raise

    def put(self, key: str, data: bytes, content_type: str) -> None:
        try:
            with timed(self._timings, "upload"):
                self._store.put(key, data, content_type)
        except Exception as exc:
            self.error = exc
            raise


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
    timings: StageTimings | None = None,
    lane: _ImageLane | None = None,
) -> dict[str, Any]:
    """One original, start to finish; the result as its progress record.
    ``timings``, when given, gets the seconds of each stage (MASK-PERF-1);
    its SQL's seconds are the batch's to time (DatabaseSeconds, run_batch).
    ``lane``, pipelined (MASK-PERF-2): the image's claim before its first
    query, its original read ahead, the hold checked before its seed and its
    turn before its batch-row writes; its seconds read ahead count as its."""
    assert rt.raw_store is not None and rt.media_store is not None
    start = rt.monotonic() - (lane.ahead if lane is not None else 0.0)
    with timed(timings, "db"):  # read again from the database once a rollback expired it
        batch_id = getattr(batch, "id", None)
    record: dict[str, Any] = {
        "event": "item",
        "blob_sha256": item.blob_sha256,
        "product_id": item.product_id,
        "head_id": item.head_id,
        "rank": item.rank,
        "batch_id": batch_id,
        "clean_sha256": None,
        "mask_sha256": None,
        "reason": None,
    }

    def failed(reason: str, outage: bool = False) -> dict[str, Any]:
        # an outage's casualty is 'interrupted': it counts toward the circuit and
        # a plain rerun retries it; an image's own fault stays 'failed'
        status = "interrupted" if outage else "failed"
        record.update(status=status, reason=reason, seconds=round(rt.monotonic() - start, 3))
        return record

    def raised(where: str, exc: Exception) -> dict[str, Any]:
        return failed(f"{where}:{type(exc).__name__}", is_outage(exc))

    def rolled_back(where: str, exc: Exception) -> dict[str, Any]:
        with timed(timings, "db"):  # a round trip like any query
            db.rollback()
        return raised(where, exc)

    media = _Witness(rt.media_store, timings)
    try:
        if lane is not None:
            lane.claim(db)  # its footprint read, and the related images it waits for
        with timed(timings, "db"):  # the image's first query: the pool's checkout too
            settled = original_settled(db, item.blob_sha256, retry_failed=retry_failed)
    except Exception as exc:
        return rolled_back("db", exc)
    if settled is not None:
        record.update(status="already", mask_status=settled)
        return record
    if item.codec not in ("identity", "gzip"):
        return failed(f"unsupported_codec:{item.codec}")
    try:
        if lane is None:
            with timed(timings, "fetch"):
                data = _decode(rt.raw_store.get(item.storage_key), item.codec)
        else:
            start -= lane.fetch_seconds  # read ahead, before the image had a worker
            data = lane.fetched()
    except Exception as exc:
        return raised("fetch", exc)
    if hashlib.sha256(data).hexdigest() != item.blob_sha256:
        return failed("raw_hash_mismatch")
    if lane is not None:
        lane.before_seed()  # held: nothing of it written, a rerun takes it
    try:
        with timed(timings, "seed"):
            clean = seed_clean_image(
                db,
                media,
                data,
                origins=[ImageOrigin(url=item.url, source=item.source)],
                figure_id=item.product_id,
            )
        clean_sha = clean.sha256
    except Exception as exc:
        return rolled_back("seed", exc)
    record["clean_sha256"] = clean_sha
    mask_start = rt.monotonic()
    try:
        report = settle_clean_copy(
            db,
            media,
            clean_sha,
            backend=backend,
            batch=batch,
            retry_failed=retry_failed,
            timings=timings,
            turn=lane.turn if lane is not None else None,
        )
    except Exception as exc:
        return rolled_back("settle", exc)
    record["mask_seconds"] = round(rt.monotonic() - mask_start, 3)
    if report is None:
        # another original's copy of the same picture was settled first
        record.update(status="already", mask_status=original_settled(db, item.blob_sha256))
    elif report.status == "failed":
        # the lane caught it and kept the class name; the bucket's error, if any, says more
        witnessed = media.error is not None and is_outage(media.error)
        failed(str(report.reason), is_outage_reason(report.reason) or witnessed)
    else:
        record.update(status=report.status, reason=report.reason, mask_sha256=report.mask_sha256)
    record["seconds"] = round(rt.monotonic() - start, 3)
    return record


# --------------------------------------------------------------------------
# the pipeline (MASK-PERF-2)
# --------------------------------------------------------------------------


@dataclass
class _Fetched:
    """An original read ahead: its bytes or the error reading them, its
    picture's keys (None when it never reaches the seed), and the seconds
    each took on the read-ahead thread."""

    data: bytes | None = None
    error: Exception | None = None
    keys: PictureKeys | None = None
    fetch_seconds: float = 0.0
    keys_seconds: float = 0.0


def _read_ahead(item: WorkItem, rt: Runtime) -> _Fetched:
    """``item``'s original from the raw bucket, decoded and checked against
    its sha, and its picture's keys, before the image has a worker. An
    original in a codec the lane refuses is never read."""
    assert rt.raw_store is not None
    if item.codec not in ("identity", "gzip"):
        return _Fetched()
    start = rt.monotonic()
    try:
        data = _decode(rt.raw_store.get(item.storage_key), item.codec)
    except Exception as exc:
        return _Fetched(error=exc, fetch_seconds=rt.monotonic() - start)
    fetched = rt.monotonic()
    keys = None
    if hashlib.sha256(data).hexdigest() == item.blob_sha256:
        # a picture that will not decode fails the seed before it writes a row
        with contextlib.suppress(Exception):
            keys = picture_keys(data)
    return _Fetched(
        data=data,
        keys=keys,
        fetch_seconds=fetched - start,
        keys_seconds=rt.monotonic() - fetched,
    )


class _BatchRow:
    """The chunk's batch row as the last image to write it left it (its
    counts and cursor), lent to the next image's own copy of the row at its
    turn: each image's UPDATE is the one one image at a time makes. Unknown
    after an image whose last write rolled back: the next one reads the row
    again, as one image at a time does after a rollback -- at its turn,
    before any write of its own is pending, so an autoflushing session
    still makes one UPDATE of both (never the cursor flushed alone first)."""

    def __init__(self, batch: MaskBatch) -> None:
        self._state: tuple[dict[str, int], str | None] | None = (
            dict(batch.counts),
            batch.cursor_sha256,
        )

    def lend(self, db: Session, batch: MaskBatch) -> None:
        if self._state is None:
            db.refresh(batch, ["counts", "cursor_sha256"])
            return
        counts, cursor = self._state
        set_committed_value(batch, "counts", dict(counts))
        set_committed_value(batch, "cursor_sha256", cursor)

    def keep(self, batch: MaskBatch) -> None:
        state = inspect(batch)
        loaded = state.dict
        if state.modified or "counts" not in loaded or "cursor_sha256" not in loaded:
            self._state = None
        else:
            self._state = (dict(loaded["counts"]), loaded["cursor_sha256"])


@dataclass
class _Done:
    """An image off its worker: its record, or None when it was held."""

    index: int
    record: dict[str, Any] | None
    stages: dict[str, float]
    error: BaseException | None = None


@dataclass(eq=False)
class _ChunkRun:
    """What one chunk's workers share: the claims, the turns, the batch row,
    the hold, and the queue to the driver's thread (a GPU job or an image
    done; at most one of either per worker, so never more than ``writers``)."""

    batch: MaskBatch
    events: queue.Queue[GpuJob | _Done]
    claims: Claims = field(default_factory=Claims)
    turns: Turns = field(default_factory=Turns)
    hold: threading.Event = field(default_factory=threading.Event)
    row: _BatchRow = field(init=False)

    def __post_init__(self) -> None:
        self.row = _BatchRow(self.batch)

    def submit(self, call: Callable[[], Any]) -> Any:
        """Run ``call`` (a model call) on the driver's thread, and wait."""
        job = GpuJob(call)
        self.events.put(job)
        return job.wait()


class _ImageLane:
    """One image's place in the pipeline, as ``_mint`` meets it."""

    def __init__(
        self,
        run: _ChunkRun,
        index: int,
        item: WorkItem,
        fetched: _Fetched,
        db: Session,
        batch: MaskBatch,
        timings: StageTimings,
    ) -> None:
        self._run, self._index, self._item = run, index, item
        self._fetched, self._db, self._batch, self._timings = fetched, db, batch, timings
        self._claim: Claim | None = None
        self._registered = False
        self._turned = False
        timings.add("claim", fetched.keys_seconds)  # its picture keys, read ahead
        self.ahead = fetched.keys_seconds
        self.fetch_seconds = fetched.fetch_seconds

    def claim(self, db: Session) -> None:
        """Before the image's first query: read its footprint, register it in
        work-list order and wait for every earlier image it meets."""
        self.before_seed()  # held before it began: not even a query
        with self._timings.stage("claim"):
            since = self._run.claims.snapshot()
            shas, digests = {self._item.blob_sha256}, set()
            keys = self._fetched.keys
            if keys is not None:
                shas.add(keys.clean_sha256)
                digests = {keys.digest, keys.own_digest}
            try:
                footprint = seed_footprint(db, shas=shas, digests=digests)
            except BaseException:
                self._registered = True
                self._run.claims.skip(self._index)
                raise
            # the seed reads its gallery link by the product and the URL
            footprint.add(f"gallery {self._item.product_id} {self._item.url or ''}")
            self._claim, waits = self._run.claims.register(self._index, footprint, since)
            self._registered = True
        with self._timings.stage("turn_wait"):
            for earlier in waits:
                earlier.done.wait()

    def fetched(self) -> bytes:
        self._timings.add("fetch", self._fetched.fetch_seconds)
        if self._fetched.error is not None:
            raise self._fetched.error
        assert self._fetched.data is not None
        return self._fetched.data

    def before_seed(self) -> None:
        if self._run.hold.is_set():
            raise Held

    def turn(self) -> None:
        """Before the image's batch-row writes: once every image before it
        has ended; the row as the last of them left it."""
        if self._turned:
            return
        with self._timings.stage("turn_wait"):
            self._run.turns.take(self._index)
        self._turned = True
        self._run.row.lend(self._db, self._batch)

    def close(self) -> None:
        """The image ended (recorded, held or broken): whoever waits goes on."""
        if not self._registered:
            self._run.claims.skip(self._index)
        if self._turned:
            self._run.row.keep(self._batch)
        if self._claim is not None:
            self._run.claims.finish(self._claim)
        self._run.turns.end(self._index)


class _Pipeline:
    """A chunk's images, ``writers`` at a time (each on a worker thread with
    a session of its own, ``_mint`` as ever), ``prefetch`` originals read
    ahead, the model on the driver's thread alone. Records are taken in
    work-list order, each once its image's writes committed. A stop, a scraper
    window or the circuit HOLDS the chunk: no further image starts or gets
    onto the GPU; the images already past the GPU finish, the others are let
    go unrecorded (``Held``) and come back first for the next chunk. At most
    ``writers`` images are under way past the last one recorded, so an outage
    opens the circuit after at most ``writers`` - 1 images more than one at a
    time would have tried."""

    def __init__(
        self,
        rt: Runtime,
        config: BatchConfig,
        backend: MattingBackend,
        sql: DatabaseSeconds,
        sessions: sessionmaker[Session],
        interrupted: set[str],
    ) -> None:
        self.rt, self.config, self.backend, self.sql = rt, config, backend, sql
        self.sessions, self.interrupted = sessions, interrupted

    def run_chunk(
        self,
        items: list[WorkItem],
        batch: MaskBatch,
        chunk: _Chunk,
        take: Callable[[dict[str, Any], dict[str, float]], bool],
    ) -> list[WorkItem]:
        """Run ``items``; ``take`` each record (True: hold, the circuit is
        open). Returns the items not done: the held ones, then the unstarted."""
        writers, prefetch = self.config.writers, self.config.prefetch
        run = _ChunkRun(batch=batch, events=queue.Queue(maxsize=2 * writers))
        gpu = on_the_gpu(self.backend, run.submit)
        pool = ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="mask-read-ahead")
        ahead: dict[int, Future[_Fetched]] = {}
        workers: list[threading.Thread] = []
        done: dict[int, _Done] = {}
        held: list[int] = []
        failure: BaseException | None = None
        started = emitted = in_flight = 0

        def holding() -> bool:
            if not run.hold.is_set():
                if self.rt.stop_requested():
                    run.hold.set()
                elif self.rt.gates.busy() is not None:
                    chunk.yielded = True
                    run.hold.set()
            return run.hold.is_set()

        def read_ahead() -> None:
            for i in range(started, min(len(items), started + prefetch)):
                if i not in ahead:
                    ahead[i] = pool.submit(_read_ahead, items[i], self.rt)

        try:
            while True:
                # at most ``writers`` images past the last one recorded: one that
                # ends early (no row to write) never lets the chunk run further ahead
                while started - emitted < writers and started < len(items) and not holding():
                    read_ahead()
                    worker = threading.Thread(
                        target=self._work,
                        args=(run, gpu, started, items[started], ahead.pop(started)),
                        name=f"mask-image-{started}",
                    )
                    workers.append(worker)
                    worker.start()
                    started += 1
                    in_flight += 1
                if not run.hold.is_set():
                    read_ahead()
                if not in_flight:
                    break
                event = run.events.get()
                if isinstance(event, GpuJob):
                    if holding():
                        event.refuse()
                    else:
                        event.run()
                    continue
                in_flight -= 1
                done[event.index] = event
                while emitted in done:
                    ended = done.pop(emitted)
                    emitted += 1
                    if ended.error is not None:
                        failure = failure or ended.error
                        run.hold.set()
                    elif ended.record is None:
                        held.append(ended.index)
                    elif take(ended.record, ended.stages):
                        run.hold.set()
        finally:
            run.hold.set()
            while in_flight:  # the driver broke off: let every worker go
                event = run.events.get()
                if isinstance(event, GpuJob):
                    event.refuse()
                else:
                    in_flight -= 1
            pool.shutdown(wait=True, cancel_futures=True)
            for worker in workers:
                worker.join()
        if failure is not None:
            raise failure
        return [items[i] for i in held] + items[started:]

    def _work(
        self,
        run: _ChunkRun,
        gpu: MattingBackend,
        index: int,
        item: WorkItem,
        ahead: Future[_Fetched],
    ) -> None:
        db = self.sessions()
        timings = StageTimings(self.rt.monotonic)
        lane: _ImageLane | None = None
        record: dict[str, Any] | None = None
        error: BaseException | None = None
        try:
            fetched = ahead.result()
            batch = db.merge(run.batch, load=False)  # the row, without reading it
            lane = _ImageLane(run, index, item, fetched, db, batch, timings)
            redrive = self.config.retry_failed or item.blob_sha256 in self.interrupted
            with self.sql.charging(timings):
                record = _mint(db, item, self.rt, gpu, batch, redrive, timings, lane)
        except Held:
            with contextlib.suppress(Exception):
                db.rollback()  # its judgment's reads: its seed, if any, is committed
        except BaseException as exc:  # a fault of the batch's own: it stops
            error = exc
        finally:
            if lane is not None:
                lane.close()
            else:
                run.claims.skip(index)
                run.turns.end(index)
            with contextlib.suppress(Exception):
                db.close()
            run.events.put(_Done(index, record, timings.seconds, error))


def _pipeline_refusal(db: Session, writers: int) -> str | None:
    """Why the pipeline cannot run on ``db``'s pool, or None: every worker
    holds a connection of its own while its image runs, and the batch its own."""
    pool = db.get_bind().engine.pool
    needed = writers + 1
    if isinstance(pool, StaticPool | SingletonThreadPool):
        lends: float = 1
    elif isinstance(pool, QueuePool):
        overflow = pool._max_overflow
        lends = math.inf if overflow < 0 else pool.size() + overflow
    else:
        return None  # a connection per checkout
    if lends >= needed:
        return None
    return (
        f"--writers {writers} needs {needed} database connections at once (one per "
        f"worker, one for the batch), the pool lends {lends:g}: fewer writers, "
        "or --writers 0 for one image at a time"
    )


class _Tally:
    """The batch's running count across chunks: the outage streak and
    whether the model's own memory is measured since it was loaded."""

    def __init__(self, max_consecutive_failures: int) -> None:
        self.max = max_consecutive_failures
        self.streak = (
            0  # outage casualties in a row (an image's own fault neither counts nor resets)
        )
        self.measured = False

    def take(
        self,
        rt: Runtime,
        chunk: _Chunk,
        gpu: GpuGate | None,
        record: dict[str, Any],
        stages: dict[str, float],
    ) -> bool:
        """One image's record, in work-list order: True once the circuit opens."""
        record["chunk"] = chunk.number
        if gpu is not None and not self.measured and record["status"] in _RAN_MODEL:
            gpu.measure_own()
            self.measured = True
        rt.progress.append(record)
        chunk.records.append(record)
        chunk.stages.append(stages)
        if record["status"] == "interrupted":
            self.streak += 1
        elif record["status"] != "failed":
            self.streak = 0
        return self.streak >= self.max


def _one_at_a_time(
    items: list[WorkItem],
    rt: Runtime,
    config: BatchConfig,
    backend: MattingBackend,
    db: Session,
    batch: MaskBatch,
    sql: DatabaseSeconds,
    interrupted: set[str],
    chunk: _Chunk,
    take: Callable[[dict[str, Any], dict[str, float]], bool],
) -> list[WorkItem]:
    """``items`` one at a time on the driver's own session (``writers`` 0).
    Returns the items not done: a stop or a scraper window stops before an
    image, the circuit after one."""
    for n, item in enumerate(items):
        if rt.stop_requested():
            return items[n:]
        if rt.gates.busy() is not None:
            chunk.yielded = True
            return items[n:]
        # an outage's casualties are re-driven without --retry-failed
        redrive = config.retry_failed or item.blob_sha256 in interrupted
        timings = StageTimings(rt.monotonic)
        with sql.charging(timings):
            record = _mint(db, item, rt, backend, batch, redrive, timings)
        if take(record, timings.seconds):
            return items[n + 1 :]
    return []


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
                if rt.gates.gpu is not None:
                    rt.gates.gpu.forget_own()
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
    if config.chunk_size < 1:
        rt.emit({"event": "refused", "detail": f"chunk size {config.chunk_size} is below 1"})
        return 2
    items = sorted(items, key=lambda i: i.rank)
    recorded = rt.progress.load()
    done = {sha for sha, r in recorded.items() if r.get("status") not in _REDO}
    failed_before = {sha for sha, r in recorded.items() if r.get("status") == "failed"}
    interrupted = {sha for sha, r in recorded.items() if r.get("status") == "interrupted"}
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
            "interrupted_before": len(interrupted & {i.blob_sha256 for i in items}),
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
    sessions: sessionmaker[Session] | None = None
    if config.writers:
        refusal = _pipeline_refusal(db, config.writers)
        if refusal is not None:
            rt.emit({"event": "refused", "detail": refusal})
            return 2
        # each worker's session as the batch's own is, on its engine
        sessions = sessionmaker(
            bind=db.get_bind(), autoflush=db.autoflush, expire_on_commit=db.expire_on_commit
        )
    totals: dict[str, int] = {}
    tally = _Tally(config.max_consecutive_failures)
    remaining = deque(pending)
    chunk_no = 0
    loaded = False
    # each image's statements, commits and rollbacks as its "db"
    sql = DatabaseSeconds(db, sessions)
    pipeline = (
        None if sessions is None else _Pipeline(rt, config, backend, sql, sessions, interrupted)
    )
    try:
        while remaining:
            if _wait_for_gate(rt, config, backend if loaded else None):
                loaded = False
            gpu = rt.gates.gpu
            if not loaded:
                if gpu is not None:
                    gpu.mark_loading()
                    tally.measured = False
                try:
                    backend.load()
                except Exception as exc:
                    rt.emit({"event": "refused", "detail": str(exc)})
                    return 2
                loaded = True
            chunk_no += 1
            chunk = _Chunk(
                number=chunk_no,
                batch_id=None,
                started=rt.monotonic(),
                baseline_vram_mib=gpu.last.used_mib if gpu is not None and gpu.last else None,
                provider=backend.active_provider,
            )
            batch = open_mask_batch(db, backend_name=backend.name)
            chunk.batch_id = batch.id
            todo = [remaining.popleft() for _ in range(min(config.chunk_size, len(remaining)))]
            take = functools.partial(tally.take, rt, chunk, gpu)
            with rt.vram() as sampler:
                if pipeline is None:
                    left = _one_at_a_time(
                        todo, rt, config, backend, db, batch, sql, interrupted, chunk, take
                    )
                else:
                    left = pipeline.run_chunk(todo, batch, chunk, take)
            remaining.extendleft(reversed(left))
            finish_mask_batch(db, batch)
            circuit_open = tally.streak >= config.max_consecutive_failures
            report = chunk.report(sampler.peak_mib, rt.monotonic() - chunk.started)
            report["held_vram_mib"] = gpu.own_mib if gpu is not None else None
            rt.emit(report)
            rt.progress.append(report)
            for status, n in report["counts"].items():
                totals[status] = totals.get(status, 0) + n
            if circuit_open:
                rt.emit(
                    {
                        "event": "circuit_open",
                        "consecutive_failures": tally.streak,
                        "interrupted": tally.streak,
                    }
                )
                return 3
            if rt.stop_requested():
                raise _Stopped
    except _Stopped:
        rt.emit({"event": "stopped", "counts": totals, "pending": len(remaining)})
        return 130
    except _GaveUp as gave_up:
        rt.emit({"event": "gave_up", **gave_up.result.as_dict()})
        return 4
    finally:
        sql.close()
    rt.emit({"event": "done", "counts": totals, "pending": len(remaining)})
    return 1 if totals.get("failed") or totals.get("interrupted") else 0


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
        writers=args.writers,
        prefetch=args.prefetch,
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
        # each worker's original, copy and mask kept until it reads its copy back
        media_store = RecentWrites(bucket, keep=4 + 3 * args.writers)
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


def _whole(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be {minimum} or more, got {value}")
        if maximum is not None and value > maximum:
            raise argparse.ArgumentTypeError(f"must be {maximum} or less, got {value}")
        return value

    return parse


def _seconds(
    minimum: float, maximum: float = math.inf, *, above: bool = False
) -> Callable[[str], float]:
    """Finite seconds from ``minimum`` (exclusive with ``above``) to ``maximum``."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
        low = value > minimum if above else value >= minimum
        if not (math.isfinite(value) and low and value <= maximum):
            bound = f"above {minimum:g} s" if above else f"{minimum:g} s or more"
            if maximum < math.inf:
                bound += f" and at most {maximum:g} s"
            raise argparse.ArgumentTypeError(f"must be {bound}, got {text}")
        return value

    return parse


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.mask_batch",
        description=(__doc__ or "").split("\n")[0],
        allow_abbrev=False,
    )
    p.add_argument("--sandbox", help="SQLite + local objects in DIR instead of DATABASE_URL/S3")
    sub = p.add_subparsers(dest="command", required=True)

    # no abbreviations: the wrapper sizes the container by the literal --device
    run = sub.add_parser(
        "run", help="mint masks for a work list (gated, chunked, resumable)", allow_abbrev=False
    )
    run.add_argument("--worklist", action="append", required=True, type=Path)
    run.add_argument("--progress", default="mask-batch-progress.jsonl")
    run.add_argument("--chunk-size", type=_whole(1), default=20)
    run.add_argument("--limit", type=_whole(0))
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
    run.add_argument("--gpu-free-min-mib", type=_whole(1024), default=DEFAULT_GPU_FREE_MIN_MIB)
    run.add_argument("--gpu-samples", type=_whole(1), default=3)
    run.add_argument("--busy-windows", default=DEFAULT_BUSY_WINDOWS)
    run.add_argument("--residential-ip")
    run.add_argument("--no-residential-check", action="store_true")
    run.add_argument("--ip-echo-url", default=DEFAULT_IP_ECHO_URL)
    run.add_argument("--residential-ttl", type=_seconds(0.0, above=True), default=300.0)
    run.add_argument("--poll-seconds", type=_seconds(1.0, 3600.0), default=60.0)
    run.add_argument("--max-wait-seconds", type=_seconds(0.0))
    run.add_argument("--max-consecutive-failures", type=_whole(1), default=5)
    run.add_argument("--writers", type=_whole(0, MAX_WRITERS), default=DEFAULT_WRITERS)
    run.add_argument("--prefetch", type=_whole(1, MAX_PREFETCH), default=DEFAULT_PREFETCH)
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
