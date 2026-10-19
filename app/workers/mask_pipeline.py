"""MASK-PERF-2: the parts of the pipelined mask batch (app.cli.mask_batch).

The batch runs a chunk's images on several workers, each with a database
session of its own, so one image's I/O (its original read, its seed, its
judgment, its uploads and rows) overlaps another's model run. Three rules
keep what it writes what one image at a time writes; a fourth bounds its
memory:

- RELATED IMAGES WAIT (``Claims``). Seeding and settling an image reads and
  writes the rows of its picture: its original, every capture of the
  picture and every clean copy of it (app.workers.mask_seed), and its
  gallery link. Each image claims those keys, in work-list order; an image
  whose keys meet an earlier image's still in flight waits for it to end
  before its first query. Images of different pictures never wait.
- THE BATCH ROW IN ORDER (``Turns``). Every image that settles writes the
  chunk's batch row (its counts and cursor) and may register a mask another
  image shares; those writes are made in work-list order, one image at a
  time, each after the image before it has ended.
- ONE GPU STREAM (``GpuJob``, ``on_the_gpu``). A worker never runs the
  model: it hands the call to the driver's thread, which runs one at a
  time and refuses any while the batch is held (a pause or a stop); the
  worker then lets its image go (``Held``), unrecorded, for a rerun.
- A FEW DECODED AT ONCE (``Slots``). An image's copy is decoded for the
  model only when a slot is free, and holds it until its mask is made: the
  memory of the images waiting for the GPU is bounded whatever the writers.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from .matting import MattingBackend
from .stage_timing import StageTimings

T = TypeVar("T")


class Held(BaseException):
    """An image the pipeline let go before the GPU: a pause or a stop came
    first. Nothing of it is recorded; a rerun takes it again (what it had
    committed, its seed, is idempotent). A BaseException, so no image's own
    ``except Exception`` takes it for that image's failure."""


@dataclass(eq=False)
class Claim:
    """One image's keys: its picture's rows, and those of every image it
    waits for (so an image after it waits for those too)."""

    index: int
    keys: set[str]
    #: the order it ended in (``Claims.finish``); None while in flight
    finished: int | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Claims:
    """The images of one chunk, registered in work-list order, each with the
    earlier images it must wait for: those still in flight whose keys meet
    its own."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next = 0  # the index registered next
        self._ended = 0  # images ended so far
        self._claims: list[Claim] = []

    def snapshot(self) -> int:
        """Now, before an image reads its keys from the database: an image
        that ends after this may be half visible to that read."""
        with self._cond:
            return self._ended

    def register(self, index: int, keys: set[str], since: int) -> tuple[Claim, list[Claim]]:
        """``index``'s claim (once every earlier index has registered) and the
        claims it waits for. It joins every earlier claim its keys meet --
        still in flight, or ended after ``since`` (its writes may be half
        visible to the read the keys came from) -- and takes their keys, so
        whatever they joined is joined too."""
        with self._cond:
            self._cond.wait_for(lambda: self._next == index)
            grown = set(keys)
            joined: list[Claim] = []
            more = True
            while more:
                more = False
                for claim in self._claims:
                    live = claim.finished is None or claim.finished > since
                    if claim not in joined and live and claim.keys & grown:
                        joined.append(claim)
                        grown |= claim.keys
                        more = True
            mine = Claim(index, grown)
            self._claims.append(mine)
            self._next += 1
            self._cond.notify_all()
            return mine, [c for c in joined if c.finished is None]

    def skip(self, index: int) -> None:
        """``index`` claims nothing (it never got as far as its first query)."""
        with self._cond:
            self._cond.wait_for(lambda: self._next == index)
            self._next += 1
            self._cond.notify_all()

    def finish(self, claim: Claim) -> None:
        """The image ended: whoever waits for it goes on."""
        with self._cond:
            self._ended += 1
            claim.finished = self._ended
            claim.done.set()


class Turns:
    """The batch-row writes of one chunk, in work-list order: ``take(i)``
    returns once every image before ``i`` has ended."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next = 0  # every index below it has ended
        self._ended: set[int] = set()

    def take(self, index: int) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._next == index)

    def end(self, index: int) -> None:
        """``index`` ended (written or not): once each one before it has too,
        the next image's turn comes."""
        with self._cond:
            self._ended.add(index)
            while self._next in self._ended:
                self._ended.discard(self._next)
                self._next += 1
            self._cond.notify_all()


class Slots:
    """The images whose copies are decoded for the model at once (the
    memory bound): an image holds a slot from its copy's decode, through its
    wait for the GPU, to its mask. A slot goes to the earliest image waiting,
    so no later image keeps the one the chunk records next waiting. A holder
    waits for nothing but the GPU, so every slot comes back."""

    def __init__(self, size: int) -> None:
        self._cond = threading.Condition()
        self._free = size
        self._waiting: list[int] = []  # a heap of work-list indexes

    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)

    def acquire(self, index: int) -> None:
        with self._cond:
            heapq.heappush(self._waiting, index)
            self._cond.wait_for(lambda: self._free > 0 and self._waiting[0] == index)
            heapq.heappop(self._waiting)
            self._free -= 1
            self._cond.notify_all()  # another slot may be free for the next one

    def release(self) -> None:
        with self._cond:
            self._free += 1
            self._cond.notify_all()


class GpuJob:
    """One image's model call, handed by its worker to the thread that owns
    the GPU, which runs it (``run``) or refuses it (``refuse``)."""

    def __init__(self, call: Callable[[], Any]) -> None:
        self._call = call
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None

    def run(self) -> None:
        try:
            self._result = self._call()
        except Exception as exc:  # the image's own failure: its worker records it
            self._error = exc
        except BaseException:  # the driver itself is going down: let the worker go
            self._error = Held()
            raise
        finally:
            self._done.set()

    def refuse(self) -> None:
        self._error = Held()
        self._done.set()

    def wait(self) -> Any:
        self._done.wait()
        if self._error is not None:
            raise self._error
        return self._result


class _OnTheGpu(MattingBackend):
    """A backend whose every model call runs on the GPU's thread (``submit``),
    the worker waiting as "gpu_wait" meanwhile."""

    def __init__(self, backend: MattingBackend, submit: Callable[[Callable[[], Any]], Any]) -> None:
        self._backend = backend
        self._submit = submit

    @property
    def active_provider(self) -> str | None:
        return self._backend.active_provider

    def matte(self, image: Image.Image) -> Image.Image:
        result: Image.Image = self._submit(lambda: self._backend.matte(image))
        return result

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        alpha: np.ndarray = self._submit(lambda: self._backend.predict_alpha(image))
        return alpha

    def predict_alpha_timed(self, image: Image.Image, timings: StageTimings) -> np.ndarray:
        # the model's own stages (inference, base_keep) open inside, on the GPU's thread
        with timings.stage("gpu_wait"):
            alpha: np.ndarray = self._submit(
                lambda: self._backend.predict_alpha_timed(image, timings)
            )
        return alpha


def on_the_gpu(
    backend: MattingBackend, submit: Callable[[Callable[[], Any]], Any]
) -> MattingBackend:
    """``backend`` as a worker sees it: the same name (recorded beside each
    mask) and the same answers, each computed on the GPU's thread."""
    named = type(
        f"{type(backend).__name__}OnTheGpu",
        (_OnTheGpu,),
        {"name": backend.name, "publishes_masks": backend.publishes_masks},
    )
    proxy: MattingBackend = named(backend, submit)
    return proxy
