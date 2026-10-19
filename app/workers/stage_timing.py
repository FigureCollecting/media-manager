"""One image's seconds by stage (MASK-PERF-1): where the mask batch's time goes.

The mask batch (app.cli.mask_batch) times every image's stages with its own
injectable clock and reports their means and medians per chunk. A stage is a
context manager around the code it times: it reads the clock on the way in
and out and lets every exception through untouched, so timing never changes
an answer, an order or a transaction.

A stage inside another counts only in the inner one (the outer keeps its own
seconds), so the stages of an image add up to the time spent in them, each
second once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

#: The stages, in the order a chunk reports them: the pipeline's order, the
#: database (reached all along the way) last.
STAGES = (
    "fetch",  # the original from the raw bucket, and its decode
    "seed",  # the clean copy minted and registered (app.workers.mask_seed)
    "readback",  # the lane re-reading the copy it hashes
    "inference",  # the mask model's answer (BiRefNet)
    "base_keep",  # the base-keep pass: the IS-Net donors and the merge
    "analysis",  # the copy's decode, grounding and appearances
    "encode",  # the mask PNG
    "upload",  # the mask's PUT
    "db",  # every other query and commit
)


class StageTimings:
    """The seconds of one image, per stage, by ``clock``."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._inner: list[float] = []  # per open stage: the seconds of the stages inside it
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        self._inner.append(0.0)
        try:
            yield
        finally:
            took = self._clock() - start
            own = took - self._inner.pop()
            self.seconds[name] = self.seconds.get(name, 0.0) + own
            if self._inner:
                self._inner[-1] += took


def timed(timings: StageTimings | None, name: str) -> AbstractContextManager[None]:
    """``timings.stage(name)``, or nothing at all when nobody is timing."""
    return nullcontext() if timings is None else timings.stage(name)
