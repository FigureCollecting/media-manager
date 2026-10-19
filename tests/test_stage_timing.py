"""MASK-PERF-1: one image's seconds by stage, on an injected clock.

The mask batch (app.cli.mask_batch) times each image's stages and reports
their means and medians per chunk; these are the timer, the model's passes
and the mask artifact under it. Timing never changes an answer: the timed
and the untimed path give the same bytes.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from app.workers.basekept import BaseKeptMattingBackend
from app.workers.mask import build_mask_artifact
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.stage_timing import StageTimings, timed
from PIL import Image


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_a_stage_adds_up_every_time_it_runs() -> None:
    clock = _Clock()
    timings = StageTimings(clock)
    for cost in (1.5, 2.0):
        with timings.stage("db"):
            clock.now += cost
    with timings.stage("fetch"):
        clock.now += 0.25
    assert timings.seconds == {"db": 3.5, "fetch": 0.25}


def test_a_stage_inside_another_is_counted_once_in_the_inner_one() -> None:
    clock = _Clock()
    timings = StageTimings(clock)
    with timings.stage("analysis"):
        clock.now += 1.0
        with timings.stage("inference"):
            clock.now += 4.0
            with timings.stage("encode"):
                clock.now += 0.5
        clock.now += 0.25
    assert timings.seconds == {"encode": 0.5, "inference": 4.0, "analysis": 1.25}
    assert sum(timings.seconds.values()) == 5.75  # the whole outer stage, once


def test_a_stage_that_raises_keeps_its_seconds_and_the_very_exception() -> None:
    clock = _Clock()
    timings = StageTimings(clock)
    boom = TimeoutError("read timed out")
    with pytest.raises(TimeoutError) as caught, timings.stage("fetch"):
        clock.now += 4.0
        raise boom
    assert caught.value is boom
    assert timings.seconds == {"fetch": 4.0}
    with timings.stage("seed"):  # nothing left open by the failure
        clock.now += 1.0
    assert timings.seconds == {"fetch": 4.0, "seed": 1.0}


def test_without_timings_a_stage_is_nothing() -> None:
    with timed(None, "db"):
        pass
    clock = _Clock()
    timings = StageTimings(clock)
    with timed(timings, "db"):
        clock.now += 0.5
    assert timings.seconds == {"db": 0.5}


class _Fixed(MattingBackend):
    """A fixed answer that costs fake seconds and logs when it ran."""

    def __init__(self, name: str, alpha: np.ndarray, clock: _Clock, cost: float, log: list[str]):
        self.name = name  # type: ignore[misc]
        self.alpha = alpha
        self.clock = clock
        self.cost = cost
        self.log = log

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        self.clock.now += self.cost
        self.log.append(self.name)
        return self.alpha.copy()

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError


H, W = 200, 100


def _body() -> np.ndarray:
    a = np.zeros((H, W), dtype=np.uint8)
    a[20:160, 40:60] = 255
    return a


def _body_and_base() -> np.ndarray:
    a = _body()
    a[160:180, 15:85] = 230
    return a


def test_a_single_model_times_its_whole_answer_as_inference() -> None:
    clock = _Clock()
    log: list[str] = []
    model = _Fixed("birefnet", _body(), clock, 0.75, log)
    timings = StageTimings(clock)
    image = Image.new("RGBA", (W, H), (9, 9, 9, 255))
    assert np.array_equal(model.predict_alpha_timed(image, timings), model.predict_alpha(image))
    assert timings.seconds == {"inference": 0.75}


def test_the_base_keep_pass_is_its_own_stage_and_changes_no_answer() -> None:
    clock = _Clock()
    log: list[str] = []
    backend = BaseKeptMattingBackend(
        primary=_Fixed("birefnet", _body(), clock, 2.0, log),
        donors=[
            _Fixed("isnet-general", _body_and_base(), clock, 0.5, log),
            _Fixed("isnet-anime", _body(), clock, 0.25, log),
        ],
    )
    image = Image.new("RGB", (W, H), (9, 9, 9))
    plain = backend.predict_alpha(image)
    plain_order = list(log)
    log.clear()
    timings = StageTimings(clock)
    timed_alpha = backend.predict_alpha_timed(image, timings)
    assert np.array_equal(timed_alpha, plain)
    assert plain[170, 50] > 200  # the base restored: the pass really ran
    assert log == plain_order == ["isnet-general", "isnet-anime", "birefnet"]
    assert timings.seconds == {"base_keep": 0.75, "inference": 2.0}


def _photo() -> bytes:
    arr = np.full((90, 60, 3), (250, 250, 248), dtype=np.uint8)
    arr[15:75, 18:42] = (40, 30, 90)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_the_mask_artifact_is_the_same_timed_or_not() -> None:
    clean = _photo()
    timings = StageTimings(_Clock())
    timed_outcome = build_mask_artifact(clean, backend=StubMattingBackend(), timings=timings)
    assert timed_outcome == build_mask_artifact(clean, backend=StubMattingBackend())
    assert timed_outcome.status == "ok" and timed_outcome.mask_png
    assert set(timings.seconds) == {"inference", "encode"}


def test_an_empty_mask_is_never_encoded() -> None:
    class _Nothing(StubMattingBackend):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            return np.zeros((image.height, image.width), dtype=np.uint8)

    timings = StageTimings(_Clock())
    outcome = build_mask_artifact(_photo(), backend=_Nothing(), timings=timings)
    assert outcome.status == "empty"
    assert set(timings.seconds) == {"inference"}
