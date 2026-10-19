"""MASK-PERF-1: one image's seconds by stage, on an injected clock.

The mask batch (app.cli.mask_batch) times each image's stages and reports
their means and medians per chunk; these are the timer, the model's passes
and the mask artifact under it. Timing never changes an answer: the timed
and the untimed path give the same bytes.
"""

from __future__ import annotations

import dataclasses
import io
import sqlite3
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from app.workers.basekept import BaseKeptMattingBackend
from app.workers.mask import build_mask_artifact
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.stage_timing import DatabaseSeconds, StageTimings, timed
from PIL import Image
from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool


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


def test_seconds_two_events_bound_are_carved_out_of_the_stage_open_at_both() -> None:
    clock = _Clock()
    timings = StageTimings(clock)
    with timings.stage("seed"):
        clock.now += 1.0
        mark = timings.mark()
        clock.now += 0.5
        timings.since("db", mark)
        clock.now += 0.25
    top = timings.mark()
    clock.now += 2.0
    timings.since("db", top)
    assert timings.seconds == {"seed": 1.25, "db": 2.5}
    assert sum(timings.seconds.values()) == 3.75  # each second once


def test_seconds_across_a_stage_boundary_are_left_where_they_fell() -> None:
    """A mark whose stage has ended, or before a stage that began since,
    times nothing: those seconds are already some stage's."""
    clock = _Clock()
    timings = StageTimings(clock)
    before = timings.mark()
    with timings.stage("seed"):
        clock.now += 1.0
        inside = timings.mark()
    clock.now += 0.5
    timings.since("db", before)  # a stage began and ended since
    timings.since("db", inside)  # its stage has ended
    with timings.stage("fetch"):
        clock.now += 0.25
        timings.since("db", inside)  # as deep, but another stage
    assert timings.seconds == {"seed": 1.0, "fetch": 0.25}


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


# --------------------------------------------------------------------------
# the database's seconds, by SQLAlchemy's events
# --------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class _Row(_Base):
    __tablename__ = "rows"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]


@dataclasses.dataclass
class _Database:
    clock: _Clock
    engine: Engine
    session: Session


@pytest.fixture()
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Database]:
    """A database whose round trips cost fake seconds: each statement
    0.0625 s and each COMMIT 0.5 s, by listeners added before any timing's;
    each DBAPI rollback of a transaction 0.25 s. As on Postgres, any
    statement opens the transaction, and a rollback with none open (the
    pool's reset of a connection it gets back) is no round trip."""
    clock = _Clock()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    _Base.metadata.create_all(engine)
    in_transaction = [False]

    def statement(*a: Any) -> None:
        clock.now += 0.0625
        in_transaction[0] = True

    def commit(conn: Any) -> None:
        clock.now += 0.5
        in_transaction[0] = False

    event.listen(engine, "before_cursor_execute", statement)
    event.listen(engine, "commit", commit)
    real_rollback = engine.dialect.do_rollback

    def rollback(dbapi_connection: Any) -> None:
        if in_transaction[0]:
            clock.now += 0.25
        in_transaction[0] = False
        real_rollback(dbapi_connection)

    monkeypatch.setattr(engine.dialect, "do_rollback", rollback)
    session = Session(engine, expire_on_commit=False)
    try:
        yield _Database(clock, engine, session)
    finally:
        session.close()
        engine.dispose()


def test_every_statement_commit_and_rollback_is_a_database_second(database: _Database) -> None:
    """Carved out of the stage that ran them: a commit with its flush
    counted once, from the session's commit to the transaction's end."""
    clock, db = database.clock, database.session
    sql = DatabaseSeconds(db)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            clock.now += 1.0  # the seed's own work
            db.execute(select(_Row)).all()
            db.add(_Row(name="a"))
            db.commit()  # its INSERT and the COMMIT
            db.execute(select(_Row)).all()
            db.rollback()
    finally:
        sql.close()
    assert timings.seconds == {"seed": 1.0, "db": 0.0625 + (0.0625 + 0.5) + (0.0625 + 0.25)}


def test_what_fails_is_a_database_second_up_to_its_failure(database: _Database) -> None:
    """A statement the database refuses, and a commit whose INSERT it
    refuses (the flush rolls back): each counted to its end, the handling
    after it the stage's, and the very same error raised."""
    clock, db = database.clock, database.session
    db.add(_Row(id=1, name="a"))
    db.commit()
    with pytest.raises(OperationalError) as plain:
        db.execute(text("SELECT 1 FROM nowhere"))
    db.rollback()
    sql = DatabaseSeconds(db)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            with pytest.raises(OperationalError) as refused:
                db.execute(text("SELECT 1 FROM nowhere"))
            clock.now += 1.0  # the failure handled
            db.rollback()
            db.add(_Row(id=1, name="again"))
            with pytest.raises(IntegrityError):
                db.commit()
            clock.now += 1.0
            db.rollback()  # the flush rolled back already: no round trip
    finally:
        sql.close()
    assert str(refused.value) == str(plain.value)
    assert timings.seconds == {"seed": 2.0, "db": (0.0625 + 0.25) + (0.0625 + 0.25)}


def test_a_commit_the_database_refuses_is_a_database_second(
    database: _Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, db = database.clock, database.session

    def refused(dbapi_connection: Any) -> None:
        clock.now += 0.75
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(database.engine.dialect, "do_commit", refused)
    sql = DatabaseSeconds(db)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            db.add(_Row(name="a"))
            with pytest.raises(OperationalError, match="disk I/O error"):
                db.commit()  # its INSERT, the commit listener's 0.5, the refusal
            clock.now += 1.0
    finally:
        sql.close()
    assert timings.seconds == {"seed": 1.0, "db": 0.0625 + 0.5 + 0.75}


def test_each_cursor_run_of_a_flush_and_the_rollback_of_a_close_are_database_seconds(
    database: _Database,
) -> None:
    clock, db = database.clock, database.session
    sql = DatabaseSeconds(db)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            db.add_all([_Row(name="a"), _Row(name="b")])
            db.flush()  # one INSERT statement, a cursor run per row
            clock.now += 1.0
            db.close()  # its transaction rolled back
    finally:
        sql.close()
    assert timings.seconds == {"seed": 1.0, "db": 2 * 0.0625 + 0.25}


def test_a_statement_straight_on_the_driver_is_timed_from_its_cursor_run(
    database: _Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exec_driver_sql has no before_execute: the cursor run bounds it, so
    the round trip is the database's and a listener added earlier, run
    before it, stays the stage's."""
    clock, db = database.clock, database.session
    real_execute = database.engine.dialect.do_execute

    def execute(*a: Any) -> None:
        clock.now += 0.125
        real_execute(*a)

    monkeypatch.setattr(database.engine.dialect, "do_execute", execute)
    sql = DatabaseSeconds(db)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            assert db.connection().exec_driver_sql("SELECT 7").scalar() == 7
    finally:
        sql.close()
    assert timings.seconds == {"seed": 0.0625, "db": 0.125}


def test_a_statement_another_listener_stopped_leaves_no_seconds_behind(
    database: _Database,
) -> None:
    """A listener added after the timing's may refuse a statement before it
    runs (no result, no error event): that statement times nothing, and the
    commit after it is counted once."""
    clock, engine, db = database.clock, database.engine, database.session
    sql = DatabaseSeconds(db)

    def refuse(*a: Any) -> None:
        raise RuntimeError("refused before it ran")

    event.listen(engine, "before_execute", refuse)
    timings = StageTimings(clock)
    try:
        with sql.charging(timings), timings.stage("seed"):
            with pytest.raises(RuntimeError, match="refused before it ran"):
                db.execute(select(_Row))
            event.remove(engine, "before_execute", refuse)
            clock.now += 1.0
            db.add(_Row(name="a"))
            db.commit()
    finally:
        sql.close()
    assert timings.seconds == {"seed": 1.0, "db": 0.0625 + 0.5}


def test_only_the_image_being_timed_is_charged_and_close_leaves_no_listener(
    database: _Database,
) -> None:
    clock, engine, db = database.clock, database.engine, database.session

    def listeners() -> list[int]:
        return [
            *(len(getattr(engine.dispatch, n)) for n in ("before_execute", "after_execute")),
            *(len(getattr(engine.dispatch, n)) for n in ("before_cursor_execute", "commit")),
            *(len(getattr(engine.dispatch, n)) for n in ("after_cursor_execute", "rollback")),
            len(engine.dialect.dispatch.handle_error),
            *(len(getattr(db.dispatch, n)) for n in ("before_commit", "after_rollback")),
            len(db.dispatch.after_transaction_end),
        ]

    before = listeners()
    sql = DatabaseSeconds(db)
    assert listeners() != before
    timings = StageTimings(clock)
    db.execute(select(_Row)).all()  # before the image
    with sql.charging(timings):
        db.execute(select(_Row)).all()  # no stage open: the database's alone
    db.add(_Row(name="a"))
    db.commit()  # after it
    sql.close()
    assert listeners() == before
    assert timings.seconds == {"db": 0.0625}
