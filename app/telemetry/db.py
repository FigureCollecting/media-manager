"""MEDIADB (U8): which workload each database connection is, and what each
run's SQL costs.

APPLICATION_NAME. Every connection a named engine opens carries
``<service>/<job>/<run id>`` as its ``application_name`` (a server, with no
run, its host for the run id): pg_stat_activity, the server's log lines and
pg_stat_statements then tell the display seeder, the sync, the replay, the
WSL mask batch and the API apart, where the role alone could not (the sync
and the replay share media_report; the batch, the API and the worker share
media). One name per process, set when its run starts (``name_connections``);
a connection opened before then, or by an engine never named, passes none,
and a DATABASE_URL that names its connections itself keeps its own.
Postgres keeps 63 bytes of it.

DB TALLY. The statements a thread sent through an engine, the rows the
driver says they answered or wrote (``cursor.rowcount``; none when it does
not know, as for SQLite's SELECTs) and their seconds, cursor execute to its
answer (the commit's round trip is not a statement and is not counted). It
only reads the clock: no statement, result or error changes.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, event

#: NAMEDATALEN - 1: the server truncates a longer application_name.
APPLICATION_NAME_MAX = 63

_name: str | None = None


def application_name(service: str, job: str, run_id: str | None = None) -> str:
    """``<service>/<job>/<run id>``, the host when there is no run, cut to
    what Postgres keeps."""
    return f"{service}/{job}/{run_id or socket.gethostname()}"[:APPLICATION_NAME_MAX]


def name_connections(service: str, job: str, run_id: str | None = None) -> str:
    """Name every connection a named engine opens from now on; the name."""
    global _name
    _name = application_name(service, job, run_id)
    return _name


def connection_name() -> str | None:
    """The name connections get now (None: none set)."""
    return _name


def _named(_dialect: Any, _conn_rec: Any, _cargs: Any, cparams: dict[str, Any]) -> None:
    if _name is not None and _dialect.name == "postgresql":
        cparams.setdefault("application_name", _name)


def name_engine(engine: Engine) -> None:
    """Every connection ``engine`` opens carries the process's name (a
    Postgres engine's; any other is left as it is)."""
    if not event.contains(engine, "do_connect", _named):
        event.listen(engine, "do_connect", _named)


@dataclass
class _Counts:
    seconds: float = 0.0
    statements: int = 0
    rows: int = 0

    def add(self, seconds: float, rows: int) -> None:
        self.seconds += seconds
        self.statements += 1
        self.rows += rows

    def fields(self) -> dict[str, float | int]:
        return {
            "db_seconds": round(self.seconds, 3),
            "db_statements": self.statements,
            "db_rows": self.rows,
        }


class DbTally:
    """Each thread's statements through ``engine``, their rows and seconds,
    from construction to ``close``: ``take`` this thread's since its last
    take, ``totals`` every thread's."""

    def __init__(self, engine: Engine, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._engine = engine
        self._clock = clock
        self._local = threading.local()
        self._lock = threading.Lock()
        self._total = _Counts()
        self._hooks: list[tuple[str, Callable[..., None]]] = [
            ("before_cursor_execute", self._starting),
            ("after_cursor_execute", self._answered),
            ("handle_error", self._failed),
        ]
        for name, fn in self._hooks:
            event.listen(engine, name, fn)

    def _mine(self) -> _Counts:
        mine: _Counts | None = getattr(self._local, "counts", None)
        if mine is None:
            mine = self._local.counts = _Counts()
        return mine

    def _starting(self, *_: Any) -> None:
        self._local.started = self._clock()

    def _ended(self, rows: int) -> None:
        started: float | None = getattr(self._local, "started", None)
        if started is None:
            return
        self._local.started = None
        seconds = self._clock() - started
        self._mine().add(seconds, rows)
        with self._lock:
            self._total.add(seconds, rows)

    def _answered(self, _conn: Any, cursor: Any, *_: Any) -> None:
        rowcount = getattr(cursor, "rowcount", -1)
        self._ended(rowcount if isinstance(rowcount, int) and rowcount > 0 else 0)

    def _failed(self, *_: Any) -> None:
        self._ended(0)

    def take(self) -> dict[str, float | int]:
        """This thread's statements, rows and seconds since its last take."""
        mine = self._mine()
        self._local.counts = _Counts()
        return mine.fields()

    def totals(self) -> dict[str, float | int]:
        """Every thread's, since construction."""
        with self._lock:
            return self._total.fields()

    def close(self) -> None:
        """Stop listening: the engine as it was."""
        for name, fn in self._hooks:
            if event.contains(self._engine, name, fn):
                event.remove(self._engine, name, fn)
