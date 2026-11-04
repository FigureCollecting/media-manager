"""MEDIADB (U8): each connection names its workload, and the seeder's and
the sync's lines carry their SQL's seconds, statements and rows.

``application_name`` is ``<service>/<job>/<run id>`` (Postgres keeps 63
bytes): pg_stat_activity, the server log and pg_stat_statements then tell
the seeder, the sync, the replay, the WSL batch and the API apart where the
role alone could not (the sync and the replay share media_report, the batch
and the API share media)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from app import db as configured_db
from app.cli import display_copy as cli
from app.object_store import LocalObjectStore
from app.telemetry import db as tdb
from app.workers import display_copy as dc
from app.workers.raw_source import ReadOnlyStore
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import Session

from spine_fakes import serve, sha
from test_display_copy import STORE, T0, _photo, _pid, _queue, _Raw
from test_display_copy_cli import _sandbox
from test_display_feed import QueueSpine, _sync
from test_display_feed import _event as feed_event


@pytest.fixture(autouse=True)
def _no_name() -> Any:
    before = tdb.connection_name()
    yield
    tdb._name = before  # noqa: SLF001 -- the process-wide name, as each test found it


# --------------------------------------------------------------------------
# application_name
# --------------------------------------------------------------------------


def test_the_name_is_service_job_and_run_and_fits_postgres() -> None:
    assert tdb.application_name("mm-display", "display-seed", "ab" * 16) == (
        "mm-display/display-seed/" + "ab" * 16
    )
    long = tdb.application_name("s" * 40, "j" * 40, "r" * 40)
    assert len(long) == tdb.APPLICATION_NAME_MAX == 63
    assert long.startswith("s" * 40 + "/")


def test_a_server_without_a_run_is_named_by_its_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tdb.socket, "gethostname", lambda: "media-origin-7d9f")
    assert tdb.application_name("media-origin", "http") == "media-origin/http/media-origin-7d9f"


def _connect_params(url: str, *, named: bool) -> dict[str, Any]:
    """The keyword arguments a postgres engine would hand the driver."""
    engine = create_engine(url)
    if named:
        tdb.name_engine(engine)
    params: dict[str, Any] = {}

    @event.listens_for(engine, "do_connect")
    def _capture(dialect: Any, conn_rec: Any, cargs: Any, cparams: dict[str, Any]) -> Any:
        params.update(cparams)
        raise RuntimeError("captured")  # no server: stop before the driver connects

    with pytest.raises(Exception, match="captured"):
        engine.connect()
    engine.dispose()
    return params


def test_every_connection_an_engine_opens_carries_the_processs_name() -> None:
    tdb.name_connections("mm-display", "display-sync", "cd" * 16)
    params = _connect_params("postgresql+psycopg2://u@127.0.0.1:1/media", named=True)
    assert params["application_name"] == "mm-display/display-sync/" + "cd" * 16
    # an engine never named, or before any name is set, passes none
    assert "application_name" not in _connect_params(
        "postgresql+psycopg2://u@127.0.0.1:1/media", named=False
    )
    tdb._name = None  # noqa: SLF001
    assert "application_name" not in _connect_params(
        "postgresql+psycopg2://u@127.0.0.1:1/media", named=True
    )


def test_a_url_that_names_its_connections_keeps_its_own_name() -> None:
    tdb.name_connections("mm-display", "display-sync")
    url = "postgresql+psycopg2://u@127.0.0.1:1/media?application_name=by-hand"
    assert _connect_params(url, named=True)["application_name"] == "by-hand"


def test_naming_an_engine_twice_listens_once() -> None:
    engine = create_engine("postgresql+psycopg2://u@127.0.0.1:1/media")
    tdb.name_engine(engine)
    tdb.name_engine(engine)
    assert len(engine.dialect.dispatch.do_connect) == 1
    engine.dispose()


def test_a_sqlite_engine_is_never_handed_a_name(tmp_path: Path) -> None:
    tdb.name_connections("mm-display", "display-seed")
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    tdb.name_engine(engine)
    with engine.connect() as conn:  # sqlite3.connect() would refuse the keyword
        assert conn.execute(text("SELECT 1")).scalar_one() == 1
    engine.dispose()


def test_the_configured_engine_is_named() -> None:
    assert tdb._named in configured_db.engine.dialect.dispatch.do_connect  # noqa: SLF001


def test_the_display_cli_names_its_connections_with_its_command_and_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox = _sandbox(tmp_path / "sb")
    assert cli.main(["status", "--sandbox", str(sandbox)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    run_id = next(x for x in lines if x["event"] == "job.start")["run_id"]
    assert tdb.connection_name() == f"mm-display/display-status/{run_id}"


# --------------------------------------------------------------------------
# the SQL's seconds, statements and rows
# --------------------------------------------------------------------------


def test_a_tally_counts_each_threads_statements_rows_and_seconds(tmp_path: Path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 't.db'}", connect_args={"check_same_thread": False}
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (n INTEGER)"))
    ticks = iter(float(n) for n in range(100))
    tally = tdb.DbTally(engine, clock=lambda: next(ticks))
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO t VALUES (1), (2), (3)"))
        conn.execute(text("SELECT n FROM t"))
    mine = tally.take()
    assert mine == {"db_seconds": 2.0, "db_statements": 2, "db_rows": 3}
    assert tally.take() == {"db_seconds": 0.0, "db_statements": 0, "db_rows": 0}

    other: list[dict[str, Any]] = []

    def elsewhere() -> None:
        with engine.begin() as conn:
            conn.execute(text("UPDATE t SET n = n + 1"))
        other.append(tally.take())

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()
    assert other == [{"db_seconds": 1.0, "db_statements": 1, "db_rows": 3}]
    assert tally.take() == {"db_seconds": 0.0, "db_statements": 0, "db_rows": 0}
    assert tally.totals() == {"db_seconds": 3.0, "db_statements": 3, "db_rows": 6}

    with pytest.raises(Exception), engine.begin() as conn:  # noqa: B017 -- any driver error
        conn.execute(text("SELECT nothing FROM nowhere"))
    assert tally.take()["db_statements"] == 1  # a statement that failed took its time too

    tally.close()
    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))
    assert tally.totals()["db_statements"] == 4
    engine.dispose()


def test_the_seeders_lines_carry_their_sql(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    origin, key = raw.put(_photo(1))
    _queue(db_session, origin, key, depictions=[(_pid(1), _pid(101), STORE.format(n=1), "amiami")])
    sha2 = hashlib.sha256(_photo(2)).hexdigest()  # queued, its raw object never stored
    _queue(
        db_session,
        sha2,
        f"raw-img/sha256/{sha2[:2]}/{sha2}.png",
        depictions=[(_pid(2), _pid(102), STORE.format(n=2), "amiami")],
        seen=T0 - dt.timedelta(minutes=1),
    )
    lines: list[dict[str, Any]] = []
    code = dc.run_seed(
        lambda: db_session,
        ReadOnlyStore(raw.store),
        LocalObjectStore(tmp_path / "media"),
        dc.SeedConfig(threads=0),
        emit=lines.append,
        now=lambda: T0,
    )
    assert code == dc.EXIT_DONE
    item = next(x for x in lines if x["event"] == "item")
    assert item["reason"] == "missing_raw"
    assert item["db_statements"] > 0 and item["db_seconds"] >= 0 and "db_rows" in item
    done = next(x for x in lines if x["event"] == "done")
    assert done["db_statements"] > item["db_statements"]  # the seeded one's too
    assert set(done) >= {"db_seconds", "db_statements", "db_rows"}


def test_the_syncs_lines_carry_their_sql(db_session: Session) -> None:
    fake = QueueSpine()
    fake.depict_full(sha("a"), seen=T0)
    with serve(fake) as url:
        _code, lines = _sync(db_session, url)
    reported, enqueued, done = (feed_event(lines, n) for n in ("reported", "enqueued", "done"))
    for line in (reported, enqueued, done):
        assert set(line) >= {"db_seconds", "db_statements", "db_rows"}
    assert enqueued["db_statements"] > 0
    # the done line's are the run's: both phases', the purge's and the lag's
    assert done["db_statements"] > reported["db_statements"] + enqueued["db_statements"]
    assert db_session.execute(select(text("1"))).scalar_one() == 1
