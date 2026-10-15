"""MM-REPORT-1: ``python -m app.cli.mask_report``, the replay's command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.cli import mask_report as cli
from app.models import Base, SpineReportRun
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from spine_fakes import FakeSpine, seed_copy, serve


def _sandbox(tmp_path: Path, *tags: str) -> tuple[str, dict[str, dict[str, object]]]:
    engine = create_engine(f"sqlite:///{tmp_path / 'masks.db'}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        seeded = {tag: seed_copy(db, tag) for tag in tags}
    engine.dispose()
    return str(tmp_path), seeded


def _lines(out: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_a_dry_run_needs_no_spine_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox, seeded = _sandbox(tmp_path, "a")
    monkeypatch.delenv("SPINE_MEDIA_URL", raising=False)
    assert cli.main(["--sandbox", sandbox, "--dry-run"]) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[0]["event"] == "plan" and lines[0]["dry_run"] is True
    lease = next(x for x in lines if x.get("rpc") == "LeasePendingImages")
    assert lease["origin_sha256"] == [seeded["a"]["origin"]]
    engine = create_engine(f"sqlite:///{tmp_path / 'masks.db'}")
    with Session(engine) as db:
        assert db.execute(select(SpineReportRun)).first() is None
    engine.dispose()


def test_a_run_reports_to_the_spine_named_by_the_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox, seeded = _sandbox(tmp_path, "a")
    fake = FakeSpine()
    pid = fake.depict(str(seeded["a"]["origin"]))
    with serve(fake) as url:
        monkeypatch.setenv("SPINE_MEDIA_URL", url)
        code = cli.main(["--sandbox", sandbox, "--batch-size", "10", "--lease-seconds", "30"])
    assert code == 0
    assert fake.depictions[pid]["state"] == "derived"
    lease = fake.sent("LeasePendingImages")[0]
    assert (lease.max_items, lease.lease_seconds) == (200, 30)
    assert lease.lease_owner.startswith("mm-report run 1 on ")
    done = _lines(capsys.readouterr().out)[-1]
    assert done["event"] == "done" and done["exit"] == 0


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        ([], "SPINE_MEDIA_URL"),
        (["--spine-url", "https://ingest-server.fc.svc:50063"], "https"),
        (["--spine-url", "http://h:1", "--batch-size", "201"], "batch_size"),
        (["--spine-url", "http://h:1", "--lease-seconds", "0"], "lease_seconds"),
        (["--spine-url", "http://h:1", "--max-attempts", "0"], "max_attempts"),
        (["--spine-url", "http://h:1", "--deadline-seconds", "0"], "deadline"),
    ],
)
def test_bad_arguments_are_refused_before_anything_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    why: str,
) -> None:
    sandbox, _ = _sandbox(tmp_path)
    monkeypatch.delenv("SPINE_MEDIA_URL", raising=False)
    assert cli.main(["--sandbox", sandbox, *argv]) == 2
    assert why in capsys.readouterr().err
    engine = create_engine(f"sqlite:///{tmp_path / 'masks.db'}")
    with Session(engine) as db:
        assert db.execute(select(SpineReportRun)).first() is None
    engine.dispose()


def test_an_unknown_option_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--dry"])  # no abbreviations
    assert caught.value.code == 2
