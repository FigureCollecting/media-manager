"""SHOW1: ``python -m app.cli.display_copy``, the lane's command line --
sync (the spine side), seed (the bucket side), status and requeue."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from app.cli import display_copy as cli
from app.models import Base, DisplayDepiction, DisplayOriginal, FigureGallery
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from spine_fakes import serve, sha
from test_display_feed import QueueSpine, _seeded

T0 = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)


def _png(seed: int) -> bytes:
    image = Image.new("RGB", (32, 48), (seed % 256, 40, 200))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def _sandbox(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{tmp_path / 'masks.db'}")
    Base.metadata.create_all(engine)
    engine.dispose()
    return tmp_path


def _db(sandbox: Path) -> Session:
    return Session(create_engine(f"sqlite:///{sandbox / 'masks.db'}"), expire_on_commit=False)


def _lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _event(lines: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(x for x in lines if x["event"] == event)


def _raw_object(raw_dir: Path, data: bytes) -> tuple[str, str]:
    origin = hashlib.sha256(data).hexdigest()
    key = f"raw-img/sha256/{origin[:2]}/{origin}.png"
    (raw_dir / key).parent.mkdir(parents=True, exist_ok=True)
    (raw_dir / key).write_bytes(data)
    return origin, key


def test_sync_then_seed_then_sync_reports_the_new_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _sandbox(tmp_path / "db")
    raw_dir, media_dir = tmp_path / "raw", tmp_path / "media"
    origin, key = _raw_object(raw_dir, _png(1))
    fake = QueueSpine()
    product = str(uuid.UUID(int=5))
    pid = fake.depict_full(
        origin, seen=T0, key=key, product=product, url="https://img.amiami.example/a.png"
    )
    with serve(fake) as url:
        monkeypatch.setenv("SPINE_MEDIA_URL", url)
        assert cli.main(["sync", "--sandbox", str(sandbox)]) == 0
        lease = fake.sent("LeasePendingImages")[0]
        assert lease.lease_owner.startswith("mm-display-sync")
        assert (
            cli.main(
                [
                    "seed",
                    "--sandbox",
                    str(sandbox),
                    "--raw-dir",
                    str(raw_dir),
                    "--media-dir",
                    str(media_dir),
                    "--threads",
                    "1",
                ]
            )
            == 0
        )
        assert cli.main(["sync", "--sandbox", str(sandbox), "--max-new", "0"]) == 0
    assert fake.depictions[pid]["state"] == "derived"
    lines = _lines(capsys.readouterr().out)
    ends = [x for x in lines if x["event"] == "job.end"]
    assert [e["code"] for e in ends] == ["ok", "ok", "ok"]
    assert _event(lines, "app.done")
    with _db(sandbox) as db:
        (link,) = db.execute(select(FigureGallery)).scalars()
        assert (link.figure_id, link.source_url) == (product, "https://img.amiami.example/a.png")
        assert db.get(DisplayDepiction, pid).outcome == "derived"


@dataclass
class _SlowReports(QueueSpine):
    """A spine whose ReportDerivative takes 0.4 s."""

    def ReportDerivative(self, request: Any, context: Any) -> Any:  # noqa: N802
        time.sleep(0.4)
        return super().ReportDerivative(request, context)


def test_a_sync_on_the_real_clock_stops_its_reports_at_their_share_and_still_enqueues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--budget-seconds 4 on time.monotonic: eight reports of 0.4 s would
    take 3.2 s, so the REPORT stops once its 2 s are spent (by the sixth row
    at the latest), and the enqueue, the purge and job.end follow in the
    2 s or so left."""
    sandbox = _sandbox(tmp_path / "db")
    fake = _SlowReports()
    with _db(sandbox) as db:
        for tag in "abcdefgh":
            _seeded(db, fake, tag)
    fake.depict_full(sha("new"), seen=T0)
    with serve(fake) as url:
        monkeypatch.setenv("SPINE_MEDIA_URL", url)
        assert cli.main(["sync", "--sandbox", str(sandbox), "--budget-seconds", "4"]) == 0
    lines = _lines(capsys.readouterr().out)
    reported = _event(lines, "app.reported")
    assert reported["partial"] == "budget" and 1 <= reported["derived"] <= 5
    assert reported["derived"] + reported["remaining"] == 8
    assert len(fake.sent("ReportDerivative")) == reported["derived"]
    assert _event(lines, "app.enqueued")["new_depictions"] == 1
    assert _event(lines, "app.done")
    end = _event(lines, "job.end")
    assert (end["exit"], end["partial"], end["remaining"]) == (0, "budget", reported["remaining"])


def test_status_prints_the_queue_and_requeue_is_a_dry_run_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox = _sandbox(tmp_path)
    with _db(sandbox) as db:
        for n, state in enumerate(("failed", "failed", "held")):
            origin = hashlib.sha256(bytes([n])).hexdigest()
            db.add(
                DisplayOriginal(
                    origin_sha256=origin,
                    storage_key=f"sha256/{origin[:2]}/{origin}",
                    codec="identity",
                    first_seen_at=T0,
                    state=state,
                    reason="undecodable"
                    if state == "failed"
                    else "would_withdraw_mask:not_granted",
                )
            )
        db.commit()
    assert cli.main(["status", "--sandbox", str(sandbox)]) == 0
    status = _event(_lines(capsys.readouterr().out), "app.status")
    assert json.loads(status["originals"]) == {"failed": 2, "held": 1}
    assert cli.main(["requeue", "--sandbox", str(sandbox), "--state", "failed"]) == 0
    line = _event(_lines(capsys.readouterr().out), "app.requeue")
    assert (line["matched"], line["applied"]) == (2, False)
    assert (
        cli.main(["requeue", "--sandbox", str(sandbox), "--state", "held", "--force", "--apply"])
        == 0
    )
    with _db(sandbox) as db:
        states = {(o.state, o.force_seed) for o in db.execute(select(DisplayOriginal)).scalars()}
    assert states == {("failed", False), ("pending", True)}


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (["sync"], "SPINE_MEDIA_URL"),
        (["sync", "--spine-url", "https://ingest-server.fc.svc:50063"], "https"),
        (["sync", "--spine-url", "http://h:1", "--lease-items", "201"], "lease_items"),
        (["sync", "--spine-url", "http://h:1", "--order", "sideways"], "order"),
        (["sync", "--spine-url", "http://h:1", "--scope", "all"], "SHOWABLE"),
        (["sync", "--spine-url", "http://h:1", "--lease-seconds", "3600"], "lease_seconds"),
        (["sync", "--spine-url", "http://h:1", "--max-attempts", "0"], "max_attempts"),
        (["seed", "--raw-dir", "/nonexistent-raw", "--threads", "-1"], "threads"),
        (["seed", "--media-dir", "/x"], "RAW_S3_"),
        (["requeue", "--state", "pending"], "requeue"),
        (["requeue", "--state", "failed", "--force"], "--force"),
    ],
)
def test_bad_arguments_are_refused_before_anything_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    why: str,
) -> None:
    sandbox = _sandbox(tmp_path)
    monkeypatch.delenv("SPINE_MEDIA_URL", raising=False)
    for name in ("RAW_S3_ENDPOINT_URL", "RAW_S3_BUCKET", "RAW_S3_ACCESS_KEY", "RAW_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert cli.main([*argv, "--sandbox", str(sandbox)]) == 2
    out = capsys.readouterr()
    end = _lines(out.out)[-1]
    assert (end["event"], end["code"]) == ("job.end", "error")
    assert why in str(end["err"])


def test_a_seed_that_finds_a_bad_credential_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.exceptions import ClientError

    sandbox = _sandbox(tmp_path / "db")
    origin = hashlib.sha256(b"x").hexdigest()
    with _db(sandbox) as db:
        db.add(
            DisplayOriginal(
                origin_sha256=origin,
                storage_key=f"sha256/{origin[:2]}/{origin}",
                codec="identity",
                first_seen_at=T0,
                state="pending",
            )
        )
        db.flush()
        db.add(
            DisplayDepiction(
                product_image_id=str(uuid.uuid4()),
                origin_sha256=origin,
                product_id=str(uuid.uuid4()),
                origin_url="https://x.example/a",
                source_site="amiami",
                first_seen_at=T0,
            )
        )
        db.commit()

    class Denied:
        def get(self, key: str) -> bytes:
            raise ClientError(
                {
                    "Error": {"Code": "InvalidAccessKeyId"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            )

    monkeypatch.setattr(cli, "_raw_store", lambda args, connections: Denied())
    code = cli.main(
        ["seed", "--sandbox", str(sandbox), "--media-dir", str(tmp_path / "m"), "--threads", "0"]
    )
    assert code == 2
    lines = _lines(capsys.readouterr().out)
    assert _event(lines, "app.stopped")["why"] == "configuration"
    with _db(sandbox) as db:
        row = db.get(DisplayOriginal, origin)
        assert (row.state, row.attempts) == ("pending", 0)


def test_an_unknown_command_or_option_is_refused() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["reap"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        cli.main(["sync", "--spine"])  # no abbreviations
    assert caught.value.code == 2


def test_without_a_sandbox_the_configured_database_is_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.db

    sandbox = _sandbox(tmp_path)
    engine = create_engine(f"sqlite:///{sandbox / 'masks.db'}")
    monkeypatch.setattr(app.db, "SessionLocal", lambda: Session(engine, expire_on_commit=False))
    assert cli.main(["status"]) == 0
    assert json.loads(_event(_lines(capsys.readouterr().out), "app.status")["originals"]) == {}
    engine.dispose()


def test_the_media_bucket_is_the_configured_one_without_media_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.s3
    from app.object_store import S3ObjectStore

    made: dict[str, Any] = {}

    def client(**kwargs: Any) -> object:
        made.update(kwargs)
        return object()

    monkeypatch.setattr(app.s3, "s3_client", client)
    args = cli._parser().parse_args(["seed"])
    store = cli._media_store(args, 12)
    assert isinstance(store, S3ObjectStore) and made["max_pool_connections"] == 12


def test_a_zero_deadline_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sandbox = _sandbox(tmp_path)
    argv = [
        "sync",
        "--spine-url",
        "http://h:1",
        "--deadline-seconds",
        "0",
        "--sandbox",
        str(sandbox),
    ]
    assert cli.main(argv) == 2
    assert "deadline" in str(_lines(capsys.readouterr().out)[-1]["err"])
