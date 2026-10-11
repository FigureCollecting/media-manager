"""``python -m app.cli.masks``: seed a sample, run the batch (dry run unless
--apply), report a batch. ``--sandbox DIR`` keeps everything in DIR (a SQLite
file and a local object directory), so a sample run needs no services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.cli import masks as cli
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "mask_sample"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else {})


@pytest.fixture()
def sandbox(tmp_path: Path) -> str:
    return str(tmp_path / "sandbox")


def _rows(sandbox: str, model: type) -> list:  # type: ignore[type-arg]
    engine = create_engine(f"sqlite:///{sandbox}/masks.db")
    with Session(engine) as db:
        return list(db.execute(select(model)).scalars())


class TestSampleRun:
    def test_seed_then_dry_run_then_apply_then_nothing_left(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from app.models import ImageMask, PublicObject

        code, seeded = _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        assert code == 0
        assert seeded == {"seeded": 9}

        code, dry = _run(capsys, "--sandbox", sandbox, "run")
        assert code == 0
        assert dry["dry_run"] is True
        assert dry["batch_id"] is None
        assert dry["counts"]["ineligible"] == 1
        assert sum(dry["counts"].values()) == 9
        assert _rows(sandbox, ImageMask) == []

        by_figure = {i["figure_ids"][0]: i for i in dry["items"]}
        assert by_figure["sample:8"]["status"] == "ineligible"
        assert by_figure["sample:8"]["reason"] == "user_photo"
        ranges = dry["grounding_ranges"]
        assert set(ranges) >= {"bottom_margin_frac", "top_margin_frac", "contact_band_width_frac"}
        lo, hi = ranges["bottom_margin_frac"]
        assert 0 <= lo <= hi < 1

        code, applied = _run(capsys, "--sandbox", sandbox, "run", "--apply")
        assert code == 0
        assert applied["dry_run"] is False
        assert applied["counts"] == dry["counts"]
        masks = [p for p in _rows(sandbox, PublicObject) if p.kind == "mask"]
        assert len(masks) == applied["counts"]["ok"] > 0

        code, again = _run(capsys, "--sandbox", sandbox, "run", "--apply")
        assert (code, again["items"]) == (0, [])

        code, report = _run(
            capsys, "--sandbox", sandbox, "report", "--batch", str(applied["batch_id"])
        )
        assert code == 0
        assert report["status"] == "done"
        assert report["counts"] == applied["counts"]

    def test_the_full_report_can_be_written_to_a_file(
        self, sandbox: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        target = tmp_path / "report.json"
        code, summary = _run(
            capsys, "--sandbox", sandbox, "run", "--limit", "2", "--report", str(target)
        )
        assert code == 0
        assert len(summary["items"]) == 2
        assert json.loads(target.read_text()) == summary


class TestRefusals:
    def test_resuming_an_unknown_batch_is_an_error(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(["--sandbox", sandbox, "run", "--apply", "--resume", "999"])
        assert code == 2
        assert "not resumable" in capsys.readouterr().err

    def test_birefnet_without_weights_is_refused_before_any_work(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MATTING_MODEL_PATH", raising=False)
        code = cli.main(["--sandbox", sandbox, "run", "--backend", "birefnet"])
        assert code == 2
        assert "--model" in capsys.readouterr().err
        assert not Path(sandbox).exists()

    def test_report_of_an_unknown_batch(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--sandbox", sandbox, "report", "--batch", "7"]) == 2
        assert "no mask batch 7" in capsys.readouterr().err

    def test_a_subcommand_is_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cli.main([])


class TestBackends:
    def test_birefnet_with_weights_is_built_lazily(self, tmp_path: Path) -> None:
        from app.workers.matting import BiRefNetMattingBackend

        backend = cli.make_backend("birefnet", str(tmp_path / "w.onnx"), None, None)
        assert isinstance(backend, BiRefNetMattingBackend)

    def test_base_keep_needs_all_three_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.workers.basekept import BaseKeptMattingBackend

        for var in ("MATTING_MODEL_PATH", "MATTING_ISNET_GENERAL_PATH", "MATTING_ISNET_ANIME_PATH"):
            monkeypatch.delenv(var, raising=False)
        backend = cli.make_backend("birefnet-basekept", "b.onnx", "g.onnx", "a.onnx")
        assert isinstance(backend, BaseKeptMattingBackend)
        assert [d._model_path for d in backend.donors] == ["g.onnx", "a.onnx"]  # type: ignore[attr-defined]
        with pytest.raises(cli.Refused, match="--isnet-anime"):
            cli.make_backend("birefnet-basekept", "b.onnx", "g.onnx", None)
        with pytest.raises(cli.Refused, match="--isnet-general"):
            cli.make_backend("birefnet-basekept", "b.onnx", None, "a.onnx")
        with pytest.raises(cli.Refused, match="--model"):
            cli.make_backend("birefnet-basekept", None, "g.onnx", "a.onnx")

    def test_base_keep_weights_can_come_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MATTING_MODEL_PATH", "b.onnx")
        monkeypatch.setenv("MATTING_ISNET_GENERAL_PATH", "g.onnx")
        monkeypatch.setenv("MATTING_ISNET_ANIME_PATH", "a.onnx")
        assert cli.make_backend("birefnet-basekept", None, None, None) is not None

    def test_stub_and_default(self) -> None:
        from app.workers.matting import StubMattingBackend

        assert isinstance(cli.make_backend("stub", None, None, None), StubMattingBackend)
        assert cli.make_backend(None, None, None, None) is None


class TestConfiguredStores:
    def test_without_a_sandbox_it_uses_the_configured_database_and_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.db
        import app.object_store

        sentinel_db = object()
        monkeypatch.setattr(app.db, "SessionLocal", lambda: sentinel_db)
        monkeypatch.setattr(app.object_store, "S3ObjectStore", lambda: "s3")
        db, store = cli.open_stores(None)
        assert (db, store) == (sentinel_db, "s3")
