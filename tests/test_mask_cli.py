"""``python -m app.cli.masks``: seed a sample, run the batch (dry run unless
--apply), report a batch. ``--sandbox DIR`` keeps everything in DIR (a SQLite
file and a local object directory), so a sample run needs no services."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from app.cli import masks as cli
from app.config import get_settings
from app.workers.matting import StubMattingBackend
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "mask_sample"


#: What every line carries beside the report (LG-U11: the estate's line).
_LINE_KEYS = {"time", "level", "service", "version", "event", "msg", "code", "duration_ms",
              "trace_id", "span_id", "job", "exit", "err"}  # fmt: skip
#: The report's lists of records: one line each, their count on job.end.
_RECORD_LISTS = {"items", "withdrawn", "restored"}


def _decode(value: object) -> object:
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _nest(report: dict, prefix: str) -> None:  # type: ignore[type-arg]
    keys = [k for k in report if k.startswith(f"{prefix}_")]
    if f"{prefix}_dry_run" in report:  # a nested report (revoke-store, restore-store)
        report[prefix] = {k[len(prefix) + 1 :]: report.pop(k) for k in keys}
        for name in _RECORD_LISTS:
            if isinstance(report[prefix].get(name), int):
                report[prefix][name] = []


def _report(out: str) -> dict:  # type: ignore[type-arg]
    """The masks CLI's report, read back from its lines: job.end carries the
    report's fields, each record list is one app.<name> line per record."""
    lines = [json.loads(line) for line in out.splitlines()]
    assert (lines[0]["event"], lines[-1]["event"]) == ("job.start", "job.end")
    assert len({line["trace_id"] for line in lines}) == 1  # one run, one trace
    report = {k: _decode(v) for k, v in lines[-1].items() if k not in _LINE_KEYS}
    records: dict[str, list] = {}  # type: ignore[type-arg]
    for line in lines[1:-1]:
        fields = {k: _decode(v) for k, v in line.items() if k not in _LINE_KEYS}
        record = fields["value"] if set(fields) == {"value"} else fields
        records.setdefault(line["event"].removeprefix("app."), []).append(record)
    for name in [k for k in report if any(k == n or k.endswith(f"_{n}") for n in _RECORD_LISTS)]:
        if isinstance(report[name], int):
            report[name] = records.get(name, [])
    _nest(report, "reevaluate")
    _nest(report, "batch")
    return report


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    code = cli.main(list(argv))
    return code, _report(capsys.readouterr().out)


def _refusal(capsys: pytest.CaptureFixture[str]) -> str:
    """Why the CLI refused: the run's job.end err (stderr stays empty)."""
    captured = capsys.readouterr()
    end = json.loads(captured.out.splitlines()[-1])
    assert (end["event"], end["code"], captured.err) == ("job.end", "error", "")
    return str(end["err"]["message"])


@pytest.fixture()
def sandbox(tmp_path: Path) -> str:
    return str(tmp_path / "sandbox")


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask model, so the sample can be
    APPLIED without a GPU. The real stub is dry run only."""

    name = "stub-model"
    publishes_masks = True


@pytest.fixture()
def model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_backend", lambda *a: _Model())


@pytest.fixture(autouse=True)
def _fresh_settings() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def store_granted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sample's one store listing (sample:4) is masked only under a grant."""
    monkeypatch.setenv("MASK_STORE_GRANTS", json.dumps({"example-store": ["example-store.test"]}))
    get_settings.cache_clear()


def _rows(sandbox: str, model: type) -> list:  # type: ignore[type-arg]
    engine = create_engine(f"sqlite:///{sandbox}/masks.db")
    with Session(engine) as db:
        return list(db.execute(select(model)).scalars())


class TestSampleRun:
    def test_seed_then_dry_run_then_apply_then_nothing_left(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None, store_granted: None
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

    def test_without_a_grant_the_store_listing_gets_no_mask(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        code, dry = _run(capsys, "--sandbox", sandbox, "run")
        assert code == 0
        by_figure = {i["figure_ids"][0]: i for i in dry["items"]}
        assert (by_figure["sample:4"]["status"], by_figure["sample:4"]["reason"]) == (
            "ineligible",
            "not_granted",
        )
        assert dry["counts"]["ineligible"] == 2

    def test_reevaluate_is_a_dry_run_unless_applied(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None
    ) -> None:
        from app.models import ImageMask

        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        _run(capsys, "--sandbox", sandbox, "run", "--apply")
        code, dry = _run(capsys, "--sandbox", sandbox, "reevaluate")
        assert code == 0
        assert (dry["dry_run"], dry["withdrawn"], dry["restored"]) == (True, [], [])
        ok = [r for r in _rows(sandbox, ImageMask) if r.status == "ok"]
        assert dry["unchanged"] == len(ok) > 0
        code, applied = _run(capsys, "--sandbox", sandbox, "reevaluate", "--apply")
        assert (code, applied["dry_run"]) == (0, False)

    def test_recheck_ineligible_reselects_refused_images(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None
    ) -> None:
        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        _run(capsys, "--sandbox", sandbox, "run", "--apply")
        code, again = _run(capsys, "--sandbox", sandbox, "run", "--apply", "--recheck-ineligible")
        assert code == 0
        assert sorted(i["figure_ids"][0] for i in again["items"]) == ["sample:4", "sample:8"]

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


class TestStoreRevocation:
    """``revoke-store`` / ``restore-store``: the single commands for a store
    that asks for its cut-outs to be removed (README runbook)."""

    def test_revoke_then_restore_a_store(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None, store_granted: None
    ) -> None:
        from app.models import ImageMask

        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        _, applied = _run(capsys, "--sandbox", sandbox, "run", "--apply")
        store_sha = next(
            i["source_sha256"] for i in applied["items"] if i["figure_ids"] == ["sample:4"]
        )
        ok_before = sorted(r.source_sha256 for r in _rows(sandbox, ImageMask) if r.status == "ok")
        assert store_sha in ok_before

        code, dry = _run(capsys, "--sandbox", sandbox, "revoke-store", "example-store")
        assert (code, dry["dry_run"], dry["reevaluate"]["withdrawn"]) == (0, True, [store_sha])
        code, revoked = _run(
            capsys, "--sandbox", sandbox, "revoke-store", "example-store", "--apply"
        )
        assert (code, revoked["reevaluate"]["withdrawn"]) == (0, [store_sha])
        rows = {r.source_sha256: (r.status, r.reason) for r in _rows(sandbox, ImageMask)}
        assert rows[store_sha] == ("withdrawn", "store_revoked")
        assert sorted(s for s, (st, _) in rows.items() if st == "ok") == sorted(
            s for s in ok_before if s != store_sha
        )

        code, restored = _run(
            capsys, "--sandbox", sandbox, "restore-store", "example-store", "--apply"
        )
        assert (code, restored["reevaluate"]["restored"]) == (0, [store_sha])
        rows = {r.source_sha256: (r.status, r.reason) for r in _rows(sandbox, ImageMask)}
        assert rows[store_sha] == ("ok", None)

    def test_restore_is_a_dry_run_unless_applied(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None, store_granted: None
    ) -> None:
        from app.models import ImageMask, MaskStoreRevocation

        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        _run(capsys, "--sandbox", sandbox, "run", "--apply")
        _run(capsys, "--sandbox", sandbox, "revoke-store", "example-store", "--apply")
        withdrawn = [r.source_sha256 for r in _rows(sandbox, ImageMask) if r.status == "withdrawn"]
        assert len(withdrawn) == 1
        code, dry = _run(capsys, "--sandbox", sandbox, "restore-store", "example-store")
        assert (code, dry["dry_run"], dry["reevaluate"]["restored"]) == (0, True, withdrawn)
        assert [r.source for r in _rows(sandbox, MaskStoreRevocation)] == ["example-store"]
        assert [
            r.source_sha256 for r in _rows(sandbox, ImageMask) if r.status == "withdrawn"
        ] == withdrawn

    def test_a_batchs_report_names_its_source(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None, store_granted: None
    ) -> None:
        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        _, applied = _run(
            capsys, "--sandbox", sandbox, "run", "--apply", "--source", "Example-Store"
        )
        code, report = _run(
            capsys, "--sandbox", sandbox, "report", "--batch", str(applied["batch_id"])
        )
        assert (code, report["source"]) == (0, "example-store")

    def test_a_batch_run_can_target_one_source(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], store_granted: None
    ) -> None:
        _run(capsys, "--sandbox", sandbox, "seed", "--sample", str(SAMPLE))
        code, dry = _run(capsys, "--sandbox", sandbox, "run", "--source", "example-store")
        assert code == 0
        assert [i["figure_ids"] for i in dry["items"]] == [["sample:4"]]

    @pytest.mark.parametrize("backend", [[], ["--backend", "stub"]])
    def test_restoring_with_the_stub_is_refused_before_any_work(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], backend: list[str]
    ) -> None:
        code = cli.main(
            ["--sandbox", sandbox, "restore-store", "example-store", "--apply", *backend]
        )
        assert code == 2
        assert "dry run only" in _refusal(capsys)
        assert not Path(sandbox).exists()

    def test_a_configured_revocation_is_refused_with_its_reason(
        self,
        sandbox: str,
        capsys: pytest.CaptureFixture[str],
        model: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MASK_STORE_REVOKED", '["example-store"]')
        get_settings.cache_clear()
        code = cli.main(["--sandbox", sandbox, "restore-store", "example-store", "--apply"])
        assert code == 2
        assert "MASK_STORE_REVOKED" in _refusal(capsys)

    def test_a_blank_source_is_refused(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--sandbox", sandbox, "revoke-store", " ", "--apply"]) == 2
        assert "source" in _refusal(capsys)


class TestRefusals:
    def test_resuming_an_unknown_batch_is_an_error(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], model: None
    ) -> None:
        code = cli.main(["--sandbox", sandbox, "run", "--apply", "--resume", "999"])
        assert code == 2
        assert "not resumable" in _refusal(capsys)

    def test_birefnet_without_weights_is_refused_before_any_work(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MATTING_MODEL_PATH", raising=False)
        code = cli.main(["--sandbox", sandbox, "run", "--backend", "birefnet"])
        assert code == 2
        assert "--model" in _refusal(capsys)
        assert not Path(sandbox).exists()

    @pytest.mark.parametrize("backend", [[], ["--backend", "stub"]])
    def test_applying_with_the_stub_is_refused_before_any_work(
        self, sandbox: str, capsys: pytest.CaptureFixture[str], backend: list[str]
    ) -> None:
        code = cli.main(["--sandbox", sandbox, "run", "--apply", *backend])
        assert code == 2
        assert "dry run only" in _refusal(capsys)
        assert not Path(sandbox).exists()

    def test_report_of_an_unknown_batch(
        self, sandbox: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--sandbox", sandbox, "report", "--batch", "7"]) == 2
        assert "no mask batch 7" in _refusal(capsys)

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
