"""MASK-1 operator CLI.

    python -m app.cli.masks [--sandbox DIR] seed --sample DIR
    python -m app.cli.masks [--sandbox DIR] run [--apply] [--limit N] [--page-size N]
                                                [--resume BATCH] [--retry-failed]
                                                [--recheck-ineligible] [--source SOURCE]
                                                [--backend stub|birefnet|birefnet-basekept]
                                                [--model PATH] [--isnet-general PATH]
                                                [--isnet-anime PATH]
                                                [--report FILE]
    python -m app.cli.masks [--sandbox DIR] report --batch ID
    python -m app.cli.masks [--sandbox DIR] reevaluate [--apply]
    python -m app.cli.masks [--sandbox DIR] revoke-store SOURCE [--apply]
    python -m app.cli.masks [--sandbox DIR] restore-store SOURCE [--apply] [--backend ...]
    python -m app.cli.masks [--sandbox DIR] official-gallery --worklist FILE [--worklist FILE ...]
                                                             [--apply]

``run`` is a DRY RUN unless ``--apply``: it computes and prints every mask's
outcome and grounding, and writes nothing. ``--apply`` needs a mask model
(birefnet, birefnet-basekept, or one configured): the stub is dry run only. ``--sandbox DIR`` keeps the whole
lane in DIR (``masks.db`` SQLite + ``objects/``), so a seeded sample runs with
no Postgres or bucket; without it the configured DATABASE_URL and S3 bucket
are used. ``--backend birefnet --model PATH`` runs the real model (CUDA
required, see app.workers.matting); the default is the configured backend.
``revoke-store`` / ``restore-store`` (MASK-G) withdraw or bring back every
mask of one source; dry runs unless ``--apply`` (README runbook "A store
asks for its cut-outs to be removed"). ``restore-store --apply`` needs a mask
model, like ``run --apply``. ``official-gallery`` (GAL3) records the gallery
list's source class on the links of the photos settled before it was carried
and clears the ``ineligible`` row of each copy the full judgment now allows,
so the next mask run mints it (app.workers.gallery_rejudge); its work lists
come from ``mask_batch sql official-gallery-worklist``; a dry run unless
``--apply``.
Exit status: 0 done, 2 refused (nothing was changed by the refused step).

Output (LG-U11): the estate's log lines on stdout, one run = one root span
(app.telemetry): job.start; one line per record of the report (``app.items``
per masked image, ``app.withdrawn`` / ``app.restored`` per mask ...); then
job.end carrying the rest of the report (counts, grounding ranges as one-line
JSON) and ``exit``. ``reevaluate`` judges its rows in batches, each an ITEM
(fc-shared's job model: a new root trace linked to the run) holding that
batch's withdrawn/restored lines and an ``app.reevaluate_batch`` line; its
job.end carries the counts. Every line carries run_id. A refusal is job.end's
err. ``run --report FILE`` still writes the whole report as one JSON document.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from .. import telemetry
from ..models import Base, FigureGallery, ImageVersion, MaskBatch, PublicObject
from ..object_store import LocalObjectStore, ObjectStore
from ..telemetry.events import report_lines
from ..workers.basekept import BaseKeptMattingBackend
from ..workers.gallery_rejudge import GalleryCapture, rejudge_official_gallery
from ..workers.mask_job import reevaluate_masks, restore_store, revoke_store, run_mask_batch
from ..workers.mask_policy import ImageOrigin
from ..workers.mask_seed import seed_clean_image
from ..workers.matting import (
    BiRefNetMattingBackend,
    IsNetMattingBackend,
    MattingBackend,
    StubMattingBackend,
    get_matting_backend,
)
from .mask_batch import read_worklists

#: Grounding fields summarised as [min, max] over a run's measured masks.
RANGE_FIELDS = (
    "bottom_margin_frac",
    "top_margin_frac",
    "contact_band_center_x_frac",
    "contact_band_width_frac",
)


@compiles(UUID, "sqlite")
def _uuid_sqlite(type_: UUID[Any], compiler: Any, **kw: Any) -> str:
    return "TEXT"


@compiles(CITEXT, "sqlite")
def _citext_sqlite(type_: CITEXT, compiler: Any, **kw: Any) -> str:
    return "TEXT"


class Refused(Exception):
    """A request the CLI refuses before changing anything (exit status 2)."""


def _weights(given: str | None, env: str, flag: str, backend: str) -> str:
    path = given or os.environ.get(env)
    if not path:
        raise Refused(f"--backend {backend} needs {flag} PATH (or {env})")
    return path


def make_backend(
    name: str | None,
    model: str | None,
    isnet_general: str | None,
    isnet_anime: str | None,
    *,
    device: str | None = None,
    cpu_threads: int | None = None,
) -> MattingBackend | None:
    """None = the configured backend (Settings.matting_backend). ``device``
    and ``cpu_threads`` place every session (default: Settings.matting_device
    and matting_cpu_threads; app.workers.matting.MATTING_DEVICES)."""
    if name is None:
        return None
    if name not in ("birefnet", "birefnet-basekept"):
        return StubMattingBackend()
    if device is None or cpu_threads is None:
        from ..config import get_settings

        settings = get_settings()
        device = device or settings.matting_device
        cpu_threads = cpu_threads if cpu_threads is not None else settings.matting_cpu_threads
    placement: dict[str, Any] = {"device": device, "cpu_threads": cpu_threads}
    primary = BiRefNetMattingBackend(
        model_path=_weights(model, "MATTING_MODEL_PATH", "--model", name), **placement
    )
    if name == "birefnet":
        return primary
    return BaseKeptMattingBackend(
        primary=primary,
        donors=[
            IsNetMattingBackend(
                model_path=_weights(
                    isnet_general, "MATTING_ISNET_GENERAL_PATH", "--isnet-general", name
                ),
                **placement,
            ),
            IsNetMattingBackend(
                model_path=_weights(isnet_anime, "MATTING_ISNET_ANIME_PATH", "--isnet-anime", name),
                **placement,
            ),
        ],
    )


def open_stores(sandbox: str | None) -> tuple[Session, ObjectStore]:
    if sandbox is None:
        from .. import db as configured_db
        from .. import object_store as configured_store

        return configured_db.SessionLocal(), configured_store.S3ObjectStore()
    root = Path(sandbox)
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{root / 'masks.db'}")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False), LocalObjectStore(root / "objects")


def _seed(db: Session, store: ObjectStore, sample: Path) -> dict[str, Any]:
    manifest = json.loads((sample / "manifest.json").read_text())
    for entry in manifest["entries"]:
        seed_clean_image(
            db,
            store,
            (sample / "originals" / f"{entry['name']}.png").read_bytes(),
            origins=[ImageOrigin(**o) for o in entry["origins"]],
            figure_id=entry["figure_id"],
        )
    return {"seeded": len(manifest["entries"])}


def _figure_ids(db: Session, source_sha256: str) -> list[str]:
    rows = db.execute(
        select(FigureGallery.figure_id)
        .join(ImageVersion, ImageVersion.image_id == FigureGallery.image_id)
        .join(PublicObject, PublicObject.image_version_id == ImageVersion.id)
        .where(PublicObject.sha256 == source_sha256)
        .order_by(FigureGallery.id)
    ).scalars()
    return list(rows)


def _run(db: Session, store: ObjectStore, args: argparse.Namespace) -> dict[str, Any]:
    backend = make_backend(args.backend, args.model, args.isnet_general, args.isnet_anime)
    try:
        report = run_mask_batch(
            db,
            store,
            backend=backend,
            dry_run=not args.apply,
            limit=args.limit,
            page_size=args.page_size,
            resume_batch_id=args.resume,
            retry_failed=args.retry_failed,
            recheck_ineligible=args.recheck_ineligible,
            source=args.source,
        )
    except ValueError as exc:
        raise Refused(str(exc)) from exc
    summary = report.as_dict()
    for item in summary["items"]:
        item["figure_ids"] = _figure_ids(db, item["source_sha256"])
    measured = [i["grounding"] for i in summary["items"] if i["grounding"]]
    summary["grounding_ranges"] = {
        f: [min(g[f] for g in measured), max(g[f] for g in measured)]
        for f in RANGE_FIELDS
        if measured
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return summary


def _store_command(db: Session, store: ObjectStore, args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.command == "revoke-store":
            return revoke_store(db, args.source, dry_run=not args.apply)
        backend = make_backend(args.backend, args.model, args.isnet_general, args.isnet_anime)
        return restore_store(db, store, args.source, dry_run=not args.apply, backend=backend)
    except ValueError as exc:
        raise Refused(str(exc)) from exc


def _report(db: Session, batch_id: int) -> dict[str, Any]:
    batch = db.get(MaskBatch, batch_id)
    if batch is None:
        raise Refused(f"no mask batch {batch_id}")
    return {
        "id": batch.id,
        "status": batch.status,
        "pipeline_version": batch.pipeline_version,
        "backend": batch.backend,
        "counts": batch.counts,
        "cursor_sha256": batch.cursor_sha256,
        "item_limit": batch.item_limit,
        "source": batch.source,
        "started_at": batch.started_at,
        "finished_at": batch.finished_at,
    }


def _official_gallery(db: Session, args: argparse.Namespace) -> dict[str, Any]:
    """GAL3: every row of the work lists, one gallery link each (not each
    original once), through app.workers.gallery_rejudge."""
    try:
        items, skipped = read_worklists([Path(p) for p in args.worklist], every_row=True)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    captures = [
        GalleryCapture(i.blob_sha256, i.product_id, i.url, i.source, i.source_class) for i in items
    ]
    report = rejudge_official_gallery(db, captures, dry_run=not args.apply)
    return {**report, "skipped_lines": skipped}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli.masks", description=__doc__.split("\n")[0])
    p.add_argument("--sandbox", help="keep the lane in this directory (SQLite + local objects)")
    sub = p.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="register a sample's originals and their clean copies")
    seed.add_argument("--sample", required=True, type=Path)

    run = sub.add_parser("run", help="mask every clean image without one (dry run by default)")
    run.add_argument("--apply", action="store_true", help="store results (default: dry run)")
    run.add_argument("--limit", type=int)
    run.add_argument("--page-size", type=int, default=50)
    run.add_argument("--resume", type=int, help="continue an unfinished batch (keeps its --source)")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--recheck-ineligible", action="store_true", help="re-judge images refused a mask"
    )
    run.add_argument("--source", help="only clean copies with a live origin of this source")
    _backend_options(run)
    run.add_argument("--report", help="also write the full JSON report here")

    rep = sub.add_parser("report", help="show one applied batch")
    rep.add_argument("--batch", required=True, type=int)

    ree = sub.add_parser(
        "reevaluate", help="withdraw or restore published masks (dry run by default)"
    )
    ree.add_argument("--apply", action="store_true", help="store the changes (default: dry run)")

    revoke = sub.add_parser(
        "revoke-store", help="withdraw every mask of one source (dry run by default)"
    )
    revoke.add_argument("source")
    revoke.add_argument("--apply", action="store_true", help="store the change (default: dry run)")

    restore = sub.add_parser(
        "restore-store", help="lift a revoke-store and mask what it refused (dry run by default)"
    )
    restore.add_argument("source")
    restore.add_argument("--apply", action="store_true", help="store the change (default: dry run)")
    _backend_options(restore)

    gallery = sub.add_parser(
        "official-gallery",
        help="re-judge refused photos of MFC's official gallery list (dry run by default)",
    )
    gallery.add_argument(
        "--worklist",
        action="append",
        required=True,
        help="mask_batch sql official-gallery-worklist output (repeatable)",
    )
    gallery.add_argument(
        "--apply", action="store_true", help="store the changes (default: dry run)"
    )
    return p


def _backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["stub", "birefnet", "birefnet-basekept"])
    parser.add_argument("--model", help="BiRefNet ONNX weights (birefnet, birefnet-basekept)")
    parser.add_argument(
        "--isnet-general", help="IS-Net general-use ONNX weights (birefnet-basekept)"
    )
    parser.add_argument("--isnet-anime", help="IS-Net anime ONNX weights (birefnet-basekept)")


SERVICE = "media-masks"


def _reevaluate(
    db: Session, args: argparse.Namespace, run: telemetry.JobRun, log: telemetry.Logger
) -> dict[str, Any]:
    """Each batch of rows is an item of the run: its records and its line in its own trace."""

    def scope(index: int, judge: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        def batch() -> dict[str, Any]:
            part = judge()
            records = {"withdrawn": part["withdrawn"], "restored": part["restored"]}
            counts = report_lines(log, records)
            log.info("", event="app.reevaluate_batch", batch=index, **counts,
                     unchanged=part["unchanged"])  # fmt: skip
            return part

        return run.item(batch, name="item.batch", attributes={"fc.batch": index})

    report = reevaluate_masks(db, dry_run=not args.apply, batch_scope=scope)
    return {**report, "withdrawn": len(report["withdrawn"]), "restored": len(report["restored"])}


def _command(
    args: argparse.Namespace, run: telemetry.JobRun, log: telemetry.Logger
) -> dict[str, Any]:
    if args.command in ("run", "restore-store"):
        # refuse before touching any store
        chosen = make_backend(args.backend, args.model, args.isnet_general, args.isnet_anime)
        active = chosen or get_matting_backend()
        if args.apply and not active.publishes_masks:
            raise Refused(f"backend {active.name!r} is not a mask model: dry run only")
    db, store = open_stores(args.sandbox)
    try:
        if args.command == "seed":
            return _seed(db, store, args.sample)
        if args.command == "run":
            return _run(db, store, args)
        if args.command == "reevaluate":
            return _reevaluate(db, args, run, log)
        if args.command in ("revoke-store", "restore-store"):
            return _store_command(db, store, args)
        if args.command == "official-gallery":
            return _official_gallery(db, args)
        return _report(db, args.batch)
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log = telemetry.configure(SERVICE)
    telemetry.start_tracing(SERVICE)
    summary: dict[str, Any] = {}

    def body(run: telemetry.JobRun) -> int:
        summary.update(report_lines(log, _command(args, run, log)))
        return 0

    try:
        return telemetry.run_job(
            f"masks-{args.command}", body, logger=log, counts=lambda code: {**summary, "exit": code}
        )
    except Refused:
        return 2  # job.end says why (err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
