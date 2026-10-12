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
model, like ``run --apply``.
Exit status: 0 done, 2 refused (nothing was changed by the refused step).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from ..models import Base, FigureGallery, ImageVersion, MaskBatch, PublicObject
from ..object_store import LocalObjectStore, ObjectStore
from ..workers.basekept import BaseKeptMattingBackend
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
    name: str | None, model: str | None, isnet_general: str | None, isnet_anime: str | None
) -> MattingBackend | None:
    """None = the configured backend (Settings.matting_backend)."""
    if name is None:
        return None
    if name == "birefnet":
        return BiRefNetMattingBackend(
            model_path=_weights(model, "MATTING_MODEL_PATH", "--model", name)
        )
    if name == "birefnet-basekept":
        return BaseKeptMattingBackend(
            primary=BiRefNetMattingBackend(
                model_path=_weights(model, "MATTING_MODEL_PATH", "--model", name)
            ),
            donors=[
                IsNetMattingBackend(
                    model_path=_weights(
                        isnet_general, "MATTING_ISNET_GENERAL_PATH", "--isnet-general", name
                    )
                ),
                IsNetMattingBackend(
                    model_path=_weights(
                        isnet_anime, "MATTING_ISNET_ANIME_PATH", "--isnet-anime", name
                    )
                ),
            ],
        )
    return StubMattingBackend()


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


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))


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
        "started_at": batch.started_at,
        "finished_at": batch.finished_at,
    }


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
    run.add_argument("--resume", type=int, help="continue an unfinished batch")
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
    return p


def _backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["stub", "birefnet", "birefnet-basekept"])
    parser.add_argument("--model", help="BiRefNet ONNX weights (birefnet, birefnet-basekept)")
    parser.add_argument(
        "--isnet-general", help="IS-Net general-use ONNX weights (birefnet-basekept)"
    )
    parser.add_argument("--isnet-anime", help="IS-Net anime ONNX weights (birefnet-basekept)")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in ("run", "restore-store"):
            # refuse before touching any store
            chosen = make_backend(args.backend, args.model, args.isnet_general, args.isnet_anime)
            active = chosen or get_matting_backend()
            if args.apply and not active.publishes_masks:
                raise Refused(f"backend {active.name!r} is not a mask model: dry run only")
        db, store = open_stores(args.sandbox)
        try:
            if args.command == "seed":
                _print(_seed(db, store, args.sample))
            elif args.command == "run":
                _print(_run(db, store, args))
            elif args.command == "reevaluate":
                _print(reevaluate_masks(db, dry_run=not args.apply))
            elif args.command in ("revoke-store", "restore-store"):
                _print(_store_command(db, store, args))
            else:
                _print(_report(db, args.batch))
        finally:
            db.close()
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
