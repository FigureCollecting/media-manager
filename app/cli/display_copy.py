"""SHOW1: the cluster display-copy lane's command line.

    python -m app.cli.display_copy sync    [--spine-url URL] [--order newest|oldest]
        [--scope showable] [--lease-items N] [--lease-seconds S<=120] [--max-new N]
        [--high-water N] [--max-failed-reports N] [--purge-days D] [--budget-seconds S]
        [--deadline-seconds S] [--max-attempts N] [--sandbox DIR]
    python -m app.cli.display_copy seed    [--threads N] [--budget-seconds S]
        [--max-failed N] [--max-failed-fraction F] [--max-errors N] [--max-outages N]
        [--raw-dir DIR] [--media-dir DIR] [--sandbox DIR]
    python -m app.cli.display_copy status  [--sandbox DIR]
    python -m app.cli.display_copy requeue --state failed|held|stuck
        [--reason-prefix P] [--force] [--apply] [--sandbox DIR]

``sync`` is the mm-display-sync CronJob (service account media-report, the
spine identity, NO bucket key): REPORT, ENQUEUE, PURGE (app.spine.display_feed).
``seed`` is the mm-display-seed CronJob (service account media-display, the
media bucket's key and the raw bucket's read-only key, NO spine identity):
app.workers.display_copy. The two meet only in the media DB.

``status`` prints the queue's counts. ``requeue`` puts failed, held or stuck
originals back to pending (a dry run unless ``--apply``; ``--force`` only
with ``--state held``: seed although it withdraws a mask).

``--sandbox DIR`` uses ``DIR/masks.db`` (SQLite) instead of DATABASE_URL;
``--raw-dir`` and ``--media-dir`` read a local raw store and write a local
media store instead of the buckets (tests and dry rehearsals).

Output (LG-U11): the estate's log line on stdout (app.telemetry): job.start,
one app.<event> line per lane event, job.end with the summary and ``exit``.

Exit status: see app.spine.display_feed (sync) and app.workers.display_copy
(seed); status and requeue 0; 2 bad arguments (nothing ran).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import grpc
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .. import telemetry
from ..object_store import LocalObjectStore, ObjectStore, S3ObjectStore
from ..spine import display_feed as feed
from ..spine.media_client import (
    DEFAULT_DEADLINE_SECONDS,
    RetryPolicy,
    SpineMediaClient,
    open_channel,
    target_from_url,
)
from ..telemetry.events import EventLines
from ..workers import display_copy as seeder
from ..workers.raw_source import RawStoreConfigError, ReadOnlyStore, raw_store_from_env

EXIT_USAGE = 2
SERVICE = "mm-display"
_SUMMARIZED = ("enqueued", "reported", "stopped", "done", "status", "requeue", "failed_held")
#: The bucket clients' timeouts and retries: the mask batch's (app.cli.mask_batch).
BUCKET_CONFIG: dict[str, Any] = {
    "connect_timeout": 10,
    "read_timeout": 60,
    "retries": {"max_attempts": 3, "mode": "standard"},
    "tcp_keepalive": True,
}


class Refused(ValueError):
    """Bad arguments: nothing ran (exit 2)."""


def _sessions(args: argparse.Namespace) -> Callable[[], Session]:
    if args.sandbox is None:
        from .. import db as configured_db

        return configured_db.SessionLocal
    engine = create_engine(
        f"sqlite:///{Path(args.sandbox) / 'masks.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    return lambda: Session(engine, expire_on_commit=False)


def _raw_store(args: argparse.Namespace, connections: int) -> ObjectStore:
    if args.raw_dir:
        return ReadOnlyStore(LocalObjectStore(args.raw_dir))
    try:
        return raw_store_from_env(connections, BUCKET_CONFIG)
    except RawStoreConfigError as exc:
        raise Refused(f"display_copy: {exc} (or --raw-dir)") from exc


def _media_store(args: argparse.Namespace, connections: int) -> ObjectStore:
    if args.media_dir:
        return LocalObjectStore(args.media_dir)
    from ..config import get_settings
    from ..s3 import s3_client

    client = s3_client(max_pool_connections=connections, **BUCKET_CONFIG)
    return S3ObjectStore(client=client, bucket=get_settings().s3_bucket)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.display_copy",
        description="The cluster display-copy lane (SHOW1).",
        allow_abbrev=False,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def command(name: str, help_: str) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=help_, allow_abbrev=False)
        c.add_argument("--sandbox", help="use DIR/masks.db (SQLite) instead of DATABASE_URL")
        return c

    d = feed.FeedConfig()
    s = command("sync", "report seeded and failed originals, then lease new ones")
    s.add_argument("--spine-url", help="SpineMedia, http://host:port (default SPINE_MEDIA_URL)")
    s.add_argument("--order", default=d.order)
    s.add_argument("--scope", default=d.scope)
    s.add_argument("--lease-items", type=int, default=d.lease_items)
    s.add_argument("--lease-seconds", type=int, default=d.lease_seconds)
    s.add_argument("--max-new", type=int, default=d.max_new)
    s.add_argument("--high-water", type=int, default=d.high_water)
    s.add_argument("--max-failed-reports", type=int, default=d.max_failed_reports)
    s.add_argument("--purge-days", type=int, default=d.purge_days)
    s.add_argument("--budget-seconds", type=float, default=d.budget_seconds)
    s.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    s.add_argument("--max-attempts", type=int, default=5, help="per call, retries included")

    c = seeder.SeedConfig()
    e = command("seed", "seed queued originals' clean copies (no spine identity)")
    e.add_argument("--threads", type=int, default=c.threads)
    e.add_argument("--budget-seconds", type=float, default=c.budget_seconds)
    e.add_argument("--max-failed", type=int, default=c.max_failed)
    e.add_argument("--max-failed-fraction", type=float, default=c.max_failed_fraction)
    e.add_argument("--max-errors", type=int, default=c.max_errors)
    e.add_argument("--max-outages", type=int, default=c.max_outages)
    e.add_argument("--raw-dir", help="a local raw store instead of the raw bucket (RAW_S3_*)")
    e.add_argument("--media-dir", help="a local media store instead of the media bucket (S3_*)")

    command("status", "print the queue's counts")
    r = command("requeue", "put failed, held or stuck originals back to pending")
    r.add_argument("--state", required=True)
    r.add_argument("--reason-prefix")
    r.add_argument(
        "--force", action="store_true", help="held only: seed though it withdraws a mask"
    )
    r.add_argument("--apply", action="store_true", help="write (default: count only)")
    return p


def _sync(args: argparse.Namespace, emit: EventLines) -> int:
    url = args.spine_url or os.environ.get("SPINE_MEDIA_URL", "")
    try:
        config = feed.FeedConfig(
            order=args.order,
            scope=args.scope,
            lease_items=args.lease_items,
            lease_seconds=args.lease_seconds,
            max_new=args.max_new,
            high_water=args.high_water,
            max_failed_reports=args.max_failed_reports,
            purge_days=args.purge_days,
            budget_seconds=args.budget_seconds,
        )
        policy = RetryPolicy(max_attempts=args.max_attempts)
        if not args.deadline_seconds > 0:
            raise ValueError("--deadline-seconds must be above 0")
        target_from_url(url)
    except ValueError as exc:
        raise Refused(f"display_copy sync: {exc}") from exc
    channels: list[grpc.Channel] = []

    def connect() -> SpineMediaClient:
        channels.append(open_channel(url))
        return SpineMediaClient(channels[-1], deadline_seconds=args.deadline_seconds, policy=policy)

    db = _sessions(args)()
    try:
        return feed.run_sync(db, connect, config, emit=emit)
    finally:
        db.close()
        for channel in channels:
            channel.close()


def _seed(args: argparse.Namespace, emit: EventLines) -> int:
    try:
        config = seeder.SeedConfig(
            threads=args.threads,
            budget_seconds=args.budget_seconds,
            max_failed=args.max_failed,
            max_failed_fraction=args.max_failed_fraction,
            max_errors=args.max_errors,
            max_outages=args.max_outages,
        )
    except ValueError as exc:
        raise Refused(f"display_copy seed: {exc}") from exc
    connections = max(10, 2 * config.threads)
    raw = _raw_store(args, connections)
    media = _media_store(args, connections)
    sessions = _sessions(args)
    if config.threads == 0:
        db = sessions()
        try:
            return seeder.run_seed(lambda: db, raw, media, config, emit=emit)
        finally:
            db.close()
    return seeder.run_seed(sessions, raw, media, config, emit=emit)


def _status(args: argparse.Namespace, emit: EventLines) -> int:
    db = _sessions(args)()
    try:
        found = seeder.status(db)
    finally:
        db.close()
    emit(
        {
            "event": "status",
            "originals": json.dumps(found["originals"], sort_keys=True),
            "unreported_depictions": found["unreported_depictions"],
            "reasons": json.dumps(found["reasons"], sort_keys=True),
        }
    )
    return 0


def _requeue(args: argparse.Namespace, emit: EventLines) -> int:
    if args.force and args.state != "held":
        raise Refused("display_copy requeue: --force is only for --state held")
    db = _sessions(args)()
    try:
        matched = seeder.requeue(
            db,
            state=args.state,
            reason_prefix=args.reason_prefix,
            force=args.force,
            apply=args.apply,
        )
    except ValueError as exc:
        raise Refused(f"display_copy {exc}") from exc
    finally:
        db.close()
    emit({"event": "requeue", "state": args.state, "matched": matched, "applied": args.apply})
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace, EventLines], int]] = {
    "sync": _sync,
    "seed": _seed,
    "status": _status,
    "requeue": _requeue,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log = telemetry.configure(SERVICE)
    telemetry.start_tracing(SERVICE)
    lines = EventLines(log, summarize=_SUMMARIZED)
    try:
        return telemetry.run_job(
            f"display-{args.command}",
            lambda run: _COMMANDS[args.command](args, lines),
            logger=log,
            counts=lambda code: {**lines.summary, "exit": code},
            ok=lambda code: code == 0,
        )
    except Refused:
        return EXIT_USAGE  # job.end says why (err)


if __name__ == "__main__":  # pragma: no cover - the module's entry point
    raise SystemExit(main())
