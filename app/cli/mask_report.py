"""MM-REPORT-1: replay the media DB's clean copies and masks into the spine.

    python -m app.cli.mask_report [--dry-run] [--spine-url URL] [--batch-size N]
        [--lease-seconds S] [--deadline-seconds S] [--max-attempts N] [--restart]
        [--sandbox DIR]

Runs in the cluster as the ``mm-report`` Job (namespace ``media``, service
account ``media-report``): it reads the media DB (``DATABASE_URL``) and calls
media.v1 SpineMedia at ``SPINE_MEDIA_URL``
(``http://ingest-server.fc.svc.cluster.local:50063``) in plaintext through its
Linkerd proxy, which carries the call over mTLS under that identity. What it
sends, in what order, and how it resumes: app.spine.replay.

``--dry-run`` reads the media DB and prints every call it would make; it
opens no gRPC channel and writes nothing. ``--restart`` abandons an
unfinished run and starts over (by default an unfinished run is resumed
after its cursor). ``--sandbox DIR`` reads ``DIR/masks.db`` (SQLite) instead
of DATABASE_URL. One JSON line per event on stdout: ``plan``, ``resume``,
``would_send`` (dry run), ``batch`` (counts per RPC, stored/matched,
warnings, refusals by status), ``stopped``, ``done``.

Exit status: 0 done; 1 done, with calls refused; 2 bad arguments (nothing
ran); 3 stopped, resumable: run it again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import grpc
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ..spine.media_client import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_LEASE_SECONDS,
    RetryPolicy,
    SpineMediaClient,
    open_channel,
    target_from_url,
)
from ..spine.replay import ReplayConfig, run_replay

EXIT_USAGE = 2


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def _open_db(sandbox: str | None) -> Session:
    if sandbox is None:
        from .. import db as configured_db

        return configured_db.SessionLocal()
    engine = create_engine(f"sqlite:///{Path(sandbox) / 'masks.db'}")
    return Session(engine, expire_on_commit=False)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.mask_report",
        description="Replay the media DB's clean copies and masks into the spine (media.v1).",
        allow_abbrev=False,
    )
    p.add_argument("--dry-run", action="store_true", help="print the calls; send nothing")
    p.add_argument("--spine-url", help="SpineMedia, http://host:port (default SPINE_MEDIA_URL)")
    p.add_argument("--batch-size", type=int, default=200, help="originals per lease (1-200)")
    p.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    p.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    p.add_argument("--max-attempts", type=int, default=5, help="per call, retries included")
    p.add_argument("--restart", action="store_true", help="abandon an unfinished run")
    p.add_argument("--sandbox", help="read DIR/masks.db (SQLite) instead of DATABASE_URL")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    url = args.spine_url or os.environ.get("SPINE_MEDIA_URL", "")
    try:
        config = ReplayConfig(
            batch_size=args.batch_size,
            lease_seconds=args.lease_seconds,
            dry_run=args.dry_run,
            restart=args.restart,
        )
        policy = RetryPolicy(max_attempts=args.max_attempts)
        if not args.deadline_seconds > 0:
            raise ValueError("--deadline-seconds must be above 0")
        if not args.dry_run:
            target_from_url(url)
    except ValueError as exc:
        print(f"mask_report: {exc}", file=sys.stderr)
        return EXIT_USAGE

    channels: list[grpc.Channel] = []

    def connect() -> SpineMediaClient:
        channels.append(open_channel(url))
        return SpineMediaClient(channels[-1], deadline_seconds=args.deadline_seconds, policy=policy)

    db = _open_db(args.sandbox)
    try:
        return run_replay(db, config, connect=connect, emit=_emit)
    finally:
        db.close()
        for channel in channels:
            channel.close()


if __name__ == "__main__":  # pragma: no cover - the module's entry point
    raise SystemExit(main())
