"""The vendored media.v1 contract, pinned.

``proto/media/v1/media.proto`` is fc-ingest-contract's file at tag
``PROTO_TAG`` (commit ``PROTO_COMMIT``), byte for byte. The file imports
nothing, so it is vendored alone, and the build generates the Python stubs
from it (``scripts/gen_spine_stubs.py``) only after its sha-256 matches
``PROTO_SHA256``: a hand edit, or a copy from another tag, stops the build
instead of shipping a client for a contract nobody published. To move to a
new contract version, copy the tagged file and update all three pins.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROTO_PATH = "proto/media/v1/media.proto"
PROTO_TAG = "v0.12.0"
PROTO_COMMIT = "9d88b4b27ed15dad992b718f17d8df7e1545f35e"
PROTO_SHA256 = "4cfab26ba87ba29aec815eb92867df6005c7d82777588c455fd048d09fbc7b6e"


class ProtoMismatch(RuntimeError):
    """The vendored proto is missing or is not the pinned file."""


def verify_vendored_proto(root: Path) -> bytes:
    """The vendored proto's bytes under ``root``, once they match the pin."""
    path = root / PROTO_PATH
    if not path.is_file():
        raise ProtoMismatch(f"{PROTO_PATH} is missing under {root}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PROTO_SHA256:
        raise ProtoMismatch(
            f"{PROTO_PATH} has sha-256 {digest}, not the pinned {PROTO_SHA256} "
            f"(fc-ingest-contract {PROTO_TAG})"
        )
    return data
