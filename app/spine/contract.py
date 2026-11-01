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
# The branch head until fc-ingest-contract squash-merges 0.12.0; then this line
# alone moves to the merge commit (the bytes, and PROTO_SHA256, stay the same).
PROTO_COMMIT = "317271207a28bec1c5872ab8e481085d996670aa"
PROTO_SHA256 = "c5cc34de2d1f4d87d2d59350be6bf3bc86e12ecf5093cf817633da14c1ecfd5c"


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
