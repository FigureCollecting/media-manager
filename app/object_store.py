"""Where the derivative lane keeps bytes: an S3/MinIO bucket in a deployment,
a local directory for the sandboxed sample run (``python -m app.cli.masks
--sandbox``) and the tests.

Public objects (clean derivatives, masks) are CONTENT-ADDRESSED: their key is
derived from the sha-256 of the bytes, so the same bytes always land at the
same key and a duplicate write is a no-op in effect.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any, Protocol

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

_EXT_FOR_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


class ObjectStore(Protocol):
    def get(self, key: str) -> bytes: ...

    def put(self, key: str, data: bytes, content_type: str) -> None: ...


def derived_key(sha256: str, mime: str) -> str:
    """``derived/<aa>/<sha>.<ext>`` for one of our own public objects."""
    if not _SHA256_HEX.match(sha256):
        raise ValueError("a derived key needs a lowercase hex sha-256")
    return f"derived/{sha256[:2]}/{sha256}.{_EXT_FOR_MIME.get(mime, 'bin')}"


class S3ObjectStore:
    """The deployment store: the configured bucket on S3/MinIO, written
    private (the public origin streams bytes; the bucket is never public)."""

    def __init__(self, *, client: Any | None = None, bucket: str | None = None) -> None:
        if client is None or bucket is None:
            from .config import get_settings
            from .s3 import get_s3

            client = client or get_s3()
            bucket = bucket or get_settings().s3_bucket
        self._client = client
        self._bucket = bucket

    def get(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        data: bytes = obj["Body"].read()
        return data

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type, ACL="private"
        )


class LocalObjectStore:
    """A directory of objects keyed by relative path. A key that would leave
    the root (absolute, ``..``, empty) is refused."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        norm = posixpath.normpath(key) if key else ""
        if not norm or norm.startswith(("/", "..")) or norm == ".":
            raise ValueError(f"object key outside the store: {key!r}")
        return self._root / norm

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def keys(self) -> set[str]:
        if not self._root.exists():
            return set()
        return {p.relative_to(self._root).as_posix() for p in self._root.rglob("*") if p.is_file()}
