"""The raw bucket as a reader of captured originals sees it, and what a
failure reading or writing a bucket says about whose fault it is.

Shared by the WSL mask batch (app.cli.mask_batch) and the cluster
display-copy lane (app.workers.display_copy), so both read an original the
same way: from the raw bucket, read only, never from where it was captured
(the lane and the batch never contact a store), its stored object under the
byte cap, a gzip one inflated only up to it.
"""

from __future__ import annotations

import os
from typing import Any

from ..object_store import ObjectStore, S3ObjectStore
from .image_limits import ImageLimitError, current_limits, gunzip_capped

#: The codecs an original may be stored under (the spine's raw.blob.codec):
#: an asset capture is stored as it came, a page as gzip.
KNOWN_CODECS = ("identity", "gzip")

#: The exception classes (by name, anywhere in the class's ancestry) of an
#: OUTAGE -- the network, the database or the bucket -- not of the image: the
#: builtin and botocore ConnectionError, timeouts, SQLAlchemy's
#: OperationalError and InterfaceError (a dropped connection), botocore's
#: HTTPClientError (a connection closed mid-response).
OUTAGE_CLASSES = frozenset(
    {"ConnectionError", "TimeoutError", "Timeout", "OperationalError", "InterfaceError",
     "HTTPClientError"}
)  # fmt: skip
#: The same, as the mask lane records a failure: its class name alone.
OUTAGE_REASONS = OUTAGE_CLASSES | {
    "ConnectionResetError", "ConnectionRefusedError", "ConnectionAbortedError",
    "BrokenPipeError", "EndpointConnectionError", "ConnectionClosedError",
    "ConnectTimeoutError", "ReadTimeoutError", "ProxyConnectionError",
}  # fmt: skip

#: The bucket's error codes that say the CREDENTIAL or the configuration is
#: wrong (a key that is not one, expired, without the permission, a clock the
#: signature is refused for, a bucket that is not there): never the image's
#: fault, and the same for every image, so a run stops on the first one.
CONFIGURATION_CODES = frozenset(
    {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "RequestTimeTooSkewed",
     "ExpiredToken", "InvalidToken", "TokenRefreshRequired", "AllAccessDisabled",
     "NoSuchBucket", "AuthorizationHeaderMalformed", "InvalidBucketName"}
)  # fmt: skip


class ReadOnlyStore:
    """The raw bucket as a reader sees it: it reads originals, never writes."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    def get(self, key: str) -> bytes:
        return self._store.get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        raise PermissionError("the raw bucket is read only to this reader")


def decode_stored(stored: bytes, codec: str) -> bytes:
    """An original's bytes from the raw bucket's object: the stored object under the byte cap (a refusal before
    anything opens it) and a gzip one inflated only up to that cap (MM-CAPS-1: a bomb stops at the ceiling)."""
    cap = current_limits().max_bytes
    if codec == "gzip":
        return gunzip_capped(stored, cap)
    if len(stored) > cap:
        raise ImageLimitError("too_large_bytes", f"{len(stored)} bytes exceeds the {cap}-byte cap")
    return stored


def _status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)  # botocore's ClientError
    if isinstance(response, dict):
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    return None


def bucket_error_code(exc: BaseException) -> str | None:
    """The bucket's own error code (``NoSuchKey``, ``AccessDenied``, ...) of a
    botocore ClientError; None for anything else."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        return code if isinstance(code, str) and code else None
    return None


def is_outage(exc: BaseException) -> bool:
    """An outage-class failure (``OUTAGE_CLASSES``, or the bucket answering
    5xx): the image is not at fault, a rerun should try it again."""
    if any(cls.__name__ in OUTAGE_CLASSES for cls in type(exc).__mro__):
        return True
    status = _status(exc)
    return status is not None and status >= 500


def is_outage_reason(reason: str | None) -> bool:
    return reason in OUTAGE_REASONS


def is_configuration_error(exc: BaseException) -> bool:
    """A bucket refusal that says the credential or the configuration is
    wrong (``CONFIGURATION_CODES``): the same for every image."""
    return bucket_error_code(exc) in CONFIGURATION_CODES


class RawStoreConfigError(ValueError):
    """The raw bucket is not configured: nothing was read."""


def raw_store_from_env(connections: int, client_config: dict[str, Any]) -> ReadOnlyStore:
    """The raw bucket from RAW_S3_ENDPOINT_URL, RAW_S3_BUCKET,
    RAW_S3_ACCESS_KEY and RAW_S3_SECRET_KEY (RAW_S3_REGION optional), read
    only; its client keeps ``connections`` and takes ``client_config``
    (botocore Config options)."""
    names = ("RAW_S3_ENDPOINT_URL", "RAW_S3_BUCKET", "RAW_S3_ACCESS_KEY", "RAW_S3_SECRET_KEY")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RawStoreConfigError(f"the raw bucket needs {', '.join(missing)}")
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["RAW_S3_ENDPOINT_URL"],
        region_name=os.environ.get("RAW_S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["RAW_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RAW_S3_SECRET_KEY"],
        config=Config(max_pool_connections=connections, **client_config),
    )
    return ReadOnlyStore(S3ObjectStore(client=client, bucket=os.environ["RAW_S3_BUCKET"]))
