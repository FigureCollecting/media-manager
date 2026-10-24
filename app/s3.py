import logging
from functools import lru_cache
from typing import Any, cast

import boto3
from botocore.client import Config

from .config import get_settings

logger = logging.getLogger(__name__)


def s3_client(*, max_pool_connections: int = 10, **config: Any) -> Any:
    """A boto3 S3 client for the configured bucket, keeping up to
    ``max_pool_connections`` connections (botocore's default: 10); ``config``
    adds botocore Config options (the mask batch's timeouts and retries)."""
    s = get_settings()
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        region_name=s.s3_region,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        config=Config(
            s3={"addressing_style": "virtual"},
            signature_version="s3v4",
            max_pool_connections=max_pool_connections,
            **config,
        ),
        use_ssl=s.s3_secure,
        verify=s.s3_secure,
    )


@lru_cache(maxsize=1)
def get_s3() -> Any:
    """Return a cached boto3 S3 client singleton."""
    return s3_client()


def presign_post_for_upload(key: str, *, content_type: str, size: int) -> dict[str, Any]:
    s = get_settings()
    client = get_s3()
    conditions = [
        ["content-length-range", max(1, size // 100), size * 2],  # basic sanity range
        {"Content-Type": content_type},
        {"acl": "private"},
    ]
    fields = {"Content-Type": content_type, "acl": "private"}
    return cast(
        dict[str, Any],
        client.generate_presigned_post(
            Bucket=s.s3_bucket,
            Key=key,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=s.s3_presign_expiry_post,
        ),
    )


def presign_get(key: str, *, expires_in: int | None = None) -> str:
    s = get_settings()
    client = get_s3()
    return cast(
        str,
        client.generate_presigned_url(
            "get_object",
            Params={"Bucket": s.s3_bucket, "Key": key},
            ExpiresIn=expires_in or s.s3_presign_expiry_get,
        ),
    )


def move_object(src_key: str, dest_key: str) -> None:
    # A self-move would copy the object onto itself and then DELETE it,
    # destroying the only copy (final keys are content-addressed -- there is
    # no second copy to fall back on). Nothing to do: the bytes are already
    # at their destination.
    if src_key == dest_key:
        return
    s = get_settings()
    client = get_s3()
    client.copy({"Bucket": s.s3_bucket, "Key": src_key}, s.s3_bucket, dest_key)
    client.delete_object(Bucket=s.s3_bucket, Key=src_key)
