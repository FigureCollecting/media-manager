"""The worker's real Celery broker transport must load against the installed redis-py.

Unit tests use fakeredis, so a kombu release whose Redis transport demands a newer
redis-py than we pin (kombu 5.7.0 wants >= 6.1.0) only failed at runtime. Opening the
transport's channel raises VersionMismatch before any socket is touched; the PING that
follows is stubbed, so no Redis is needed.
"""

from __future__ import annotations

from unittest.mock import patch

from app.workers import celery_app


def test_celery_broker_channel_opens_without_version_mismatch() -> None:
    with patch("redis.Redis.ping", return_value=True), celery_app.connection_for_write() as conn:
        channel = conn.default_channel
        assert channel.Client is not None
