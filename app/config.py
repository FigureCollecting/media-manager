from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # App
    app_name: str = "media-manager"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    allow_dev_tokens: bool = False

    # Security
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    token_exp_minutes: int = 60
    refresh_token_exp_minutes: int = 60 * 24 * 7

    # Data stores
    database_url: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/media_manager"
    redis_url: str = "redis://redis:6379/0"

    # S3/MinIO
    s3_endpoint_url: str | None = "http://minio:9000"
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "images"
    s3_secure: bool = False
    s3_presign_expiry_post: int = 15 * 60
    s3_presign_expiry_get: int = 10 * 60

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5051",
        "https://figurecollecting.com",
    ]

    # The public derivative origin (app.derivative_main): the app origins
    # that may read /d/<sha256> bytes cross-origin (the service worker caches
    # them; the case view composites a mask over its derivative). fc-infra
    # edge README "The public image address (ADDR-1)". Env MEDIA_CORS_ORIGINS
    # as a JSON list; a canary origin is added by its deployment's env, never
    # by default.
    media_cors_origins: list[str] = ["https://figurecollecting.com"]

    # MASK-1 per-source mask grants (app.workers.mask_policy): per source, the
    # domains (subdomains included) whose listing photos may get a mask. Env
    # MASK_STORE_GRANTS as a JSON object, e.g. {"amiami": ["amiami.com"]}.
    # None by default: MFC item-main images are the only maskable origin
    # until a store is granted by name.
    mask_store_grants: dict[str, list[str]] = {}

    # Rate limiting
    rate_limit_per_minute: int = 120

    # DB connection pool
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800  # 30 minutes

    # Matting pipeline: "stub" (default, no model) or "birefnet" (real
    # subject segmentation -- see app.workers.matting.BiRefNetMattingBackend).
    matting_backend: str = "stub"
    # Path to the BiRefNet ONNX weights baked into the worker Docker image
    # (see Dockerfile.worker). Only consulted when matting_backend is
    # "birefnet"; unset in dev/tests, where the stub backend is used.
    matting_model_path: str | None = None
    # "birefnet_basekept" (the July base-keep pass, app.workers.basekept)
    # also needs the two IS-Net base donors' ONNX weights.
    matting_isnet_general_path: str | None = None
    matting_isnet_anime_path: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
