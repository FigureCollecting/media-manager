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
    # REQUIRED, no default: a deployment that forgets ENVIRONMENT must not boot as
    # development (dev routes mounted, app.boot_guard skipped). Unset, Settings()
    # fails with "environment  Field required".
    environment: Literal["development", "production", "test"]
    log_level: str = "INFO"
    allow_dev_tokens: bool = False
    # Whether the deployment grants a caller the REST /auth routes (token refresh).
    # Outside development and test they are mounted only when this is true, and
    # the API refuses to start if they are mounted while it is false
    # (app.boot_guard, which also refuses dev tokens and a guessable jwt_secret).
    auth_routes_granted: bool = False

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

    # MASK-1 store masks (app.workers.mask_policy). MASK-G, Ross 2026-10-12
    # (option b): cut-out masks for EVERY store's listing photos, revocable
    # per store. MASK_STORE_GRANTS registers the store sources: per source,
    # the domains (subdomains included) its listing photos come from -- the
    # spine's source.domains for that retailer. Every registered source is
    # granted; there is no per-store opt-in. Env as a JSON object, e.g.
    # {"amiami": ["amiami.com"]}. Empty here: no store's images reach
    # media-manager yet (gallery ingest records source "mfc" only).
    mask_store_grants: dict[str, list[str]] = {}
    # The per-source deny list that wins over any grant (MFC included):
    # MASK_STORE_REVOKED as a JSON list, e.g. '["shop"]' (a bare or
    # comma-separated value fails to load), plus what `masks revoke-store`
    # persists at run time (mask_store_revocations). README runbook "A store
    # asks for its cut-outs to be removed".
    mask_store_revoked: list[str] = []

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
    # Where those sessions run (app.workers.matting.MATTING_DEVICES): "cuda"
    # requires the CUDA execution provider and refuses a session without it;
    # "cpu" is an explicit lean CPU session (no CPU arena, no memory pattern);
    # "auto" asks for CUDA and falls back to that CPU session with a warning.
    matting_device: Literal["cuda", "auto", "cpu"] = "cuda"
    # A CPU session's intra-op threads; unset = ONNX Runtime's default.
    matting_cpu_threads: int | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # environment comes from the env, never a default
