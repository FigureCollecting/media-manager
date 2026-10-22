"""The API's boot guard (MT-24 M1): outside a dev environment it refuses to start
with a dev or ungranted auth surface, or with a jwt_secret anyone could guess.

Why it matters: an HS256 jwt_secret is a mint-anything key. Whoever holds it can
sign `ten=<any tenant>`, any `sub`, or `sub=service:x`, which every route trusts,
so a guessable secret opens every tenant. The dev-token route mints tokens on
request, and the refresh route renews any leaked refresh token with no revocation.
The guard runs when app.main is imported, before uvicorn binds the port, so
`--lifespan off` cannot skip it; a refusal names every reason at once and never
the secret.

"Non-dev" is every environment except development and test, so a new value
fails closed. MM-TENANCY-1 (asymmetric, tenant-pinned tokens) replaces the
HS256 scheme itself; this only stops the API booting in a state that is unsafe
under it.
"""

from __future__ import annotations

from fastapi import FastAPI

from .config import Settings
from .routes.auth_routes import dev_router as dev_auth_router
from .routes.auth_routes import router as auth_router

DEV_ENVIRONMENTS = frozenset({"development", "test"})

# The config default, and the test suite's value: both are in this repository.
KNOWN_DEFAULT_SECRETS = frozenset({str(Settings.model_fields["jwt_secret"].default), "test-secret"})

# RFC 7518 3.2: an HS256 key must be at least as long as the hash, 256 bits.
MIN_SECRET_BYTES = 32


class BootRefused(RuntimeError):
    """The API must not start with this configuration."""


def is_dev_environment(settings: Settings) -> bool:
    return settings.environment in DEV_ENVIRONMENTS


# By endpoint, not path: a router mounted under another prefix is still itself.
DEV_AUTH_ENDPOINTS = frozenset(getattr(r, "endpoint", None) for r in dev_auth_router.routes)
REST_AUTH_ENDPOINTS = frozenset(getattr(r, "endpoint", None) for r in auth_router.routes)


def _mounted(app: FastAPI, endpoints: frozenset[object]) -> list[str]:
    return sorted(
        getattr(r, "path", "") for r in app.routes if getattr(r, "endpoint", None) in endpoints
    )


def mount_auth_routers(app: FastAPI, settings: Settings) -> None:
    """Mount /auth/dev-token in a dev environment only, and /auth/refresh in a dev
    environment or where the deployment grants it (AUTH_ROUTES_GRANTED)."""
    if is_dev_environment(settings):
        app.include_router(dev_auth_router)
    if is_dev_environment(settings) or settings.auth_routes_granted:
        app.include_router(auth_router)


def boot_violations(settings: Settings, app: FastAPI) -> list[str]:
    """Every reason this configuration must not start, in a fixed order; empty in a
    dev environment."""
    if is_dev_environment(settings):
        return []
    found: list[str] = []
    if settings.allow_dev_tokens:
        found.append("ALLOW_DEV_TOKENS is true")
    if settings.jwt_secret in KNOWN_DEFAULT_SECRETS:
        found.append("JWT_SECRET is a known default")
    elif len(settings.jwt_secret.encode("utf-8")) < MIN_SECRET_BYTES:
        found.append(
            f"JWT_SECRET is shorter than {MIN_SECRET_BYTES} bytes"
            " (RFC 7518 3.2: an HS256 key is at least 256 bits)"
        )
    if dev := _mounted(app, DEV_AUTH_ENDPOINTS):
        found.append(f"the dev auth router is mounted ({', '.join(dev)})")
    if (rest := _mounted(app, REST_AUTH_ENDPOINTS)) and not settings.auth_routes_granted:
        found.append(
            f"the REST auth router is mounted ({', '.join(rest)}) but AUTH_ROUTES_GRANTED is false"
        )
    return found


def enforce_boot_guard(settings: Settings, app: FastAPI) -> None:
    if found := boot_violations(settings, app):
        raise BootRefused(
            f"media-manager refuses to start in environment {settings.environment!r}: "
            + "; ".join(found)
        )
