"""The boot guard (MT-24 M1): outside a dev environment the API refuses to start
with a dev or ungranted auth surface, or with a jwt_secret anyone could guess.

Each refusal has its own red case; the clean production case is the control
that keeps the guard from refusing everything. The last class starts the real
module in a subprocess, so "refuses to start" is observed, not inferred.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path

import pytest
from app.boot_guard import (
    BootRefused,
    boot_violations,
    enforce_boot_guard,
    mount_auth_routers,
)
from app.config import Settings
from app.routes.auth_routes import dev_router
from app.routes.auth_routes import router as auth_router
from fastapi import FastAPI
from pydantic import ValidationError

REPO = Path(__file__).resolve().parent.parent


def _prod(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "sqlite://",
        "environment": "production",
        "jwt_secret": secrets.token_urlsafe(48),
        "allow_dev_tokens": False,
        "auth_routes_granted": False,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _mounted(settings: Settings) -> FastAPI:
    app = FastAPI()
    mount_auth_routers(app, settings)
    return app


def _auth_paths(app: FastAPI) -> set[str]:
    return {p for p in (getattr(r, "path", "") for r in app.routes) if p.startswith("/auth/")}


class TestProductionRefusals:
    def test_a_clean_production_app_starts(self) -> None:
        settings = _prod()
        app = _mounted(settings)
        assert boot_violations(settings, app) == []
        enforce_boot_guard(settings, app)

    def test_allow_dev_tokens_refuses(self) -> None:
        settings = _prod(allow_dev_tokens=True)
        assert boot_violations(settings, FastAPI()) == ["ALLOW_DEV_TOKENS is true"]

    def test_the_config_default_secret_refuses(self) -> None:
        default = Settings.model_fields["jwt_secret"].default
        settings = _prod(jwt_secret=default)
        assert boot_violations(settings, FastAPI()) == ["JWT_SECRET is a known default"]

    def test_the_test_suites_secret_refuses(self) -> None:
        settings = _prod(jwt_secret="test-secret")
        assert boot_violations(settings, FastAPI()) == ["JWT_SECRET is a known default"]

    def test_a_secret_one_byte_short_of_hs256s_key_size_refuses(self) -> None:
        settings = _prod(jwt_secret=secrets.token_hex(16)[:31])
        assert boot_violations(settings, FastAPI()) == [
            "JWT_SECRET is shorter than 32 bytes (RFC 7518 3.2: an HS256 key is at least 256 bits)"
        ]

    def test_a_secret_of_exactly_32_bytes_starts(self) -> None:
        settings = _prod(jwt_secret=secrets.token_hex(16))
        assert boot_violations(settings, FastAPI()) == []

    def test_the_length_is_counted_in_bytes_not_characters(self) -> None:
        settings = _prod(jwt_secret="é" * 16)  # 16 characters, 32 bytes of UTF-8
        assert boot_violations(settings, FastAPI()) == []

    def test_the_dev_auth_router_mounted_refuses(self) -> None:
        settings = _prod()
        app = _mounted(_prod(environment="development"))
        assert "the dev auth router is mounted (/auth/dev-token)" in boot_violations(settings, app)

    def test_the_dev_auth_router_under_another_prefix_still_refuses(self) -> None:
        app = FastAPI()
        app.include_router(dev_router, prefix="/v1")
        app.include_router(auth_router, prefix="/v1")
        assert boot_violations(_prod(), app) == [
            "the dev auth router is mounted (/v1/auth/dev-token)",
            "the REST auth router is mounted (/v1/auth/refresh) but AUTH_ROUTES_GRANTED is false",
        ]

    def test_the_rest_auth_router_mounted_while_ungranted_refuses(self) -> None:
        settings = _prod(auth_routes_granted=False)
        app = _mounted(_prod(auth_routes_granted=True))
        assert boot_violations(settings, app) == [
            "the REST auth router is mounted (/auth/refresh) but AUTH_ROUTES_GRANTED is false"
        ]

    def test_the_rest_auth_router_mounted_while_granted_starts(self) -> None:
        settings = _prod(auth_routes_granted=True)
        assert boot_violations(settings, _mounted(settings)) == []

    def test_every_violation_is_named_in_one_refusal(self) -> None:
        settings = _prod(allow_dev_tokens=True, jwt_secret="short")
        app = _mounted(_prod(environment="test"))
        with pytest.raises(BootRefused) as refused:
            enforce_boot_guard(settings, app)
        message = str(refused.value)
        assert "refuses to start in environment 'production'" in message
        for reason in (
            "ALLOW_DEV_TOKENS is true",
            "JWT_SECRET is shorter than 32 bytes",
            "the dev auth router is mounted",
            "the REST auth router is mounted",
        ):
            assert reason in message

    def test_the_refusal_never_carries_the_secret(self) -> None:
        secret = "s3cr3t-" + secrets.token_hex(4)
        with pytest.raises(BootRefused) as refused:
            enforce_boot_guard(_prod(jwt_secret=secret), FastAPI())
        assert secret not in str(refused.value)


class TestDevEnvironmentsAreExempt:
    @pytest.mark.parametrize("environment", ["development", "test"])
    def test_a_dev_environment_starts_with_every_dev_affordance(self, environment: str) -> None:
        settings = _prod(environment=environment, allow_dev_tokens=True, jwt_secret="test-secret")
        app = _mounted(settings)
        assert boot_violations(settings, app) == []


class TestMounting:
    def test_production_mounts_no_auth_route_unless_granted(self) -> None:
        assert _auth_paths(_mounted(_prod())) == set()

    def test_a_granted_production_mounts_refresh_and_never_dev_token(self) -> None:
        assert _auth_paths(_mounted(_prod(auth_routes_granted=True))) == {"/auth/refresh"}

    @pytest.mark.parametrize("environment", ["development", "test"])
    def test_a_dev_environment_mounts_both(self, environment: str) -> None:
        paths = _auth_paths(_mounted(_prod(environment=environment)))
        assert paths == {"/auth/dev-token", "/auth/refresh"}


def _start(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Import the API module as uvicorn does, in a fresh interpreter with only `env` set."""
    base = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.main as m; "
            "print(sorted(r.path for r in m.app.routes if r.path.startswith('/auth/')))",
        ],
        cwd=REPO,
        env={**base, "DATABASE_URL": "sqlite://", **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestEnvironmentIsRequired:
    def test_settings_without_an_environment_do_not_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        with pytest.raises(ValidationError) as missing:
            Settings(_env_file=None)  # type: ignore[call-arg]
        assert "environment" in str(missing.value)
        assert "Field required" in str(missing.value)


class TestTheModuleRefusesToStart:
    def test_an_unset_environment_does_not_start_and_mounts_nothing(self) -> None:
        run = _start({"JWT_SECRET": secrets.token_urlsafe(48)})
        assert run.returncode != 0
        assert "environment" in run.stderr and "Field required" in run.stderr
        assert "/auth/" not in run.stdout

    def test_production_with_the_default_secret_does_not_start(self) -> None:
        run = _start({"ENVIRONMENT": "production"})
        assert run.returncode != 0
        assert "BootRefused" in run.stderr
        assert "JWT_SECRET is a known default" in run.stderr

    def test_production_with_dev_tokens_allowed_does_not_start(self) -> None:
        run = _start(
            {
                "ENVIRONMENT": "production",
                "JWT_SECRET": secrets.token_urlsafe(48),
                "ALLOW_DEV_TOKENS": "true",
            }
        )
        assert run.returncode != 0
        assert "ALLOW_DEV_TOKENS is true" in run.stderr

    def test_a_clean_production_starts_with_no_auth_route(self) -> None:
        run = _start({"ENVIRONMENT": "production", "JWT_SECRET": secrets.token_urlsafe(48)})
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == "[]"
