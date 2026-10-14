"""MASK-BATCH-1: scripts/mask-batch.sh, the thin wrapper that runs the driver in
the CUDA worker image. A fake ``docker`` on PATH records the command line: the
GPU, the host network (the operator's port-forward is on 127.0.0.1), the state
mount, configuration passed by NAME only (no secret value on any argv), and
the driver's exit status passed through."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mask-batch.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

SECRETS = {
    "DATABASE_URL": "postgresql+psycopg2://media@127.0.0.1:15432/media",
    "PGPASSWORD": "pg-secret-value",
    "S3_SECRET_KEY": "media-secret-value",
    "RAW_S3_SECRET_KEY": "raw-secret-value",
    "MASK_BATCH_RESIDENTIAL_IP": "198.51.100.7",
}


def _fake_docker(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$FAKE_DOCKER_ARGV"\n'
        'echo \'{"event": "done"}\'\n'
        'exit "${FAKE_DOCKER_RC:-0}"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run(
    tmp_path: Path, *args: str, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = _fake_docker(tmp_path)
    argv_file = tmp_path / "argv.txt"
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MASK_BATCH", "S3_", "RAW_S3", "PG"))
    }
    base.pop("DATABASE_URL", None)
    full = {
        **base,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DOCKER_ARGV": str(argv_file),
        **env,
    }
    done = subprocess.run(
        ["bash", str(SCRIPT), *args], env=full, capture_output=True, text=True, timeout=30
    )
    argv = argv_file.read_text().splitlines() if argv_file.exists() else []
    return done, argv


def test_the_image_is_required(tmp_path: Path) -> None:
    done, argv = _run(tmp_path, str(tmp_path / "state"))
    assert done.returncode != 0
    assert "MASK_BATCH_IMAGE" in done.stderr
    assert argv == []


def test_help_prints_the_usage(tmp_path: Path) -> None:
    done, _ = _run(tmp_path, "--help")
    assert done.returncode == 2
    assert "STATE_DIR" in done.stdout


def test_the_run_is_on_the_gpu_with_names_not_values(tmp_path: Path) -> None:
    state = tmp_path / "state"
    ca = tmp_path / "ca.crt"
    ca.write_text("x")
    done, argv = _run(
        tmp_path,
        str(state),
        "--worklist",
        "/state/ross.jsonl",
        "--dry-run",
        MASK_BATCH_IMAGE="worker@sha256:abc",
        PGSSLROOTCERT=str(ca),
        **SECRETS,
    )
    assert done.returncode == 0, done.stderr
    assert argv[0] == "run"
    joined = " ".join(argv)
    assert "--gpus all" in joined and "--network host" in joined and "--init" in joined
    assert "--name mask-batch" in joined
    assert "PYTHONPATH=/app" in argv  # the image's code, run from /state
    assert f"{state}:/state" in argv
    assert f"{ca}:/run/mask-batch/pg-ca.crt:ro" in argv
    assert "PGSSLROOTCERT=/run/mask-batch/pg-ca.crt" in argv
    for name, value in SECRETS.items():
        assert name in argv  # passed by name...
        assert value not in joined  # ...never by value
    assert "S3_ACCESS_KEY" not in argv  # an unset variable is not passed
    tail = argv[argv.index("worker@sha256:abc") :]
    assert tail == [
        "worker@sha256:abc",
        "python",
        "-m",
        "app.cli.mask_batch",
        "run",
        "--progress",
        "/state/progress.jsonl",
        "--worklist",
        "/state/ross.jsonl",
        "--dry-run",
    ]
    assert (state / "batch.log").read_text() == '{"event": "done"}\n'


def test_the_drivers_exit_status_comes_back(tmp_path: Path) -> None:
    done, _ = _run(tmp_path, str(tmp_path / "s"), MASK_BATCH_IMAGE="w", FAKE_DOCKER_RC="3")
    assert done.returncode == 3


def test_a_cpu_run_and_a_test_mount_are_opt_in(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    done, argv = _run(
        tmp_path,
        str(tmp_path / "s"),
        MASK_BATCH_IMAGE="w",
        MASK_BATCH_GPUS="none",
        MASK_BATCH_APP_DIR=str(app_dir),
        MASK_BATCH_NAME="mask-batch-test",
    )
    assert done.returncode == 0
    assert "--gpus" not in argv
    assert f"{app_dir}:/app/app:ro" in argv
    assert "mask-batch-test" in argv
