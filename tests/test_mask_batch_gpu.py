"""MASK-BATCH-1 on a real GPU: the whole batch against a LOCAL stack.

Runs only where ``MASK_BATCH_GPU_IMAGE`` names a built worker image
(``docker build -f Dockerfile.worker``) and docker can start GPU containers;
everywhere else (CI included) it skips. A disposable PostgreSQL 17 (pg-spine's
major) gets the media schema from this repo's alembic chain, a disposable
S3 server holds a raw bucket and a media bucket (SeaweedFS: the MinIO images
are no longer pullable from Docker Hub or quay.io without an account, as of
2026-10-14), and ``scripts/mask-batch.sh``
runs the driver in the worker image with ``--gpus all`` on this checkout's
``app/`` (``MASK_BATCH_APP_DIR``) over the IMG-SERVER-1 bench set: the nine
sample originals plus each upscaled to 1200 px wide (JPEG q90), eighteen
originals, the nine PNGs ranked as Ross's. Nothing leaves the box: synthetic
pictures, fixture URLs, local containers removed at the end.

``MASK_BATCH_GPU_REPORT=FILE`` also writes the measured numbers there.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import boto3
import pytest
from PIL import Image
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "tests" / "fixtures" / "mask_sample"
IMAGE = os.environ.get("MASK_BATCH_GPU_IMAGE", "")
PG_IMAGE = "postgres:17-alpine"
S3_IMAGE = "chrislusf/seaweedfs:4.48"
KEY, SECRET = "local-key", "local-secret"  # the local S3 server checks no credentials

#: The IMG-SERVER-1 CPU bench on this box (~/tmp/img-server-1/bench/SUMMARY.txt):
#: the base-keep pass, median wall seconds per image.
CPU_BASELINE = {"basekept-t2-noarena": 22.9, "basekept-t4": 13.1}

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker"),
    pytest.mark.skipif(not IMAGE, reason="set MASK_BATCH_GPU_IMAGE to a built worker image"),
]


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True, timeout=300
    ).stdout.strip()


def _port(name: str, inner: str) -> str:
    return _docker("port", name, inner).splitlines()[0].rsplit(":", 1)[-1]


@pytest.fixture(scope="module")
def stack() -> Generator[dict[str, str], None, None]:
    tag = uuid.uuid4().hex[:8]
    pg, s3_name = f"mm-maskbatch-pg-{tag}", f"mm-maskbatch-s3-{tag}"
    try:
        _docker(
            "run", "-d", "--rm", "--name", pg, "-e", "POSTGRES_PASSWORD=postgres",
            "-p", "127.0.0.1::5432", PG_IMAGE,
        )  # fmt: skip
        _docker(
            "run", "-d", "--rm", "--name", s3_name, "-p", "127.0.0.1::8333", S3_IMAGE, "server", "-s3",
        )  # fmt: skip
        admin = f"postgresql+psycopg2://postgres:postgres@127.0.0.1:{_port(pg, '5432/tcp')}"
        engine = create_engine(f"{admin}/postgres", isolation_level="AUTOCOMMIT")
        deadline = time.monotonic() + 60
        while True:
            try:
                with engine.connect() as conn:
                    for db in ("media", "media_cpu"):
                        conn.execute(text(f"CREATE DATABASE {db}"))
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        engine.dispose()
        for db in ("media", "media_cpu"):
            subprocess.run(
                ["alembic", "upgrade", "head"],
                cwd=REPO,
                env={**os.environ, "DATABASE_URL": f"{admin}/{db}"},
                check=True,
                capture_output=True,
                timeout=120,
            )
        s3_url = f"http://127.0.0.1:{_port(s3_name, '8333/tcp')}"
        s3 = boto3.client(
            "s3", endpoint_url=s3_url, aws_access_key_id=KEY, aws_secret_access_key=SECRET,
            region_name="us-east-1",
        )  # fmt: skip
        deadline = time.monotonic() + 60
        while True:
            try:
                for bucket in ("raw", "media", "media-cpu"):
                    s3.create_bucket(Bucket=bucket)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        yield {"admin": admin, "s3": s3_url}
    finally:
        subprocess.run(["docker", "rm", "-f", "-v", pg, s3_name], capture_output=True)


def _bench_set() -> list[tuple[str, bytes, dict[str, Any], int]]:
    """``(name, original bytes, origin, rank)``: the 18 bench originals."""
    manifest = json.loads((SAMPLE / "manifest.json").read_text())
    out: list[tuple[str, bytes, dict[str, Any], int]] = []
    for entry in manifest["entries"]:
        raw = (SAMPLE / "originals" / f"{entry['name']}.png").read_bytes()
        origin = entry["origins"][0]
        out.append((entry["name"], raw, origin, 0))
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        big = im.resize((1200, int(1200 * im.height / im.width)), Image.BICUBIC)
        buf = io.BytesIO()
        big.save(buf, format="JPEG", quality=90)
        out.append((f"{entry['name']}@1200w", buf.getvalue(), origin, 9))
    return out


def _upload_worklist(stack: dict[str, str], state: Path) -> list[dict[str, Any]]:
    s3 = boto3.client(
        "s3", endpoint_url=stack["s3"], aws_access_key_id=KEY, aws_secret_access_key=SECRET,
        region_name="us-east-1",
    )  # fmt: skip
    rows = []
    for n, (name, data, origin, rank) in enumerate(_bench_set()):
        sha = hashlib.sha256(data).hexdigest()
        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        key = f"raw-img/sha256/{sha[:2]}/{sha}.{ext}"
        s3.put_object(Bucket="raw", Key=key, Body=data)
        rows.append(
            {
                "blob_sha256": sha,
                "storage_key": key,
                "codec": "identity",
                "url": origin["url"],
                "source": origin["source"],
                "product_id": f"00000000-0000-4000-8000-{n:012d}",
                "head_id": None,
                "rank": rank,
                "name": name,
            }
        )
    state.mkdir(parents=True, exist_ok=True)
    (state / "worklist.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _batch(
    stack: dict[str, str], state: Path, db: str, bucket: str, *args: str, **env: str
) -> tuple[int, list[dict[str, Any]]]:
    full = {
        **os.environ,
        "MASK_BATCH_IMAGE": IMAGE,
        "MASK_BATCH_APP_DIR": str(REPO / "app"),
        "MASK_BATCH_NAME": f"mm-maskbatch-run-{uuid.uuid4().hex[:8]}",
        "DATABASE_URL": f"{stack['admin']}/{db}",
        "S3_ENDPOINT_URL": stack["s3"],
        "S3_BUCKET": bucket,
        "S3_ACCESS_KEY": KEY,
        "S3_SECRET_KEY": SECRET,
        "S3_SECURE": "false",
        "RAW_S3_ENDPOINT_URL": stack["s3"],
        "RAW_S3_BUCKET": "raw",
        "RAW_S3_ACCESS_KEY": KEY,
        "RAW_S3_SECRET_KEY": SECRET,
        **env,
    }
    done = subprocess.run(
        ["bash", str(REPO / "scripts" / "mask-batch.sh"), str(state), "--worklist",
         "/state/worklist.jsonl", "--no-residential-check", "--busy-windows", "", *args],
        env=full, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    events = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    (state / f"run-{len(list(state.glob('run-*')))}.err").write_text(done.stderr)
    return done.returncode, events


def _of(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event"] == name]


def test_the_bench_set_on_this_gpu(stack: dict[str, str], tmp_path: Path) -> None:
    state = tmp_path / "gpu"
    rows = _upload_worklist(stack, state)
    code, events = _batch(
        stack, state, "media", "media", "--chunk-size", "6", "--max-wait-seconds", "900"
    )
    assert code == 0, (state / "run-0.err").read_text()[-3000:]
    chunks = _of(events, "chunk")
    assert [c["count"] for c in chunks] == [6, 6, 6]
    assert set(chunks[0]["ranks"]) == {0} and set(chunks[2]["ranks"]) == {9}  # Ross first
    assert all(c["provider"] == "CUDAExecutionProvider" for c in chunks)
    assert all(isinstance(c["peak_vram_mib"], int) for c in chunks)
    done = _of(events, "done")[0]
    counts = done["counts"]
    # spike-collector-photo (a collector's photo) and miku-nendo (an unregistered
    # store) get no mask, at both sizes; every other original gets one.
    assert counts.get("ineligible") == 4 and counts.get("failed") is None
    assert counts.get("ok", 0) + counts.get("held", 0) == 14

    engine = create_engine(f"{stack['admin']}/media")
    with engine.connect() as conn:
        statuses = dict(
            conn.execute(text("SELECT status, count(*) FROM image_masks GROUP BY 1")).all()
        )
        published = conn.execute(
            text("SELECT count(*) FROM public_objects WHERE kind = 'mask'")
        ).scalar_one()
    engine.dispose()
    assert sum(statuses.values()) == 18 and statuses.get("ineligible") == 4
    assert published == statuses.get("ok", 0)

    # a re-run with the same progress file has nothing to do...
    code2, again = _batch(stack, state, "media", "media")
    assert code2 == 0 and _of(again, "start")[0]["pending"] == 0
    # ...and without it the database says every image is settled
    (state / "progress.jsonl").rename(state / "progress-first.jsonl")
    code3, third = _batch(stack, state, "media", "media", "--chunk-size", "18")
    assert code3 == 0
    assert _of(third, "chunk")[0]["counts"] == {"already": 18}

    minted = [c for c in chunks if c["seconds_per_image"] is not None]
    report = {
        "image": IMAGE,
        "chunks": [
            {
                k: c[k]
                for k in (
                    "chunk",
                    "count",
                    "counts",
                    "seconds_per_image",
                    "median_seconds",
                    "mask_seconds_per_image",
                    "peak_vram_mib",
                    "baseline_vram_mib",
                    "provider",
                    "elapsed_seconds",
                )
            }
            for c in chunks
        ],  # fmt: skip
        "counts": counts,
        "seconds_per_image": round(
            sum(c["seconds_per_image"] * c["count"] for c in minted)
            / sum(c["count"] for c in minted),
            3,
        ),
        "peak_vram_mib": max(c["peak_vram_mib"] for c in chunks),
        "cpu_baseline_seconds_per_image": CPU_BASELINE,
        "originals": len(rows),
    }
    if os.environ.get("MASK_BATCH_GPU_REPORT"):
        Path(os.environ["MASK_BATCH_GPU_REPORT"]).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def test_without_a_gpu_auto_falls_back_to_the_cpu_and_cuda_waits(
    stack: dict[str, str], tmp_path: Path
) -> None:
    state = tmp_path / "cpu"
    _upload_worklist(stack, state)
    # --device cuda (the default) never runs without a GPU: its gate cannot read one
    waited, events = _batch(
        stack, state, "media_cpu", "media-cpu", "--limit", "1", "--max-wait-seconds", "0",
        MASK_BATCH_GPUS="none",
    )  # fmt: skip
    assert waited == 4
    assert _of(events, "gave_up")[0]["gate"] == "gpu"
    assert _of(events, "chunk") == []
    code, events = _batch(
        stack, state, "media_cpu", "media-cpu", "--device", "auto", "--cpu-threads", "4",
        "--limit", "2", "--chunk-size", "2", MASK_BATCH_GPUS="none", MASK_BATCH_MEMORY="10g",
    )  # fmt: skip
    assert code == 0, (state / "run-1.err").read_text()[-3000:]
    (chunk,) = _of(events, "chunk")
    assert chunk["provider"] == "CPUExecutionProvider"
    assert chunk["counts"].get("ok", 0) + chunk["counts"].get("held", 0) == 2
    if os.environ.get("MASK_BATCH_GPU_REPORT"):
        path = Path(os.environ["MASK_BATCH_GPU_REPORT"]).with_suffix(".cpu.json")
        path.write_text(json.dumps(chunk, indent=1))
    print(json.dumps(chunk))
