"""Generate the media.v1 Python stubs from the vendored, pinned proto.

    python scripts/gen_spine_stubs.py [REPO_ROOT]

Writes ``app/spine/media_v1/media_pb2.py``, ``media_pb2.pyi`` and
``media_pb2_grpc.py`` (git-ignored). Refuses, writing nothing, unless the
vendored proto matches its pin (app.spine.contract). Needs grpcio-tools (a
dev dependency; the images run it in a build stage).

The proto is staged at ``app/spine/media_v1/media.proto`` in a scratch
include root, so protoc emits package-relative imports
(``from app.spine.media_v1 import media_pb2``). The descriptor's file name
changes with it; the package, and so every wire name
(``/media.v1.SpineMedia/...``), does not.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent)
    sys.path.insert(0, str(root))
    from app.spine.contract import ProtoMismatch, verify_vendored_proto
    from grpc_tools import protoc

    try:
        data = verify_vendored_proto(root)
    except ProtoMismatch as exc:
        print(f"gen_spine_stubs: {exc}", file=sys.stderr)
        return 2
    out = root / "app" / "spine" / "media_v1"
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "app" / "spine" / "media_v1" / "media.proto"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(data)
        rc = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{scratch}",
                f"--python_out={root}",
                f"--pyi_out={root}",
                f"--grpc_python_out={root}",
                str(staged),
            ]
        )
    if rc != 0:
        print(f"gen_spine_stubs: protoc exited {rc}", file=sys.stderr)
        return 1
    print(f"gen_spine_stubs: wrote {sorted(p.name for p in out.glob('media_pb2*'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
