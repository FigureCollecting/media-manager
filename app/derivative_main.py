"""The public derivative origin: ``GET``/``HEAD /d/<64 lowercase hex>``.

Its contract is fc-infra ``nodes/fc-app-01/edge/README.md``, "The public image
address (ADDR-1)": ``MEDIA_PUBLIC_BASE_URL = https://images.figurecollecting.com/d``
names every derivative and every mask a client is sent, and this app is what
answers it.

A SEPARATE ASGI APP, NEVER ``app.main`` (image track I11): ``app.main`` mounts
every router -- auth, images, the integer-id serve routes, external refs,
albums -- and none of them may be reachable through the public address.
This app mounts exactly ``/healthz`` and ``/d/{sha256}``; anything else,
including any other method on ``/d/``, is a bodiless 404.

THE ADDRESS IS THE CAPABILITY. Only rows of ``public_objects`` are served --
our clean derivatives and our masks -- and an original's bytes are never
registered, so an original, a held mask or any other stored object has no
address here. The one overlap is an original whose bytes ARE one of our
objects -- a clean copy captured again, before or after it was minted, or
a mask or cut-out captured after it was registered (before, the
registration refuses those bytes): it is linked behind the object by those
bytes, so a clean copy is judged with it and a mask or cut-out is served
only while it is live. A registered
object is served only while app.publication.servable says so: a
soft-deleted image or version, a withdrawn mask, a mask whose image is no
longer mask-eligible, or a mask of a revoked store (MASK-G, from the moment
the revocation is committed) is a 404. The bytes are re-hashed before they
are sent: content cached forever on phones must be exactly what its address
names, and a store that returns anything else is a 500 that is never cached.

CACHING. ``public, max-age=31536000, immutable``: the address can never name
other bytes. The request's ``Cache-Control``/``Pragma``/``If-None-Match`` are
not read -- the app's service worker sends ``no-cache`` on every fetch, and
the answer is the same bytes whatever it asks. 404s and 500s are
``no-store`` -- a judgment that raises included: a mask registered tomorrow
must not be shadowed by a cached miss today.

CORS. ``Access-Control-Allow-Origin`` echoes an app origin
(``Settings.media_cors_origins``, exact match) and ``Vary: Origin`` is always
sent: the service worker must read the bytes to cache them and the case view
composites the mask over its derivative, neither of which an opaque
response allows. No credentials, ever.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_db
from .models import PublicObject
from .object_store import ObjectStore, S3ObjectStore
from .publication import servable, store_revocations

logger = logging.getLogger(__name__)

#: The whole grammar, matched on the path as the server presents it (ASGI
#: ``raw_path``, not the decoded path), so a percent-encoded alias
#: (``/d%2F<sha>``, ``/d/%62...``) is not the address.
_DERIVATIVE_PATH = re.compile(rb"/d/[0-9a-f]{64}")

IMMUTABLE = "public, max-age=31536000, immutable"

_HARDENING = {
    "X-Content-Type-Options": "nosniff",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}


def get_object_store() -> ObjectStore:
    return S3ObjectStore()


def _bare(status: int) -> Response:
    """A bodiless, never-cached answer for anything that is not a hit."""
    return Response(status_code=status, headers={"Cache-Control": "no-store", **_HARDENING})


app = FastAPI(
    title="media-manager derivative origin", openapi_url=None, docs_url=None, redoc_url=None
)


@app.middleware("http")
async def only_the_grammar(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Everything that is not GET/HEAD /d/<64 lowercase hex> (no query) or
    GET /healthz, as the server presents it, is a 404 -- decided before any
    route, database or store. The app cannot see what the server drops
    before it: uvicorn strips an empty ``?`` and a ``#fragment``, so
    ``/d/<sha>?`` reaches it as ``/d/<sha>`` (the same bytes, harmless)."""
    raw = request.scope.get("raw_path") or request.url.path.encode()
    method = request.method
    if method == "GET" and raw == b"/healthz":
        return await call_next(request)
    if method in ("GET", "HEAD") and _DERIVATIVE_PATH.fullmatch(raw) and not request.url.query:
        return await call_next(request)
    return _bare(404)


@app.get("/healthz")
def healthz(db: Session = Depends(get_db)) -> Response:  # noqa: B008
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        return JSONResponse(status_code=503, content={"status": "degraded"})
    return JSONResponse(content={"status": "ok"})


@app.api_route("/d/{sha256}", methods=["GET", "HEAD"])
def derivative(
    sha256: str,
    request: Request,
    db: Session = Depends(get_db),  # noqa: B008
    store: ObjectStore = Depends(get_object_store),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    try:
        obj = db.get(PublicObject, sha256)
        live = obj is not None and servable(
            db,
            obj,
            settings.mask_store_grants,
            store_revocations(db, settings.mask_store_revoked),
        )
    except Exception as exc:
        logger.error("derivative_unjudged", extra={"sha256": sha256, "error": type(exc).__name__})
        return _bare(500)
    if obj is None or not live:
        return _bare(404)
    try:
        data = store.get(obj.storage_key)
    except Exception as exc:
        logger.error("derivative_unreadable", extra={"sha256": sha256, "error": type(exc).__name__})
        return _bare(500)
    if hashlib.sha256(data).hexdigest() != sha256:
        logger.error("derivative_integrity_mismatch", extra={"sha256": sha256})
        return _bare(500)

    headers = {
        "Cache-Control": IMMUTABLE,
        "ETag": f'"{sha256}"',
        "Vary": "Origin",
        **_HARDENING,
    }
    origin = request.headers.get("origin")
    if origin is not None and origin in settings.media_cors_origins:
        headers["Access-Control-Allow-Origin"] = origin
    return Response(content=data, media_type=obj.mime, headers=headers)
