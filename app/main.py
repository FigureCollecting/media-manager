from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from . import telemetry
from .boot_guard import enforce_boot_guard, mount_auth_routers
from .config import Settings, get_settings
from .db import get_db
from .rate_limit import RateLimitMiddleware
from .routes.album_routes import router as album_router
from .routes.external_routes import router as external_router
from .routes.gallery_routes import router as gallery_router
from .routes.image_routes import router as image_router
from .routes.search_routes import router as search_router
from .routes.serve_routes import router as serve_router
from .routes.tag_routes import router as tag_router
from .telemetry.http import AccessLogMiddleware

SERVICE = "media-api"
# Tracing first, before any request: every line under a request carries its trace_id.
telemetry.start_tracing(SERVICE)
#: The http.in lines' logger; configure() routes Python's logging to the same shape.
LOG = telemetry.Logger(service=telemetry.service_name(SERVICE))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log = telemetry.configure(SERVICE, level=settings.log_level)
    log.info("media-api started", event="app.start", env=settings.environment)
    yield
    log.info("media-api stopping", event="app.stop")


app = FastAPI(title="media-manager", lifespan=lifespan)

_settings_cors = get_settings()
app.add_middleware(RateLimitMiddleware)
# Never allow credentials with wildcard origins
_cors_credentials = "*" not in _settings_cors.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_cors.cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("x-request-id")
    if not request_id:
        # generate a simple UUID-like value without import overhead
        import uuid

        request_id = str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# Outermost of the app's middleware: one http.in line per request, 429s included.
app.add_middleware(AccessLogMiddleware, logger=LOG)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import logging
    import uuid

    logger = logging.getLogger(__name__)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    logger.exception("unhandled_exception", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
        headers={"x-request-id": request_id},
    )


@app.get("/healthz")
def healthz(
    settings: Settings = Depends(get_settings),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> Response:
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    status = "ok" if db_status == "ok" else "degraded"
    code = 200 if status == "ok" else 503
    return JSONResponse(
        status_code=code,
        content={"status": status, "service": settings.app_name, "db": db_status},
    )


mount_auth_routers(app, get_settings())
app.include_router(image_router)
app.include_router(serve_router)
app.include_router(external_router)
app.include_router(album_router)
app.include_router(tag_router)
app.include_router(search_router)
app.include_router(gallery_router)

# Last, once every router is mounted: outside a dev environment the import fails
# here (app.boot_guard), so uvicorn never serves an unsafe configuration.
enforce_boot_guard(get_settings(), app)
