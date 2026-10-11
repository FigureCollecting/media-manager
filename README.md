# media-manager

Production-ready image microservice built with FastAPI, SQLAlchemy 2.x, Postgres, Celery, and S3/MinIO. Includes multi-tenant links, albums, tags, age-gating with safe variants, content-addressable storage, background transforms, and search.

Local dev (Docker/Coolify-friendly):

- Copy `.env.example` to `.env` and adjust if needed.
- Start stack: `make dev-up`
- Initialize DB: `make db-upgrade`
- Watch logs: `make logs`
- Get a local token: `make token USER_ID=<uuid> TENANT_ID=<uuid>`

Endpoints (highlights):

- Auth: `POST /auth/dev-token` (local only)
- Upload: `POST /images/initiate-upload`, `POST /images/complete`
- Versions: `POST /images/{id}/versions`, visibility and safe-alt operations
- Serving: `GET /serve/{image_id}@{version_id}`, `GET /public/{image_id}@{version_id}`
- Albums: CRUD, share link, cover mosaic
- Tags/Search: tag images/albums and search by text and tags

Storage layout:

- Originals: `{sha[:2]}/{sha[2:4]}/{sha}.{ext}`
- Staging: `uploads/{uuid}/{filename}`
- Derived: `versions/{image_id}/v{no}-{shortid}.{ext}`
- Album covers: `albums/{album_id}/cover-{hash}.webp`

Switching MinIO → other S3 (R2/B2/Wasabi):

- Update `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET` in `.env` or compose env.
- Keep `S3_REGION` to a supported value; set `S3_SECURE=true` for TLS endpoints.

Hosted Postgres (Neon/Supabase):

- Set `DATABASE_URL` accordingly; ensure extensions `citext` and `pg_trgm` are available. Run `make db-upgrade`.

Coolify deploy tips:

- Use the provided `Dockerfile.api` and `Dockerfile.worker` images.
- Provision Postgres, Redis, and MinIO (or external S3) services; set env vars in the app.
- Configure healthcheck on `/healthz`.

Testing:

- Run locally: `make test` (requires Python 3.12 + Poetry). Integration tests are skipped unless backing services are available.


Mask lane (MASK-1):

- A separate alpha mask per clean image (Ross 2026-09-26: non-destructive; the clean copy is never cut out). The mask is a PNG whose alpha channel is the mask (`LA`, luminance white), the same size as its derivative; the client composites it over the clean copy (CSS `mask-image` / canvas `destination-in`). Grounding and appearance per fc-api-contract 0.4.0 `ProductImage`, plus the top margin and opaque box (CONTRACT-1 follow-up). See `app/workers/mask.py`.
- Eligibility keys on origin: MFC `/upload/items/` and store listing images get masks; collector photos (`/upload/pictures/`, role `user`, class `user_photo`) do not (`app/workers/mask_policy.py`).
- Batch job, dry run by default: `python -m app.cli.masks [--sandbox DIR] seed --sample tests/fixtures/mask_sample`, then `... run` (report only) or `... run --apply`; `--resume BATCH`, `--retry-failed`, `--backend stub|birefnet|birefnet-basekept` (the July base-keep pass: BiRefNet plus IS-Net base donors). Celery task `generate_masks(limit, apply=False)`.
- Public origin: `app/derivative_main.py` is its own ASGI app (never `app.main`) serving `GET`/`HEAD /d/<sha256>` for registered derivatives and masks only: immutable caching, CORS for `MEDIA_CORS_ORIGINS`, `Vary: Origin`, nosniff, CORP cross-origin, CSP sandbox; anything else is a bodiless 404.
