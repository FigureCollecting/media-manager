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
- Eligibility keys on origin and fails closed: MFC `/upload/items/` images get masks, collector photos (`/upload/pictures/`) do not; a store's listing photos get masks when its source is registered with the domains they come from (`MASK_STORE_GRANTS`; MASK-G, Ross 2026-10-12: every registered store is granted, revocable per store; a store is registered only by its entry there, so the default-allow reaches every retailer once follow-up (4) below renders their domains into it) and the source is not revoked (`MASK_STORE_REVOKED` or `revoke-store`, below). Anything else gets none (`app/workers/mask_policy.py`).
- Batch job, dry run by default: `python -m app.cli.masks [--sandbox DIR] seed --sample tests/fixtures/mask_sample`, then `... run` (report only) or `... run --apply`; `--resume BATCH` (keeps the batch's `--source`), `--retry-failed`, `--recheck-ineligible` (after a grant or an origin change), `--source SOURCE` (one source only), `--backend stub|birefnet|birefnet-basekept` (the July base-keep pass: BiRefNet plus IS-Net base donors). Celery task `generate_masks(limit, apply=False)`. `--apply` needs a mask model: the stub (the default backend) is dry run only. `... reevaluate [--apply]` (task `reevaluate_masks`) withdraws published masks whose image was deleted or is no longer eligible and restores those that qualify again.
- Public origin: `app/derivative_main.py` is its own ASGI app (never `app.main`) serving `GET`/`HEAD /d/<sha256>` for registered derivatives and masks only, and only while they are live (`app/publication.py`: a soft-deleted image or version, a withdrawn mask, a mask whose image is no longer eligible, a mask of a revoked store, or a mask or cut-out whose very bytes a soft-deleted original has is a 404): immutable caching, a constant `Access-Control-Allow-Origin: *` with no credentials and no `Vary` on every answer (the edge caches by URL alone and replays the stored headers), nosniff, CORP cross-origin, CSP sandbox; anything else is a bodiless 404.

GPU mask batch (MASK-BATCH-1):

- `python -m app.cli.mask_batch run --worklist FILE ...` mints masks for a work list of captured originals (JSON lines: the raw bucket key, sha and origin of each, Ross's holdings ranked first), in chunks: per original it is skipped when its clean copy already has a mask result (idempotent per image sha), else read from the raw bucket (`RAW_S3_*`, read only), seeded and masked into the media database and bucket. Before every chunk it waits for Ross's gate: outside the scraper roll windows (`--busy-windows`, default `03-26,41-43,50-56`), the GPU idle (< 5 %) or 9 GiB free (`nvidia-smi`; while its own model is held: no other consumer, i.e. the card's memory not grown more than 512 MiB since the model was measured and not busy on every reading 2 s after the chunk), and the residential address (`--residential-ip`; a VPN window pauses it). It releases the model while it waits, appends every result to `--progress` (a restart resumes from it), prints one JSON line per chunk (count, s/img, peak VRAM, failures), and `--dry-run` opens nothing. `python -m app.cli.mask_batch sql ross-heads|ross-worklist|catalog-worklist` prints the read-only work-list queries (coordinator, then spine). `scripts/mask-batch.sh STATE_DIR [run options]` runs it in the worker image with `--gpus all`, configuration passed by name only; its container memory is 8g on the GPU and 10g when the model may run on the CPU (`--device auto` or `cpu`, or `MASK_BATCH_GPUS=none`: the CPU base-keep pass is OOM-killed at 8g), and `MASK_BATCH_MEMORY` overrides both.
- `MATTING_DEVICE`: `cuda` (default) refuses a session without the CUDA execution provider; `auto` falls back to a lean CPU session (no CPU arena, no memory pattern) and logs `matting_cpu_fallback`; `cpu` runs there on purpose (`MATTING_CPU_THREADS`). On the GPU each model's arena is capped (BiRefNet 8 GiB, IS-Net 2 GiB): uncapped, the base-keep pass outgrew a 12 GB card and slowed to ~5.5 s per image; capped it measured ~0.7 s per image on an RTX 4070 Ti with ~9.5 GB in use (the CPU path: ~15 s with 4 threads).

Runbook -- a store asks for its cut-outs to be removed (MASK-G):

- The single command, on the deployed media-manager (its `DATABASE_URL` and bucket): `python -m app.cli.masks revoke-store <source> --apply` (Celery task `revoke_mask_store(source, apply=True)`). Without `--apply` it only prints which masks it would withdraw. `<source>` is the gallery source name (case and surrounding spaces do not matter). To keep the revocation through a database restore, also add the source to `MASK_STORE_REVOKED` in the deployment's environment, written as a JSON list, exactly `MASK_STORE_REVOKED='["shop"]'` (several: `'["shop", "other"]'`). A bare or comma-separated value (`shop`, `shop,other`), or an empty one (`MASK_STORE_REVOKED=`), is not accepted: the origin and the worker refuse to start until it is corrected.
- media-manager (this repo, now): the revocation is committed first, and from that moment the origin answers 404 (`no-store`) for every mask of an image with an origin of that store, a picture it shares with another source included, and no batch makes a new one (`store_revoked`). The command then withdraws each such published row (reason `store_revoked`), and nothing else is judged. Clean copies stay: display is decided elsewhere. A mask whose bytes another source's picture also produced stays served as that source's mask.
- Not done here, separate follow-up units: (1) fc-coordinator, a `display_restriction` by source so `GetProductImages` withholds `maskUrl` for that store's images; the phone's owned sync then evicts its cached masks (ADDR-1 withheld path), within one sync once that unit lands. Until then a phone keeps a mask it already cached. (2) fc-aggregation, the `image_mask` withdrawal write path (a status column or a delete grant; PR #109's 0032 note), so the spine read stops sending a withdrawn mask. (3) The edge purge by URL once the images tunnel or cache rule lands (`app/publication.py`). (4) fc-infra, rendering `MASK_STORE_GRANTS` from the spine's retailer `source.domains` once store images reach media-manager (today gallery ingest records source `mfc` only).
- Undo: `python -m app.cli.masks restore-store <source> --apply` (task `restore_mask_store`). It lifts the persisted revocation, restores the withdrawn masks that qualify again and re-checks the store's refused images (`--recheck-ineligible --source`), so it needs the mask model. Re-run it if it is interrupted. A source listed in `MASK_STORE_REVOKED` is refused: lift it in the environment first, by removing it from the list. To lift the last one, remove the variable or set it to `'[]'`, never to an empty value, which stops the origin and the worker like a bare one.
