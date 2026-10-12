from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_auth_ctx
from ..models import Album, AlbumTag, Image, ImageTag, Tag, UserImageLink
from ..policy import AuthCtx
from ..schemas import (
    AlbumSearchResponse,
    AlbumSearchResult,
    ImageSearchResponse,
    ImageSearchResult,
)
from ..search import build_text_filter

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/images", response_model=ImageSearchResponse)
def search_images(
    query: str | None = None,
    tags: str | None = None,
    limit: int = 50,
    after: int | None = None,
    db: Session = Depends(get_db),  # noqa: B008
    ctx: AuthCtx = Depends(require_auth_ctx),  # noqa: B008
) -> ImageSearchResponse:
    limit = max(1, min(limit, 100))
    stmt = select(Image).where(Image.deleted_at.is_(None))
    if not ctx.is_service:
        stmt = stmt.join(UserImageLink, UserImageLink.image_id == Image.id).where(
            UserImageLink.user_id == ctx.subject
        )
    if query:
        stmt = stmt.where(build_text_filter(db, query, Image.mime, Image.storage_key))
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if tag_list:
            stmt = (
                stmt.join(ImageTag, ImageTag.image_id == Image.id, isouter=True)
                .join(Tag, Tag.id == ImageTag.tag_id, isouter=True)
                .where(Tag.name.in_(tag_list))
            )
    if after is not None:
        stmt = stmt.where(Image.id > after)
    stmt = stmt.order_by(Image.id).limit(limit + 1)
    rows = db.execute(stmt).scalars().all()
    has_next = len(rows) > limit
    results = rows[:limit]
    next_cursor = results[-1].id if has_next else None
    return ImageSearchResponse(
        results=[ImageSearchResult(id=r.id, mime=r.mime, sha256=r.sha256) for r in results],
        next_cursor=next_cursor,
    )


@router.get("/albums", response_model=AlbumSearchResponse)
def search_albums(
    query: str | None = None,
    tags: str | None = None,
    limit: int = 50,
    after: int | None = None,
    db: Session = Depends(get_db),  # noqa: B008
    ctx: AuthCtx = Depends(require_auth_ctx),  # noqa: B008
) -> AlbumSearchResponse:
    limit = max(1, min(limit, 100))
    stmt = select(Album).where(Album.deleted_at.is_(None))
    # TODO(C1): album search scopes by tenant while image search scopes by
    # owner link; reconcile owner-vs-tenant consistency under the grant model.
    if not ctx.is_service:
        # Non-service callers see ONLY their own tenant's albums. A caller
        # without a tenant sees nothing, and null-tenant albums are never
        # world-visible (fail closed).
        if ctx.tenant_id is None:
            return AlbumSearchResponse(results=[], next_cursor=None)
        stmt = stmt.where(Album.tenant_id == ctx.tenant_id)
    if query:
        stmt = stmt.where(build_text_filter(db, query, Album.title, Album.description))
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if tag_list:
            stmt = (
                stmt.join(AlbumTag, AlbumTag.album_id == Album.id, isouter=True)
                .join(Tag, Tag.id == AlbumTag.tag_id, isouter=True)
                .where(Tag.name.in_(tag_list))
            )
    if after is not None:
        stmt = stmt.where(Album.id > after)
    stmt = stmt.order_by(Album.id).limit(limit + 1)
    rows = db.execute(stmt).scalars().all()
    has_next = len(rows) > limit
    results = rows[:limit]
    next_cursor = results[-1].id if has_next else None
    return AlbumSearchResponse(
        results=[AlbumSearchResult(id=r.id, title=r.title) for r in results],
        next_cursor=next_cursor,
    )
