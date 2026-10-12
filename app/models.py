from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    phash: Mapped[str | None] = mapped_column(String(64))
    mime: Mapped[str | None] = mapped_column(String(100))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_key: Mapped[str | None] = mapped_column(String(512), unique=True)
    #: The PIXEL DIGEST of this capture (``mask_seed.pixel_digest``), recorded
    #: by every capture route (the seed, the gallery ingest) through
    #: ``mask_seed.link_capture``: the key a clean copy minted LATER finds
    #: this capture by, so it is linked behind that copy whichever came
    #: first. NULL for an image no capture route has recorded (an upload).
    pixel_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    versions: Mapped[list[ImageVersion]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ImageVersion(Base):
    __tablename__ = "image_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    derived_from_version: Mapped[int | None] = mapped_column(Integer)
    transform_spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    mime: Mapped[str | None] = mapped_column(String(100))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    visibility: Mapped[str] = mapped_column(String(16), default="private", nullable=False)
    age_rating: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)
    alt_for_version_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    matted: Mapped[bool] = mapped_column(Boolean, server_default=false(), nullable=False)
    bottom_margin_frac: Mapped[float | None] = mapped_column(Float)
    contact_band_center_x_frac: Mapped[float | None] = mapped_column(Float)
    contact_band_width_frac: Mapped[float | None] = mapped_column(Float)
    thumbhash: Mapped[str | None] = mapped_column(String(64))
    dominant_color: Mapped[str | None] = mapped_column(String(32))

    __table_args__ = (
        UniqueConstraint("image_id", "version_no", name="uq_image_version_no"),
        CheckConstraint(
            "visibility in ('private','tenant','public','catalog')", name="ck_visibility"
        ),
    )

    image: Mapped[Image] = relationship(back_populates="versions")


class UserImageLink(Base):
    __tablename__ = "user_image_links"

    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    current_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_versions.id", ondelete="SET NULL")
    )
    role: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    actor_service: Mapped[str | None] = mapped_column(String(100))
    tenant_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    image_id: Mapped[int | None] = mapped_column(Integer)
    version_id: Mapped[int | None] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ts: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )


class Album(Base):
    __tablename__ = "albums"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    owner_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    default_visibility: Mapped[str] = mapped_column(String(16), default="private", nullable=False)
    is_shareable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allow_item_override: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    share_token_hash: Mapped[str | None] = mapped_column(String(128))
    share_age_threshold: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "default_visibility in ('private','tenant','public')", name="ck_album_visibility"
        ),
    )


class AlbumItem(Base):
    __tablename__ = "album_items"

    album_id: Mapped[int] = mapped_column(
        ForeignKey("albums.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    version_id: Mapped[int | None] = mapped_column(Integer)
    item_visibility: Mapped[str | None] = mapped_column(String(16))


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    owner_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))

    __table_args__ = (CheckConstraint("scope in ('global','tenant','user')", name="ck_tag_scope"),)


class ImageTag(Base):
    __tablename__ = "image_tags"

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default="00000000-0000-0000-0000-000000000000"
    )
    owner_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))


class AlbumTag(Base):
    __tablename__ = "album_tags"

    album_id: Mapped[int] = mapped_column(
        ForeignKey("albums.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)


class ExternalRef(Base):
    __tablename__ = "external_refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ref_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ref_id: Mapped[str] = mapped_column(String(200), nullable=False)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    version_id: Mapped[int | None] = mapped_column(Integer)
    tenant_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))

    __table_args__ = (UniqueConstraint("ref_type", "ref_id", name="uq_ref_type_id"),)


class ServiceClient(Base):
    __tablename__ = "service_clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    secret_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    scopes: Mapped[str] = mapped_column(Text, nullable=False)


class FigureGallery(Base):
    __tablename__ = "figure_galleries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    figure_id: Mapped[str] = mapped_column(String(200), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    caption: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="mfc")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("figure_id", "source_url", name="uq_figure_source_url"),
        Index("ix_figure_galleries_figure_id", "figure_id"),
    )


class PublicObject(Base):
    """The content-addressed registry the public origin answers from:
    ``GET /d/<sha256>`` serves a row of this table and nothing else.

    ONLY our own work products are registered here -- a clean (sanitized)
    derivative, a matted fallback, or a MASK-1 mask. An original's bytes are
    never registered (``mask_seed.register_public_object`` refuses an
    ``images.sha256``). The one overlap: an original whose bytes ARE one of
    our objects -- a clean copy captured again, before or after it was
    minted, or a mask or cut-out captured after it was registered (before,
    the registration refuses those bytes) -- is linked behind the object by
    those bytes: a clean copy is judged with it, a mask or cut-out is served
    only while it is live (app.publication).
    """

    __tablename__ = "public_objects"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    mime: Mapped[str] = mapped_column(String(100), nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    #: The version whose bytes these are (derivatives); NULL for masks, which
    #: point back at their clean image from image_masks.
    image_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_versions.id", ondelete="CASCADE")
    )
    #: A clean copy's PIXEL DIGEST (``mask_seed.pixel_digest``) of the
    #: original it was minted from: a capture with those pixels, whatever its
    #: bytes, is linked behind it. Set on ``sanitized`` objects and only on
    #: them; one clean copy minted per digest.
    pixel_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    #: The pixel digest of the clean copy's OWN bytes, decoded: the same as
    #: ``pixel_sha256`` for a lossless (PNG) copy, another for a JPEG copy,
    #: whose decoded pixels are not its original's. A capture of the copy
    #: itself -- its bytes, or its pixels under other bytes -- has it, and is
    #: linked behind it. Set on ``sanitized`` objects and only on them.
    own_pixel_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )

    __table_args__ = (
        CheckConstraint("kind in ('sanitized','matted','mask')", name="ck_public_object_kind"),
        CheckConstraint(
            "(kind = 'sanitized') = (pixel_sha256 IS NOT NULL)", name="ck_public_object_pixels"
        ),
        CheckConstraint(
            "(kind = 'sanitized') = (own_pixel_sha256 IS NOT NULL)",
            name="ck_public_object_own_pixels",
        ),
    )


class PublicObjectImage(Base):
    """Every image a clean public object is the clean copy of: its own
    image, and every other capture of the same picture. For a mask or a
    cut-out: a capture of its very bytes.

    A capture belongs to a picture by its pixels (the copy's
    ``pixel_sha256``), by the copy's own decoded pixels
    (``own_pixel_sha256``: our JPEG copy captured again, re-saved or with
    metadata added) or by the copy's very bytes. ``mask_seed.link_capture``
    links every capture of a picture behind every clean copy of it, in
    whichever order they arrive and whichever route captured them (the seed,
    the gallery ingest), so each copy, and any mask made from it, is judged
    over all of them (app.publication).
    """

    __tablename__ = "public_object_images"

    sha256: Mapped[str] = mapped_column(
        ForeignKey("public_objects.sha256", ondelete="CASCADE"), primary_key=True
    )
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )


class MaskStoreRevocation(Base):
    """A source whose masks are revoked (MASK-G), persisted by
    ``masks revoke-store``. With MASK_STORE_REVOKED it wins over any grant:
    no mask of an image with an origin of that source is made or served
    (app.workers.mask_policy, app.publication)."""

    __tablename__ = "mask_store_revocations"

    #: The source, lower-cased and stripped (``mask_policy.source_key``).
    source: Mapped[str] = mapped_column(String(50), primary_key=True)
    revoked_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )


class MaskBatch(Base):
    """One applied run of the MASK-1 batch job (dry runs are never stored).
    ``cursor_sha256`` is the last clean image settled, so a killed run
    resumes after it; ``counts`` accumulate across resumes; ``source`` is the
    scope a resume keeps."""

    __tablename__ = "mask_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    cursor_sha256: Mapped[str | None] = mapped_column(String(64))
    item_limit: Mapped[int | None] = mapped_column(Integer)
    #: The one source the batch is limited to (``run --source``, as
    #: ``mask_policy.source_key``); NULL for every source.
    source: Mapped[str | None] = mapped_column(String(50))
    counts: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (CheckConstraint("status in ('running','done')", name="ck_mask_batch_status"),)


class ImageMask(Base):
    """MASK-1's answer for one clean image at one mask pipeline version --
    the idempotency key of the batch job. ``ok`` rows carry a mask that is
    registered in public_objects; ``held`` rows keep their mask privately
    for review; ``empty``/``ineligible``/``failed`` rows carry no mask; a
    ``withdrawn`` row's mask stays registered but is not served (its image
    was deleted or is no longer eligible; ``reevaluate_masks``).

    Grounding is measured on the mask's alpha (contract ProductImage:
    bottom_margin_frac, contact_band) plus the CONTRACT-1 follow-up fields
    (top margin, opaque box as half-open fractions). The appearance is kept
    twice: ``masked_*`` with the mask applied, ``plain_*`` without it, so a
    caller denied the mask (``no_mask``) is described without it.
    """

    __tablename__ = "image_masks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_sha256: Mapped[str] = mapped_column(
        ForeignKey("public_objects.sha256", ondelete="CASCADE"), nullable=False
    )
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(64))
    origin_class: Mapped[str | None] = mapped_column(String(32))
    backend: Mapped[str | None] = mapped_column(String(32))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("mask_batches.id", ondelete="SET NULL"))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    mask_sha256: Mapped[str | None] = mapped_column(String(64))
    mask_storage_key: Mapped[str | None] = mapped_column(String(512))
    bottom_margin_frac: Mapped[float | None] = mapped_column(Float)
    top_margin_frac: Mapped[float | None] = mapped_column(Float)
    box_top_frac: Mapped[float | None] = mapped_column(Float)
    box_bottom_frac: Mapped[float | None] = mapped_column(Float)
    box_left_frac: Mapped[float | None] = mapped_column(Float)
    box_right_frac: Mapped[float | None] = mapped_column(Float)
    contact_band_center_x_frac: Mapped[float | None] = mapped_column(Float)
    contact_band_width_frac: Mapped[float | None] = mapped_column(Float)
    masked_thumbhash: Mapped[str | None] = mapped_column(String(64))
    masked_dominant_color: Mapped[str | None] = mapped_column(String(7))
    plain_thumbhash: Mapped[str | None] = mapped_column(String(64))
    plain_dominant_color: Mapped[str | None] = mapped_column(String(7))
    #: Grounding on the clean copy's OWN alpha, when it has one (rare: a
    #: transparent PNG source); the AlphaGrounding fields by name.
    own_alpha_grounding: Mapped[dict[str, float] | None] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("source_sha256", "pipeline_version", name="uq_image_mask_source_version"),
        CheckConstraint(
            "status in ('ok','held','empty','ineligible','failed','withdrawn')",
            name="ck_image_mask_status",
        ),
        Index("ix_image_masks_mask_sha256", "mask_sha256"),
    )


# Useful indexes
Index("ix_image_versions_image_id_visibility", ImageVersion.image_id, ImageVersion.visibility)
Index("ix_album_title_trgm", Album.title)
Index("ix_album_desc_trgm", Album.description)
