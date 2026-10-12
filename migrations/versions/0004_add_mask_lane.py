"""MASK-1: the public object registry, mask results and mask batches.

Revision ID: 0004_add_mask_lane
Revises: 0003_add_matte_grounding_fields
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_add_mask_lane"
down_revision = "0003_add_matte_grounding_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "public_objects",
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("mime", sa.String(100), nullable=False),
        sa.Column("bytes", sa.BigInteger, nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False, unique=True),
        sa.Column(
            "image_version_id",
            sa.Integer,
            sa.ForeignKey("image_versions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("kind in ('sanitized','matted','mask')", name="ck_public_object_kind"),
    )
    op.create_table(
        "mask_batches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("backend", sa.String(32), nullable=False),
        sa.Column("dry_run", sa.Boolean, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("cursor_sha256", sa.String(64), nullable=True),
        sa.Column("item_limit", sa.Integer, nullable=True),
        sa.Column("counts", sa.JSON, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status in ('running','done')", name="ck_mask_batch_status"),
    )
    op.create_table(
        "image_masks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "source_sha256",
            sa.String(64),
            sa.ForeignKey("public_objects.sha256", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("origin_class", sa.String(32), nullable=True),
        sa.Column("backend", sa.String(32), nullable=True),
        sa.Column(
            "batch_id",
            sa.Integer,
            sa.ForeignKey("mask_batches.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("mask_sha256", sa.String(64), nullable=True),
        sa.Column("mask_storage_key", sa.String(512), nullable=True),
        sa.Column("bottom_margin_frac", sa.Float, nullable=True),
        sa.Column("top_margin_frac", sa.Float, nullable=True),
        sa.Column("box_top_frac", sa.Float, nullable=True),
        sa.Column("box_bottom_frac", sa.Float, nullable=True),
        sa.Column("box_left_frac", sa.Float, nullable=True),
        sa.Column("box_right_frac", sa.Float, nullable=True),
        sa.Column("contact_band_center_x_frac", sa.Float, nullable=True),
        sa.Column("contact_band_width_frac", sa.Float, nullable=True),
        sa.Column("masked_thumbhash", sa.String(64), nullable=True),
        sa.Column("masked_dominant_color", sa.String(7), nullable=True),
        sa.Column("plain_thumbhash", sa.String(64), nullable=True),
        sa.Column("plain_dominant_color", sa.String(7), nullable=True),
        sa.Column("own_alpha_grounding", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_sha256", "pipeline_version", name="uq_image_mask_source_version"
        ),
        sa.CheckConstraint(
            "status in ('ok','held','empty','ineligible','failed','withdrawn')", name="ck_image_mask_status"
        ),
    )
    op.create_index("ix_image_masks_mask_sha256", "image_masks", ["mask_sha256"])


def downgrade() -> None:
    op.drop_index("ix_image_masks_mask_sha256", table_name="image_masks")
    op.drop_table("image_masks")
    op.drop_table("mask_batches")
    op.drop_table("public_objects")
