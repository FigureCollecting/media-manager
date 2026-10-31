"""SHOW1: the display-copy lane's queue -- the originals it owes a clean copy
(display_originals) and the spine depictions it reports them for
(display_depictions).

Revision ID: 0007_add_display_copy_queue
Revises: 0006_add_gallery_source_class
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_add_display_copy_queue"
down_revision = "0006_add_gallery_source_class"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "display_originals",
        sa.Column("origin_sha256", sa.String(64), primary_key=True),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("codec", sa.String(32), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("claimed_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_by", sa.String(128), nullable=True),
        sa.Column("reason", sa.String(256), nullable=True),
        sa.Column("force_seed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("copy_sha256", sa.String(64), nullable=True),
        sa.Column("copy_mime", sa.String(100), nullable=True),
        sa.Column("copy_width", sa.Integer, nullable=True),
        sa.Column("copy_height", sa.Integer, nullable=True),
        sa.Column("copy_generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("thumbhash", sa.String(64), nullable=True),
        sa.Column("dominant_color", sa.String(7), nullable=True),
        sa.Column("own_grounding", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state in ('pending','seeded','failed','held','stuck')",
            name="ck_display_original_state",
        ),
    )
    op.create_index(
        "ix_display_originals_state_seen", "display_originals", ["state", "first_seen_at"]
    )
    op.create_table(
        "display_depictions",
        sa.Column("product_image_id", sa.String(36), primary_key=True),
        sa.Column(
            "origin_sha256",
            sa.String(64),
            sa.ForeignKey("display_originals.origin_sha256", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(36), nullable=False),
        sa.Column("origin_url", sa.String(2048), nullable=False),
        sa.Column("source_site", sa.String(50), nullable=False),
        sa.Column("source_class", sa.String(32), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("outcome", sa.String(16), nullable=True),
        sa.Column("note", sa.String(256), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_display_depictions_origin", "display_depictions", ["origin_sha256"])
    op.create_index("ix_display_depictions_reported_at", "display_depictions", ["reported_at"])


def downgrade() -> None:
    op.drop_index("ix_display_depictions_reported_at", table_name="display_depictions")
    op.drop_index("ix_display_depictions_origin", table_name="display_depictions")
    op.drop_table("display_depictions")
    op.drop_index("ix_display_originals_state_seen", table_name="display_originals")
    op.drop_table("display_originals")
