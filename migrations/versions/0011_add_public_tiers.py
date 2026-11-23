"""IMGREND (IR-MM-1): the registry of display-sized renditions ("tiers") of
clean copies and masks (public_tiers) and the minting ledger (tier_parents).

TWO NEW TABLES AND NOTHING ELSE. public_objects gains no column, no index and
no kind: the WSL mask batch runs pinned code against it and the origin reads
it on every request. Nothing is minted by this migration; the origin reads
public_tiers only once MEDIA_TIERS_SERVED is set (after fc-infra grants it
SELECT), so applying this is inert.

THE ONE LOCK ON A LIVE TABLE. Each table's foreign key to public_objects
takes SHARE ROW EXCLUSIVE on public_objects for the instant of the DDL. No
read waits for it (the origin's included), but it waits for every open
transaction that WROTE public_objects (a seeder's, the mask batch's), and
writers that start meanwhile queue behind it. Under the migrate Job's
lock_timeout=5s (fc-infra db/media-migrate-job.yaml) it therefore fails
rather than hold writers up for longer; the migration is one transaction, so
a failure leaves nothing behind and a re-run applies it. Apply it between
seeder runs.

Revision ID: 0011_add_public_tiers
Revises: 0010_drop_prefix_indexes
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_add_public_tiers"
down_revision = "0010_drop_prefix_indexes"
branch_labels = None
depends_on = None


def _parent() -> sa.Column[str]:
    return sa.Column(
        "parent_sha256",
        sa.String(64),
        sa.ForeignKey("public_objects.sha256", ondelete="CASCADE"),
        primary_key=True,
    )


def upgrade() -> None:
    op.create_table(
        "public_tiers",
        _parent(),
        sa.Column("pipeline_version", sa.String(32), primary_key=True),
        sa.Column("long_edge_px", sa.Integer, primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("parent_kind", sa.String(16), nullable=False),
        sa.Column("mime", sa.String(100), nullable=False),
        sa.Column("bytes", sa.BigInteger, nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "parent_kind in ('sanitized','mask')", name="ck_public_tier_parent_kind"
        ),
        sa.CheckConstraint("mime = 'image/webp'", name="ck_public_tier_mime"),
        sa.CheckConstraint("bytes > 0", name="ck_public_tier_bytes"),
        sa.CheckConstraint(
            "long_edge_px between 1 and 4096"
            " and width between 1 and long_edge_px and height between 1 and long_edge_px"
            " and (width = long_edge_px or height = long_edge_px)",
            name="ck_public_tier_edge",
        ),
    )
    op.create_index("ix_public_tiers_sha256", "public_tiers", ["sha256"])
    op.create_table(
        "tier_parents",
        _parent(),
        sa.Column("pipeline_version", sa.String(32), primary_key=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("max_edge_px", sa.Integer, nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("state in ('done','refused','failed')", name="ck_tier_parent_state"),
    )
    op.create_index(
        "ix_tier_parents_state", "tier_parents", ["pipeline_version", "state", "parent_sha256"]
    )


def downgrade() -> None:
    op.drop_index("ix_tier_parents_state", table_name="tier_parents")
    op.drop_table("tier_parents")
    op.drop_index("ix_public_tiers_sha256", table_name="public_tiers")
    op.drop_table("public_tiers")
