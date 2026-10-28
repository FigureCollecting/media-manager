"""GAL3: the source class a gallery link was captured with
(figure_galleries.source_class), so the mask policy can tell MFC's official
gallery list from a collector's own upload by provenance.

Revision ID: 0006_add_gallery_source_class
Revises: 0005_add_spine_report_ledger
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_add_gallery_source_class"
down_revision = "0005_add_spine_report_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, no default: a catalog-only change, every existing link NULL
    # (unclassified) until the mask work list or `masks official-gallery` says.
    op.add_column("figure_galleries", sa.Column("source_class", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("figure_galleries", "source_class")
