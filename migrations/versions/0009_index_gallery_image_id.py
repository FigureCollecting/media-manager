"""MASKRATE: index figure_galleries by image_id. Every copy judged reads the
origins of each image behind it by image_id (app.publication.gallery_origins:
the mask batch's settle, the display seeder's withdraw check); with no index
each read walked the whole table (a Parallel Seq Scan in prod, Nov 2026).

PROD HAS IT ALREADY: built by hand on 2026-11-03, CONCURRENTLY, with this
name and definition (btree on image_id). IF NOT EXISTS makes this migration a
no-op there: the index is kept, only the version moves.

Built as 0008 builds its two (read it for why): CONCURRENTLY, outside a
transaction, with no lock timeout while it builds; an index of this name a
cut-short build left INVALID is dropped and built again.

Revision ID: 0009_index_gallery_image_id
Revises: 0008_index_copy_lookups
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_index_gallery_image_id"
down_revision = "0008_index_copy_lookups"
branch_labels = None
depends_on = None

NAME, TABLE, COLUMN = "ix_figure_galleries_image_id", "figure_galleries", "image_id"


def _left_invalid() -> bool:
    if op.get_context().as_sql:  # `alembic upgrade --sql`: no database to ask
        return False
    return bool(
        op.get_bind()
        .execute(
            sa.text("SELECT NOT indisvalid FROM pg_index WHERE indexrelid = to_regclass(:name)"),
            {"name": NAME},
        )
        .scalar()
    )


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = 0")
        if _left_invalid():
            op.drop_index(NAME, table_name=TABLE, postgresql_concurrently=True)
        op.create_index(NAME, TABLE, [COLUMN], postgresql_concurrently=True, if_not_exists=True)
        op.execute("RESET lock_timeout")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = 0")
        op.drop_index(NAME, table_name=TABLE, postgresql_concurrently=True, if_exists=True)
        op.execute("RESET lock_timeout")
