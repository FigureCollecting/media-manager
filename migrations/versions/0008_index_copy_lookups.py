"""SHOW1-THROUGHPUT: index the two copy lookups the display lane makes for
every original -- public_objects by image_version_id (the copies an image
minted: app.spine.copies.reported_copy) and public_object_images by image_id
(the copies an image is linked behind: reported_copy's fallback, the seeder's
_picture_copies and mask_seed._join_picture). Without them each lookup walks
the whole table; public_object_images' primary key (sha256, image_id) does not
lead on image_id.

CONCURRENTLY, so the seeders' INSERTs are not blocked while an index builds:
outside a transaction (alembic's autocommit block, which first commits the run
so far).

NO LOCK TIMEOUT WHILE BUILDING. The migrate Job runs alembic with
lock_timeout=5s (fc-infra db/media-migrate-job.yaml PGOPTIONS), so that DDL
behind a long transaction fails instead of queueing every read of its table.
A concurrent build takes a lock no read or write waits for, but it waits for
every transaction that wrote its table to end (a seeder's runs 40-85 s), and
that wait is a lock wait: under the Job's timeout it would fail. So the build
waits with none; RESET puts the Job's back for any later migration in the run.
The Job's 300 s activeDeadlineSeconds does NOT bound the build: it ends the
pod, not the build, which keeps running in postgres and can finish, leaving a
valid index with the version still at 0007.

A RE-RUN IS SAFE. IF NOT EXISTS keeps an index a killed Job's build finished.
A build cut short leaves an INVALID index of that name,
never read and still written on every insert, that IF NOT EXISTS would keep:
it is dropped and built again. (Offline, `alembic upgrade --sql` cannot see
one: drop it by hand before running that SQL.)

Revision ID: 0008_index_copy_lookups
Revises: 0007_add_display_copy_queue
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_index_copy_lookups"
down_revision = "0007_add_display_copy_queue"
branch_labels = None
depends_on = None

INDEXES = (
    ("ix_public_objects_image_version_id", "public_objects", "image_version_id"),
    ("ix_public_object_images_image_id", "public_object_images", "image_id"),
)


def _left_invalid(name: str) -> bool:
    if op.get_context().as_sql:  # `alembic upgrade --sql`: no database to ask
        return False
    return bool(
        op.get_bind()
        .execute(
            sa.text("SELECT NOT indisvalid FROM pg_index WHERE indexrelid = to_regclass(:name)"),
            {"name": name},
        )
        .scalar()
    )


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = 0")
        for name, table, column in INDEXES:
            if _left_invalid(name):
                op.drop_index(name, table_name=table, postgresql_concurrently=True)
            op.create_index(name, table, [column], postgresql_concurrently=True, if_not_exists=True)
        op.execute("RESET lock_timeout")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = 0")
        for name, table, _ in reversed(INDEXES):
            op.drop_index(name, table_name=table, postgresql_concurrently=True, if_exists=True)
        op.execute("RESET lock_timeout")
