"""MEDIADB (M-5, the verifier's part of it): drop two indexes that are each the
leading column of a UNIQUE index their readers move to at the same buffer
counts (measured at prod scale, Nov 2026):

  ix_image_versions_image_id_visibility (image_id, visibility), 59 MB: no
    statement filters visibility with image_id; reported_copy, the display
    meta and max(version_no) read uq_image_version_no (image_id, version_no).
  ix_figure_galleries_figure_id (figure_id), 30 MB: the gallery routes and
    the seed's link read by figure_id read uq_figure_source_url (figure_id,
    source_url).

Each seeded original wrote an entry in both for nothing (three, with its
second version). NOT the three storage_key UNIQUEs the audit also named: they
are constraints, not plain indexes, and image_versions' is the only thing
that keeps two versions from naming one object (the verifier produced the
collision); they stay.

CONCURRENTLY, outside a transaction, as 0008 builds. NO LOCK OR STATEMENT
TIMEOUT WHILE DROPPING: a concurrent drop waits for every transaction that
holds any lock on its table to end, READS INCLUDED (the verifier's probe: an
idle read-only transaction stalls it), and that wait is a lock wait, so
under the migrate Job's lock_timeout=5s (fc-infra db/media-migrate-job.yaml)
it would fail while a long reader runs (the display seeder's, before
MEDIADB's commits; the WSL mask batch's). It waits instead; meanwhile its
lock (SHARE UPDATE EXCLUSIVE) queues no read or write, only other DDL and
vacuum. Apply it after the seeder runs MEDIADB (or between its runs, :05-:20)
so the wait is short; RESET puts the Job's timeouts back for any later
migration in the run.

A RE-RUN IS SAFE. A drop cut short (the Job's deadline ends the pod, not the
drop) leaves the index INVALID, never read; IF EXISTS drops it the next time.
The downgrade builds both as 0008 builds, an invalid one rebuilt.

Revision ID: 0010_drop_prefix_indexes
Revises: 0009_index_gallery_image_id
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_drop_prefix_indexes"
down_revision = "0009_index_gallery_image_id"
branch_labels = None
depends_on = None

#: (name, table, columns), in the order the upgrade drops them.
INDEXES = (
    ("ix_image_versions_image_id_visibility", "image_versions", ["image_id", "visibility"]),
    ("ix_figure_galleries_figure_id", "figure_galleries", ["figure_id"]),
)


def _left_invalid(name: str) -> bool:
    if op.get_context().as_sql:  # `alembic downgrade --sql`: no database to ask
        return False
    return bool(
        op.get_bind()
        .execute(
            sa.text("SELECT NOT indisvalid FROM pg_index WHERE indexrelid = to_regclass(:name)"),
            {"name": name},
        )
        .scalar()
    )


def _no_timeouts() -> None:
    op.execute("SET lock_timeout = 0")
    op.execute("SET statement_timeout = 0")


def _timeouts_back() -> None:
    op.execute("RESET statement_timeout")
    op.execute("RESET lock_timeout")


def upgrade() -> None:
    with op.get_context().autocommit_block():
        _no_timeouts()
        for name, table, _columns in INDEXES:
            op.drop_index(name, table_name=table, postgresql_concurrently=True, if_exists=True)
        _timeouts_back()


def downgrade() -> None:
    with op.get_context().autocommit_block():
        _no_timeouts()
        for name, table, columns in reversed(INDEXES):
            if _left_invalid(name):  # a build cut short, as 0008 rebuilds one
                op.drop_index(name, table_name=table, postgresql_concurrently=True)
            op.create_index(name, table, columns, postgresql_concurrently=True, if_not_exists=True)
        _timeouts_back()
