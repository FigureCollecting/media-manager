"""MM-REPORT-1: the media.v1 replay's ledger (spine_report_runs) and the
mask withdrawals it has sent the spine (spine_mask_withdrawals).

Revision ID: 0005_add_spine_report_ledger
Revises: 0004_add_mask_lane
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_add_spine_report_ledger"
down_revision = "0004_add_mask_lane"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "spine_report_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("cursor_sha256", sa.String(64), nullable=True),
        sa.Column("resume_not_before", sa.DateTime(timezone=True), nullable=True),
        sa.Column("batch_size", sa.Integer, nullable=False),
        sa.Column("counts", sa.JSON, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('running','done','abandoned')", name="ck_spine_report_run_status"
        ),
    )
    op.create_table(
        "spine_mask_withdrawals",
        sa.Column("mask_sha256", sa.String(64), primary_key=True),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("spine_mask_withdrawals")
    op.drop_table("spine_report_runs")
