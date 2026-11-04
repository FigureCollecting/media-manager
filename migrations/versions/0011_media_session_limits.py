"""MEDIADB (U5's media limits): bound every session of the media database.

  idle_in_transaction_session_timeout 180 s: a session left idle inside a
    transaction is ended (FATAL 25P03, its transaction rolled back). An idle
    transaction holds its locks, stalls DROP INDEX CONCURRENTLY and every
    ACCESS EXCLUSIVE DDL even when it only read, and from its first write
    holds back vacuum for the whole database.
  statement_timeout 120 s: one statement running longer is cancelled. The
    longest the media workloads run are the replay's _dead_copies (2.2 s at
    prod scale) and its streamed reads, a FETCH at a time.

ORDER: only once nothing holds a transaction open that long on purpose. The
display seeder did, 40-134 s from its depictions read through the raw GET,
the decode and the PUTs, until MEDIADB's commits; the WSL mask batch still
reads inside a transaction across its GPU queue and its turn waits (its own
unit, not this one). This revision is separate from 0010 so it can wait.

Set on the database (CNPG's managed roles have no SET field), as the one
running the migration finds it: current_database(), never a name. A session
that sets its own value keeps it (pg_dump sets 0). THE OWNER'S DDL IS NOT
BOUND: the role that runs the migrations (in prod media_migrator, the
database's owner: no superuser needed for either statement) keeps no
statement limit in this database, since a CONCURRENTLY build waits out
every writer and reads its table twice; its lock waits stay the migrate
Job's lock_timeout. The downgrade resets all three.

Revision ID: 0011_media_session_limits
Revises: 0010_drop_prefix_indexes
"""

from __future__ import annotations

from alembic import op

revision = "0011_media_session_limits"
down_revision = "0010_drop_prefix_indexes"
branch_labels = None
depends_on = None

IDLE_IN_TRANSACTION = "180s"
STATEMENT = "120s"


def _on_this_database(*statements: str) -> None:
    """Each of ``statements`` (a format() string with %I for the database)
    for the database the migration runs in."""
    body = " ".join(f"EXECUTE format('{s}', current_database());" for s in statements)
    op.execute(f"DO $$ BEGIN {body} END $$")


def upgrade() -> None:
    _on_this_database(
        f"ALTER DATABASE %I SET idle_in_transaction_session_timeout = ''{IDLE_IN_TRANSACTION}''",
        f"ALTER DATABASE %I SET statement_timeout = ''{STATEMENT}''",
        "ALTER ROLE CURRENT_USER IN DATABASE %I SET statement_timeout = 0",
    )


def downgrade() -> None:
    _on_this_database(
        "ALTER ROLE CURRENT_USER IN DATABASE %I RESET statement_timeout",
        "ALTER DATABASE %I RESET statement_timeout",
        "ALTER DATABASE %I RESET idle_in_transaction_session_timeout",
    )
