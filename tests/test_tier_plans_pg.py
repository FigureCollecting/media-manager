"""IMGREND (IR-MM-1) on a real postgres: migration 0011 (public_tiers,
tier_parents) and the plans of every lookup the tiers add.

THE PLANS. The origin judges a tier on every request, and the lane reads the
registry and the ledger for every parent, so each of those reads must be an
index probe at prod scale: the MASKRATE plan guard (test_per_photo_plans_pg)
run over the tier lookups, on its ROWS-scale fill plus five rungs for every
public object (2.9 M tier rows) and a ledger row for each (576 k), at prod's
planner cost and at U5's. Hosts without docker skip.

THE MIGRATION. Two new tables and nothing else: public_objects gains no
column. Their foreign keys need a SHARE ROW EXCLUSIVE lock on public_objects
for the instant of the DDL, which waits for a writer's open transaction:
under the migrate Job's lock_timeout it fails whole (nothing half-made) and
a re-run applies it; a reader never holds it up.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import string
import subprocess
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.models import ImageMask, PublicObject, PublicTier, TierParent
from app.object_store import LocalObjectStore
from app.publication import tier_servable
from app.workers import tiers
from app.workers.mask import MASK_PIPELINE_VERSION, encode_mask_png
from app.workers.mask_seed import register_public_object
from app.workers.tiers import (
    TIER_PIPELINE_VERSION,
    ladder_for,
    mint_parent,
    refused_parents,
    register_public_tier,
    tier_key,
)
from PIL import Image
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import test_per_photo_plans_pg as plans
from test_migration_chain_replay import (  # noqa: F401  (fixtures)
    REPO_ROOT,
    _run_alembic,
    _schema_snapshot,
    empty_database_url,
    pg_admin_url,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker"),
    pytest.mark.timeout(600),  # the first plan test fills the tables (its fixture's setup)
]

BEFORE = "0010_drop_prefix_indexes"
HEAD = "0011_add_public_tiers"

#: Five rungs for every public object the plan guard's fill made, a ledger
#: row for each: 1 parent in 4096 refused, 1 failed, the rest done.
TIER_FILL = """
INSERT INTO public_tiers (parent_sha256, pipeline_version, long_edge_px, sha256, parent_kind,
                          mime, bytes, width, height, storage_key, created_at)
  SELECT p.sha256, '$version', e.edge,
         encode(sha256(convert_to('tier:' || p.sha256 || ':' || e.edge, 'UTF8')), 'hex'),
         p.kind, 'image/webp', 20000, e.edge, 1, 'tiers/' || p.sha256 || '/' || e.edge, now()
  FROM public_objects p CROSS JOIN (VALUES (128), (256), (512), (1024), (2048)) AS e(edge);
INSERT INTO tier_parents (parent_sha256, pipeline_version, state, reason, max_edge_px,
                          attempts, created_at, updated_at)
  SELECT sha256, '$version',
         CASE left(sha256, 3) WHEN '000' THEN 'refused' WHEN '001' THEN 'failed' ELSE 'done' END,
         CASE left(sha256, 3) WHEN '000' THEN 'parent_too_large' WHEN '001' THEN 'OSError' END,
         CASE WHEN left(sha256, 3) IN ('000', '001') THEN 0 ELSE 2048 END, 1, now(), now()
  FROM public_objects;
ANALYZE public_tiers;
ANALYZE tier_parents;
"""

TIER_INDEXES = {
    "public_tiers.public_tiers_pkey",
    "public_tiers.ix_public_tiers_sha256",
    "tier_parents.tier_parents_pkey",
    "tier_parents.ix_tier_parents_state",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _tier_of(parent: str, edge: int) -> str:
    return _sha(f"tier:{parent}:{edge}")


@pytest.fixture(scope="module")
def pg() -> Iterator[sessionmaker[Session]]:
    with plans.filled_postgres() as factory:
        with factory() as db:
            db.connection().exec_driver_sql("SET LOCAL session_replication_role = replica")
            db.connection().execution_options(no_parameters=True).exec_driver_sql(
                string.Template(TIER_FILL).substitute(version=TIER_PIPELINE_VERSION)
            )
            db.commit()
        yield factory


def _unprobed(pg: sessionmaker[Session], run: Any, used: set[str]) -> list[str]:
    """The whole-table reads among the statements ``run(db)`` sends, at both
    planner costs; ``used`` collects the indexes probed."""
    with pg() as db:
        statements = plans._captured(db.get_bind(), lambda: run(db))
        db.rollback()
    assert statements
    return plans._guarded(pg, statements, used)


# --------------------------------------------------------------------------
# the plans
# --------------------------------------------------------------------------


def test_the_fill_is_at_prod_scale(pg: sessionmaker[Session]) -> None:
    with pg() as db:
        tiers = db.execute(text("SELECT count(*) FROM public_tiers")).scalar_one()
        ledger = db.execute(text("SELECT count(*) FROM tier_parents")).scalar_one()
        refused = db.execute(
            text("SELECT count(*) FROM tier_parents WHERE state = 'refused'")
        ).scalar_one()
    assert tiers >= 2_000_000 and ledger >= 500_000 and 20 <= refused <= 1000


def test_the_guard_knows_the_tier_tables() -> None:
    assert {"public_tiers", "tier_parents"} <= set(plans.MEDIA_TABLES)


def _served(db: Session, sha: str) -> tuple[str, int] | None:
    """Whose rung the origin would serve at ``sha`` (no grants, no revocations)."""
    row = tier_servable(db, sha, {}, frozenset())
    return None if row is None else (row.parent_sha256, row.long_edge_px)


@pytest.mark.parametrize("kind", ["copy", "mask"])
def test_the_origins_judgment_of_a_tier_probes_every_table(
    pg: sessionmaker[Session], kind: str
) -> None:
    """The tier by its sha, a capture of its bytes, its parent, and the
    parent's whole judgment (app.publication.servable): a clean copy's rung
    is a hit; a mask's goes on through the mask rows and its image's origins
    (the fill's origins grant no mask, so that one ends a refusal)."""
    used: set[str] = set()
    sha = _tier_of(_sha(f"{kind}:1"), 256)
    found: list[Any] = []
    assert _unprobed(pg, lambda db: found.append(_served(db, sha)), used) == []
    assert "public_tiers.ix_public_tiers_sha256" in used
    assert "public_objects.public_objects_pkey" in used
    assert "images.images_sha256_key" in used
    if kind == "copy":
        assert found == [(_sha("copy:1"), 256)]
    else:
        assert found == [None]
        assert "image_masks.ix_image_masks_mask_sha256" in used
        assert "figure_galleries.ix_figure_galleries_image_id" in used


def test_a_tier_of_a_soft_deleted_image_is_judged_by_probes_too(
    pg: sessionmaker[Session],
) -> None:
    used: set[str] = set()
    sha = _tier_of(_sha("copy:97"), 512)  # the fill soft-deletes 1 image in 97
    found: list[Any] = []
    assert _unprobed(pg, lambda db: found.append(_served(db, sha)), used) == []
    assert found == [None]


def test_a_miss_reads_one_index_and_nothing_else(pg: sessionmaker[Session]) -> None:
    used: set[str] = set()
    found: list[Any] = []
    assert _unprobed(pg, lambda db: found.append(tier_servable(db, "0" * 64)), used) == []
    assert found == [None] and used == {"public_tiers.ix_public_tiers_sha256"}


def test_registering_a_tier_probes_the_three_registries(pg: sessionmaker[Session]) -> None:
    used: set[str] = set()
    parent = _sha("copy:2")

    def run(db: Session) -> None:
        register_public_tier(
            db,
            sha256=_sha("a new tier"),
            parent_sha256=parent,
            pipeline_version="mm-tier-plan",
            long_edge_px=256,
            size=10,
            width=256,
            height=100,
        )

    assert _unprobed(pg, run, used) == []
    assert used >= {
        "public_tiers.public_tiers_pkey",
        "public_objects.public_objects_pkey",
        "images.images_sha256_key",
    }


def _real_pair(
    db: Session, store: LocalObjectStore, label: str, picture: bytes | None = None
) -> tuple[str, str]:
    """A real clean copy and a real published mask of it, as rows and bytes."""
    clean = plans._seed(db, store, picture or plans._png(label), label)
    alpha = np.zeros((clean.height, clean.width), dtype=np.uint8)
    alpha[8:40, 8:30] = 255
    mask = encode_mask_png(alpha)
    mask_sha = hashlib.sha256(mask).hexdigest()
    key = f"derived/{mask_sha[:2]}/{mask_sha}.png"
    store.put(key, mask, "image/png")
    register_public_object(
        db,
        sha256=mask_sha,
        kind="mask",
        mime="image/png",
        size=len(mask),
        width=clean.width,
        height=clean.height,
        storage_key=key,
    )
    db.add(
        ImageMask(
            source_sha256=clean.sha256,
            pipeline_version=MASK_PIPELINE_VERSION,
            status="ok",
            mask_sha256=mask_sha,
            mask_storage_key=key,
        )
    )
    db.flush()
    return clean.sha256, mask_sha


def test_minting_a_parent_probes_the_ledger_the_registry_and_the_masks(
    pg: sessionmaker[Session], tmp_path: Path
) -> None:
    """The whole of mint_parent for a clean copy and for its mask, then the
    settled repeat of each (the ledger answers, the store is not read)."""
    used: set[str] = set()
    store = LocalObjectStore(tmp_path / "media")
    states: list[tuple[str, int]] = []

    def run(db: Session) -> None:
        for parent in _real_pair(db, store, "tier-plan-1"):
            first = mint_parent(db, store, parent)
            again = mint_parent(db, store, parent)
            states.extend([(first.state, first.minted), (again.state, again.minted)])

    assert _unprobed(pg, run, used) == []
    assert states == [("done", 1), ("done", 0)] * 2  # a 48 x 64 picture: its native rung
    assert used >= {
        "public_tiers.public_tiers_pkey",
        "tier_parents.tier_parents_pkey",
        "public_objects.public_objects_pkey",
        "image_masks.ix_image_masks_mask_sha256",
    }


def test_the_refused_parents_are_paged_on_the_ledgers_state_index(
    pg: sessionmaker[Session],
) -> None:
    used: set[str] = set()
    listed: list[str] = []

    def run(db: Session) -> None:
        first = [p.parent_sha256 for p in refused_parents(db, limit=10)]
        listed.extend(first)
        listed.extend(p.parent_sha256 for p in refused_parents(db, after=first[-1], limit=10))

    assert _unprobed(pg, run, used) == []
    assert len(listed) == 20 and listed == sorted(set(listed))
    assert all(sha.startswith("000") for sha in listed)
    assert used == {"tier_parents.ix_tier_parents_state"}


def test_every_tier_index_is_one_some_lookup_uses(pg: sessionmaker[Session]) -> None:
    """No index of the two tables is written for nothing: each is the one a
    lookup above probes (and the parents' cascade deletes by the two keys)."""
    with pg() as db:
        names = {
            f"{table}.{ix['name']}"
            for table in ("public_tiers", "tier_parents")
            for ix in inspect(db.connection()).get_indexes(table)
        } | {
            f"{table}.{inspect(db.connection()).get_pk_constraint(table)['name']}"
            for table in ("public_tiers", "tier_parents")
        }
    assert names == TIER_INDEXES


# --------------------------------------------------------------------------
# migration 0011
# --------------------------------------------------------------------------


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _version(url: str) -> str:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return str(conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one())
    finally:
        engine.dispose()


def test_0011_adds_the_two_tables_and_changes_no_other(
    empty_database_url: str,  # noqa: F811
    pg_admin_url: str,  # noqa: F811
) -> None:
    before = _run_alembic("upgrade", BEFORE, database_url=empty_database_url)
    assert before.returncode == 0, before.stdout + before.stderr
    engine = create_engine(empty_database_url)
    try:
        old_tables, old_indexes = _schema_snapshot(engine)
        up = _run_alembic("upgrade", "head", database_url=empty_database_url)
        assert up.returncode == 0, up.stdout + up.stderr
        assert _version(empty_database_url) == HEAD
        new_tables, new_indexes = _schema_snapshot(engine)
    finally:
        engine.dispose()
    assert set(new_tables) - set(old_tables) == {"public_tiers", "tier_parents"}
    assert {t: new_tables[t] for t in old_tables} == old_tables
    assert {t: new_indexes[t] for t in old_indexes} == old_indexes
    assert new_indexes["public_tiers"] == {("ix_public_tiers_sha256", ("sha256",))}
    assert new_indexes["tier_parents"] == {
        ("ix_tier_parents_state", ("parent_sha256", "pipeline_version", "state"))
    }


def test_0011_downgrades_to_exactly_the_schema_before_it(
    empty_database_url: str,  # noqa: F811
    pg_admin_url: str,  # noqa: F811
) -> None:
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    down = _run_alembic("downgrade", BEFORE, database_url=empty_database_url)
    assert down.returncode == 0, down.stdout + down.stderr
    assert not {"public_tiers", "tier_parents"} & _tables(empty_database_url)

    name = f"migtest_0010_{uuid.uuid4().hex[:8]}"
    admin = create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    before_url = pg_admin_url.rsplit("/", 1)[0] + f"/{name}"
    try:
        only = _run_alembic("upgrade", BEFORE, database_url=before_url)
        assert only.returncode == 0, only.stdout + only.stderr
        downgraded, before = create_engine(empty_database_url), create_engine(before_url)
        try:
            assert _schema_snapshot(downgraded) == _schema_snapshot(before)
        finally:
            downgraded.dispose()
            before.dispose()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
    again = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert again.returncode == 0, again.stdout + again.stderr


def test_0011_offline_sql_is_two_creates_in_one_transaction(pg_admin_url: str) -> None:  # noqa: F811
    sql = _run_alembic("upgrade", f"{BEFORE}:{HEAD}", "--sql", database_url=pg_admin_url)
    assert sql.returncode == 0, sql.stdout + sql.stderr
    out = sql.stdout
    assert out.count("CREATE TABLE public_tiers") == 1 == out.count("CREATE TABLE tier_parents")
    assert "CONCURRENTLY" not in out and "ALTER TABLE public_objects" not in out
    assert out.index("BEGIN;") < out.index("CREATE TABLE public_tiers") < out.index("COMMIT;")
    down = _run_alembic("downgrade", f"{HEAD}:{BEFORE}", "--sql", database_url=pg_admin_url)
    assert down.returncode == 0, down.stdout + down.stderr
    assert "DROP TABLE public_tiers" in down.stdout and "DROP TABLE tier_parents" in down.stdout


def _under_the_jobs_timeout(url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": url, "PGOPTIONS": "-c lock_timeout=1s"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_0011_behind_a_writers_open_transaction_fails_whole_and_applies_on_a_rerun(
    empty_database_url: str,  # noqa: F811
) -> None:
    """The foreign keys to public_objects need its SHARE ROW EXCLUSIVE lock,
    which a transaction that wrote public_objects holds off. Under the
    migrate Job's lock_timeout the migration fails instead of queueing every
    writer behind it, leaves nothing behind, and a re-run applies it."""
    before = _run_alembic("upgrade", BEFORE, database_url=empty_database_url)
    assert before.returncode == 0, before.stdout + before.stderr
    engine = create_engine(empty_database_url)
    try:
        with engine.connect() as seeder:
            seeder.execute(text("LOCK TABLE public_objects IN ROW EXCLUSIVE MODE"))
            blocked = _under_the_jobs_timeout(empty_database_url)
            assert blocked.returncode != 0
            assert "lock timeout" in blocked.stdout + blocked.stderr
            seeder.rollback()
        assert _version(empty_database_url) == BEFORE
        assert not {"public_tiers", "tier_parents"} & _tables(empty_database_url)
        rerun = _under_the_jobs_timeout(empty_database_url)
        assert rerun.returncode == 0, rerun.stdout + rerun.stderr
        assert _version(empty_database_url) == HEAD
    finally:
        engine.dispose()


def test_0011_is_not_held_up_by_a_reader_and_holds_up_no_reader(
    empty_database_url: str,  # noqa: F811
) -> None:
    """The origin reads public_objects on every request: a read inside an
    open transaction neither blocks the migration nor waits for it."""
    before = _run_alembic("upgrade", BEFORE, database_url=empty_database_url)
    assert before.returncode == 0, before.stdout + before.stderr
    engine = create_engine(empty_database_url)
    try:
        with engine.connect() as reader:
            reader.execute(text("SELECT count(*) FROM public_objects")).scalar_one()
            up = _under_the_jobs_timeout(empty_database_url)
            assert up.returncode == 0, up.stdout + up.stderr
            reader.execute(text("SELECT count(*) FROM public_objects")).scalar_one()
    finally:
        engine.dispose()
    assert _version(empty_database_url) == HEAD


def test_postgres_holds_the_tier_shape_and_the_ledger_states(
    empty_database_url: str,  # noqa: F811
) -> None:
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    parent = "a" * 64

    def tier(**over: object) -> PublicTier:
        fields: dict[str, object] = {
            "sha256": "1" * 64,
            "parent_sha256": parent,
            "parent_kind": "mask",
            "pipeline_version": "v",
            "long_edge_px": 256,
            "mime": "image/webp",
            "bytes": 10,
            "width": 256,
            "height": 192,
            "storage_key": tier_key("1" * 64),
        }
        fields.update(over)
        return PublicTier(**fields)

    try:
        with factory() as db:
            db.add(
                PublicObject(
                    sha256=parent,
                    kind="mask",
                    mime="image/png",
                    bytes=1,
                    width=1,
                    height=1,
                    storage_key="k",
                )
            )
            db.commit()
            for bad in (
                tier(parent_kind="matted"),
                tier(long_edge_px=0, width=0, height=0),
                tier(long_edge_px=4097, width=4097),
                tier(width=100),
                tier(height=300),
                tier(bytes=0),
                tier(mime="image/png"),
                tier(parent_sha256="0" * 64),
                TierParent(
                    parent_sha256=parent,
                    pipeline_version="v",
                    state="skipped",
                    max_edge_px=0,
                    attempts=0,
                ),  # fmt: skip
                TierParent(
                    parent_sha256="0" * 64,
                    pipeline_version="v",
                    state="done",
                    max_edge_px=0,
                    attempts=0,
                ),  # fmt: skip
            ):
                db.add(bad)
                with pytest.raises(IntegrityError):
                    db.commit()
                db.rollback()
            db.add(tier())
            db.add(tier(sha256="1" * 64, long_edge_px=4096, width=4096, height=1))
            db.add(
                TierParent(
                    parent_sha256=parent,
                    pipeline_version="v",
                    state="done",
                    max_edge_px=2048,
                    attempts=1,
                )  # fmt: skip
            )
            db.commit()
            # the parent's row takes its tiers and its ledger row with it
            db.execute(text("DELETE FROM public_objects WHERE sha256 = :s"), {"s": parent})
            db.commit()
            assert db.execute(select(PublicTier)).all() == []
            assert db.execute(select(TierParent)).all() == []
    finally:
        engine.dispose()


class _CountingStore(LocalObjectStore):
    """A store that counts the parent reads (``get``) it is asked for."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes:
        with self._lock:
            self.reads += 1
        return super().get(key)


def _a_big_picture(width: int, height: int) -> bytes:
    """A picture of its own, noisy enough that every rung differs, that is
    still cheap to make."""
    rng = np.random.default_rng(width * 10_000 + height)
    arr = rng.integers(0, 256, (height // 20, width // 20, 3), dtype=np.uint8)
    picture = Image.fromarray(arr).resize((width, height), Image.Resampling.NEAREST)
    out = io.BytesIO()
    picture.save(out, format="PNG")
    return out.getvalue()


@pytest.mark.parametrize("parent_at", [0, 1], ids=["clean copy", "mask"])
def test_two_workers_minting_one_parent_at_once_leave_one_set_of_rows_and_one_loser(
    empty_database_url: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_at: int,
) -> None:
    """Both workers read the empty registry and ledger, then a barrier holds
    each at its first register until the other is there, so both insert the
    same keys. The first insert wins; the second fails on the primary key with
    an IntegrityError and settles nothing (the caller rolls it back). The rows
    are exactly the ladder, once, with one ledger row; the stored bytes are the
    rows'; and the loser's retry settles from the ledger without reading the
    store (the contract U5's parallel lane relies on)."""
    up = _run_alembic("upgrade", "head", database_url=empty_database_url)
    assert up.returncode == 0, up.stdout + up.stderr
    engine = create_engine(empty_database_url, pool_size=6)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = _CountingStore(tmp_path / "media")
    try:
        with factory() as db:
            pair = _real_pair(db, store, "tier-race", _a_big_picture(1300, 900))
            db.commit()
        parent = pair[parent_at]

        barrier = threading.Barrier(2, timeout=30)
        arrived: set[int] = set()
        real_register = tiers.register_public_tier

        def gated(db: Session, **fields: Any) -> PublicTier:
            me = threading.get_ident()
            if me not in arrived:
                arrived.add(me)
                barrier.wait()  # both are past their ledger and registry reads
            return real_register(db, **fields)

        monkeypatch.setattr(tiers, "register_public_tier", gated)
        outcomes: dict[str, object] = {}

        def worker(name: str) -> None:
            with factory() as db:
                try:
                    outcomes[name] = mint_parent(db, store, parent)
                    db.commit()
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    outcomes[name] = exc

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("w1", "w2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert not any(thread.is_alive() for thread in threads)
        monkeypatch.setattr(tiers, "register_public_tier", real_register)

        losers = [v for v in outcomes.values() if isinstance(v, Exception)]
        winners = [v for v in outcomes.values() if not isinstance(v, Exception)]
        assert len(losers) == 1 and len(winners) == 1, outcomes
        assert isinstance(losers[0], IntegrityError)
        assert "public_tiers_pkey" in str(losers[0])
        want = list(ladder_for(1300, 900))
        assert (winners[0].state, winners[0].minted) == ("done", len(want))  # type: ignore[union-attr]

        with factory() as db:
            rows = list(
                db.execute(select(PublicTier).where(PublicTier.parent_sha256 == parent)).scalars()
            )
            ledger = list(
                db.execute(select(TierParent).where(TierParent.parent_sha256 == parent)).scalars()
            )
            assert sorted(row.long_edge_px for row in rows) == want  # none twice, none missing
            assert [(t.state, t.max_edge_px, t.attempts) for t in ledger] == [("done", 2048, 1)]
            for row in rows:
                assert hashlib.sha256(store.get(row.storage_key)).hexdigest() == row.sha256

            store.reads = 0
            again = mint_parent(db, store, parent)
            db.commit()
            assert (again.state, again.minted, store.reads) == ("done", 0, 0)
            assert [t.long_edge_px for t in again.tiers] == want
    finally:
        engine.dispose()
