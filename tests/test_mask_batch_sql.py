"""MASK-BATCH-1: the work-list SQL, run the way the operator runs it.

The queries are printed by ``python -m app.cli.mask_batch sql ...`` and run by
the operator against the coordinator and spine databases through psql, in a
read-only transaction (``default_transaction_read_only=on``). This runs them
through psql in a disposable PostgreSQL 17 container (pg-spine's major)
against a MIRROR of the tables they read: the columns, types and keys copied
from fc-coordinator 0002/0003 (``facet_state``) and fc-aggregation
0001/0005/0026/0027 (``product``, ``source``, ``raw.url``, ``raw.blob``,
``raw.capture``, ``product_image`` with its ``source_class``). It proves the SQL parses and selects the
right rows on those shapes; it is not the real migration chain. Hosts without
docker skip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Generator

import pytest
from app.cli import mask_batch as mb

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")

IMAGE = "postgres:17-alpine"

ROSS = "a0000000-0000-4000-8000-000000000001"
OTHER = "a0000000-0000-4000-8000-000000000002"
OWNED, ORDERED, WISHED, REMOVED, NOT_HIS = (
    f"b0000000-0000-4000-8000-00000000000{n}" for n in range(1, 6)
)
MERGED_INTO_OWNED = "b0000000-0000-4000-8000-000000000006"
CATALOG_ONLY = "b0000000-0000-4000-8000-000000000007"
GALLERY = "https://static.myfigurecollection.net/upload/pictures/2026/10/27/{n}.jpeg"

COORDINATOR_DDL = """
CREATE TABLE facet_state (
  user_id     uuid NOT NULL,
  facet_key   text COLLATE "C" NOT NULL,
  version     text COLLATE "C" NOT NULL,
  op          text NOT NULL CHECK (op IN ('upsert', 'delete')),
  payload     text NOT NULL,
  seq         bigint NOT NULL,
  PRIMARY KEY (user_id, facet_key),
  CHECK ((op = 'delete') = (payload = ''))
);
"""

SPINE_DDL = """
CREATE SCHEMA raw;
CREATE TABLE source (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  site text NOT NULL UNIQUE,
  kind text NOT NULL
);
CREATE TABLE product (
  id uuid PRIMARY KEY,
  status text NOT NULL DEFAULT 'active',
  redirect_to_product_id uuid REFERENCES product(id),
  CONSTRAINT product_merge_consistent CHECK ((status='merged') = (redirect_to_product_id IS NOT NULL))
);
CREATE TABLE raw.url (
  id bigserial PRIMARY KEY,
  url text NOT NULL,
  url_sha256 bytea NOT NULL,
  host text NOT NULL
);
CREATE TABLE raw.blob (
  sha256 bytea PRIMARY KEY,
  bytes bigint NOT NULL,
  codec text NOT NULL DEFAULT 'zstd',
  storage_key text NOT NULL,
  content_type text,
  visibility text NOT NULL DEFAULT 'visible'
);
CREATE TABLE raw.capture (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  fetched_at timestamptz NOT NULL,
  url_id bigint NOT NULL REFERENCES raw.url(id),
  lane text NOT NULL,
  blob_sha256 bytea REFERENCES raw.blob(sha256),
  engine_version text NOT NULL,
  PRIMARY KEY (id, fetched_at)
);
CREATE TABLE product_image (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  product_id uuid NOT NULL REFERENCES product(id),
  source_id uuid NOT NULL REFERENCES source(id),
  blob_sha256 bytea NOT NULL REFERENCES raw.blob(sha256),
  origin_capture_id uuid,
  origin_captured_at timestamptz,
  role text NOT NULL,
  position integer,
  state text NOT NULL DEFAULT 'captured',
  suppressed_at timestamptz,
  source_class text CHECK (source_class IS NULL OR source_class IN
    ('manufacturer_press','retailer_studio','user_photo','unknown')),
  FOREIGN KEY (origin_capture_id, origin_captured_at) REFERENCES raw.capture(id, fetched_at)
);
"""


def _facets(user: str, occ: str, head: str, status: str | None) -> str:
    rows = [
        f"('{user}', 'occ/{occ}/head', 'v', 'upsert', '{json.dumps({'head_id': head})}', 1)",
    ]
    if status is None:
        rows.append(f"('{user}', 'occ/{occ}/status', 'v', 'delete', '', 2)")
    else:
        rows.append(
            f"('{user}', 'occ/{occ}/status', 'v', 'upsert', '{json.dumps({'status': status})}', 2)"
        )
    return "INSERT INTO facet_state VALUES " + ", ".join(rows) + ";\n"


COORDINATOR_ROWS = "".join(
    [
        _facets(ROSS, "o1", OWNED, "owned"),
        _facets(ROSS, "o2", OWNED, "wished"),  # a second copy: the better rank wins
        _facets(ROSS, "o3", ORDERED, "ordered"),
        _facets(ROSS, "o4", WISHED, "wished"),
        _facets(ROSS, "o5", REMOVED, None),  # the copy was removed
        _facets(OTHER, "o6", NOT_HIS, "owned"),
    ]
)


def _sha(n: int) -> str:
    return f"{n:02x}" * 32


def _image(n: int, product: str, *, site: str = "mfc", position: int | None = 0, **kw: str) -> str:
    url = kw.get("url", f"https://static.myfigurecollection.net/upload/items/1/{n}.jpg")
    state = kw.get("state", "captured")
    visibility = kw.get("visibility", "visible")
    source_class = f"'{kw['source_class']}'" if "source_class" in kw else "NULL"

    return f"""
INSERT INTO raw.url (id, url, url_sha256, host) VALUES ({n}, '{url}', '\\x{_sha(n)}', 'h');
INSERT INTO raw.blob (sha256, bytes, codec, storage_key, visibility)
  VALUES ('\\x{_sha(n)}', 10, 'identity', 'raw-img/sha256/{_sha(n)[:2]}/{_sha(n)}.jpg', '{visibility}');
INSERT INTO raw.capture (id, fetched_at, url_id, lane, blob_sha256, engine_version)
  VALUES ('c0000000-0000-4000-8000-0000000000{n:02d}', '2026-09-10', {n}, 'asset', '\\x{_sha(n)}', 'e');
INSERT INTO product_image (product_id, source_id, blob_sha256, origin_capture_id, origin_captured_at,
                           role, position, state, suppressed_at, source_class)
  VALUES ('{product}', (SELECT id FROM source WHERE site = '{site}'), '\\x{_sha(n)}',
          'c0000000-0000-4000-8000-0000000000{n:02d}', '2026-09-10', 'gallery',
          {"NULL" if position is None else position}, '{state}',
          {"now()" if state == "suppressed" else "NULL"}, {source_class});
"""


SPINE_ROWS = (
    "INSERT INTO source (site, kind) VALUES ('mfc', 'database'), ('amiami', 'retailer');\n"
    + "".join(
        f"INSERT INTO product (id) VALUES ('{p}');\n"
        for p in (OWNED, ORDERED, WISHED, REMOVED, NOT_HIS, CATALOG_ONLY)
    )
    + f"INSERT INTO product (id, status, redirect_to_product_id) "
    f"VALUES ('{MERGED_INTO_OWNED}', 'merged', '{OWNED}');\n"
    + _image(11, OWNED, position=1)
    + _image(12, OWNED, position=0)
    + _image(13, MERGED_INTO_OWNED)
    + _image(14, OWNED, site="amiami", url="https://img.amiami.example/1.jpg")
    + _image(21, ORDERED, position=None)
    + _image(22, ORDERED, state="suppressed")
    + _image(31, WISHED)
    + _image(32, WISHED, visibility="suppressed")
    + _image(41, REMOVED)
    + _image(51, NOT_HIS)
    + _image(71, CATALOG_ONLY)
    # GAL3: the official gallery list, as the gallery-capture lane recorded it
    + _image(81, OWNED, url=GALLERY.format(n=81), source_class="manufacturer_press")
    + _image(82, CATALOG_ONLY, url=GALLERY.format(n=82), source_class="manufacturer_press")
    + _image(
        83, OWNED, url=GALLERY.format(n=83), source_class="manufacturer_press", state="suppressed"
    )
    + _image(84, OWNED, url=GALLERY.format(n=84))  # the same path, no class: not the list
    + _image(85, OWNED, source_class="unknown")
    + _image(
        86,
        OWNED,
        site="amiami",
        url="https://img.amiami.example/86.jpg",
        source_class="manufacturer_press",
    )
)


@pytest.fixture(scope="module")
def pg() -> Generator[str, None, None]:
    name = f"mm-maskbatch-sql-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-e", "POSTGRES_PASSWORD=postgres", IMAGE],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start {IMAGE}: {started.stderr}")
    try:
        deadline = time.monotonic() + 60
        while True:
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres", "-h", "127.0.0.1"],
                capture_output=True,
            )
            if ready.returncode == 0:
                break
            if time.monotonic() > deadline:  # pragma: no cover - a stuck container
                pytest.skip("postgres never became ready")
            time.sleep(0.5)
        for db, ddl in (
            ("coordinator", COORDINATOR_DDL + COORDINATOR_ROWS),
            ("spine", SPINE_DDL + SPINE_ROWS),
        ):
            _psql(name, "postgres", f"CREATE DATABASE {db};", read_only=False)
            _psql(name, db, ddl, read_only=False)
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, text=True)


def _psql(container: str, db: str, sql: str, *, read_only: bool = True) -> list[str]:
    """psql exactly as the operator's runner calls it (-X -A, ON_ERROR_STOP,
    the SQL on stdin), read only by default as there."""
    env = ["-e", "PGOPTIONS=-c default_transaction_read_only=on"] if read_only else []
    done = subprocess.run(
        ["docker", "exec", "-i", *env, container, "psql", "-U", "postgres", "-d", db]
        + ["-X", "-A", "-v", "ON_ERROR_STOP=1", "-f", "-"],
        input=sql,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_ross_heads_are_his_live_copies_ranked(pg: str) -> None:
    lines = _psql(pg, "coordinator", mb.ross_heads_sql(ROSS))
    rows = [json.loads(line) for line in lines if line.startswith("{")]
    assert rows == [
        {"head_id": OWNED, "rank": 0},
        {"head_id": ORDERED, "rank": 1},
        {"head_id": WISHED, "rank": 2},
    ]


def test_ross_worklist_is_his_heads_live_mfc_item_images(pg: str) -> None:
    heads = _psql(pg, "coordinator", mb.ross_heads_sql(ROSS))
    lines = _psql(pg, "spine", mb.ross_worklist_sql(heads))
    items, skipped = mb.parse_worklist(lines)
    assert skipped == 2  # psql echoes BEGIN and ROLLBACK; tuples only, no header or footer
    got = [(i.blob_sha256, i.rank, i.head_id, i.product_id) for i in items]
    assert got == [
        (_sha(12), 0, OWNED, OWNED),  # position 0 first
        (_sha(13), 0, OWNED, MERGED_INTO_OWNED),  # merged into his head
        *((_sha(n), 0, OWNED, OWNED) for n in (81, 84, 85)),  # position 0, by sha
        (_sha(11), 0, OWNED, OWNED),
        (_sha(21), 1, ORDERED, ORDERED),  # the suppressed depiction is left out
        (_sha(31), 2, WISHED, WISHED),  # the suppressed blob is left out
    ]
    first = items[0]
    assert first.storage_key == f"raw-img/sha256/{_sha(12)[:2]}/{_sha(12)}.jpg"
    assert (first.codec, first.source) == ("identity", "mfc")
    assert first.url == "https://static.myfigurecollection.net/upload/items/1/12.jpg"


def test_the_catalog_worklist_is_every_live_mfc_item_image(pg: str) -> None:
    items, _ = mb.parse_worklist(_psql(pg, "spine", mb.catalog_worklist_sql()))
    assert sorted(i.blob_sha256 for i in items) == sorted(
        _sha(n) for n in (11, 12, 13, 21, 31, 41, 51, 71, 81, 82, 84, 85)
    )
    classes = {i.blob_sha256: i.source_class for i in items}
    assert (classes[_sha(81)], classes[_sha(85)], classes[_sha(11)]) == (
        "manufacturer_press",
        "unknown",
        None,
    )
    assert {i.rank for i in items} == {9} and {i.head_id for i in items} == {None}
    shards = [
        {
            i.blob_sha256
            for i in mb.parse_worklist(_psql(pg, "spine", mb.catalog_worklist_sql(f"{k}/3")))[0]
        }
        for k in range(3)
    ]
    assert set().union(*shards) == {i.blob_sha256 for i in items}
    assert sum(len(s) for s in shards) == len(items)  # each original in exactly one shard


def test_the_official_gallery_worklist_is_the_gallery_lists_live_mfc_images(pg: str) -> None:
    """GAL3: only what the gallery-capture lane recorded as manufacturer_press,
    on MFC, live -- never a picture by its path alone."""
    items, _ = mb.parse_worklist(_psql(pg, "spine", mb.official_gallery_worklist_sql()))
    assert sorted(i.blob_sha256 for i in items) == sorted(_sha(n) for n in (81, 82))
    assert {i.source_class for i in items} == {"manufacturer_press"}
    assert {i.source for i in items} == {"mfc"} and {i.rank for i in items} == {9}
    by_sha = {i.blob_sha256: i for i in items}
    assert by_sha[_sha(81)].url == GALLERY.format(n=81)
    assert by_sha[_sha(81)].product_id == OWNED
    shards = [
        {
            i.blob_sha256
            for i in mb.parse_worklist(
                _psql(pg, "spine", mb.official_gallery_worklist_sql(f"{k}/2"))
            )[0]
        }
        for k in range(2)
    ]
    assert set().union(*shards) == set(by_sha) and sum(map(len, shards)) == len(items)
