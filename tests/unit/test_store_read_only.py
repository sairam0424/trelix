"""
Read-only opens must be read-only AND must open the file they were asked for.

`Database(..., read_only=True)` and `ReadOnlyConnectionPool` build their connection
string as a SQLite URI. While the path was interpolated raw, a `#` in it ended the URI
as a fragment and a `?` in it opened a second query section, so `mode=ro` was dropped
and SQLite opened — and *created* — a different file in read-write mode. That handle
accepted writes (the flag was a lie) and could not see the index's tables (the reads
were answered from an empty database), and a stray database appeared at the truncated
path. `trelix index --dry-run --prune` is the shipped caller, a command whose whole
promise is that it changes nothing.

Every other test of this behaviour builds its path from pytest's `tmp_path`, which
never contains a URI metacharacter, so it passed against the broken code. These tests
name the directory themselves. Both halves of each pair matter: a raise alone would
also be produced by a handle pointed at some other read-only file, and a visible table
alone would also be produced by a writable handle on the right file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database, read_only_uri

# A directory name carrying both URI metacharacters at once: `#` truncates the URI and
# `?` starts a query section, and they fail differently, so neither substitutes for the
# other. Kept off `tmp_path`'s own name because pytest owns that.
HOSTILE_DIR = "my #2 repo?v=1"

SEEDED = IndexedFile(
    path="/repo/src/auth/login.py",
    rel_path="src/auth/login.py",
    language=Language.PYTHON,
    hash="abc123",
    size_bytes=1024,
)


def _seed_index(tmp_path: Path) -> Path:
    """Write a real index (schema + one `files` row) under a URI-hostile directory."""
    db_path = tmp_path / HOSTILE_DIR / "index.db"
    db = Database(db_path)
    db.upsert_file(SEEDED)
    db.close()
    return db_path


class TestReadOnlyUri:
    def test_uri_metacharacters_are_percent_encoded(self, tmp_path: Path) -> None:
        uri = read_only_uri(tmp_path / HOSTILE_DIR / "index.db")

        # The mode parameter has to survive as the only query section; a bare `#` or `?`
        # in the path would strip or shadow it.
        assert uri.endswith("?mode=ro")
        assert "%23" in uri and "%3F" in uri
        assert "#" not in uri
        assert uri.count("?") == 1

    def test_relative_path_still_produces_a_usable_uri(self) -> None:
        """`Path.as_uri()` raises on a relative path, and the documented constructor
        example is `Database(Path(".trelix/index.db"))` — so the encoding must not
        require an absolute path."""
        assert read_only_uri(Path(".trelix/index.db")) == "file:.trelix/index.db?mode=ro"


class TestDatabaseReadOnlyUnderUriMetacharacters:
    def test_write_through_read_only_handle_raises(self, tmp_path: Path) -> None:
        db_path = _seed_index(tmp_path)

        with Database(db_path, read_only=True) as db:
            with pytest.raises(sqlite3.OperationalError):
                db._conn.execute("CREATE TABLE audit_smuggled (id INTEGER)")
                db._conn.commit()

    def test_seeded_table_is_visible_through_read_only_handle(self, tmp_path: Path) -> None:
        db_path = _seed_index(tmp_path)

        with Database(db_path, read_only=True) as db:
            assert db.get_file_hash(SEEDED.rel_path) == SEEDED.hash

    def test_no_stray_database_is_created_beside_the_index(self, tmp_path: Path) -> None:
        """The truncated path (`…/my ` for a `#`) is where SQLite used to create the
        replacement database — a dry run writing a file into the parent directory.
        Asserted without reading, so this stays a statement about the filesystem even
        when the handle happens to answer queries."""
        db_path = _seed_index(tmp_path)
        before = sorted(p.name for p in tmp_path.iterdir())

        with Database(db_path, read_only=True):
            pass

        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_relative_path_open_still_works(self, tmp_path: Path, monkeypatch) -> None:
        """Guards the encoding choice, not the escaping: `as_uri()` would reject this."""
        _seed_index(tmp_path)
        monkeypatch.chdir(tmp_path)

        with Database(Path(HOSTILE_DIR) / "index.db", read_only=True) as db:
            assert db.get_file_hash(SEEDED.rel_path) == SEEDED.hash


class TestReadPoolUnderUriMetacharacters:
    def test_pooled_connection_is_read_only_and_sees_the_index(self, tmp_path: Path) -> None:
        from trelix.store.read_pool import ReadOnlyConnectionPool

        db_path = _seed_index(tmp_path)
        pool = ReadOnlyConnectionPool(db_path, pool_size=2)
        try:
            with pool.acquire() as conn:
                row = conn.execute(
                    "SELECT hash FROM files WHERE rel_path = ?", (SEEDED.rel_path,)
                ).fetchone()
                assert row is not None and row[0] == SEEDED.hash

                with pytest.raises(sqlite3.OperationalError):
                    conn.execute("CREATE TABLE audit_smuggled (id INTEGER)")
                    conn.commit()
        finally:
            pool.close_all()
