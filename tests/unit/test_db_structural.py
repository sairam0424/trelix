"""Structural guarantees of the SQLite schema layer.

Two properties are pinned here, both of which were silently violated on the
live index (`.trelix/index.db`, 192 MB, 30 user tables) before this file existed:

1. Deleting a symbol removes its `def_use_edges`. That table is the largest in
   the index (117,814 rows on the live index) and carries no foreign key, so
   until `_purge_def_use_edges()` shipped every file deletion leaked rows.
   Measured on the live index at the time of the fix: 4,768 of 117,814 rows
   (4.0%) pointed at a `symbols.id` that no longer existed.

2. `pragma user_version` identifies the schema generation, so an older reader
   refuses a newer index instead of misreading it. It read 0 on the live index
   across ~17 unversioned `CREATE TABLE IF NOT EXISTS`/`ALTER` migration blocks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.analysis.defuse import DefUseEdge
from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import SCHEMA_VERSION, Database, SchemaVersionError


def _make_file(db: Database, rel_path: str = "a.py") -> int:
    return db.upsert_file(
        IndexedFile(
            path=f"/r/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash="h",
            size_bytes=10,
        )
    )


def _make_symbol(db: Database, file_id: int, name: str) -> int:
    return db.insert_symbol(
        Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=10,
            signature=f"def {name}()",
            body=f"def {name}(): x = 1; return x",
        )
    )


def _add_edges(db: Database, symbol_id: int, count: int = 2) -> None:
    db.insert_def_use_edges(
        [
            DefUseEdge(
                symbol_id=symbol_id,
                var_name=f"v{i}",
                def_line=1 + i,
                use_line=2 + i,
                edge_type="def",
            )
            for i in range(count)
        ]
    )


def _edge_count(db: Database) -> int:
    return int(db._conn.execute("SELECT COUNT(*) FROM def_use_edges").fetchone()[0])


def _orphan_edge_count(db: Database) -> int:
    return int(
        db._conn.execute(
            "SELECT COUNT(*) FROM def_use_edges d "
            "LEFT JOIN symbols s ON s.id = d.symbol_id WHERE s.id IS NULL"
        ).fetchone()[0]
    )


class TestDefUseEdgeCleanup:
    """Every path that removes a symbol must remove its def-use edges."""

    def test_watcher_file_deletion_removes_def_use_edges(self, tmp_path: Path) -> None:
        """delete_file_by_path() is the watcher's path — it ran on every deleted
        file in a watched repo and left every def-use edge behind."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db)
        _add_edges(db, _make_symbol(db, file_id, "fn"), count=3)
        assert _edge_count(db) == 3

        assert db.delete_file_by_path("/r/a.py", "a.py") is True

        assert _edge_count(db) == 0
        assert _orphan_edge_count(db) == 0
        db.close()

    def test_reindex_symbol_wipe_removes_def_use_edges(self, tmp_path: Path) -> None:
        """delete_file_symbols() is the full re-index path: edges accumulated
        once per re-index of a file whose symbols all changed."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db)
        _add_edges(db, _make_symbol(db, file_id, "fn"), count=3)

        db.delete_file_symbols(file_id)

        assert _edge_count(db) == 0
        db.close()

    def test_partial_symbol_delete_removes_only_that_symbol_edges(self, tmp_path: Path) -> None:
        """delete_symbols_by_qualified_names() is the content-hash-diffed path.
        It must purge the changed symbol's edges and preserve the unchanged
        symbol's, which the current pass never re-inserts."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db)
        changed = _make_symbol(db, file_id, "changed")
        unchanged = _make_symbol(db, file_id, "unchanged")
        _add_edges(db, changed, count=2)
        _add_edges(db, unchanged, count=4)

        db.delete_symbols_by_qualified_names(file_id, ["changed"])

        assert _edge_count(db) == 4
        assert {e.symbol_id for e in db.get_data_flows(unchanged)} == {unchanged}
        assert db.get_data_flows(changed) == []
        db.close()

    def test_other_files_edges_survive_a_deletion(self, tmp_path: Path) -> None:
        """The purge is keyed on the deleted symbol ids, not a blanket wipe."""
        db = Database(tmp_path / "index.db")
        doomed_file = _make_file(db, "doomed.py")
        kept_file = _make_file(db, "kept.py")
        _add_edges(db, _make_symbol(db, doomed_file, "gone"), count=2)
        kept_symbol = _make_symbol(db, kept_file, "kept")
        _add_edges(db, kept_symbol, count=5)

        db.delete_file_by_path("/r/doomed.py", "doomed.py")

        assert _edge_count(db) == 5
        assert len(db.get_data_flows(kept_symbol)) == 5
        db.close()

    def test_sparse_embeddings_cleared_for_deleted_chunks(self, tmp_path: Path) -> None:
        """sparse_embeddings.chunk_id has the same shape as def_use_edges.symbol_id:
        no REFERENCES, and SparseStore only ever deletes a chunk_id it is about to
        re-upsert. Chunks die by cascade from symbols, so the rows were unreachable
        and permanent."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db)
        symbol_id = _make_symbol(db, file_id, "fn")
        chunk_id = db.insert_chunk(Chunk(symbol_id=symbol_id, chunk_text="x = 1", token_count=3))
        db._conn.execute(
            "INSERT INTO sparse_embeddings (chunk_id, token_id, weight) VALUES (?, ?, ?)",
            (chunk_id, 7, 0.5),
        )
        db._conn.commit()

        db.delete_file_symbols(file_id)

        remaining = db._conn.execute("SELECT COUNT(*) FROM sparse_embeddings").fetchone()[0]
        assert remaining == 0
        db.close()


class TestSchemaVersion:
    """`pragma user_version` gates the schema generation."""

    def test_new_index_is_stamped(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        assert db.schema_version() == SCHEMA_VERSION
        assert SCHEMA_VERSION >= 1
        db.close()

    def test_legacy_unversioned_index_is_upgraded_not_rejected(self, tmp_path: Path) -> None:
        """Every index in the wild reads 0. That means "pre-versioning", NOT
        corrupt — it must open, keep its rows, and get stamped."""
        db_path = tmp_path / "index.db"
        db = Database(db_path)
        file_id = _make_file(db)
        _add_edges(db, _make_symbol(db, file_id, "fn"), count=2)
        db.close()

        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA user_version = 0")
        raw.commit()
        raw.close()

        reopened = Database(db_path)
        assert reopened.schema_version() == SCHEMA_VERSION
        assert reopened.get_file_hash("a.py") == "h"
        assert _edge_count(reopened) == 2
        reopened.close()

    def test_newer_index_is_refused(self, tmp_path: Path) -> None:
        """An older reader must refuse rather than misread. Without this, a
        3.1.1 install opens a 3.2.0-written index silently."""
        db_path = tmp_path / "index.db"
        Database(db_path).close()

        raw = sqlite3.connect(str(db_path))
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        raw.commit()
        raw.close()

        with pytest.raises(SchemaVersionError) as exc:
            Database(db_path)
        assert str(SCHEMA_VERSION + 1) in str(exc.value)
        assert "trelix" in str(exc.value)

    def test_upgrade_reclaims_preexisting_orphans(self, tmp_path: Path) -> None:
        """The 4,768 orphans already on the live index have no other reclaimer.
        The one-time sweep is gated on the version stamp so it costs nothing on
        every subsequent open."""
        db_path = tmp_path / "index.db"
        db = Database(db_path)
        file_id = _make_file(db)
        live_symbol = _make_symbol(db, file_id, "fn")
        _add_edges(db, live_symbol, count=2)
        # Orphans of the exact shape the leak produced: symbol row already gone.
        db._conn.execute(
            "INSERT INTO def_use_edges (symbol_id, var_name, def_line, use_line, edge_type) "
            "VALUES (?, 'ghost', 1, 2, 'def')",
            (live_symbol + 9000,),
        )
        db._conn.commit()
        db._conn.execute("PRAGMA user_version = 0")
        db._conn.commit()
        db.close()

        reopened = Database(db_path)
        assert _orphan_edge_count(reopened) == 0
        assert len(reopened.get_data_flows(live_symbol)) == 2
        reopened.close()

    def test_reopen_at_current_version_skips_the_sweep(self, tmp_path: Path) -> None:
        """An already-stamped index must not pay for the reclaim sweep again —
        it is an anti-join over the largest table in the index."""
        db_path = tmp_path / "index.db"
        Database(db_path).close()

        reopened = Database(db_path)
        calls: list[int] = []
        original = reopened._reclaim_orphaned_def_use_edges
        reopened._reclaim_orphaned_def_use_edges = lambda: calls.append(1)  # type: ignore[method-assign]
        reopened.init_schema()
        reopened._reclaim_orphaned_def_use_edges = original  # type: ignore[method-assign]
        assert calls == []
        reopened.close()


class TestForeignKeyEnforcement:
    """CASCADE is only load-bearing if the pragma is actually on."""

    def test_foreign_keys_pragma_is_on(self, tmp_path: Path) -> None:
        """`PRAGMA foreign_keys = ON` sits in the DDL executescript. If it ever
        stopped taking effect, every ON DELETE CASCADE in this schema would
        become a silent no-op and the explicit purges here would be the only
        cleanup left."""
        db = Database(tmp_path / "index.db")
        assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        db.close()


class TestFileSummaryVectorIsDeletedWithTheFile:
    """`delete_file_by_path` must remove the file-summary vector too.

    The summary vector lives in the same `chunk_embeddings` table as chunk vectors, under
    the `chunk_id = -(file_id)` sentinel. So it is NOT returned by
    `get_chunk_ids_for_file()`, and the `delete_batch(chunk_ids)` call never reached it —
    while the `file_summaries` row itself cascades away with the `files` row. The result was
    a vector with neither a summary nor a file behind it.

    Found by `verify-index.sh`'s "summary vectors == summaries" gate the first time a real
    `--prune` removed a file: 482 vectors against 481 summaries, the orphan being `-475` for
    a `file_id` that no longer existed. The leak predates `--prune` — the watcher's delete
    path has taken it for as long as file summaries have existed.
    """

    class _RecordingStore:
        """Captures what delete_batch was asked to remove.

        A recorder rather than a real store because the assertion is about which IDS the
        caller passes, and the sentinel is a convention of that caller — a real vec0 table
        would confirm the rows vanish without showing that the negative id was ever sent.
        """

        def __init__(self) -> None:
            self.deleted: list[int] = []

        def delete_batch(self, chunk_ids: list[int]) -> None:
            self.deleted.extend(chunk_ids)

    def test_the_summary_sentinel_is_included_in_the_delete(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.core.models import IndexedFile, Language
        from trelix.store.db import Database

        with Database(tmp_path / "index.db") as db:
            file_id = db.upsert_file(
                IndexedFile(
                    path="/repo/gone.py",
                    rel_path="gone.py",
                    language=Language.PYTHON,
                    hash="h",
                    size_bytes=10,
                )
            )
            db._conn.commit()

            store = self._RecordingStore()
            assert db.delete_file_by_path("/repo/gone.py", "gone.py", store) is True

        assert -file_id in store.deleted, (
            f"the file-summary sentinel -{file_id} was not deleted, so its vector outlives "
            f"both the file and the summary; delete_batch saw {store.deleted}"
        )

    def test_a_missing_file_deletes_nothing(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Guards the test above against passing on a delete that fires unconditionally."""
        from trelix.store.db import Database

        with Database(tmp_path / "index.db") as db:
            store = self._RecordingStore()
            assert db.delete_file_by_path("/repo/absent.py", "absent.py", store) is False

        assert store.deleted == [], store.deleted
