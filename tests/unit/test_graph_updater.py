"""Tests for incremental graph updates after file changes."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.updater import GraphUpdater
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> tuple[Database, int, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="h1", size_bytes=50
        )
    )
    sid = db.insert_symbol(Symbol(
        file_id=fid, name="my_func", qualified_name="my_func",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
        signature="def my_func()", body="def my_func(): pass"
    ))
    return db, fid, sid


class TestGraphUpdater:
    def test_update_file_does_not_raise_on_valid_file(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should complete without error
        updater.update_file("a.py")

    def test_update_file_on_unknown_file_is_noop(self, tmp_path: Path) -> None:
        db, _, _ = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should not raise even if file not found
        updater.update_file("nonexistent.py")

    def test_graph_metadata_refreshed_after_update(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        updater.update_file("a.py")
        # graph_metadata should have an entry for the symbol
        row = db._conn.execute(
            "SELECT symbol_id FROM graph_metadata WHERE symbol_id = ?", (sid,)
        ).fetchone()
        assert row is not None
