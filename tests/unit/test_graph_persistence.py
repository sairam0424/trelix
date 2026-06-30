"""Tests for graph persistence — save/load community assignments."""

from __future__ import annotations

from pathlib import Path

from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.persistence import load_graph_metadata, save_graph_metadata
from trelix.store.db import Database


def _make_db_with_symbol(tmp_path: Path) -> tuple[Database, int]:
    db = Database(tmp_path / "index.db")
    f = IndexedFile(
        path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10
    )
    fid = db.upsert_file(f)
    s = Symbol(
        file_id=fid,
        name="fn",
        qualified_name="fn",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature="def fn()",
        body="def fn(): pass",
    )
    sid = db.insert_symbol(s)
    return db, sid


class TestGraphPersistence:
    def test_save_then_load_community(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        # Manually set community
        cg.nx.nodes[sid]["community"] = 42
        save_graph_metadata(db, cg)

        # Fresh graph — community should be None before load
        cg2 = CodeGraph(db)
        assert cg2.nx.nodes[sid]["community"] is None

        # After load, community should be restored
        load_graph_metadata(db, cg2)
        assert cg2.nx.nodes[sid]["community"] == 42

    def test_save_idempotent(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        cg.nx.nodes[sid]["community"] = 1
        save_graph_metadata(db, cg)
        cg.nx.nodes[sid]["community"] = 2
        save_graph_metadata(db, cg)  # Should overwrite, not error

        cg2 = CodeGraph(db)
        load_graph_metadata(db, cg2)
        assert cg2.nx.nodes[sid]["community"] == 2

    def test_missing_nodes_skipped_gracefully(self, tmp_path: Path) -> None:
        db, sid = _make_db_with_symbol(tmp_path)
        cg = CodeGraph(db)
        # Save with one community
        cg.nx.nodes[sid]["community"] = 5
        save_graph_metadata(db, cg)

        # New graph with no nodes — load should not crash
        import networkx as nx

        cg_empty = CodeGraph.__new__(CodeGraph)
        cg_empty._g = nx.MultiDiGraph()
        cg_empty._db = db
        load_graph_metadata(db, cg_empty)  # no crash
