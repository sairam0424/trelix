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


class TestGetTopCentralSymbolsOnAFreshIndex:
    """`get_top_central_symbols` must work before `trelix graph` has ever run.

    `graph_metadata` is not part of the base schema — it is created on demand by
    `_ensure_table`. `save_graph_metadata` and `load_graph_metadata` both call it;
    `get_top_central_symbols` did not, so on any index that had not had `trelix graph`
    run it raised `OperationalError: no such table: graph_metadata`.

    `Retriever._apply_pagerank_boost` catches that and logs at DEBUG, while the CLI runs
    at WARNING. The result was an enabled retrieval feature
    (`TRELIX_RETRIEVAL_PAGERANK_BOOST=true`) that had never once fired and emitted no
    diagnostic saying so.
    """

    def test_returns_empty_instead_of_raising(self, tmp_path: Path) -> None:
        from trelix.graph.persistence import get_top_central_symbols
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        assert "graph_metadata" not in {
            r[0] for r in db._conn.execute("SELECT name FROM sqlite_master").fetchall()
        }, "precondition: graph_metadata must be absent for this test to mean anything"

        assert get_top_central_symbols(db, top_n=10) == []

    def test_pagerank_boost_is_a_no_op_not_a_crash(self, tmp_path: Path) -> None:
        """The retriever's boost path must survive a graph-less index."""
        from trelix.graph.persistence import get_top_central_symbols
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        get_top_central_symbols(db, top_n=200)  # would previously raise
        # Calling it twice must also be safe (idempotent table creation).
        assert get_top_central_symbols(db, top_n=200) == []

    def test_still_returns_rows_once_metadata_exists(self, tmp_path: Path) -> None:
        """The fix must not mask a working graph."""
        from trelix.graph.persistence import _ensure_table, get_top_central_symbols
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        _ensure_table(db)
        db._conn.executemany(
            "INSERT INTO graph_metadata (symbol_id, community, centrality, node_type) "
            "VALUES (?, ?, ?, ?)",
            [(1, 0, 0.10, "symbol"), (2, 0, 0.90, "symbol"), (3, 0, 0.50, "symbol")],
        )
        db._conn.commit()

        assert get_top_central_symbols(db, top_n=2) == [2, 3]
