"""Tests for CodeGraph — unified MultiDiGraph over trelix edge tables."""

from __future__ import annotations

from pathlib import Path

from trelix.core.models import (
    CallEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
    TypeEdge,
)
from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "index.db")
    return db


def _insert_file(db: Database, rel_path: str, lang: Language = Language.PYTHON) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=lang,
        hash="abc",
        size_bytes=100,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent_id: int | None = None,
) -> int:
    s = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=kind,
        line_start=1,
        line_end=10,
        signature=f"def {name}()",
        body=f"def {name}(): pass",
    )
    return db.insert_symbol(s)


class TestCodeGraphConstruction:
    def test_empty_db_builds_empty_graph(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        cg = CodeGraph(db)
        assert cg.node_count == 0
        assert cg.edge_count == 0

    def test_nodes_from_symbols(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        _insert_symbol(db, fid, "login")
        _insert_symbol(db, fid, "logout")
        cg = CodeGraph(db)
        assert cg.node_count == 2

    def test_call_edge_added(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid1 = _insert_symbol(db, fid, "login")
        sid2 = _insert_symbol(db, fid, "hash_password")
        db.insert_call_edges(
            [CallEdge(caller_id=sid1, callee_name="hash_password", callee_id=sid2, line=5)]
        )
        cg = CodeGraph(db)
        # CALLS edge: login → hash_password
        assert cg.edge_count >= 1
        neighbors = cg.neighbors(sid1)
        assert sid2 in neighbors

    def test_type_edge_extends(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "models.py")
        sid1 = _insert_symbol(db, fid, "AdminUser", SymbolKind.CLASS)
        sid2 = _insert_symbol(db, fid, "User", SymbolKind.CLASS)
        db.insert_type_edges(
            [
                TypeEdge(
                    from_symbol_id=sid1,
                    to_type_name="User",
                    edge_kind="extends",
                    to_symbol_id=sid2,
                )
            ]
        )
        cg = CodeGraph(db)
        neighbors = cg.neighbors(sid1)
        assert sid2 in neighbors

    def test_node_attributes(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        cg = CodeGraph(db)
        attrs = cg.nx.nodes[sid]
        assert attrs["name"] == "login"
        assert attrs["kind"] == SymbolKind.FUNCTION.value
        assert attrs["file"] == "auth.py"
        assert attrs["community"] is None

    def test_shortest_path_connected(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "handle_request")
        sid2 = _insert_symbol(db, fid, "authenticate")
        sid3 = _insert_symbol(db, fid, "hash_password")
        db.insert_call_edges(
            [
                CallEdge(caller_id=sid1, callee_name="authenticate", callee_id=sid2, line=3),
                CallEdge(caller_id=sid2, callee_name="hash_password", callee_id=sid3, line=7),
            ]
        )
        cg = CodeGraph(db)
        path = cg.shortest_path(sid1, sid3)
        assert path is not None
        assert path[0] == sid1
        assert path[-1] == sid3

    def test_shortest_path_disconnected(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "fn_a")
        sid2 = _insert_symbol(db, fid, "fn_b")
        cg = CodeGraph(db)
        assert cg.shortest_path(sid1, sid2) is None

    def test_subgraph(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "app.py")
        sid1 = _insert_symbol(db, fid, "fn_a")
        sid2 = _insert_symbol(db, fid, "fn_b")
        sid3 = _insert_symbol(db, fid, "fn_c")
        cg = CodeGraph(db)
        sg = cg.subgraph([sid1, sid2])
        assert sid1 in sg.nodes
        assert sid2 in sg.nodes
        assert sid3 not in sg.nodes
