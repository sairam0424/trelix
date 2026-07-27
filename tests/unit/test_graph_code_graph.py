"""Tests for CodeGraph — unified MultiDiGraph over trelix edge tables."""

from __future__ import annotations

from pathlib import Path

from trelix.core.models import (
    CallEdge,
    GenericEdge,
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


class TestCodeGraphGenericEdges:
    """Cross-source edges (e.g. git-log ticket links) surface as synthetic
    string-keyed artifact nodes, distinct from int-keyed symbol/file nodes."""

    def test_generic_edge_creates_artifact_node(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        db.insert_generic_edges(
            [GenericEdge(from_symbol_id=sid, source_ref="ticket:PROJ-1", edge_kind="references_ticket")]
        )
        cg = CodeGraph(db)
        assert "ticket:PROJ-1" in cg.nx
        assert cg.nx.nodes["ticket:PROJ-1"]["type"] == "artifact"

    def test_generic_edge_connects_symbol_to_artifact_node(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        db.insert_generic_edges(
            [GenericEdge(from_symbol_id=sid, source_ref="ticket:PROJ-1", edge_kind="references_ticket")]
        )
        cg = CodeGraph(db)
        neighbors = cg.neighbors(sid)
        assert "ticket:PROJ-1" in neighbors

    def test_generic_edge_label_uses_edge_kind_mapping(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        db.insert_generic_edges(
            [GenericEdge(from_symbol_id=sid, source_ref="ticket:PROJ-1", edge_kind="references_ticket")]
        )
        cg = CodeGraph(db)
        edge_data = cg.nx.get_edge_data(sid, "ticket:PROJ-1")
        assert any(d["label"] == "REFERENCES_TICKET" for d in edge_data.values())

    def test_unknown_generic_edge_kind_falls_back_gracefully(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid = _insert_symbol(db, fid, "login")
        db.insert_generic_edges(
            [GenericEdge(from_symbol_id=sid, source_ref="doc:readme", edge_kind="documents")]
        )
        cg = CodeGraph(db)
        edge_data = cg.nx.get_edge_data(sid, "doc:readme")
        assert any(d["label"] == "GENERIC_REL" for d in edge_data.values())

    def test_shared_artifact_node_deduplicated_across_symbols(self, tmp_path: Path) -> None:
        """Two symbols referencing the same ticket must share ONE artifact
        node, not create two separate nodes for the same source_ref."""
        db = _make_db(tmp_path)
        fid = _insert_file(db, "auth.py")
        sid1 = _insert_symbol(db, fid, "login")
        sid2 = _insert_symbol(db, fid, "logout")
        db.insert_generic_edges(
            [
                GenericEdge(from_symbol_id=sid1, source_ref="ticket:PROJ-1", edge_kind="references_ticket"),
                GenericEdge(from_symbol_id=sid2, source_ref="ticket:PROJ-1", edge_kind="references_ticket"),
            ]
        )
        cg = CodeGraph(db)
        artifact_nodes = [n for n, attrs in cg.nx.nodes(data=True) if attrs.get("type") == "artifact"]
        assert artifact_nodes == ["ticket:PROJ-1"]
        assert sid1 in cg.neighbors(sid2) or "ticket:PROJ-1" in cg.neighbors(sid2)
