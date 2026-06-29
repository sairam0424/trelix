"""Tests for community detection on CodeGraph."""

from __future__ import annotations

from pathlib import Path

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.store.db import Database


def _build_clustered_db(tmp_path: Path) -> tuple[Database, list[int]]:
    """Build a DB with two clearly separated clusters."""
    db = Database(tmp_path / "index.db")

    def _file(name: str) -> int:
        f = IndexedFile(
            path=f"/r/{name}", rel_path=name, language=Language.PYTHON, hash="x", size_bytes=10
        )
        return db.upsert_file(f)

    def _sym(fid: int, name: str) -> int:
        s = Symbol(
            file_id=fid,
            name=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature=f"def {name}()",
            body="pass",
        )
        return db.insert_symbol(s)

    # Cluster A: auth module (3 symbols, densely connected)
    fid_a = _file("auth.py")
    a1 = _sym(fid_a, "login")
    a2 = _sym(fid_a, "logout")
    a3 = _sym(fid_a, "hash_password")

    # Cluster B: db module (3 symbols, densely connected)
    fid_b = _file("db.py")
    b1 = _sym(fid_b, "query")
    b2 = _sym(fid_b, "insert")
    b3 = _sym(fid_b, "connect")

    # Dense intra-cluster edges
    db.insert_call_edges(
        [
            CallEdge(caller_id=a1, callee_name="hash_password", callee_id=a3, line=2),
            CallEdge(caller_id=a2, callee_name="hash_password", callee_id=a3, line=3),
            CallEdge(caller_id=b1, callee_name="connect", callee_id=b3, line=2),
            CallEdge(caller_id=b2, callee_name="connect", callee_id=b3, line=3),
        ]
    )

    return db, [a1, a2, a3, b1, b2, b3]


class TestCommunityDetection:
    def test_returns_mapping_for_all_nodes(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert isinstance(mapping, dict)
        for sid in sids:
            assert sid in mapping

    def test_community_ids_are_ints(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        for cid in mapping.values():
            assert isinstance(cid, int)

    def test_assign_sets_node_attrs(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        for sid in sids:
            assert cg.nx.nodes[sid]["community"] is not None

    def test_community_summary_structure(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        summary = get_community_summary(cg)
        assert isinstance(summary, list)
        assert len(summary) >= 1
        for item in summary:
            assert "community_id" in item
            assert "size" in item
            assert "top_files" in item
            assert "top_symbols" in item

    def test_empty_graph(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert mapping == {}
        summary = get_community_summary(cg)
        assert summary == []

    def test_clusters_get_distinct_community_ids(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        # auth cluster (sids[0..2]) and db cluster (sids[3..5]) should be in different communities
        auth_community = mapping.get(sids[0])
        db_community = mapping.get(sids[3])
        # With dense intra-cluster and no inter-cluster edges, they must differ
        if auth_community is not None and db_community is not None:
            assert auth_community != db_community, (
                f"Expected auth cluster (community {auth_community}) != "
                f"db cluster (community {db_community})"
            )
