"""Tests for graph-aware search using CodeGraph traversal."""
from __future__ import annotations

from pathlib import Path

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities
from trelix.graph.search import get_community_context, graph_search
from trelix.store.db import Database


def _build_db(tmp_path: Path) -> tuple[Database, list[int]]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(IndexedFile(path="/r/auth.py", rel_path="auth.py",
                                      language=Language.PYTHON, hash="x", size_bytes=100))
    sids = []
    for name in ["login", "logout", "hash_password", "check_token"]:
        s = Symbol(file_id=fid, name=name, qualified_name=name, kind=SymbolKind.FUNCTION,
                   line_start=1, line_end=5, signature=f"def {name}()", body=f"def {name}(): pass")
        sids.append(db.insert_symbol(s))
        db.insert_chunk_for_symbol(sids[-1], f"def {name}(): pass", 5)
    # login → hash_password → check_token
    db.insert_call_edges([
        CallEdge(caller_id=sids[0], callee_name="hash_password", callee_id=sids[2], line=3),
        CallEdge(caller_id=sids[2], callee_name="check_token", callee_id=sids[3], line=2),
    ])
    return db, sids


class TestGraphSearch:
    def test_graph_search_returns_search_results(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[sids[0]], depth=2, max_results=10)
        assert isinstance(results, list)
        # Should find hash_password and check_token as neighbors
        found_ids = {r.symbol.id for r in results}
        assert sids[2] in found_ids  # hash_password

    def test_graph_search_source_label(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[sids[0]], depth=1, max_results=10)
        for r in results:
            assert r.source == "graph_search"

    def test_graph_search_empty_query(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        results = graph_search(db, cg, query_symbol_ids=[], depth=1, max_results=10)
        assert results == []

    def test_get_community_context(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        # All 4 symbols in one file with edges — likely same community
        community_members = get_community_context(cg, sids[0])
        assert isinstance(community_members, list)
        assert sids[0] in community_members

    def test_rerank_scores_decrease_with_hop_distance(self, tmp_path: Path) -> None:
        db, sids = _build_db(tmp_path)
        cg = CodeGraph(db)
        # login(sids[0]) → hash_password(sids[2]) at hop 1, → check_token(sids[3]) at hop 2
        results = graph_search(db, cg, query_symbol_ids=[sids[0]], depth=2, max_results=10)
        by_id = {r.symbol.id: r.score for r in results}
        # hop-1 neighbor (hash_password) should score higher than hop-2 (check_token)
        if sids[2] in by_id and sids[3] in by_id:
            assert by_id[sids[2]] > by_id[sids[3]]
        # hop-1 score should be exactly 0.5
        if sids[2] in by_id:
            assert abs(by_id[sids[2]] - 0.5) < 0.01
