"""Tests for PageRank symbol importance scoring."""
from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import compute_pagerank
from trelix.graph.persistence import get_top_central_symbols, save_graph_metadata
from trelix.store.db import Database


def _build_star_graph(tmp_path: Path) -> tuple[Database, CodeGraph, int]:
    """Build a star graph: hub calls 3 leaves. Hub should have highest PageRank."""
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10)
    )
    hub = db.insert_symbol(Symbol(file_id=fid, name="hub", qualified_name="hub",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=5, signature="def hub()", body=""))
    leaf1 = db.insert_symbol(Symbol(file_id=fid, name="leaf1", qualified_name="leaf1",
        kind=SymbolKind.FUNCTION, line_start=10, line_end=14, signature="def leaf1()", body=""))
    leaf2 = db.insert_symbol(Symbol(file_id=fid, name="leaf2", qualified_name="leaf2",
        kind=SymbolKind.FUNCTION, line_start=20, line_end=24, signature="def leaf2()", body=""))
    leaf3 = db.insert_symbol(Symbol(file_id=fid, name="leaf3", qualified_name="leaf3",
        kind=SymbolKind.FUNCTION, line_start=30, line_end=34, signature="def leaf3()", body=""))
    # leaf1, leaf2, leaf3 all call hub (hub is the target, gets PageRank from incoming)
    db.insert_call_edges([
        CallEdge(caller_id=leaf1, callee_name="hub", callee_id=hub, line=11),
        CallEdge(caller_id=leaf2, callee_name="hub", callee_id=hub, line=21),
        CallEdge(caller_id=leaf3, callee_name="hub", callee_id=hub, line=31),
    ])
    cg = CodeGraph(db)
    return db, cg, hub


class TestComputePagerank:
    def test_returns_dict_of_node_scores(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())

    def test_hub_has_higher_score_than_leaves(self, tmp_path: Path) -> None:
        _, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        hub_score = scores.get(hub_id, 0.0)
        leaf_scores = [v for k, v in scores.items() if k != hub_id]
        assert hub_score > max(leaf_scores, default=0.0)

    def test_scores_normalized_0_to_1(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert max(scores.values()) <= 1.0 + 1e-9
        assert min(scores.values()) >= 0.0 - 1e-9

    def test_empty_graph_returns_empty_dict(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        assert compute_pagerank(cg) == {}


class TestGetTopCentralSymbols:
    def test_returns_sorted_by_centrality(self, tmp_path: Path) -> None:
        db, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        # Assign centrality scores to graph nodes
        for node_id, score in scores.items():
            cg.nx.nodes[node_id]["centrality"] = score
        save_graph_metadata(db, cg)
        top = get_top_central_symbols(db, top_n=1)
        assert len(top) == 1
        assert top[0] == hub_id
