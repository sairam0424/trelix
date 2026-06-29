"""Tests for GraphVisualizer — Pyvis HTML export."""

from __future__ import annotations

import json
from pathlib import Path

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuildResult
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities
from trelix.graph.visualizer import GraphVisualizer
from trelix.store.db import Database


def _build_simple_graph(tmp_path: Path) -> tuple[Database, CodeGraph]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10
        )
    )
    sid1 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="fn_a",
            qualified_name="fn_a",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature="def fn_a()",
            body="def fn_a(): pass",
        )
    )
    sid2 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="fn_b",
            qualified_name="fn_b",
            kind=SymbolKind.FUNCTION,
            line_start=7,
            line_end=12,
            signature="def fn_b()",
            body="def fn_b(): pass",
        )
    )
    db.insert_call_edges([CallEdge(caller_id=sid1, callee_name="fn_b", callee_id=sid2, line=3)])
    cg = CodeGraph(db)
    mapping = detect_communities(cg)
    assign_communities(cg, mapping)
    return db, cg


class TestGraphVisualizer:
    def test_export_html_creates_file(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        out = str(tmp_path / "graph.html")
        viz = GraphVisualizer()
        result_path = viz.export_html(cg, out)
        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "<html" in content.lower()

    def test_export_html_max_nodes_truncates(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        out = str(tmp_path / "graph_small.html")
        viz = GraphVisualizer()
        # max_nodes=1 should not crash even when graph has 2 nodes
        result_path = viz.export_html(cg, out, max_nodes=1)
        assert Path(result_path).exists()

    def test_export_html_empty_graph(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        out = str(tmp_path / "empty.html")
        viz = GraphVisualizer()
        result_path = viz.export_html(cg, out)
        assert Path(result_path).exists()

    def test_export_community_report_json(self, tmp_path: Path) -> None:
        _, cg = _build_simple_graph(tmp_path)
        from trelix.graph.community import get_community_summary

        result = GraphBuildResult(
            code_graph=cg,
            community_count=1,
            node_count=cg.node_count,
            edge_count=cg.edge_count,
            concept_count=0,
            elapsed_seconds=0.1,
            community_summary=get_community_summary(cg),
        )
        out = str(tmp_path / "report.json")
        viz = GraphVisualizer()
        report_path = viz.export_community_report(result, out)
        assert Path(report_path).exists()
        data = json.loads(Path(report_path).read_text())
        assert "node_count" in data
        assert "communities" in data
