"""Tests for DefUseEdge model and DB schema."""
from __future__ import annotations

from pathlib import Path

from trelix.analysis.defuse import DefUseEdge
from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> tuple[Database, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10
        )
    )
    sid = db.insert_symbol(
        Symbol(file_id=fid, name="fn", qualified_name="fn", kind=SymbolKind.FUNCTION,
               line_start=1, line_end=10, signature="def fn()", body="def fn(): x = 1; return x")
    )
    return db, sid


class TestDefUseEdge:
    def test_dataclass_fields(self) -> None:
        edge = DefUseEdge(symbol_id=1, var_name="x", def_line=3, use_line=5, edge_type="def")
        assert edge.symbol_id == 1
        assert edge.var_name == "x"
        assert edge.edge_type == "def"

    def test_edge_type_values(self) -> None:
        for t in ["def", "use"]:
            edge = DefUseEdge(symbol_id=1, var_name="y", def_line=1, use_line=2, edge_type=t)
            assert edge.edge_type == t


class TestDefUseDB:
    def test_insert_and_retrieve_def_use_edges(self, tmp_path: Path) -> None:
        db, sid = _make_db(tmp_path)
        edges = [
            DefUseEdge(symbol_id=sid, var_name="x", def_line=2, use_line=3, edge_type="def"),
            DefUseEdge(symbol_id=sid, var_name="x", def_line=2, use_line=4, edge_type="use"),
        ]
        db.insert_def_use_edges(edges)
        result = db.get_data_flows(sid)
        assert len(result) == 2
        names = {e.var_name for e in result}
        assert "x" in names

    def test_get_data_flows_empty(self, tmp_path: Path) -> None:
        db, sid = _make_db(tmp_path)
        result = db.get_data_flows(sid)
        assert result == []


class TestDataFlowExtractor:
    def test_extracts_simple_assignment(self) -> None:
        from trelix.analysis.defuse import DataFlowExtractor
        from trelix.core.models import Symbol, SymbolKind
        sym = Symbol(id=1, file_id=1, name="fn", qualified_name="fn",
                     kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
                     signature="def fn()", body="def fn():\n    x = 1\n    return x\n")
        extractor = DataFlowExtractor()
        edges = extractor.extract(sym)
        var_names = {e.var_name for e in edges}
        assert "x" in var_names

    def test_returns_empty_for_empty_body(self) -> None:
        from trelix.analysis.defuse import DataFlowExtractor
        from trelix.core.models import Symbol, SymbolKind
        sym = Symbol(id=2, file_id=1, name="fn", qualified_name="fn",
                     kind=SymbolKind.FUNCTION, line_start=1, line_end=2,
                     signature="def fn()", body="")
        extractor = DataFlowExtractor()
        edges = extractor.extract(sym)
        assert edges == []

    def test_never_raises(self) -> None:
        from trelix.analysis.defuse import DataFlowExtractor
        from trelix.core.models import Symbol, SymbolKind
        sym = Symbol(id=3, file_id=1, name="fn", qualified_name="fn",
                     kind=SymbolKind.FUNCTION, line_start=1, line_end=2,
                     signature="def fn()", body="not valid python {{{{")
        extractor = DataFlowExtractor()
        edges = extractor.extract(sym)  # must not raise
        assert isinstance(edges, list)
