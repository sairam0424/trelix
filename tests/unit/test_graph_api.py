"""
Unit tests for Retriever.get_callers / get_callees / get_importers.

All db calls are mocked — no real SQLite required.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.retriever import Retriever

# Use the repo root as a valid path for IndexConfig (path validation requires existence)
_REPO_ROOT = str(Path(__file__).parent.parent.parent)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config() -> IndexConfig:
    return IndexConfig(
        repo_path=_REPO_ROOT,
        embedder=EmbedderConfig(provider="local"),
    )


def _make_file(file_id: int = 1, rel_path: str = "src/foo.py") -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{file_id}",
        size_bytes=1000,
        id=file_id,
        indexed_at=datetime(2024, 1, 1),
    )


def _make_symbol(
    sym_id: int = 1,
    file_id: int = 1,
    name: str = "my_func",
    line_start: int = 10,
) -> Symbol:
    return Symbol(
        file_id=file_id,
        name=name,
        kind=SymbolKind.FUNCTION,
        line_start=line_start,
        line_end=line_start + 5,
        qualified_name=name,
        signature=f"def {name}()",
        id=sym_id,
        body=f"def {name}(): pass",
    )


def _make_chunk(symbol_id: int = 1) -> Chunk:
    return Chunk(
        symbol_id=symbol_id,
        chunk_text="def my_func(): pass",
        token_count=10,
    )


def _make_retriever_with_mock_db() -> tuple[Retriever, MagicMock]:
    """
    Instantiate Retriever without calling __init__ infrastructure,
    then inject a mock db attribute.
    """
    config = _make_config()
    retriever = Retriever.__new__(Retriever)
    retriever.config = config
    mock_db = MagicMock()
    retriever.db = mock_db
    return retriever, mock_db


# ---------------------------------------------------------------------------
# _hydrate_symbol_id
# ---------------------------------------------------------------------------


class TestHydrateSymbolId:
    def test_returns_search_result_when_symbol_exists(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        sym = _make_symbol(sym_id=5)
        f = _make_file(file_id=1)
        chunk = _make_chunk(symbol_id=5)
        mock_db.get_symbol_with_file.return_value = (sym, f)
        mock_db.get_first_chunk_for_symbol.return_value = chunk

        result = retriever._hydrate_symbol_id(5, "test_source")

        assert result is not None
        assert isinstance(result, SearchResult)
        assert result.score == 1.0
        assert result.source == "test_source"
        assert result.symbol.id == 5

    def test_returns_none_for_stale_id(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        mock_db.get_symbol_with_file.return_value = None

        result = retriever._hydrate_symbol_id(999, "test_source")

        assert result is None

    def test_falls_back_to_body_when_no_chunk(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        sym = _make_symbol(sym_id=5)
        f = _make_file(file_id=1)
        mock_db.get_symbol_with_file.return_value = (sym, f)
        mock_db.get_first_chunk_for_symbol.return_value = None

        result = retriever._hydrate_symbol_id(5, "test_source")

        assert result is not None
        assert result.chunk.chunk_text == sym.body[:2000]
        assert result.chunk.token_count == 0


# ---------------------------------------------------------------------------
# get_callers
# ---------------------------------------------------------------------------


class TestGetCallers:
    def test_returns_two_search_results(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        target_sym = _make_symbol(sym_id=10, name="foo")
        caller_a = _make_symbol(sym_id=20, file_id=1, name="caller_a", line_start=1)
        caller_b = _make_symbol(sym_id=30, file_id=2, name="caller_b", line_start=5)
        file_a = _make_file(file_id=1, rel_path="src/a.py")
        file_b = _make_file(file_id=2, rel_path="src/b.py")

        mock_db.get_symbol_by_name.return_value = [target_sym]
        mock_db.get_callers.return_value = [20, 30]
        mock_db.get_symbol_with_file.side_effect = lambda sid: {
            20: (caller_a, file_a),
            30: (caller_b, file_b),
        }.get(sid)
        mock_db.get_first_chunk_for_symbol.side_effect = lambda sid: _make_chunk(sid)

        results = retriever.get_callers("foo")

        assert len(results) == 2
        assert all(r.source == "graph_callers" for r in results)
        assert all(r.score == 1.0 for r in results)

    def test_rank_is_one_indexed_and_contiguous(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        target_sym = _make_symbol(sym_id=10, name="foo")
        caller_a = _make_symbol(sym_id=20, file_id=1, name="a", line_start=1)
        caller_b = _make_symbol(sym_id=30, file_id=2, name="b", line_start=1)
        file_a = _make_file(file_id=1, rel_path="src/a.py")
        file_b = _make_file(file_id=2, rel_path="src/b.py")

        mock_db.get_symbol_by_name.return_value = [target_sym]
        mock_db.get_callers.return_value = [20, 30]
        mock_db.get_symbol_with_file.side_effect = lambda sid: {
            20: (caller_a, file_a),
            30: (caller_b, file_b),
        }.get(sid)
        mock_db.get_first_chunk_for_symbol.side_effect = lambda sid: _make_chunk(sid)

        results = retriever.get_callers("foo")

        ranks = sorted(r.rank for r in results)
        assert ranks == list(range(1, len(results) + 1))

    def test_unknown_symbol_returns_empty(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        mock_db.get_symbol_by_name.return_value = []

        results = retriever.get_callers("nonexistent")

        assert results == []
        mock_db.get_callers.assert_not_called()

    def test_deduplicates_callers_across_overloads(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        # Two overloads of the target symbol
        overload_a = _make_symbol(sym_id=10, name="foo")
        overload_b = _make_symbol(sym_id=11, name="foo")
        # Both overloads are called by the same caller (id=20)
        caller = _make_symbol(sym_id=20, file_id=1, name="shared_caller")
        file_a = _make_file(file_id=1, rel_path="src/a.py")

        mock_db.get_symbol_by_name.return_value = [overload_a, overload_b]
        mock_db.get_callers.side_effect = lambda sid: [20]  # both return same caller
        mock_db.get_symbol_with_file.return_value = (caller, file_a)
        mock_db.get_first_chunk_for_symbol.return_value = _make_chunk(20)

        results = retriever.get_callers("foo")

        # Deduplicated — only one result despite two overloads
        assert len(results) == 1

    def test_skips_stale_caller_ids(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        target_sym = _make_symbol(sym_id=10, name="foo")
        valid_caller = _make_symbol(sym_id=20, file_id=1, name="valid")
        file_a = _make_file(file_id=1, rel_path="src/a.py")

        mock_db.get_symbol_by_name.return_value = [target_sym]
        mock_db.get_callers.return_value = [20, 999]  # 999 is stale
        mock_db.get_symbol_with_file.side_effect = lambda sid: {
            20: (valid_caller, file_a),
            999: None,  # stale
        }.get(sid)
        mock_db.get_first_chunk_for_symbol.side_effect = lambda sid: _make_chunk(sid)

        results = retriever.get_callers("foo")

        # Only the valid caller is returned
        assert len(results) == 1
        assert results[0].symbol.id == 20


# ---------------------------------------------------------------------------
# get_callees
# ---------------------------------------------------------------------------


class TestGetCallees:
    def test_returns_search_results(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        target_sym = _make_symbol(sym_id=10, name="foo")
        callee_a = _make_symbol(sym_id=40, file_id=1, name="callee_a")
        file_a = _make_file(file_id=1, rel_path="src/a.py")

        mock_db.get_symbol_by_name.return_value = [target_sym]
        mock_db.get_callees.return_value = [40]
        mock_db.get_symbol_with_file.return_value = (callee_a, file_a)
        mock_db.get_first_chunk_for_symbol.return_value = _make_chunk(40)

        results = retriever.get_callees("foo")

        assert len(results) == 1
        assert results[0].source == "graph_callees"
        assert results[0].score == 1.0

    def test_no_resolved_edges_returns_empty(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        target_sym = _make_symbol(sym_id=10, name="foo")

        mock_db.get_symbol_by_name.return_value = [target_sym]
        mock_db.get_callees.return_value = []  # all calls were to external libs

        results = retriever.get_callees("foo")

        assert results == []

    def test_unknown_symbol_returns_empty(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        mock_db.get_symbol_by_name.return_value = []

        results = retriever.get_callees("nonexistent")

        assert results == []


# ---------------------------------------------------------------------------
# get_importers
# ---------------------------------------------------------------------------


class TestGetImporters:
    def test_returns_search_results(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        importer_sym_a = _make_symbol(sym_id=40, file_id=7, name="func_a", line_start=10)
        importer_sym_b = _make_symbol(sym_id=50, file_id=8, name="func_b", line_start=5)
        file_7 = _make_file(file_id=7, rel_path="src/cli/main.py")
        file_8 = _make_file(file_id=8, rel_path="tests/test_foo.py")

        mock_db.get_file_by_rel_path_suffix.return_value = 5
        mock_db.get_files_importing.return_value = [7, 8]
        mock_db.get_symbols_for_file.side_effect = lambda fid: {
            7: [importer_sym_a],
            8: [importer_sym_b],
        }.get(fid, [])
        mock_db.get_symbol_with_file.side_effect = lambda sid: {
            40: (importer_sym_a, file_7),
            50: (importer_sym_b, file_8),
        }.get(sid)
        mock_db.get_first_chunk_for_symbol.side_effect = lambda sid: _make_chunk(sid)

        results = retriever.get_importers("retriever")

        assert len(results) == 2
        assert all(r.source == "graph_importers" for r in results)
        assert all(r.score == 1.0 for r in results)

    def test_unknown_module_returns_empty(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        mock_db.get_file_by_rel_path_suffix.return_value = None

        results = retriever.get_importers("nomodule")

        assert results == []
        mock_db.get_files_importing.assert_not_called()

    def test_picks_lowest_line_start_symbol_per_file(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        # Two symbols in file 7 — only the first (lower line_start) should be returned
        sym_early = _make_symbol(sym_id=40, file_id=7, name="early", line_start=3)
        sym_late = _make_symbol(sym_id=41, file_id=7, name="late", line_start=100)
        file_7 = _make_file(file_id=7, rel_path="src/a.py")

        mock_db.get_file_by_rel_path_suffix.return_value = 5
        mock_db.get_files_importing.return_value = [7]
        mock_db.get_symbols_for_file.return_value = [sym_late, sym_early]  # unsorted input
        mock_db.get_symbol_with_file.return_value = (sym_early, file_7)
        mock_db.get_first_chunk_for_symbol.return_value = _make_chunk(40)

        results = retriever.get_importers("mod")

        assert len(results) == 1
        assert results[0].symbol.name == "early"

    def test_skips_file_with_no_symbols(self) -> None:
        retriever, mock_db = _make_retriever_with_mock_db()
        sym = _make_symbol(sym_id=40, file_id=7, name="f", line_start=1)
        file_7 = _make_file(file_id=7, rel_path="src/a.py")

        mock_db.get_file_by_rel_path_suffix.return_value = 5
        mock_db.get_files_importing.return_value = [7, 9]  # file 9 has no symbols
        mock_db.get_symbols_for_file.side_effect = lambda fid: {
            7: [sym],
            9: [],  # empty — should be silently skipped
        }.get(fid, [])
        mock_db.get_symbol_with_file.return_value = (sym, file_7)
        mock_db.get_first_chunk_for_symbol.return_value = _make_chunk(40)

        results = retriever.get_importers("mod")

        assert len(results) == 1


# ---------------------------------------------------------------------------
# CLI: trelix graph
# ---------------------------------------------------------------------------

from typer.testing import CliRunner  # noqa: E402

from trelix.cli.main import app as trelix_app  # noqa: E402


class TestGraphCLI:
    def test_graph_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(trelix_app, ["call-graph", "--help"])
        assert result.exit_code == 0
        assert "symbol" in result.output.lower()

    def test_graph_missing_args_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(trelix_app, ["call-graph"])
        # Missing required positional args — typer exits 2
        assert result.exit_code != 0

    def test_graph_bad_repo_exits_one(self, tmp_path) -> None:
        runner = CliRunner()
        # "call-graph" is the renamed command that takes repo + symbol args
        result = runner.invoke(
            trelix_app,
            ["call-graph", str(tmp_path / "does_not_exist"), "some_symbol"],
        )
        assert result.exit_code == 1
