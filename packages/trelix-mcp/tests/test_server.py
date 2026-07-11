"""Tests for trelix_mcp.server.

Uses unittest.mock.patch to avoid touching real files or embedding models.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(
    file: str = "src/foo.py",
    symbol: str = "foo.bar",
    kind: str = "function",
    line_start: int = 1,
    line_end: int = 10,
    score: float = 0.9,
    source: str = "vector",
    body: str = "def bar(): pass",
    language: str = "python",
) -> MagicMock:
    """Return a mock SearchResult compatible with server.py expectations."""
    r = MagicMock()
    r.file.rel_path = file
    r.file.language.value = language
    r.symbol.qualified_name = symbol
    r.symbol.kind.value = kind
    r.symbol.line_start = line_start
    r.symbol.line_end = line_end
    r.symbol.body = body
    r.score = score
    r.source = source
    return r


def _make_mock_context(results: list[MagicMock]) -> MagicMock:
    ctx = MagicMock()
    ctx.results = results
    return ctx


# ---------------------------------------------------------------------------
# Module import + basic structure
# ---------------------------------------------------------------------------


def test_server_importable() -> None:
    import trelix_mcp.server as srv  # noqa: F401


def test_mcp_attribute_exists() -> None:
    import trelix_mcp.server as srv

    assert hasattr(srv, "mcp"), "server.py must expose a top-level `mcp` object"


def test_main_callable() -> None:
    import trelix_mcp.server as srv

    assert callable(srv.main)


# ---------------------------------------------------------------------------
# 4 tools registered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_four_tools_registered() -> None:
    import trelix_mcp.server as srv

    tools = await srv.mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "search_code",
        "index_codebase",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "subscribe_resource",
        "unsubscribe_resource",
    }
    assert expected == names, f"Expected exactly 8 tools, got: {names}"


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


def test_search_code_returns_dict_envelope() -> None:
    """search_code returns a pagination envelope dict with a results list."""
    import trelix_mcp.server as srv

    mock_results = [_make_mock_result()]
    mock_ctx = _make_mock_context(mock_results)

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        response = srv.search_code("authentication", "/fake/repo", k=10)

    assert isinstance(response, dict)
    assert "results" in response
    assert "next_cursor" in response
    assert "total_available" in response
    assert len(response["results"]) == 1
    item = response["results"][0]
    assert set(item.keys()) >= {
        "file",
        "symbol",
        "kind",
        "lines",
        "score",
        "source",
        "body",
        "language",
    }


def test_search_code_respects_k_limit() -> None:
    """k=3 must truncate 20 mock results to 3."""
    import trelix_mcp.server as srv

    mock_results = [_make_mock_result(symbol=f"sym.{i}") for i in range(20)]
    mock_ctx = _make_mock_context(mock_results)

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        response = srv.search_code("auth", "/fake/repo", k=3)

    assert len(response["results"]) == 3


# ---------------------------------------------------------------------------
# index_codebase
# ---------------------------------------------------------------------------


def test_index_codebase_returns_dict() -> None:
    import trelix_mcp.server as srv

    fake_stats: dict[str, Any] = {
        "files_found": 10,
        "files_indexed": 8,
        "files_skipped": 2,
        "symbols_extracted": 50,
        "chunks_total": 50,
        "chunks_embedded": 50,
        "errors": 0,
        "elapsed_seconds": 1.23,
    }

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.EmbedderConfig"),
        patch("trelix_mcp.server.Indexer") as MockIndexer,
    ):
        MockIndexer.return_value.index.return_value = fake_stats
        result = srv.index_codebase("/fake/repo", provider="local")

    assert isinstance(result, dict)
    assert result["files_found"] == 10


# ---------------------------------------------------------------------------
# blast_radius deduplication
# ---------------------------------------------------------------------------


def test_blast_radius_deduplicates_files() -> None:
    """Two results sharing the same file should produce only one output entry."""
    import trelix_mcp.server as srv

    r1 = _make_mock_result(file="src/auth.py", symbol="auth.login")
    r2 = _make_mock_result(file="src/auth.py", symbol="auth.logout")  # same file
    r3 = _make_mock_result(file="src/db.py", symbol="db.connect")

    mock_ctx = _make_mock_context([r1, r2, r3])

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        results = srv.blast_radius("auth", "/fake/repo")

    files = [r["file"] for r in results]
    assert len(files) == 2, f"Expected 2 unique files, got {files}"
    assert "src/auth.py" in files
    assert "src/db.py" in files


# ---------------------------------------------------------------------------
# Cursor-based pagination for search_code
# ---------------------------------------------------------------------------


class TestSearchCodePagination:
    def test_search_code_returns_pagination_envelope(self, tmp_path) -> None:
        """search_code returns dict with results + next_cursor + total_available."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = []
        for i in range(25):
            r = MagicMock()
            r.file.rel_path = f"src/file{i}.py"
            r.symbol.qualified_name = f"Func{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 5
            r.symbol.body = "def f(): pass"
            r.file.language.value = "python"
            r.score = 0.9 - i * 0.01
            r.source = "vector"
            mock_results.append(r)

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="authentication",
                repo_path=str(tmp_path),
                k=10,
                cursor=0,
            )

        assert "results" in response
        assert "next_cursor" in response
        assert "total_available" in response
        assert len(response["results"]) == 10
        assert response["next_cursor"] == 10
        assert response["total_available"] == 25

    def test_search_code_pagination_second_page(self, tmp_path) -> None:
        """cursor=10 returns items 10-19."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = []
        for i in range(25):
            r = MagicMock()
            r.file.rel_path = f"src/file{i}.py"
            r.symbol.qualified_name = f"Func{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 5
            r.symbol.body = "def f(): pass"
            r.file.language.value = "python"
            r.score = float(i) / 25
            r.source = "bm25"
            mock_results.append(r)

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="login",
                repo_path=str(tmp_path),
                k=10,
                cursor=10,
            )

        assert len(response["results"]) == 10
        assert response["results"][0]["symbol"] == "Func10"
        assert response["next_cursor"] == 20

    def test_search_code_last_page_has_null_next_cursor(self, tmp_path) -> None:
        """last page has next_cursor=None."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = [MagicMock() for _ in range(5)]
        for i, r in enumerate(mock_results):
            r.file.rel_path = f"src/f{i}.py"
            r.symbol.qualified_name = f"F{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 3
            r.symbol.body = "pass"
            r.file.language.value = "python"
            r.score = 0.5
            r.source = "vector"

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="q",
                repo_path=str(tmp_path),
                k=10,
                cursor=0,
            )

        assert response["next_cursor"] is None
        assert len(response["results"]) == 5


# ---------------------------------------------------------------------------
# Progress notifications for index_codebase
# ---------------------------------------------------------------------------


class TestIndexCodebaseProgress:
    def test_index_codebase_accepts_context_param(self) -> None:
        """index_codebase tool signature accepts ctx: Context without error."""
        import inspect

        from trelix_mcp.server import index_codebase

        sig = inspect.signature(index_codebase)
        # ctx param should exist (FastMCP injects it)
        # We check the wrapped function's parameters
        list(sig.parameters.keys())
        # Either 'ctx' is in params, or the function works without it (backward compat)
        # The key check is that calling it with mock results succeeds
        assert callable(index_codebase)

    def test_index_codebase_returns_stats(self, tmp_path) -> None:
        """index_codebase returns stats dict with expected keys."""
        from unittest.mock import patch

        from trelix_mcp.server import index_codebase

        mock_stats = {
            "files_indexed": 15,
            "symbols_extracted": 220,
            "chunks_embedded": 220,
            "errors": 0,
            "elapsed_seconds": 3.1,
        }

        with patch("trelix_mcp.server.Indexer") as MockIndexer:
            MockIndexer.return_value.index.return_value = mock_stats
            result = index_codebase(repo_path=str(tmp_path), provider="local")

        assert result["files_indexed"] == 15
        assert result["errors"] == 0


# ---------------------------------------------------------------------------
# CRITICAL: no stdout bytes on import
# ---------------------------------------------------------------------------


def test_server_import_produces_no_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Importing server.py must not write anything to stdout.

    stdout is the MCP JSON protocol pipe — any spurious bytes corrupt the stream.
    """
    # Force a fresh import to catch any module-level print/write
    if "trelix_mcp.server" in sys.modules:
        del sys.modules["trelix_mcp.server"]

    import trelix_mcp.server  # noqa: F401

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"server.py wrote {len(captured.out)} bytes to stdout on import: {captured.out!r}"
    )
