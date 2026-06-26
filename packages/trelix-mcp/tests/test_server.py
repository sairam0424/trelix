"""Tests for trelix_mcp.server.

Uses unittest.mock.patch to avoid touching real files or embedding models.
"""

from __future__ import annotations

import sys
import types
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
    r.file.language = language
    r.symbol.qualified_name = symbol
    r.symbol.kind = kind
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
    assert {"search_code", "index_codebase", "get_symbol", "blast_radius"} == names, (
        f"Expected exactly 4 tools, got: {names}"
    )


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------

def test_search_code_returns_list_of_dicts() -> None:
    import trelix_mcp.server as srv

    mock_results = [_make_mock_result()]
    mock_ctx = _make_mock_context(mock_results)

    with (
        patch("trelix_mcp.server.IndexConfig") as MockConfig,
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        results = srv.search_code("authentication", "/fake/repo", k=10)

    assert isinstance(results, list)
    assert len(results) == 1
    item = results[0]
    assert set(item.keys()) >= {"file", "symbol", "kind", "lines", "score", "source", "body", "language"}


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
        results = srv.search_code("auth", "/fake/repo", k=3)

    assert len(results) == 3


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
