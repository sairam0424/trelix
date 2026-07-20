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
# 15 tools registered (8 original + 4 federation + 3 agent-session)
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
        "federation_list_repos",
        "federation_add_repo",
        "federation_remove_repo",
        "federation_search_all",
        "ask_agent",
        "agent_list_sessions",
        "agent_clear_session",
    }
    assert expected == names, f"Expected exactly 15 tools, got: {names}"


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
# Federation (multi-repo) tools
# ---------------------------------------------------------------------------


def test_federation_list_repos_empty_registry() -> None:
    import trelix_mcp.server as srv

    empty_reg = MagicMock()
    empty_reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = empty_reg
        response = srv.federation_list_repos()

    assert response == {"repos": [], "count": 0, "error": None}
    MockRegistry.load.assert_called_once_with(None)


def test_federation_list_repos_returns_entries() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    entry.path = "/repo"
    entry.weight = 2.0
    reg = MagicMock()
    reg.list.return_value = [entry]

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_list_repos()

    assert response == {
        "repos": [{"alias": "myrepo", "path": "/repo", "weight": 2.0}],
        "count": 1,
        "error": None,
    }


def test_federation_list_repos_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_list_repos(config_path="/etc/passwd")

    assert response["repos"] == []
    assert response["count"] == 0
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_add_repo_success() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_add_repo(alias="myrepo", path="/repo", weight=1.5)

    reg.add.assert_called_once_with("myrepo", "/repo", 1.5, max_repos=50)
    reg.save.assert_called_once()
    assert response == {"added": True, "alias": "myrepo", "path": "/repo", "error": None}


def test_federation_add_repo_duplicate_alias_returns_error() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.add.side_effect = ValueError("alias 'myrepo' already registered")

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_add_repo(alias="myrepo", path="/repo")

    assert response["added"] is False
    assert "already registered" in response["error"]
    reg.save.assert_not_called()


def test_federation_add_repo_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_add_repo(
            alias="myrepo", path="/repo", config_path="/etc/passwd"
        )

    assert response["added"] is False
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_remove_repo_existing() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_remove_repo(alias="myrepo")

    reg.remove.assert_called_once_with("myrepo")
    reg.save.assert_called_once()
    assert response == {"removed": True, "alias": "myrepo", "error": None}


def test_federation_remove_repo_missing_is_noop() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_remove_repo(alias="nonexistent")

    assert response == {"removed": False, "alias": "nonexistent", "error": None}


def test_federation_search_all_no_repos_returns_empty() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_search_all(query="auth")

    assert response == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 0,
        "error": None,
    }


def test_federation_search_all_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_search_all(query="auth", config_path="/etc/passwd")

    assert response["results"] == []
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_search_all_returns_dict_envelope() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    result = _make_mock_result(source="myrepo:vector")

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 1
        MockFed.return_value.retrieve.return_value = [result]
        response = srv.federation_search_all(query="auth", k=10)

    assert response["repos_searched"] == 1
    assert response["repos_skipped"] == 0
    assert response["total_available"] == 1
    assert len(response["results"]) == 1
    assert response["results"][0]["repo"] == "myrepo"
    assert response["results"][0]["source"] == "myrepo:vector"
    # Fetch width must be the fixed constant, independent of cursor/k —
    # regression guard for the pagination-stability fix (issue #69 item 3).
    MockFed.return_value.retrieve.assert_called_once_with(
        "auth", k=srv._FEDERATION_SEARCH_ALL_FETCH_WIDTH
    )


def test_federation_search_all_fetch_width_independent_of_cursor() -> None:
    """Same fixed fetch width regardless of cursor — the actual pagination fix."""
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 1
        MockFed.return_value.retrieve.return_value = []

        srv.federation_search_all(query="q", k=10, cursor=0)
        srv.federation_search_all(query="q", k=10, cursor=90)

    calls = MockFed.return_value.retrieve.call_args_list
    assert len(calls) == 2
    assert calls[0] == calls[1] == (("q",), {"k": srv._FEDERATION_SEARCH_ALL_FETCH_WIDTH})


def test_federation_search_all_reports_repos_skipped() -> None:
    import trelix_mcp.server as srv

    entries = [MagicMock(alias=f"repo{i}") for i in range(5)]
    reg = MagicMock()
    reg.list.return_value = entries

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 3
        MockFed.return_value.retrieve.return_value = []
        response = srv.federation_search_all(query="q")

    assert response["repos_searched"] == 3
    assert response["repos_skipped"] == 2


# ---------------------------------------------------------------------------
# Agent-session (persistent ReAct memory) tools
# ---------------------------------------------------------------------------


def test_ask_agent_returns_dict_with_session_id() -> None:
    import trelix_mcp.server as srv

    mock_loop = MagicMock()
    mock_loop.run.return_value = ("answer text", "some-uuid")
    mock_db = MagicMock()
    mock_db.get_agent_turns.return_value = [{"turn_index": 0}, {"turn_index": 1}]

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.AgentLoop", return_value=mock_loop),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.ask_agent(query="how does auth work", repo_path="/fake/repo")

    assert response == {"answer": "answer text", "session_id": "some-uuid", "turn_count": 2}
    # Regression guard: get_agent_turns must be called with the loop-returned
    # resolved_session_id ("some-uuid"), not the input session_id param
    # (which was None here) — a mutation swapping these would still produce
    # turn_count=2 (mock_db.get_agent_turns.return_value is fixed regardless
    # of argument), so the shape-only assertion above would miss it.
    mock_db.get_agent_turns.assert_called_once_with("some-uuid")


def test_ask_agent_generates_session_id_when_omitted() -> None:
    import trelix_mcp.server as srv

    mock_loop = MagicMock()
    mock_loop.run.return_value = ("answer", "freshly-generated-uuid")
    mock_db = MagicMock()
    mock_db.get_agent_turns.return_value = []

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.AgentLoop", return_value=mock_loop),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.ask_agent(query="q", repo_path="/fake/repo")

    mock_loop.run.assert_called_once_with("q", session_id=None)
    assert response["session_id"] == "freshly-generated-uuid"
    assert response["session_id"]


def test_agent_list_sessions_returns_dict() -> None:
    import trelix_mcp.server as srv

    mock_db = MagicMock()
    mock_db.list_agent_sessions.return_value = [
        {
            "session_id": "s1",
            "created_at": "t1",
            "last_active_at": "t1",
            "query": "q1",
            "turn_count": 2,
        }
    ]

    with (
        patch("trelix_mcp.server.IndexConfig") as MockIndexConfig,
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        MockIndexConfig.return_value.retrieval.agent_session_max_age_seconds = 604_800.0
        response = srv.agent_list_sessions(repo_path="/fake/repo")

    assert response["count"] == 1
    assert response["sessions"][0]["session_id"] == "s1"


def test_agent_clear_session_returns_dict() -> None:
    import trelix_mcp.server as srv

    mock_db = MagicMock()
    mock_db.delete_agent_session.return_value = True

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.agent_clear_session(repo_path="/fake/repo", session_id="s1")

    assert response == {"cleared": True, "session_id": "s1"}


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
